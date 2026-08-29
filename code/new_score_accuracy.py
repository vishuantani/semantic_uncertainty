'''Stage 3: correctness scoring.

NEW FILE - no upstream equivalent. This logic lived inside generate.py /
generate_improved.py as GenerationExperiment.score_against_references.

CHANGED FROM UPSTREAM - why this is a separate stage:
Upstream scored correctness during generation, which is before
clean_generated_strings runs. ROUGE therefore compared the raw beam output - a
~171-word repetition loop - against a one-to-three-word reference answer, so
rougeL_to_target was ~0 for every question, `correct` was 0/40, and every AUROC
in analyze_results.py came out nan. Scoring after cleaning fixes that.

CHANGED FROM UPSTREAM - both variants are stored:
  *_raw      scored against `most_likely_generation`         (upstream behaviour)
  *_cleaned  scored against `cleaned_most_likely_generation` (new)
`--score_on` decides which pair is aliased onto the bare keys. Keeping both means
the delta is reportable, which matters because the cleaned number is no longer
directly comparable to the paper.

SCOPE OF `--score_on raw` - read this before quoting a reproduction. The *_raw
scoring path is byte-faithful to upstream, so it reproduces upstream ACCURACY
(exact_match, rouge*_to_target and everything analyze_results.py derives from
them) and nothing more. It does NOT give you an upstream run end to end:
get_likelihoods_improved.py._sequence_ids always prefers the cleaned ids once
cleaning has been run and offers no raw switch, so every likelihood-derived
measure (average_neg_log_likelihood_of_{most,second_most}_likely_gen, the margin
measure, predictive entropy and semantic entropy) reflects whatever the cleaning
stage produced. `--score_on raw` on a cleaned pickle is therefore a MIXED
configuration - upstream accuracy against cleaned-id likelihoods - reproducing
neither side exactly. A full upstream reproduction additionally requires running
the likelihoods stage over an UNCLEANED generations pickle, where _sequence_ids
takes its raw fallback path and matches upstream exactly, padding included.

The scores are written back into the generations pickle in place, under the same
bare key names upstream used, so analyze_results.py keeps working unmodified.

Usage:
    python code/new_score_accuracy.py --generation_model=opt-350m --run_id=run_1
'''

import argparse
import os
import pickle

import evaluate
import numpy as np
import wandb

import config

os.environ['HF_DATASETS_CACHE'] = config.hf_datasets_cache

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
ALIAS_KEYS = ['exact_match'] + [r + '_to_target' for r in ROUGE_TYPES]
VARIANTS = {'raw': 'most_likely_generation',
            'cleaned': 'cleaned_most_likely_generation'}


def reference_answers(sample, dataset):
    """The gold answers to score against.

    CHANGED FROM UPSTREAM: upstream's GenerationExperiment._reference_answers read
    the live dataset batch, where coqa answers are a dict and additional_answers is
    nested. This stage reads the PICKLED sample instead, and generation already
    flattened both (generate_improved.py:405-408) - `answer` is stored as
    batch['answer']['text'] and additional_answers as [x[0] for x in ...]. Re-applying
    those transforms here would raise TypeError on the dict access and would take the
    first character of each additional answer. So both are consumed flat.
    """
    if dataset == 'coqa':
        return list(sample['answer']) + list(sample['additional_answers'])
    return list(sample['answer'])


def check_dataset_matches(sequences, dataset):
    """Fail loudly when --dataset disagrees with the run that generated the pickle.

    NEW - NO UPSTREAM EQUIVALENT. Upstream never needed this: scoring happened
    inside generate.py, where the dataset was the one argument that had just built
    the dataloader, so a mismatch was impossible. Now that scoring is its own stage
    reading a pickle, --dataset is supplied independently and defaults to
    trivia_qa. Scoring a coqa pickle without --dataset coqa would silently drop
    `additional_answers` from reference_answers(), giving a strictly smaller
    reference set, a lower max-over-references score, and a quietly understated
    accuracy with no error anywhere.

    generate_improved.py writes `additional_answers` as None for trivia_qa and as a
    list for coqa (see its run()), which makes it a reliable dataset signal.
    """
    has_additional = any(sample.get('additional_answers') is not None
                         for sample in sequences)
    pickle_dataset = 'coqa' if has_additional else 'trivia_qa'

    if pickle_dataset != dataset:
        raise ValueError(
            f'--dataset={dataset} but this generations pickle looks like '
            f'{pickle_dataset}: additional_answers is '
            f'{"present" if has_additional else "None"} on its samples. '
            f'Scoring with the wrong --dataset changes the reference set and '
            f'silently shifts accuracy, so re-run with --dataset={pickle_dataset}.')


def apply_aliases(sample, score_on):
    """Copy the chosen variant onto the bare keys analyze_results.py reads.

    NEW - NO UPSTREAM EQUIVALENT. Upstream wrote the bare keys directly because
    it only ever computed one variant.
    """
    for key in ALIAS_KEYS:
        sample[key] = sample[f'{key}_{score_on}']
    return sample


class AccuracyScorer:
    def __init__(self, args):
        self.args = args
        self.rouge = evaluate.load('rouge')
        self.exact_match_metric = evaluate.load('exact_match')

    def _path(self, run_name):
        return (f'{config.output_dir}/sequences/{run_name}/'
                f'{self.args.generation_model}_generations.pkl')

    def load(self, run_name):
        with open(self._path(run_name), 'rb') as infile:
            return pickle.load(infile)

    def score_one(self, sample, variant, text_key):
        """Score `sample[text_key]` against every reference, keeping the best.

        The max-over-references and the .lstrip() are both upstream behaviour,
        preserved so *_raw reproduces the original number exactly.
        """
        best = {'exact_match': 0.0}
        for rouge_type in ROUGE_TYPES:
            best[rouge_type + '_to_target'] = 0.0

        prediction = [sample[text_key].lstrip()]
        for answer in reference_answers(sample, self.args.dataset):
            results = self.exact_match_metric.compute(predictions=prediction,
                                                      references=[answer],
                                                      ignore_case=True,
                                                      ignore_punctuation=True)
            best['exact_match'] = max(results['exact_match'], best['exact_match'])

            rouge_results = self.rouge.compute(predictions=prediction, references=[answer])
            for rouge_type in ROUGE_TYPES:
                key = rouge_type + '_to_target'
                best[key] = max(rouge_results[rouge_type], best[key])

        for key, value in best.items():
            sample[f'{key}_{variant}'] = float(value)
        return sample

    def score(self, sequences):
        # NEW - NO UPSTREAM EQUIVALENT: guard against --dataset disagreeing with the
        # generating run before any scoring happens. See check_dataset_matches.
        check_dataset_matches(sequences, self.args.dataset)
        for sample in sequences:
            for variant, text_key in VARIANTS.items():
                if text_key not in sample:
                    raise KeyError(
                        f'{text_key!r} missing - run clean_generated_strings_improved.py '
                        'before this stage.')
                self.score_one(sample, variant, text_key)
            apply_aliases(sample, self.args.score_on)
        return sequences

    def save(self, sequences, run_name):
        path = self._path(run_name)
        with open(path, 'wb') as outfile:
            pickle.dump(sequences, outfile)
        return path

    def run(self, run_name):
        sequences = self.score(self.load(run_name))
        return sequences, self.save(sequences, run_name)

    @staticmethod
    def summarise(sequences, score_on):
        """Aggregates for W&B. The raw/cleaned gap is the headline number."""
        summary = {'n_questions': len(sequences), 'score_on': score_on}
        for variant in VARIANTS:
            rouge_l = np.array([s[f'rougeL_to_target_{variant}'] for s in sequences])
            summary[f'accuracy_{variant}'] = float((rouge_l > 0.3).mean())
            summary[f'n_correct_{variant}'] = int((rouge_l > 0.3).sum())
            summary[f'mean_rougeL_{variant}'] = float(rouge_l.mean())
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generation_model', type=str, default='opt-350m')
    parser.add_argument('--run_id', type=str, default='run_1')
    parser.add_argument('--dataset', type=str, default='trivia_qa',
                        choices=['trivia_qa', 'coqa'])
    parser.add_argument('--score_on', type=str, default='cleaned',
                        choices=['cleaned', 'raw'],
                        help='Which variant is aliased to the bare metric keys that '
                             'analyze_results.py reads. Both are always stored. '
                             "'raw' reproduces upstream's ACCURACY only - the "
                             'likelihood-derived measures still follow whatever the '
                             'cleaning stage produced, because the likelihoods stage '
                             'always prefers cleaned ids. A full upstream reproduction '
                             'also needs a likelihoods run over an uncleaned pickle.')
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')
    run_name = wandb.run.name or args.run_id
    print(f'run_id={args.run_id}  run_name={run_name}')

    scorer = AccuracyScorer(args)
    sequences, path = scorer.run(run_name)
    print(f'wrote {path}')

    summary = scorer.summarise(sequences, args.score_on)
    summary['scoring_complete'] = True
    wandb.log(summary)
    print('\n' + '  '.join(f'{k}={v}' for k, v in summary.items()))
    wandb.finish()


if __name__ == '__main__':
    main()
