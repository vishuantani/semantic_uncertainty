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
the upstream number stays reproducible and the delta is reportable, which matters
because the cleaned number is no longer directly comparable to the paper.

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

    Same selection upstream's GenerationExperiment._reference_answers made, but
    reading from the pickled sample rather than the live dataset batch.
    """
    if dataset == 'coqa':
        return list(sample['answer']['text']) + [x[0] for x in sample['additional_answers']]
    return list(sample['answer'])


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
                             "'raw' reproduces upstream's number.")
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
