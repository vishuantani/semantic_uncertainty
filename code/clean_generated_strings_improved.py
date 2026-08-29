import argparse
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import config
import wandb

from device_utils import DEVICE, DTYPE

# CHANGED FROM UPSTREAM: the inline filter loop moved to new_text_cleaning so the
# same rule can also be applied to the beam-search outputs below.
from new_text_cleaning import build_cleaned_ids, filter_generated_text

UPSTREAM_STRINGS_TO_FILTER_ON = [
    '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
    'ANSWER:'
]

UPDATED_STRINGS_TO_FILTER_ON = [
    '.', '\n', 
    'Q:', 'A:', 'q:', 'a:',
    'question:', 'answer:', 'questions:', 'answers:',
    'Question:', 'Answer:', 'Questions:', "Answers:",
    'QUESTION:', 'ANSWER:', 'QUESTIONS:', 'ANSWERS:'
]

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

class CleanGeneratedStrings:

    def __init__(self, args):
        self.args = args
        self.seed_value = args.seed

        self._set_seeds()
        self.load_model_and_tokenizer()

        self.strings_to_filter_on = UPDATED_STRINGS_TO_FILTER_ON if args.strings_to_filter_on == 'updated' else UPSTREAM_STRINGS_TO_FILTER_ON

    def load_model_and_tokenizer(self):
        self.generation_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.args.generation_model}",
                                                                  use_fast=False,
                                                                  cache_dir=config.data_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.args.generation_model}",
                                                       use_fast=False,
                                                       cache_dir=config.data_dir)

        return self.generation_tokenizer, self.tokenizer

    def _set_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)

    def _path(self, run_name):
        return f'{config.output_dir}/sequences/{run_name}/{self.args.generation_model}_generations.pkl'

    def load(self, run_name):
        with open(self._path(run_name), 'rb') as infile:
            return pickle.load(infile)

    def clean(self, sequences):
        cleaned_sequences = []

        for sample in tqdm(sequences):
            cleaned_generations = torch.ones_like(sample['generations'])
            question = sample['question']
            generated_texts = sample['generated_texts']
            cleaned_generated_texts = []

            max_len_of_generations = cleaned_generations.shape[-1]

            for i, generated_text in enumerate(generated_texts):
                # CHANGED FROM UPSTREAM: identical logic, now shared via
                # new_text_cleaning.filter_generated_text. Behaviour unchanged.
                generated_text = filter_generated_text(generated_text, self.strings_to_filter_on)
                cleaned_generated_texts.append(generated_text)
                # Combine the prompt with the cleaned answer so it seems like Question: question Answer: cleaned_answer
                clean_ids = torch.cat(
                    [sample['prompt'].to(DEVICE),
                     torch.tensor(self.tokenizer(generated_text)['input_ids'][1:], device=DEVICE)])
                cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

            sample['cleaned_generated_texts'] = cleaned_generated_texts
            sample['cleaned_generations'] = cleaned_generations

            # NEW - NO UPSTREAM EQUIVALENT.
            # Upstream cleaned only the sampled generations. The beam-search
            # outputs were left raw, which meant:
            #   - rougeL_to_target scored a ~171-word ramble against a 1-3 word
            #     answer, driving correctness to 0/40 and every AUROC to nan
            #   - average_neg_log_likelihood_of_{most,second_most}_likely_gen was
            #     computed over that same ramble, feeding the margin measure
            # Both the text and the token ids are cleaned so the two stay
            # consistent. The raw fields are left untouched so the raw-vs-cleaned
            # comparison remains available downstream.
            for field in ('most_likely_generation', 'second_most_likely_generation'):
                cleaned = filter_generated_text(sample[field], self.strings_to_filter_on)
                sample['cleaned_' + field] = cleaned
                sample['cleaned_' + field + '_ids'] = build_cleaned_ids(
                    sample['prompt'], cleaned, self.tokenizer,
                    len(sample[field + '_ids']))

            cleaned_sequences.append(sample)

        return cleaned_sequences

    def save(self, cleaned_sequences, run_name):
        path = self._path(run_name)

        with open(path, 'wb') as outfile:
            pickle.dump(cleaned_sequences, outfile)

        return path

    def run(self, run_name):
        sequences = self.load(run_name)
        cleaned_sequences = self.clean(sequences)
        path = self.save(cleaned_sequences, run_name)
        return cleaned_sequences, path

    @staticmethod
    def summarise(cleaned_sequences):
        """One line of aggregates so runs can be compared without opening the pickle.

        clean() leaves the original 'generated_texts' on the sample, so how much the
        filter list actually removed can be measured here.
        """
        original = [t for s in cleaned_sequences for t in s['generated_texts']]
        cleaned = [t for s in cleaned_sequences for t in s['cleaned_generated_texts']]
        cleaned_words = np.array([len(t.split()) for t in cleaned])

        # NEW - NO UPSTREAM EQUIVALENT: makes the beam-search cleaning visible in
        # W&B, since that is the change this stage now carries.
        most_likely_raw = np.array([len(s['most_likely_generation'].split())
                                    for s in cleaned_sequences])
        most_likely_cleaned = np.array([len(s['cleaned_most_likely_generation'].split())
                                        for s in cleaned_sequences])

        return {
            'n_questions': len(cleaned_sequences),
            'n_generations': len(cleaned),
            'fraction_generations_changed': float(np.mean([c != o for c, o in zip(cleaned, original)])),
            # A generation cleaned down to nothing means it opened with a filter string.
            'n_empty_after_cleaning': int((cleaned_words == 0).sum()),
            'mean_cleaned_words': float(cleaned_words.mean()),
            'median_cleaned_words': float(np.median(cleaned_words)),
            'mean_most_likely_words_raw': float(most_likely_raw.mean()),
            'mean_most_likely_words_cleaned': float(most_likely_cleaned.mean()),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_model', type=str, default='opt-350m')
    parser.add_argument('--run_id', type=str, default='run_1')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--strings_to_filter_on', type=str, default='upstream', help="'upstream' or 'updated' - whether to select the upstream (original) version of strings to filter on or used the updated version")
    return parser.parse_args()


def main():
    args = parse_args()

    wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')

    # CHANGED FROM UPSTREAM: upstream (and this file previously) used
    # `wandb.run.name` alone. Offline - wandb disabled or no network - that is None,
    # so this stage wrote to output/sequences/None/... and every later stage raised
    # FileNotFoundError looking under the real run name. Falling back to run_id
    # matches what generate_improved / get_likelihoods_improved / new_score_accuracy
    # already do, keeping the whole pipeline on one directory offline.
    run_name = wandb.run.name or args.run_id

    cleaner = CleanGeneratedStrings(args)
    cleaned_sequences, path = cleaner.run(run_name)

    summary = CleanGeneratedStrings.summarise(cleaned_sequences)
    summary['cleaning_complete'] = True
    wandb.log(summary)
    print(f'\ncleaning complete -> {path}')
    print('  '.join(f'{k}={v}' for k, v in summary.items()))

    wandb.finish()


if __name__ == '__main__':
    main()
