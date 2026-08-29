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
                for string in self.strings_to_filter_on:
                    if string in generated_text:
                        generated_text = generated_text.split(string)[0]
                cleaned_generated_texts.append(generated_text)
                # Combine the prompt with the cleaned answer so it seems like Question: question Answer: cleaned_answer
                clean_ids = torch.cat(
                    [sample['prompt'].to(DEVICE),
                     torch.tensor(self.tokenizer(generated_text)['input_ids'][1:], device=DEVICE)])
                cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

            sample['cleaned_generated_texts'] = cleaned_generated_texts
            sample['cleaned_generations'] = cleaned_generations
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

        return {
            'n_questions': len(cleaned_sequences),
            'n_generations': len(cleaned),
            'fraction_generations_changed': float(np.mean([c != o for c, o in zip(cleaned, original)])),
            # A generation cleaned down to nothing means it opened with a filter string.
            'n_empty_after_cleaning': int((cleaned_words == 0).sum()),
            'mean_cleaned_words': float(cleaned_words.mean()),
            'median_cleaned_words': float(np.median(cleaned_words)),
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

    run_name = wandb.run.name

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
