'''
This file groups semantically similar answers into buckets per question in our training/test set

Improvements over original:
1. [PERFORMANCE] torch.no_grad when evaluating from deberta
2. [PERFORMANCE] moved loading of metric outside the loop
3. [PERFORMANCE] batch the deberta requests for a single single question together
4. [LOCAL RUN] Moved to Base DeBERTa for entailment instead of large.
'''

import argparse
import csv
import os
import pickle
import random

from device_utils import DEVICE
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import wandb
import uuid

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']

class EntailmentExperiment:
    def __init__(self, args):
        self.args = args
        self.seed_value = args.seed

        self._set_seeds()
        self._load_model_and_tokenizer()
        self._load_evaluators()

    def _set_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)

    def _load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{self.args.entailment_model}")
        self.model = AutoModelForSequenceClassification.from_pretrained(f"microsoft/{self.args.entailment_model}").to(DEVICE)

    def _load_evaluators(self):
        # self.meteor = evaluate.load('meteor') # This isn't used anywhere
        self.rouge = evaluate.load('rouge')

    def _build_batch_from_sample(self, question, unique_generated_texts, batch_size=None):
        '''
        Take unique_generated_texts from a sample as an input and generate 2 * [n * (n - 1) / 2] queries for deberta
        n * (n - 1) / 2 is the number of pairs
        2 is because each will have a forward and a reverse query
        batch_size of None implies that all of those will be returned in a single batch
        Also, batch_size must be even for not splitting forward, reverse query across batches (might be aesthetic)
        '''
        assert (batch_size is None) or (batch_size % 2 == 0), f"batch_size must be None (for a single batch per sample) or an even number"

        inputs = []
        pair_indices = []
        for i in range(len(unique_generated_texts)):
            for j in range(i + 1, len(unique_generated_texts)):
                qa_1 = question + ' ' + unique_generated_texts[i]
                qa_2 = question + ' ' + unique_generated_texts[j]

                input = qa_1 + ' [SEP] ' + qa_2
                inputs.append(input)
                reverse_input = qa_2 + ' [SEP] ' + qa_1
                inputs.append(reverse_input)

                pair_indices.append((i, j))

        step = batch_size or len(inputs)
        encoded_inputs = [
            self.tokenizer(inputs[start:start+step], padding=True, return_tensors='pt') 
            for start in range(0, len(inputs), step)
        ]

        # What does this mean? Switching return batches to a generator (yield the tokenizer call, return pair_indices separately, or yield (batch, pair_slice) tuples) avoids that.
        return encoded_inputs, pair_indices

    def _path(self, run_name):
        return f'{config.output_dir}/sequences/{run_name}/{self.args.generation_model}_generations.pkl'

    def _load_sequences(self, run_name):
        with open(self._path(run_name), 'rb') as infile:
            return pickle.load(infile)

    def _init_semantic_sets_for_answers(self, unique_answers):
        semantic_set_ids = {}
        for index, answer in enumerate(unique_answers):
            semantic_set_ids[answer] = index

        return semantic_set_ids

    def _check_semantic_difference(self, forward, reverse):
        if self.args.similarity_style == "ENSURE_ENTAILMENT":
            # If not entailment both sides, then not similar
            return not (forward == 2 and reverse == 2)
        else:
            # If any side is a contradiction, then not similar
            # This is the default implementation of the repository, but not of the paper
            return forward == 0 or reverse == 0

    def _merge_semantic_sets(self, pair_indices, labels, semantic_set_ids, unique_generated_texts):
        has_semantically_different_answers = False
        deberta_predictions = []
        for (i, j), (forward, reverse) in zip(pair_indices, labels):
            deberta_prediction = 1
            if self._check_semantic_difference(forward, reverse):
                has_semantically_different_answers = True
                deberta_prediction = 0
            else:
                semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

            deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])

        return deberta_predictions, semantic_set_ids, has_semantically_different_answers

    def _get_syntactic_similarities(self, generated_texts):
        answer_list_1 = []
        answer_list_2 = []
        syntactic_similarities = {}
        for i in generated_texts:
            for j in generated_texts:
                if i != j:
                    answer_list_1.append(i)
                    answer_list_2.append(j)

        results = self.rouge.compute(predictions=answer_list_1, references=answer_list_2)

        for rouge_type in ROUGE_TYPES:
            syntactic_similarities[rouge_type] = results[rouge_type]

        return syntactic_similarities

    def _process_sample(self, sample):
        question = sample['question']

        if 'cleaned_generated_texts' in sample:
            generated_texts = sample['cleaned_generated_texts']
        else:
            generated_texts = sample['generated_texts']

        unique_generated_texts = sorted(set(generated_texts))

        syntactic_similarities = {}
        for rouge_type in ROUGE_TYPES:
            syntactic_similarities[rouge_type] = 0.0

        semantic_set_ids = self._init_semantic_sets_for_answers(unique_generated_texts)
        deberta_predictions = []
        has_semantically_different_answers = False

        if len(unique_generated_texts) > 1:
            encoded_input_batches, pair_indices = self._build_batch_from_sample(question, unique_generated_texts)
            chunk_pred_labels = []
            for batch in encoded_input_batches:
                with torch.no_grad():
                    batch_pred = self.model(**batch.to(DEVICE))['logits']
                predicted_labels = torch.argmax(batch_pred, dim=1)
                chunk_pred_labels.append(predicted_labels)

            # Now convert the labels to the format of (forward, reverse) so it is easy to compare
            labels = torch.cat(chunk_pred_labels).view(-1, 2).tolist()

            deberta_predictions, semantic_set_ids, has_semantically_different_answers = self._merge_semantic_sets(pair_indices, labels, semantic_set_ids, unique_generated_texts)
            syntactic_similarities = self._get_syntactic_similarities(generated_texts)
            
        sample_result = {
            'syntactic_similarities': syntactic_similarities,
            'has_semantically_different_answers': has_semantically_different_answers,
            'semantic_set_ids': [semantic_set_ids[x] for x in generated_texts]
        }

        return deberta_predictions, sample_result

    def save(self, deberta_predictions, result_dict, run_name):
        deberta_predictions_output_path = f'{config.output_dir}/entailment/{run_name}/{self.args.entailment_model}_preds.csv'
        os.makedirs(os.path.dirname(deberta_predictions_output_path), exist_ok=True)
        with open(deberta_predictions_output_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['qa_1', 'qa_2', 'prediction'])
            writer.writerows(deberta_predictions)
        print(f"Wrote DeBERTa predictions to: {deberta_predictions_output_path}")

        similarities_output_path = f'{config.output_dir}/sequences/{run_name}/{self.args.generation_model}_generations_similarities.pkl'
        os.makedirs(os.path.dirname(similarities_output_path), exist_ok=True)
        with open(similarities_output_path, 'wb') as outfile:
            pickle.dump(result_dict, outfile)
        print(f"Wrote similarities to: {similarities_output_path}")

        return deberta_predictions_output_path, similarities_output_path


    def run(self, run_name):
        sequences = self._load_sequences(run_name)

        result_dict = {}
        deberta_predictions = []

        for sample in tqdm(sequences):
            deberta_sample_predictions, sample_result = self._process_sample(sample)
            result_dict[sample['id'][0]] = sample_result
            deberta_predictions.extend(deberta_sample_predictions)

        deberta_predictions_output_path, similarities_output_path = self.save(deberta_predictions, result_dict, run_name)

        return deberta_predictions, result_dict, deberta_predictions_output_path, similarities_output_path

    @staticmethod
    def summarise(result_dict, deberta_predictions):
        """One line of aggregates so entailment configurations can be compared without opening the pickle.

        Number of semantic sets is the headline: entropy is computed over those sets, so a
        configuration that judges more answers equivalent produces fewer sets and lower entropy.
        """
        n_sets = np.array([len(set(r['semantic_set_ids'])) for r in result_dict.values()])
        n_generations = np.array([len(r['semantic_set_ids']) for r in result_dict.values()])
        flagged = np.array([r['has_semantically_different_answers'] for r in result_dict.values()])
        predictions = np.array([row[2] for row in deberta_predictions])

        return {
            'n_questions': len(result_dict),
            'n_generations': int(n_generations.sum()),
            'n_pairs_compared': len(predictions),
            'mean_semantic_sets': float(n_sets.mean()),
            'median_semantic_sets': float(np.median(n_sets)),
            'fraction_questions_multi_set': float((n_sets > 1).mean()),
            'fraction_pairs_contradicting': float((predictions == 0).mean()) if len(predictions) else 0.0,
            'fraction_flagged_different': float(flagged.mean()),
            # The flag trips on any contradicting pair, but the clustering can still collapse to a
            # single set when entailment is non-transitive. This measures how often they disagree.
            'fraction_flag_disagrees_with_sets': float((flagged != (n_sets > 1)).mean()),
            'mean_rouge1': float(np.mean([r['syntactic_similarities']['rouge1'] for r in result_dict.values()])),
            'mean_rougeL': float(np.mean([r['syntactic_similarities']['rougeL'] for r in result_dict.values()])),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--generation_model', type=str, default='opt-350m')
    parser.add_argument('--entailment_model', type=str, default='deberta-base-mnli')
    parser.add_argument('--similarity_style', type=str, default='NOT_CONTRADICTION')
    parser.add_argument('--run_id', type=str, default='run_1')
    parser.add_argument('--seed', type=int, default=10)

    return parser.parse_args()

def main():
    args = parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:8]
    wandb.init(project='nlg_uncertainty', id=run_id, config=args, resume='allow')

    run_name = wandb.run.name or run_id
    print(f'run_id={run_id}  run_name={run_name}')

    experiment = EntailmentExperiment(args)
    deberta_predictions, result_dict, _, _ = experiment.run(run_name)

    summary = EntailmentExperiment.summarise(result_dict, deberta_predictions)
    summary['entailment_complete'] = True
    wandb.log(summary)
    print('  '.join(f'{k}={v}' for k, v in summary.items()))

    wandb.finish()

if __name__ == '__main__':
    main()

