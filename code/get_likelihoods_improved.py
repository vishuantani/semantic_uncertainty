import argparse
import os
import pickle
import random
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from device_utils import DEVICE, DTYPE
import wandb
import config
import uuid

OPT_MODELS = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

class NegLogLikelihood:
    def __init__(self, args):
        self.args = args
        
        self._set_seeds(args.seed)
        self._load_model()

    def _set_seeds(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(f"facebook/{self.args.evaluation_model}",
                                             torch_dtype=DTYPE,
                                             cache_dir=config.data_dir).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.args.evaluation_model}",
                                                use_fast=False,
                                                cache_dir=config.data_dir)

    def _read_data(self, run_name):
        with open(f'{config.output_dir}/sequences/{run_name}/{self.args.generation_model}_generations.pkl', 'rb') as infile:
            sequences = pickle.load(infile)

        with open(f'{config.output_dir}/sequences/{run_name}/{self.args.generation_model}_generations_similarities.pkl', 'rb') as infile:
            similarities_dict = pickle.load(infile)

        return sequences, similarities_dict


    def get_neg_log_likelihood_for_generation(self, generation, prompt_len, most_likely = False):
        # most_likely indicates whether this calculation is for most_likely_generation
        #  which does not calcualte the unconditioned likelihoods
        #  Note: It results in returning a different set of keys

        output = {}

        target_ids = generation.clone()
        target_ids[:prompt_len] = -100
        model_output = self.model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
        hidden_states = model_output['hidden_states']
        average_neg_log_likelihood = model_output['loss']
        average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
        neg_log_likelihood = average_neg_log_likelihood * (len(generation) - prompt_len)

        output['average_neg_log_likelihood'] = average_neg_log_likelihood
        output['neg_log_likelihood'] = neg_log_likelihood
        output['average_of_last_layer_token_embeddings'] = average_of_last_layer_token_embeddings

        if not most_likely:
            generation_only = generation.clone()[(prompt_len - 1):]
            unconditioned_model_output = self.model(torch.reshape(generation_only, (1, -1)),
                                        labels=generation_only,
                                        output_hidden_states=True)
            average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
            unconditioned_neg_log_likelihood = average_unconditioned_neg_log_likelihood * (len(generation) - prompt_len)

            output['average_unconditioned_neg_log_likelihood'] = average_unconditioned_neg_log_likelihood
            output['unconditioned_neg_log_likelihood'] = unconditioned_neg_log_likelihood

        return output


    def _sequence_ids(self, sequence, field):
        """Prefer the cleaned ids when the cleaning stage produced them.

        NEW - NO UPSTREAM EQUIVALENT. Upstream read
        most_likely_generation_ids / second_most_likely_generation_ids raw and
        passed them straight to get_neg_log_likelihood_for_generation, so the NLL
        was averaged over a ~171-word repetition loop - which then fed the margin
        measure. The tensor returned here also becomes 'sequence_embeddings' and
        'most_likely_sequence_embedding' below, so this affects more than the NLL.

        CHANGED FROM UPSTREAM - padding, stated correctly. An earlier version of
        this docstring claimed "the raw ids never had padding". That is false:
        beam search with num_return_sequences=2 pads both returned beams out to
        the longer of the two whenever they finish at different lengths, which is
        exactly the situation --ban_list extended and --stop_on_turn_marker are
        designed to create. Upstream never stripped those pads, so upstream's
        average_neg_log_likelihood_of_*_likely_gen averaged over them.
        We therefore strip pads ONLY on the cleaned path, where the padding is our
        own artifact (build_cleaned_ids re-pads the cleaned completion back out to
        the original width) and stripping is required for the loss to be per real
        token. The raw fallback is left byte-identical to upstream, padding
        included, so running this stage over an UNCLEANED pickle still reproduces
        the published number instead of silently deviating from it.

        CHANGED FROM UPSTREAM - degenerate cleaned sequences. If a beam output
        opens with a filter string ('.' or '\\n', say), filter_generated_text
        truncates it to '' and build_cleaned_ids yields prompt-plus-padding only.
        After the strip that tensor equals the prompt, so
        target_ids[:prompt_len] = -100 masks every label and the model returns a
        nan loss. Upstream could not hit this (it never cleaned these fields). The
        nan would flow into average_neg_log_likelihood_of_{most,second_most}_
        likely_gen, get stacked by compute_confidence_measure_improved.py, and
        finally make roc_auc_score raise "Input contains NaN" three stages later.
        The cleaning stage's n_empty_after_cleaning shows these empties do occur,
        so for that one sequence we fall back to the RAW ids: an uncleaned, worse
        number, but a finite and traceable one rather than a silent nan.
        """
        cleaned_key = 'cleaned_' + field
        # Raw path: no strip. This is upstream's exact input tensor.
        raw = sequence[field].to(DEVICE)

        if cleaned_key not in sequence:
            return raw

        cleaned = sequence[cleaned_key].to(DEVICE)
        # Cleaned path only: strip the padding build_cleaned_ids added back.
        cleaned = cleaned[cleaned != self.tokenizer.pad_token_id]

        prompt = sequence['prompt'].to(DEVICE)
        prompt = prompt[prompt != self.tokenizer.pad_token_id]
        if len(cleaned) <= len(prompt):
            # Nothing survived cleaning - every label would be masked. Fall back.
            return raw

        return cleaned


    def get_neg_loglikelihoods_for_sequence(self, sequence, semantic_set_ids):
        # Pointwise mutual information:
        #   PMI(answer) = log(p(answer|question)) - log(p(answer))
        #   which is the same as log(p(answer|question) / p(answer))
        #   p(answer|question) = how likely is this answer given the question
        #   p(answer) = how likely is this answer (just overall, eg. "I don't know" might be quite likely irrespective of the question)
        #   p(answer|question) / p(answer) is therefore a measure of how much information does this answer provide given the question.
        with torch.no_grad():
            result_dict = {}
            
            if 'cleaned_generations' in sequence:
                generations = sequence['cleaned_generations'].to(DEVICE)
            else:
                generations = sequence['generations'].to(DEVICE)
            id_ = sequence['id']

            prompt = sequence['prompt']
            prompt = prompt[prompt != self.tokenizer.pad_token_id] # Cut the prompt short to remove any padding IDs

            average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
            pointwise_mutual_information = torch.zeros((generations.shape[0],))
            sequence_embeddings = []

            for generation_index in range(generations.shape[0]):
                generation = generations[generation_index][generations[generation_index] != self.tokenizer.pad_token_id] # And do the same with the generated text

                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                generation_output = self.get_neg_log_likelihood_for_generation(generation, len(prompt), most_likely=False)
                average_neg_log_likelihoods[generation_index] = generation_output['average_neg_log_likelihood']
                average_unconditioned_neg_log_likelihoods[generation_index] = generation_output['average_unconditioned_neg_log_likelihood']
                neg_log_likelihoods[generation_index] = generation_output['neg_log_likelihood']
                neg_unconditioned_log_likelihoods[generation_index] = generation_output['unconditioned_neg_log_likelihood']

                sequence_embeddings.append(generation_output['average_of_last_layer_token_embeddings'])

                pointwise_mutual_information[generation_index] = -generation_output['neg_log_likelihood'] + generation_output['unconditioned_neg_log_likelihood']

            # CHANGED FROM UPSTREAM: prefer cleaned ids (stripping only the padding
            # cleaning itself added), falling back to raw ids when cleaning emptied
            # the sequence. See _sequence_ids.
            most_likely_generation = self._sequence_ids(sequence, 'most_likely_generation_ids')
            most_likely_generation_generation_output = self.get_neg_log_likelihood_for_generation(most_likely_generation, len(prompt), most_likely=True)

            # CHANGED FROM UPSTREAM: same treatment as the most-likely ids above.
            second_most_likely_generation = self._sequence_ids(sequence, 'second_most_likely_generation_ids')
            second_most_likely_generation_generation_output = self.get_neg_log_likelihood_for_generation(second_most_likely_generation, len(prompt), most_likely=True)

            sequence_embeddings = torch.stack(sequence_embeddings)
            result_dict['prompt'] = prompt
            result_dict['generations'] = generations
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods
            # CHANGED FROM UPSTREAM: this embedding now summarises the CLEANED,
            # pad-stripped most-likely sequence, where upstream embedded the raw
            # beam output including its repetition loop. It is not a debug field -
            # analyze_results.py pickles it out as sequence_embeddings.pkl, so any
            # downstream consumer of that file sees cleaned-sequence embeddings now.
            result_dict['sequence_embeddings'] = most_likely_generation_generation_output['average_of_last_layer_token_embeddings']
            # CHANGED FROM UPSTREAM: these are the pad-stripped cleaned ids, so this
            # field is now variable-length across questions (upstream's were all the
            # same padded width). It has no consumer today, but a future
            # torch.stack over it would fail - pad before stacking.
            result_dict['most_likely_sequence_embedding'] = most_likely_generation
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = most_likely_generation_generation_output['average_neg_log_likelihood']
            result_dict['average_neg_log_likelihood_of_second_most_likely_gen'] = second_most_likely_generation_generation_output['average_neg_log_likelihood']
            result_dict['neg_log_likelihood_of_most_likely_gen'] = most_likely_generation_generation_output['neg_log_likelihood']
            result_dict['semantic_set_ids'] = torch.tensor(semantic_set_ids)
            result_dict['id'] = id_

            return result_dict


    def _save_likelihoods(self, likelihoods, run_name):
        filename = f'{config.output_dir}/likelihoods/{run_name}/{self.args.generation_model}_generations_{self.args.evaluation_model}_likelihoods.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename,'wb') as outfile:
            pickle.dump(likelihoods, outfile)

        return filename

    def run(self, run_name):
        sequences, similarities_dict = self._read_data(run_name)
        likelihoods = [
            self.get_neg_loglikelihoods_for_sequence(sequence, similarities_dict[sequence['id'][0]]['semantic_set_ids'])
            for sequence in tqdm(sequences, desc='likelihoods')
        ]

        likelihoods_pickle_path = self._save_likelihoods(likelihoods, run_name)

        return likelihoods, likelihoods_pickle_path

    @staticmethod
    def summarise(likelihoods):
        """One line of aggregates so runs can be compared without opening the pickle.

        These likelihoods are the input to every entropy measure downstream, so their central
        tendency is the headline: a shift in mean negative log likelihood moves predictive
        entropy directly.
        """
        def stack(key):
            return np.concatenate([l[key].detach().cpu().numpy() for l in likelihoods])

        average_nll = stack('average_neg_log_likelihoods')
        nll = stack('neg_log_likelihoods')
        pmi = stack('pointwise_mutual_information')
        most_likely = np.array([float(l['average_neg_log_likelihood_of_most_likely_gen']) for l in likelihoods])
        second = np.array([float(l['average_neg_log_likelihood_of_second_most_likely_gen']) for l in likelihoods])

        return {
            'n_questions': len(likelihoods),
            'n_generations': len(average_nll),
            'mean_average_neg_log_likelihood': float(average_nll.mean()),
            'mean_neg_log_likelihood': float(nll.mean()),
            'mean_pointwise_mutual_information': float(pmi.mean()),
            # The question should make its own answer more likely, so PMI is positive almost
            # everywhere. A low fraction points at the prompt masking or at the token count used
            # to turn the mean loss back into a sum.
            'fraction_pmi_positive': float((pmi > 0).mean()),
            'mean_average_neg_log_likelihood_of_most_likely_gen': float(most_likely.mean()),
            # Beam search returns its two candidates already ranked by length-normalised score, so
            # the most likely generation should score better than the second nearly always. It also
            # trips when the beam outputs and the prompt length fall out of alignment.
            'fraction_most_likely_beats_second': float((most_likely < second).mean()),
            # fp16 overflow surfaces here rather than as a NaN entropy three stages later.
            # CHANGED FROM UPSTREAM (this fork's earlier version): the two most-likely
            # NLLs are now counted too. They were omitted, so a degenerate cleaned
            # sequence - one masked down to zero labels, yielding a nan loss - passed
            # this check unnoticed and only surfaced as roc_auc_score's "Input contains
            # NaN" in analyze_results.py. _sequence_ids now guards against producing
            # that nan; this counter is the second line of defence, so it can never
            # reach the analysis silently again.
            'n_non_finite': int((~np.isfinite(
                np.concatenate([average_nll, nll, pmi, most_likely, second]))).sum()),
        }



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_model', type=str, default='opt-350m')
    parser.add_argument('--generation_model', type=str, default='opt-350m')
    parser.add_argument('--run_id', type=str, default='run_1')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:8]
    wandb.init(project='nlg_uncertainty', id=run_id, config=args, resume='allow')
    run_name = wandb.run.name or run_id
    print(f'run_id={run_id}  run_name={run_name}')

    experiment = NegLogLikelihood(args)
    likelihoods, likelihoods_pickle_path = experiment.run(run_name)

    summary = NegLogLikelihood.summarise(likelihoods)
    summary['likelihoods_complete'] = True
    wandb.log(summary)
    print('\n' + '  '.join(f'{k}={v}' for k, v in summary.items()))

    wandb.finish()

if __name__ == "__main__":
    main()