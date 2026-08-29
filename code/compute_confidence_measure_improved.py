import argparse
import os
import pickle
import random
import uuid

import config
import numpy as np
import torch
import wandb


class ConfidenceMeasures:
    def __init__(self, args):
        self.args = args

        self._set_seeds(args.seed)

        # Try with this set to 0
        self.llh_shift = torch.tensor(float(args.llh_shift))

    def _set_seeds(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _likelihoods_path(self, run_name):
        return (f'{config.output_dir}/likelihoods/{run_name}/'
                f'{self.args.generation_model}_generations_{self.args.evaluation_model}_likelihoods.pkl')

    def _confidence_path(self, run_name):
        return (f'{config.output_dir}/confidence/{run_name}/'
                f'aggregated_likelihoods_{self.args.generation_model}_generations.pkl')

    def _read_data(self, run_name):
        list_of_results = []

        with open(self._likelihoods_path(run_name), 'rb') as infile:
            sequences = pickle.load(infile)
            list_of_results.append((self.args.evaluation_model, sequences))

        return list_of_results

    @staticmethod
    def get_overall_log_likelihoods(list_of_results):
        """Compute log likelihood of all generations under their given context.

        list_of_results: list of dictionaries with keys:

        returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
                 that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
        """

        result_dict = {}

        list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods', 'sequence_embeddings',\
                        'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                        'average_neg_log_likelihood_of_second_most_likely_gen',\
                        'neg_log_likelihood_of_most_likely_gen', 'semantic_set_ids']

        for key in list_of_keys:
            list_of_ids = []
            overall_results = []
            for model_size, result in list_of_results:
                results_per_model = []
                for sample in result:
                    average_neg_log_likelihoods = sample[key]
                    list_of_ids.append(sample['id'][0])
                    results_per_model.append(average_neg_log_likelihoods)

                results_per_model = torch.stack(results_per_model)
                overall_results.append(results_per_model)

            if key != 'sequence_embeddings':
                overall_results = torch.stack(overall_results)

            result_dict[key] = overall_results

        result_dict['ids'] = list_of_ids
        return result_dict

    # The get_mutual_information function does nothing informative
    #   It tells us whether different different validator models agreed or disagreed and whether they did so confidently or ambiguously.
    #   That is a property of the validators and not of the generator model

    @staticmethod
    def get_log_likelihood_variance(neg_log_likelihoods):
        """Compute log likelihood variance of approximate posterior predictive"""
        mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
        variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

        return variance_of_neg_log_likelihoods

    @staticmethod
    def get_log_likelihood_mean(neg_log_likelihoods):
        """Compute softmax variance of approximate posterior predictive"""
        mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
        mean_of_neg_log_likelihoods = torch.mean(mean_across_models, dim=1)

        return mean_of_neg_log_likelihoods

    @staticmethod
    def get_mean_of_poinwise_mutual_information(pointwise_mutual_information):
        """Compute mean of pointwise mutual information"""
        mean_across_models = torch.mean(pointwise_mutual_information, dim=0)
        return torch.mean(mean_across_models, dim=1)

    # This is the Monte Carlo estimate of the entropy per answer: p*log(p) <=> (1/N)*log(p)
    #   Reason: Our model has geenrated the sequence s_m of length m given prompt x
    #   That sequence was generated given the model parameters, which are a factor of the training of the model
    #   There is no way of us to gauge the "actual" likelihood of the answer, by assessing the likelihood straight from the logits, we can get an understanding of the likelihood in the answer given the model's own understanding
    #   But as mentioned, that likelihood is subject to the parameters
    #   Instead, the accurate measure would be if we marginalise out the training data altogether - given all the data in the world, what is the likelihood of the answer
    #   The closest we can get to that is a monte carlo estimate using a variety of models. As different models are trained on different data,
    #   if we use all of them to judge and then combine their judgement of the likelihood of the answer into a single MC estimate, that is the closest we get to the true likelihood of the answer
    #   true likelihood = p(s_m | x)
    #   likelihood from model = p(s_m | x, theta)
    #   p(s_m | x) = integrate(likelihood from model * probability of that model * dtheta)
    #   Where probability of that model is saying what is the probability we end up with there parameters given training data
    #   MC estimate of that integration (if using N samples of the judge model) = (1 / N) * sum(likelihood from sample model)
    #   p_hat(s_m | x) = sum_over_i(p(s_m | x, theta_i)), where theta_i represents a judge model
    #   That is what mean across all models gets to (well, the log of that).
    #   Then entropy is sum over samples as it is the MC estimate of the entropy number in general: p * log(p) estimated by (1 / M) * sum_over_j(log(p_j)) where j represents a sample question
    #   Finally, as it is the generative model that has generated these outputs (s_m), all properties of the output relate back to the generation model
    @staticmethod
    def get_predictive_entropy(log_likelihoods):
        """Compute predictive entropy of approximate posterior predictive"""
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
        return entropy

    def get_predictive_entropy_over_concepts(self, log_likelihoods, semantic_set_ids):
        """Compute the semantic entropy"""
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        # This is ok because all the models have the same semantic set ids
        semantic_set_ids = semantic_set_ids[0]
        # get_likelihoods writes the ids on the accelerator and the likelihood buffers on the CPU,
        # so the mask below can end up on a different device than the row it indexes.
        semantic_set_ids = semantic_set_ids.to(mean_across_models.device)
        entropies = []
        for question_id in range(mean_across_models.shape[0]):
            aggregated_likelihoods = []
            row = mean_across_models[question_id]
            semantic_set_ids_row = semantic_set_ids[question_id]
            # This is where there is a fallacy that likelihood of a class is estimated as the sum of likelihoods of the samples in that class
            #   But that is a biased estimator - read explainer/estimating-semantic-entropy.html
            for semantic_set_id in torch.unique(semantic_set_ids_row):
                aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
            aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - self.llh_shift
            entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
            entropies.append(entropy)

        return torch.tensor(entropies)

    @staticmethod
    def get_margin_probability_uncertainty_measure(log_likelihoods):
        """Compute margin probability uncertainty measure"""
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
        margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

        return margin_probabilities

    @staticmethod
    def get_number_of_unique_elements_per_row(tensor):
        assert len(tensor.shape) == 2
        return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor), dim=1), dim=1)

    def compute(self, overall_results):
        predictive_entropy = self.get_predictive_entropy(-overall_results['neg_log_likelihoods'])
        predictive_entropy_over_concepts = self.get_predictive_entropy_over_concepts(
            -overall_results['average_neg_log_likelihoods'], overall_results['semantic_set_ids'])
        unnormalised_entropy_over_concepts = self.get_predictive_entropy_over_concepts(
            -overall_results['neg_log_likelihoods'], overall_results['semantic_set_ids'])

        margin_measures = self.get_margin_probability_uncertainty_measure(
            -overall_results['average_neg_log_likelihoods'])
        unnormalised_margin_measures = self.get_margin_probability_uncertainty_measure(
            -overall_results['neg_log_likelihoods'])

        number_of_semantic_sets = self.get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0])
        average_predictive_entropy = self.get_predictive_entropy(-overall_results['average_neg_log_likelihoods'])

        average_predictive_entropy_on_subsets = []
        predictive_entropy_on_subsets = []
        semantic_predictive_entropy_on_subsets = []
        number_of_semantic_sets_on_subsets = []
        num_predictions = overall_results['average_neg_log_likelihoods'].shape[-1]
        for i in range(1, num_predictions + 1):
            average_predictive_entropy_on_subsets.append(
                self.get_predictive_entropy(-overall_results['average_neg_log_likelihoods'][:, :, :int(i)]))
            predictive_entropy_on_subsets.append(
                self.get_predictive_entropy(-overall_results['neg_log_likelihoods'][:, :, :int(i)]))
            semantic_predictive_entropy_on_subsets.append(
                self.get_predictive_entropy_over_concepts(
                    -overall_results['average_neg_log_likelihoods'][:, :, :int(i)],
                    overall_results['semantic_set_ids'][:, :, :int(i)]))
            number_of_semantic_sets_on_subsets.append(
                self.get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0][:, :i]))

        average_pointwise_mutual_information = self.get_mean_of_poinwise_mutual_information(
            overall_results['pointwise_mutual_information'])

        overall_results['predictive_entropy'] = predictive_entropy
        overall_results['predictive_entropy_over_concepts'] = predictive_entropy_over_concepts
        overall_results['unnormalised_entropy_over_concepts'] = unnormalised_entropy_over_concepts
        overall_results['number_of_semantic_sets'] = number_of_semantic_sets
        overall_results['margin_measures'] = margin_measures
        overall_results['unnormalised_margin_measures'] = unnormalised_margin_measures

        overall_results['average_predictive_entropy'] = average_predictive_entropy
        for i in range(len(average_predictive_entropy_on_subsets)):
            overall_results[f'average_predictive_entropy_on_subset_{i + 1}'] = average_predictive_entropy_on_subsets[i]
            overall_results[f'predictive_entropy_on_subset_{i + 1}'] = predictive_entropy_on_subsets[i]
            overall_results[f'semantic_predictive_entropy_on_subset_{i + 1}'] = semantic_predictive_entropy_on_subsets[i]
            overall_results[f'number_of_semantic_sets_on_subset_{i + 1}'] = number_of_semantic_sets_on_subsets[i]
        overall_results['average_pointwise_mutual_information'] = average_pointwise_mutual_information

        return overall_results

    def _save(self, overall_results, run_name):
        filename = self._confidence_path(run_name)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as outfile:
            pickle.dump(overall_results, outfile)

        return filename

    def run(self, run_name):
        list_of_results = self._read_data(run_name)
        overall_results = self.get_overall_log_likelihoods(list_of_results)
        overall_results = self.compute(overall_results)

        confidence_pickle_path = self._save(overall_results, run_name)

        return overall_results, confidence_pickle_path

    def summarise(self, overall_results):
        """One line of aggregates so runs can be compared without opening the pickle.

        Every number here is a per-question measure reduced to its mean, so a change in any of
        them is a change in what the run would report as uncertainty.
        """
        def numpy(key):
            return overall_results[key].detach().cpu().numpy()

        predictive_entropy = numpy('predictive_entropy')
        average_predictive_entropy = numpy('average_predictive_entropy')
        semantic_entropy = numpy('predictive_entropy_over_concepts')
        unnormalised_semantic_entropy = numpy('unnormalised_entropy_over_concepts')
        number_of_semantic_sets = numpy('number_of_semantic_sets')
        margin_measures = numpy('margin_measures')

        num_models, num_questions, num_generations = overall_results['average_neg_log_likelihoods'].shape

        return {
            'n_questions': num_questions,
            'n_generations': num_generations,
            # The ensemble axis. While this is 1 the logsumexp over dim=0 is an identity and any
            # measure defined as disagreement between models is identically zero.
            'n_models': num_models,
            'mean_predictive_entropy': float(predictive_entropy.mean()),
            'mean_average_predictive_entropy': float(average_predictive_entropy.mean()),
            'mean_semantic_entropy': float(semantic_entropy.mean()),
            'mean_unnormalised_semantic_entropy': float(unnormalised_semantic_entropy.mean()),
            # llh_shift adds itself to every semantic entropy unchanged - it is subtracted from each
            # cluster likelihood and the same cluster count then divides the sum. Reported without it
            # so semantic entropy sits on the same scale as the predictive entropies above.
            'mean_semantic_entropy_unshifted': float(semantic_entropy.mean() - float(self.llh_shift)),
            'mean_number_of_semantic_sets': float(number_of_semantic_sets.mean()),
            # Every generation landing in one cluster makes semantic entropy degenerate for that
            # question, and it is the clustering stage rather than this one that decides the rate.
            'fraction_single_semantic_set': float((number_of_semantic_sets == 1).mean()),
            'mean_margin_measure': float(margin_measures.mean()),
            # The margin is taken between the top two *sampled* generations, so duplicate samples
            # give two identical likelihoods and a margin of ~0 - reading as maximal uncertainty
            # exactly when the model is most concentrated.
            'fraction_margin_near_zero': float((np.abs(margin_measures) < 1e-6).mean()),
            'n_non_finite': int((~np.isfinite(np.concatenate(
                [predictive_entropy, average_predictive_entropy, semantic_entropy]))).sum()),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_model', type=str, default='opt-350m')
    parser.add_argument('--evaluation_model', type=str, default='opt-350m')
    parser.add_argument('--run_id', type=str, default='run_1')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--llh_shift', type=float, default=5.0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:8]
    wandb.init(project='nlg_uncertainty', id=run_id, config=args, resume='allow')
    run_name = wandb.run.name or run_id
    print(f'run_id={run_id}  run_name={run_name}')

    experiment = ConfidenceMeasures(args)
    overall_results, confidence_pickle_path = experiment.run(run_name)

    summary = experiment.summarise(overall_results)
    summary['confidence_complete'] = True
    wandb.log(summary)
    print('\n' + '  '.join(f'{k}={v}' for k, v in summary.items()))

    wandb.finish()


if __name__ == '__main__':
    main()
