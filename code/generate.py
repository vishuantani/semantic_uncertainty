'''
What this file does:
1. After all the necessary setup (arguments, seeds, etc), load the Pretrained OPT model (Causal LM = autoregressive) and its tokenizer
2. The tokenizer of the model is needed to shape a few things while it generates the responses:
    i. Since OPT is a base model, it does not know when to stop generation. We force stopping generation when the period ('.') token is hit. For that, we need to know from the tokenizer what that is.
    ii. However, as it is going to answer from the prompt which has examples of question answers, the LM is most likely to give an answer and carry on asking itself questions and answering
        - In fact, the prompt examples don't have a period in them after the answer and the LM will try to mimic that behaviour by not adding a [eriod once it has answered and instead just continuing with a self created question
    iii. So, we ban certain words from the output -- tokens that we believe the model to most likely generate once it has reached an end to its answer
        - These are variants of Question, Answer or new line. By banning those words, when they appear (which would be when the model has answered) the next likely token should be the period and we stop generation upon finding a period.
    iv. We then load our dataset of QAs (triviaqa or coqa)
    v. Then we run the samples through a function that has two tasks:
        a. Mark the accuracy:
            - First generate the most likely (deterministic) answer by the model.
            - This is done either in a greedy fashion or with beam search, but no sampling
            - We use the beam search so that we get th most likely sequence instead of picking just the top token, we may be exploring 5 beams, but we only care about the top 2 answers
            - We will later mark this against the actual answer based on some similarity metric (mostly Rogue-L) which tells us how correct our answer
        b. Understand the uncertainty:
            - Generate multiple (set in args.num_generations_per_prompt) different answers for each question
            - This will give us the spread in the answers that the model generates that is later going to be used to calculate uncertainty
            - To get these generations, we run a beam search with args.num_beams beams with sampling, but return only the top answer. We repeat that process args.num_generations_per_prompt times per prompt to get those many generations
        c. Finally, all that data along with some data around the question, prompt & correct answer is packaged into a response from the function
3. Finally, we get the output of that function and pickle it out for later use.



What can be done next:
0. [MUST] First step idea is to keep this script with as minimal changes as needed and use new scripts to make smallest updates. Then track the different runs on W&B.
1. [EASY] Run the analysis over a variety of seeds
2. [MEDIUM] Stop on the turn marker instead of banning it. The few-shot prompt teaches
   "Answer: <bare span> Question: ..." and never ends an answer with a period, so the
   period EOS only fires if the model breaks pattern into prose. Banning 'Question'/
   'answer' forces that break, which is why prose answers get scored against bare-span
   references and only earn partial ROUGE credit. Using ' Question'/' question' as
   stopping tokens instead would let the model answer in the demonstrated format,
   stopping in ~4 tokens rather than ~17 and matching the reference style directly.
   Note this deviates from the paper's setup and would shift accuracy upward.
'''

import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize

import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
import tqdm
import wandb
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from device_utils import DEVICE, DTYPE

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
# For a trial run, updating the fraction of data to use to 0.2 (so we have a very small sample)
# parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.2)
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='trivia_qa')

# Below is currently only for triviaqa
parser.add_argument('--dataset_num_examples', type=int, default=200, help='Identify the dataset filepath based on the number of examples downloaded. Currently used only for triviaqa')
args = parser.parse_args()

wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')

run_name = wandb.run.name

# device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.model}",
                                             torch_dtype=DTYPE,
                                             cache_dir=config.data_dir).to(DEVICE)

if args.model == 'opt-30b':
    accelerate.dispatch_model(model, device_map=config.device_map)

tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model}", use_fast=False, cache_dir=config.data_dir)

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

if args.dataset == 'coqa':
    dataset = datasets.load_from_disk(f'{config.output_dir}/coqa_dataset')
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
elif args.dataset == 'trivia_qa':
    # CURRENTLY: working through this to match the output directory from parse_triviaqa with this
    dataset = datasets.load_from_disk(config.trivia_qa_path(args.dataset_num_examples))

if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset


def encode(examples):
    return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset


if args.dataset == 'coqa':
    questions = encode_and_format_dataset(train_dataset)
elif args.dataset == 'trivia_qa':
    questions = train_dataset

dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

period_token_id = tokenizer('. ')['input_ids'][1]

# ISSUE: Some answers often start using "Questions" and "Answers"
eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']

question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")


def get_generations(model, dataloader, number_of_generations):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = 256
        sequences = []
        for batch in tqdm.tqdm(dataloader):

            input_ids = batch['input_ids'].to(DEVICE)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=5,
                                                        num_return_sequences=2,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)

            input_length = input_ids.shape[1] if args.dataset == 'trivia_qa' else batch['input_ids'].shape[1]
            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                     dtype=torch.long,
                                     device=DEVICE)
            for i in range(number_of_generations):

                generation = model.generate(input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                            eos_token_id=period_token_id,
                                            temperature=args.temperature,
                                            bad_words_ids=question_framing_ids,
                                            top_p=args.top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):

                if args.dataset == 'coqa':
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }
                elif args.dataset == 'trivia_qa':
                    few_shot_question = tokenizer.decode(input_ids[0])
                    question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                    sequence_dict = {
                        'prompt': input_ids[0],
                        'generations': generations[i],
                        'id': batch['question_id'],
                        'few_shot_question': tokenizer.decode(input_ids[0]),
                        'question': question
                    }

                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[1][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['semantic_variability_reference_answers'] = batch[
                    'semantic_variability'] if 'semantic_variability' in batch else None
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None

                    sequence_dict[rouge_type + '_to_target'] = 0.0

                sequence_dict['answer'] = batch['answer']['text'] if args.dataset == 'coqa' else batch['answer']
                sequence_dict['additional_answers'] = [x[0] for x in batch['additional_answers']
                                                      ] if args.dataset == 'coqa' else None

                sequence_dict['exact_match'] = 0.0

                reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']
                                                              ] if args.dataset == 'coqa' else batch['answer']

                for answer in reference_answers:
                    predictions = [sequence_dict['most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type],
                                                                       sequence_dict[rouge_type + '_to_target'])

                sequences.append(sequence_dict)

    return sequences


sequences = get_generations(model, dataloader, args.num_generations_per_prompt)

pathlib.Path(f'{config.output_dir}/sequences/' + run_name).mkdir(parents=True, exist_ok=True)

generations_path = f'{config.output_dir}/sequences/{run_name}/{args.model}_generations.pkl'

with open(generations_path, 'wb') as outfile:
    pickle.dump(sequences, outfile)
