'''
This file does the following:
1. Downloads a pre-trained tokenizer from HuggingFace - we use Facebook/opt-350m for now
2. Downloads the TriviaQA dataset from HuggingFace: It contains a list of trivia style question and answers
3. The aim is to then tokenize the question and answers bbased on the tokenizer:
    ii. Before tokenizing, pre-append the examples from the prompt to each question and end the question with an "Answer:" tag so the LLM downstream knows it has to continue the text from that.
    iii. Tokenize the question
    iv. Tokenize the answer (separate to the question)
4. Collect them into tensors and save

A few notes:
1. The questions will be of different lengths and therefore their tokenized versions will be of different lengths (why is it a problem?). To prevent this, we normally pad so all represnetations become equal to the size of the longest input.
    However, in the tokenization process, we do so in a batched fashion. And if we pad, it will not be a global padding, but a batch level padding and still result in sequences of differing lengths across the dataset. Padding is thus saved for when we actually are going to feed the data in.
2. For now, we are subsetting this to a smaller list of `args.num_examples` number of examples instead of the whole dataset for a quicker run.
3. OPT is a base completion model not finetuned for instruction following. If we just give it the question with a "Answer:" it will just continue with the most likely tokens - like more questions.
    That is why we tokenize each question with a "prompt" which is a personality + a collection of examples so the agent knows how to answer.
'''


import argparse
import pathlib

import datasets
from transformers import AutoTokenizer

import config

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--num_examples', type=int, default=200,
                    help='cap the validation split at N examples; 0 uses the full split')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
args = parser.parse_args()

# Commenting out as it is not used in this script
# model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.model}",
#                                              torch_dtype=DTYPE,
#                                              cache_dir=config.data_dir).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model}", cache_dir=config.data_dir)

# if args.model == 'opt-30b' and DEVICE.type == 'cuda':
#     accelerate.dispatch_model(model, device_map=config.device_map)

seed_value = 10

# keep the slice size in the path so a trimmed run can't be mistaken for a full one
dataset_path = config.trivia_qa_path(args.num_examples)


if not pathlib.Path(dataset_path).exists():

    print(f'Preprocessing dataset -> {dataset_path}')
    val_data = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    if args.num_examples:
        val_data = val_data.select(range(args.num_examples))

    # streamed: pulls one shard instead of downloading all 138k train rows for these 10
    train_data = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", streaming=True)

    few_shot_prompt = 'This is a bot that correctly answers questions. \n'
    for sample in train_data.take(10):
        few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + sample['answer']['value'] + ' '

    batch_size = 4  # change to 16 for full training

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        answers = [answer["value"] for answer in batch["answer"]]

        batch_with_prompt = [few_shot_prompt + "Question: " + question + " Answer:" for question in batch["question"]]
        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        outputs = tokenizer(answers, padding=False, truncation=False)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch['answer'] = answers

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["search_results", "question_source", "entity_pages"])
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    val_data.save_to_disk(dataset_path)
else:

    val_data = datasets.load_from_disk(dataset_path)
