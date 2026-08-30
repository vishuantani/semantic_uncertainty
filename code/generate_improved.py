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
        a. Produce the answer that will later be marked for accuracy:
            - First generate the most likely (deterministic) answer by the model.
            - This is done either in a greedy fashion or with beam search, but no sampling
            - We use the beam search so that we get th most likely sequence instead of picking just the top token, we may be exploring 5 beams, but we only care about the top 2 answers
            - CHANGED FROM UPSTREAM: this stage no longer marks the answer. Upstream
              scored it here with ROUGE/exact-match; that scoring moved to
              new_score_accuracy.py so it runs AFTER clean_generated_strings_improved.py
              (see change 4 below). This stage only stores the generation.
        b. Understand the uncertainty:
            - Generate multiple (set in args.num_generations_per_prompt) different answers for each question
            - This will give us the spread in the answers that the model generates that is later going to be used to calculate uncertainty
            - To get these generations, we run a beam search with args.num_beams beams with sampling, but return only the top answer. We repeat that process args.num_generations_per_prompt times per prompt to get those many generations
        c. Finally, all that data along with some data around the question, prompt & correct answer is packaged into a response from the function
3. Finally, we get the output of that function and pickle it out for later use.


HOW TO USE THIS FILE

This is the experiment harness version of generate.py. Every improvement is behind a
flag that defaults to the CURRENT behaviour of generate.py, so running it with no
arguments reproduces the baseline. Turn one flag on at a time and compare runs in W&B.

    python generate_improved.py                          # baseline
    python generate_improved.py --stop_on_turn_marker    # docstring item 2
    python generate_improved.py --batch_samples          # one call instead of N calls
    python generate_improved.py --ban_list upstream      # the paper's narrow list

Each invocation gets a fresh W&B run id unless you pass --run_id, so experiments do not
overwrite each other's pickles or collapse into one run in the UI. The id is printed at
the start and end of the run - pass it to the downstream stages.

The improvement flags are logged to the W&B config, so you can group/filter runs by them.


What can be done next:
1. [EASY] Run the analysis over a variety of seeds
2. [MEDIUM] Stop on the turn marker instead of banning it. The few-shot prompt teaches
   "Answer: <bare span> Question: ..." and never ends an answer with a period, so the
   period EOS only fires if the model breaks pattern into prose. Banning 'Question'/
   'answer' forces that break, which is why prose answers get scored against bare-span
   references and only earn partial ROUGE credit. Using ' Question'/' question' as
   stopping tokens instead would let the model answer in the demonstrated format,
   stopping in ~4 tokens rather than ~17 and matching the reference style directly.
   Note this deviates from the paper's setup and would shift accuracy upward.
   -> now available as --stop_on_turn_marker


Changes to the original implementation:
1. --ban_list {upstream,extended}, default extended. Upstream's six-entry list only bans 
    capitalised markers; opt-350m escapes via lowercase 'question:' and never emits a
    period, so it runs to max_length. Larger models hold the format and do not need this.
2. --stop_on_turn_marker (off): stop on Question:/Answer: instead of banning them, so the
    model answers as a bare span like the few-shot examples. Also keeps '\n' banned and
    strips the marker token from the decoded text, since it is part of the returned
    sequence and would otherwise inflate length and hurt ROUGE precision.
3. --fix_question_parsing (off): the upstream split uses 'Answer: ' with a trailing space
     while the prompt ends 'Answer:', so the marker stays glued to every stored question.
4. Correctness scoring REMOVED from this stage (not a flag - unconditional). Upstream's
    GenerationExperiment loaded the rouge/exact_match metrics and ran
    _reference_answers + score_against_references here, writing exact_match and
    rouge*_to_target into the pickle during generation. That is before
    clean_generated_strings runs, so ROUGE compared the raw beam output - a ~171-word
    repetition loop - against a one-to-three-word reference, making rougeL_to_target ~0
    for every question, `correct` 0/40, and every AUROC in analyze_results.py nan. All of
    it now lives in new_score_accuracy.py, a stage that runs after cleaning and writes the
    same bare keys back into the same pickle, so analyze_results.py is unchanged. This
    stage's W&B summary correspondingly no longer reports accuracy_rougeL_over_0.3,
    n_correct, mean_rougeL_to_target or mean_exact_match.
'''

import argparse
import os
import pathlib
import pickle
import random
import uuid
import warnings

import accelerate
import config
import datasets
import numpy as np
import torch
import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from device_utils import DEVICE, DTYPE

# RENAMED FROM UPSTREAM: generate.py calls this list `eos_tokens`, but it is NOT used
# as end-of-sequence anywhere. It is passed as `bad_words_ids`, so these tokens are
# SUPPRESSED - the model is forbidden from ever emitting them. The only real EOS in the
# default configuration is the period (see _build_generation_controls). The upstream
# name inverts the actual behaviour, which is what makes the stopping bug hard to see:
# banning 'Question:' and '\n' removes the very tokens the few-shot format teaches the
# model to end an answer with, so it cannot terminate and rambles to max_length.
# The six entries the paper/upstream repo uses.
UPSTREAM_BANNED_TOKENS = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']

# ISSUE: opt-350m is weak enough to drift into lowercase prose and slip past the
# upstream list, so it never emits a period and runs to max_length. Larger models hold
# the few-shot format and do not need these extra entries.
EXTENDED_BANNED_TOKENS = UPSTREAM_BANNED_TOKENS + [
    'question:', ' question:', 'answer:', ' answer:', 'q:',
    ' Questions:', ' Answers:',
    ' questions:', ' answers:'
    # 'Questions:', 'Answers:', 'questions:', 'answers:',  # These here cause tokenising issues: they break up as multiple tokens (eg. "Answers:" is broken into "An" + whatever) and silently, only the first token is picked, banning unwanted tokens and not banning the ones we want. The space prefixed ones work fine
]

# Used by --stop_on_turn_marker: rather than banning these, treat them as end-of-sequence
# so the model may answer in the bare-span format the few-shot prompt demonstrates.
# These are the ONLY tokens in this file that genuinely become `eos_token_id`, and only
# when that flag is set.
#
# CAUTION: every entry is reduced to its FIRST token id by _first_token_id, so a marker
# must tokenize to one whole word. The space-prefixed forms are safe (' Answers:' ->
# [' Answers', ':']); the bare plural is not ('Answers:' -> ['An', 'swers', ':'], so it
# would stop on 'An' and truncate any answer starting "An..."). Verify before adding to
# --turn_marker_tokens. Note this list omits the plural forms the model actually emits,
# so --stop_on_turn_marker alone does not stop an ' Answers:' loop.
TURN_MARKER_TOKENS = ['Question:', ' Question:', 'question:', ' question:',
                      'Answer:', ' Answer:', 'answer:', ' answer:']

OPT_MODELS = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']


class GenerationExperiment:
    """Stage 1 of the pipeline, with each proposed improvement behind a flag.

    All flags default to the behaviour of generate.py, so an argument-free run is the
    baseline that every experiment should be compared against.
    """

    def __init__(self, args):
        self.args = args
        self.seed_value = args.seed

        self._set_seeds()
        self._load_model_and_tokenizer()
        self._build_dataloader()
        self._build_generation_controls()
        # REMOVED - CHANGED FROM UPSTREAM: upstream loaded the rouge and exact_match
        # metrics here. They moved to new_score_accuracy.py along with the scoring they
        # served, so this stage no longer needs `evaluate` at construction time.

    # ------------------------------------------------------------------ setup

    def _set_seeds(self):
        # PYTHONHASHSEED only takes effect if set before the interpreter starts, so this
        # assignment is a no-op. Kept to match generate.py.
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)

    def _load_model_and_tokenizer(self):
        self.model = AutoModelForCausalLM.from_pretrained(f'facebook/{self.args.model}',
                                                          dtype=DTYPE,
                                                          cache_dir=config.data_dir).to(DEVICE)

        if self.args.model == 'opt-30b' and DEVICE.type == 'cuda':
            accelerate.dispatch_model(self.model, device_map=config.device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(f'facebook/{self.args.model}',
                                                       use_fast=False,
                                                       cache_dir=config.data_dir)

    def _encode_coqa(self, examples):
        return self.tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:',
                              truncation=False, padding=False)

    def _build_dataloader(self):
        self.id_to_question_mapping = None

        if self.args.dataset == 'coqa':
            dataset = datasets.load_from_disk(f'{config.output_dir}/coqa_dataset')
            self.id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
        elif self.args.dataset == 'trivia_qa':
            dataset = datasets.load_from_disk(config.trivia_qa_path(self.args.dataset_num_examples))
        else:
            raise ValueError(f'unknown dataset {self.args.dataset!r}')

        if self.args.fraction_of_data_to_use < 1.0:
            train_dataset = dataset.train_test_split(
                test_size=(1 - self.args.fraction_of_data_to_use), seed=self.seed_value)['train']
        else:
            train_dataset = dataset

        if self.args.dataset == 'coqa':
            questions = train_dataset.map(self._encode_coqa, batched=False, load_from_cache_file=False)
            questions.set_format(type='torch', columns=['input_ids', 'attention_mask'],
                                 output_all_columns=True)
        else:
            questions = train_dataset

        self.dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

    def _first_token_id(self, text):
        """OPT prepends BOS, so index 1 is the first real token of `text`."""
        return self.tokenizer(text)['input_ids'][1]

    def _token_ids_without_bos(self, text):
        ids = self.tokenizer(text)['input_ids']
        return ids[1:] if ids and ids[0] == self.tokenizer.bos_token_id else ids

    def _validate_single_token_markers(self, markers, list_name):
        """Reject markers that do not reduce to exactly one token.

        NEW - NO UPSTREAM EQUIVALENT. _first_token_id keeps only the FIRST token of a
        marker, so a marker whose word is absent from the BPE vocabulary is silently
        truncated to a fragment: 'Answers:' tokenizes as ['An', 'swers', ':'], so the ban
        (or EOS, under --stop_on_turn_marker) lands on 'An'. That bans every bare
        "An..." while leaving the marker the model actually emits unbanned - the failure
        is invisible without inspecting token ids. Upstream never checked, which is how
        three fragmenting entries sat in the extended ban list unnoticed.

        The space-prefixed form is normally the fix: ' Answers' is a single token even
        though bare 'Answers' is not, and in generated text a turn marker always follows
        a space anyway.
        """
        broken = []
        for marker in markers:
            word = marker[:-1] if marker.endswith(':') else marker
            ids = self._token_ids_without_bos(word)
            if len(ids) != 1:
                pieces = [self.tokenizer.decode([i]) for i in ids]
                broken.append(f'    {marker!r} -> {pieces} ({len(ids)} tokens)')

        if broken:
            raise ValueError(
                f'{list_name}: {len(broken)} marker(s) do not tokenize to a single token, '
                f'so only a fragment would be used:\n' + '\n'.join(broken) +
                "\n  Use the space-prefixed form (e.g. ' Answers:' not 'Answers:'), or "
                'drop the entry.')

    def _build_generation_controls(self):
        """Decide how generation is stopped: ban the turn markers, or stop on them."""
        # RENAMED FROM UPSTREAM: `eos_tokens` -> `banned_tokens` and
        # `question_framing_ids` -> `banned_token_ids`. Same values and same behaviour as
        # generate.py; only the names change, to say what these actually do (they feed
        # `bad_words_ids`, not `eos_token_id`).
        banned_tokens = (UPSTREAM_BANNED_TOKENS if self.args.ban_list == 'upstream'
                         else EXTENDED_BANNED_TOKENS)
        # Throw an error if any of the banned tokens are multiple tokens in length
        self._validate_single_token_markers(banned_tokens,
                                            f'--ban_list {self.args.ban_list}')
        self.banned_token_ids = [[self._first_token_id(t)] for t in banned_tokens]

        self.period_token_id = self._first_token_id('. ')

        # The marker token that ends generation decodes as e.g. ' Question', so
        # _decode_completion strips the bare word. Derived from the marker list rather
        # than hardcoded so the two cannot drift apart. Longest first, in case one word
        # is a suffix of another.
        self.turn_marker_words = sorted(
            {t.strip().rstrip(':') for t in self.args.turn_marker_tokens},
            key=len, reverse=True)

        if self.args.stop_on_turn_marker:
            # Do not ban the markers; end the sequence when one appears. This lets the
            # model answer as a bare span the way the few-shot examples demonstrate.
            # '\n' is still banned - it is a formatting artifact, not a turn marker, and
            # without it the model restarts the few-shot preamble.
            # Same fragmentation trap as the ban list, and this list is settable
            # from the CLI via --turn_marker_tokens, so validate before use.
            self._validate_single_token_markers(self.args.turn_marker_tokens,
                                                '--turn_marker_tokens')
            marker_ids = sorted({self._first_token_id(t) for t in self.args.turn_marker_tokens}) # Sorting and set are cosmetic
            self.generation_control_kwargs = {'eos_token_id': [self.period_token_id] + marker_ids,
                                'bad_words_ids': [[self._first_token_id('\n')]]}
        else:
            self.generation_control_kwargs = {'eos_token_id': self.period_token_id,
                                'bad_words_ids': self.banned_token_ids}

    def _decode_completion(self, sequence, prompt_length):
        """Decode the generated continuation, minus the prompt.

        Under --stop_on_turn_marker the marker token that ended generation is part of the
        returned sequence, so ' Golding Question' comes back instead of ' Golding'. Left
        in place it would inflate length and hurt ROUGE precision.
        """
        text = self.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)

        if self.args.stop_on_turn_marker:
            stripped = text.rstrip()
            for word in self.turn_marker_words:
                if stripped.endswith(word):
                    return stripped[:-len(word)].rstrip()

        return text

    # ------------------------------------------------------------- generation

    def _max_length_for(self, input_ids):
        return input_ids.shape[1] + self.args.max_length_of_generated_sequence

    def generate_most_likely(self, input_ids):
        """The deterministic 'best answer', scored later for correctness."""
        if self.args.decoding_method == 'beam_search':
            # num_return_sequences=2 is what supplies second_most_likely_generation_*, and
            # HF requires num_return_sequences <= num_beams, so 2 is the floor.
            if self.args.most_likely_num_beams < 2:
                raise ValueError(
                    f'--most_likely_num_beams must be >= 2, got '
                    f'{self.args.most_likely_num_beams}. Beam search returns 2 sequences '
                    'so that second_most_likely_generation_ids exists, which '
                    'get_likelihoods.py needs for '
                    'average_neg_log_likelihood_of_second_most_likely_gen. Note this is '
                    'separate from --num_beams, which controls the sampled generations '
                    'and should stay at 1.')

            return self.model.generate(input_ids,
                                       num_beams=self.args.most_likely_num_beams,
                                       num_return_sequences=2,
                                       do_sample=False,
                                       max_length=self._max_length_for(input_ids),
                                       **self.generation_control_kwargs)

        if self.args.decoding_method == 'greedy':
            # HF rejects num_return_sequences > 1 when num_beams == 1 and do_sample is
            # False, so there is no second-best sequence to record. warnings.warn dedupes
            # by call site, so this prints once rather than once per question.
            warnings.warn(
                "--decoding_method greedy returns a single sequence, so "
                "second_most_likely_generation_ids cannot be produced. run() will raise "
                "IndexError on the first question, and get_likelihoods.py needs that "
                "field for average_neg_log_likelihood_of_second_most_likely_gen. Use "
                "--decoding_method beam_search for any run you intend to pass downstream.",
                RuntimeWarning,
                stacklevel=2)

            return self.model.generate(input_ids,
                                       num_beams=1,
                                       do_sample=False,
                                       max_length=self._max_length_for(input_ids),
                                       **self.generation_control_kwargs)

        raise ValueError(f'unknown decoding_method {self.args.decoding_method!r}')

    def sample_generations(self, input_ids, number_of_generations):
        """The N sampled answers whose disagreement becomes the uncertainty signal."""
        sample_kwargs = dict(do_sample=True,
                             num_beams=self.args.num_beams,
                             max_length=self._max_length_for(input_ids),
                             temperature=self.args.temperature,
                             top_p=self.args.top_p,
                             **self.generation_control_kwargs)

        generations = torch.ones((number_of_generations, self._max_length_for(input_ids)),
                                 dtype=torch.long,
                                 device=DEVICE)

        if self.args.batch_samples:
            # One call returning N sequences. Only equivalent to the loop when
            # num_beams == 1: with beam search this returns the N ranked beams of a
            # SINGLE search, which share prefixes and collapse the diversity that the
            # uncertainty measure depends on.
            if self.args.num_beams != 1:
                raise ValueError('--batch_samples requires --num_beams 1; with beam search '
                                 'it returns near-duplicate beams from one search')
            out = self.model.generate(input_ids,
                                      num_return_sequences=number_of_generations,
                                      **sample_kwargs)
            generations[:, :out.shape[1]] = out
        else:
            for i in range(number_of_generations):
                out = self.model.generate(input_ids, num_return_sequences=1, **sample_kwargs)
                generations[i, :out.shape[1]] = out

        return generations

    # ---------------------------------------------------------------- scoring

    # REMOVED - CHANGED FROM UPSTREAM: _reference_answers and
    # score_against_references moved to new_score_accuracy.py. Upstream scored
    # correctness here, during generation, which is before cleaning runs - so
    # ROUGE compared a ~171-word ramble against a 1-3 word answer and produced
    # 0/40 correct. Scoring now happens after clean_generated_strings_improved.py.

    def _build_sequence_dict(self, batch, input_ids, generations, index):
        if self.args.dataset == 'coqa':
            return {
                'prompt': batch['input_ids'][index].to('cpu'),
                'generations': generations[index].to('cpu'),
                'id': batch['id'],
                'question': self.id_to_question_mapping[batch['id'][0]],
            }

        few_shot_question = self.tokenizer.decode(input_ids[0])
        # NOTE: upstream splits on 'Answer: ' (trailing space) but the prompt ends with
        # 'Answer:', so the marker stays attached. Kept for fidelity with the paper;
        # --fix_question_parsing removes it.
        marker = 'Answer:' if self.args.fix_question_parsing else 'Answer: '
        question = few_shot_question.split('Question: ')[-1].split(marker)[0]
        if self.args.fix_question_parsing:
            question = question.strip()

        return {
            'prompt': input_ids[0].to('cpu') if self.args.pickle_on_cpu else input_ids[0],
            'generations': (generations[index].to('cpu') if self.args.pickle_on_cpu
                            else generations[index]),
            'id': batch['question_id'],
            'few_shot_question': few_shot_question,
            'question': question,
        }

    # -------------------------------------------------------------------- run

    def run(self):
        number_of_generations = self.args.num_generations_per_prompt
        sequences = []

        with torch.no_grad():
            for batch in tqdm.tqdm(self.dataloader):
                input_ids = batch['input_ids'].to(DEVICE)

                most_likely_generation = self.generate_most_likely(input_ids)
                generations = self.sample_generations(input_ids, number_of_generations)
                generations = torch.reshape(generations, (-1, number_of_generations,
                                                          generations.shape[-1]))

                # batch_size is 1, so this loop runs exactly once.
                for index in range(generations.shape[0]):
                    sequence_dict = self._build_sequence_dict(batch, input_ids, generations, index)

                    prompt_length = len(batch['input_ids'][index])
                    sequence_dict['generated_texts'] = [
                        self._decode_completion(generation, prompt_length)
                        for generation in generations[index]
                    ]

                    sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                    sequence_dict['most_likely_generation'] = self._decode_completion(
                        most_likely_generation[0], prompt_length)
                    sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                    sequence_dict['second_most_likely_generation'] = self._decode_completion(
                        most_likely_generation[1], prompt_length)

                    sequence_dict['semantic_variability_reference_answers'] = (
                        batch['semantic_variability'] if 'semantic_variability' in batch else None)
                    for rouge_type in ROUGE_TYPES:
                        sequence_dict[rouge_type + '_reference_answers'] = (
                            batch[rouge_type] if rouge_type in batch else None)

                    sequence_dict['answer'] = (batch['answer']['text'] if self.args.dataset == 'coqa'
                                               else batch['answer'])
                    sequence_dict['additional_answers'] = (
                        [x[0] for x in batch['additional_answers']] if self.args.dataset == 'coqa'
                        else None)

                    sequences.append(sequence_dict)

        return sequences

    def save(self, sequences, run_name):
        directory = pathlib.Path(f'{config.output_dir}/sequences/{run_name}')
        directory.mkdir(parents=True, exist_ok=True)
        path = str(directory / f'{self.args.model}_generations.pkl')

        with open(path, 'wb') as outfile:
            pickle.dump(sequences, outfile)

        return path

    # ------------------------------------------------------------ inspection

    @staticmethod
    def test_output_pickle(path, number_of_entries=3):
        """Reload the pickle we just wrote and print the first few entries.

        Reads from disk rather than reusing `sequences` in memory so this also checks the
        round-trip actually works before the downstream stages depend on it.
        """

        def describe(value):
            """Tensors print as shape/device rather than dumping hundreds of token ids."""
            if isinstance(value, torch.Tensor):
                return f'Tensor{tuple(value.shape)} on {value.device}'
            return repr(value)

        with open(path, 'rb') as infile:
            reloaded = pickle.load(infile)

        print(f'\n{path}\n{len(reloaded)} entries')
        print('keys:', sorted(reloaded[0].keys()))

        for i, sample in enumerate(reloaded[:number_of_entries]):
            print('=' * 78)
            print(f'[{i}] id={describe(sample["id"])}')
            print(f'  question   : {sample["question"]!r}')
            print(f'  answer     : {describe(sample["answer"])}')
            print(f'  most likely: {sample["most_likely_generation"]!r}')
            print(f'  2nd likely : {sample["second_most_likely_generation"]!r}')
            print(f'  sampled generations ({len(sample["generated_texts"])}):')
            for j, text in enumerate(sample['generated_texts']):
                print(f'    {j}: {text!r}')
            # CHANGED FROM UPSTREAM: correctness is no longer known at this stage.
            # See new_score_accuracy.py, which scores after cleaning.
            print(f'  prompt tensor     : {describe(sample["prompt"])}')
            print(f'  generations tensor: {describe(sample["generations"])}')

    @staticmethod
    def summarise(sequences):
        """One line of aggregates so runs can be compared without opening the pickle.

        CHANGED FROM UPSTREAM: the accuracy aggregates
        (accuracy_rougeL_over_0.3, n_correct, mean_rougeL_to_target,
        mean_exact_match) moved to new_score_accuracy.py, which scores after
        cleaning. Only generation-side statistics remain. max_answer_words is new
        and is here because a value equal to --max_length_of_generated_sequence is
        the signature of a generation that never stopped.
        """
        answer_words = np.array([len(s['most_likely_generation'].split()) for s in sequences])
        return {
            'n_questions': len(sequences),
            'mean_answer_words': float(answer_words.mean()),
            'median_answer_words': float(np.median(answer_words)),
            'max_answer_words': int(answer_words.max()),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- unchanged from generate.py -------------------------------------------------
    parser.add_argument('--type_of_question', type=str)
    parser.add_argument('--num_generations_per_prompt', type=int, default=5)
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.2)
    parser.add_argument('--model', type=str, default='opt-350m', choices=OPT_MODELS)
    parser.add_argument('--run_id', type=str, default=None,
                        help='W&B run id. Omit to generate a fresh one per experiment.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_beams', type=int, default=5,
                        help='beams for the SAMPLED generations. 1 = plain multinomial '
                             'sampling, which is what the uncertainty estimate wants.')
    parser.add_argument('--decoding_method', type=str, default='beam_search',
                        choices=['beam_search', 'greedy'])
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='trivia_qa',
                        choices=['trivia_qa', 'coqa'])
    parser.add_argument('--dataset_num_examples', type=int, default=200,
                        help='Identifies the dataset filepath written by parse_triviaqa.py')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--max_length_of_generated_sequence', type=int, default=256)
    parser.add_argument('--most_likely_num_beams', type=int, default=5,
                        help='beams for the deterministic answer. Must be >= 2 because '
                             'num_return_sequences=2 supplies second_most_likely_*.')

    # --- improvement toggles, all defaulting to current behaviour --------------------
    parser.add_argument('--ban_list', type=str, default='extended',
                        choices=['upstream', 'extended'],
                        help="'upstream' is the paper's six-entry list; 'extended' adds "
                             'lowercase/plural variants that opt-350m slips past.')
    parser.add_argument('--turn_marker_tokens', nargs='+', default=TURN_MARKER_TOKENS,
                        metavar='TOKEN',
                        help='Markers used by --stop_on_turn_marker. Only the first token '
                             'of each is used as an eos id, and the bare word (colon and '
                             'whitespace stripped) is removed from the decoded text.')
    parser.add_argument('--stop_on_turn_marker', action='store_true',
                        help='Stop on Question:/Answer: instead of banning them, so the '
                             'model answers as a bare span. Deviates from the paper.')
    parser.add_argument('--batch_samples', action='store_true',
                        help='Draw all N samples in one generate() call. Requires '
                             '--num_beams 1.')
    parser.add_argument('--fix_question_parsing', action='store_true',
                        help="Strip the trailing 'Answer:' left on the question by the "
                             'upstream split. Deviates from the paper.')
    parser.add_argument('--pickle_on_cpu', action='store_true',
                        help='Move prompt/generations tensors to CPU before pickling '
                             'so the pickle is not tied to the GPU it was made on.')
    parser.add_argument('--inspect_entries', type=int, default=3,
                        help='How many entries to print from the saved pickle. 0 skips.')

    return parser.parse_args()


def main():
    args = parse_args()

    # wandb.util.generate_id() was removed in wandb 0.28, so mint our own.
    run_id = args.run_id or uuid.uuid4().hex[:8]
    wandb.init(project='nlg_uncertainty', id=run_id, config=args, resume='allow')
    # wandb.run.name is None in offline mode, which would write everything to
    # output/sequences/None/ and collide across runs.
    run_name = wandb.run.name or run_id
    print(f'run_id={run_id}  run_name={run_name}')

    experiment = GenerationExperiment(args)
    sequences = experiment.run()
    path = experiment.save(sequences, run_name)

    summary = GenerationExperiment.summarise(sequences)
    wandb.log(summary)
    print('\n' + '  '.join(f'{k}={v}' for k, v in summary.items()))

    if args.inspect_entries:
        GenerationExperiment.test_output_pickle(path, args.inspect_entries)

    print(f'\nrun_id={run_id}  ->  pass this as --run_id to the downstream stages')
    wandb.finish()


if __name__ == '__main__':
    main()
