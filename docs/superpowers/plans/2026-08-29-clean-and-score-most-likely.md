# Clean & Score Most-Likely Generations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move correctness scoring out of the generation stage into its own stage that runs *after* cleaning, and clean the most-likely / second-most-likely generations (text **and** token ids) so that both the accuracy label and the negative log-likelihoods are computed on de-rambled text.

**Architecture:** `generate` becomes purely generative. `clean_generated_strings` gains two more fields to clean, producing `cleaned_{most,second_most}_likely_generation{,_ids}` alongside the existing `cleaned_generations`. A new stage `score_accuracy.py` computes ROUGE/exact-match on **both** raw and cleaned text, writes both under suffixed keys, and aliases the chosen one to the bare key `analyze_results.py` already reads. `get_likelihoods` is taught to prefer the cleaned ids, mirroring the `cleaned_generations` pattern it already has.

**Tech Stack:** Python 3.11/3.12, uv, PyTorch, HuggingFace transformers + evaluate, pytest, W&B (offline).

**Spec:** This document. Requirements were established by diagnosis in the originating session; the key findings are restated inline in each task so no task depends on external context.

## Global Constraints

- Python `>=3.11,<3.13`; all commands run via `uv run` from the **repo root** (not `code/`), because `config.py` uses relative paths `./output` and `./data`.
- W&B runs offline: prefix commands with `WANDB_MODE=offline`.
- OPT pad token id is **1**. `torch.ones_like` is the established padding idiom in this codebase and depends on this.
- Existing harness convention: improvement flags default to upstream behaviour so an argument-free run reproduces the baseline. **Exception:** the changes in this plan are bug fixes, not experiment variants, and are unconditional. The one new flag (`--score_on`) selects which score is aliased, and defaults to `cleaned`.
- Do not reformat or restructure code beyond what each task specifies.
- Pipeline order after this work:
  `generate` → `clean_generated_strings` → **`score_accuracy`** → `get_semantic_similarities` → `get_likelihoods` → `compute_confidence_measure` → `analyze_results`

## Key facts established by diagnosis (do not re-derive)

- `most_likely_generation_ids` layout is `prompt ++ 256 generated tokens` (verified: `most_likely_ids[:len(prompt)] == prompt`). Same shape as one row of `generations`.
- `get_likelihoods_improved.py` strips pad tokens for sampled generations (line ~107) but **not** for the most-likely ids (lines 121/124). Cleaned ids are pad-padded, so the pad strip must be added or NLL is computed over hundreds of `<pad>` tokens.
- `analyze_results.py` reads `rougeL_to_target` off the generations pickle and defines `correct = rougeL_to_target > 0.3`.
- `generate_improved.py` reads `rougeL_to_target` / `exact_match` in **two** places besides scoring: the `--inspect_entries` printer (~line 458) and `summarise` (~line 467). Both break if scoring is removed without updating them.
- There is currently **no test infrastructure** in this repo.

---

### Task 1: Test infrastructure and shared cleaning helpers

Extract the text-filter logic into a dependency-light module so it can be unit tested without loading a model, and add pytest.

**Files:**
- Modify: `pyproject.toml`
- Create: `code/text_cleaning.py`
- Create: `tests/test_text_cleaning.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `filter_generated_text(text: str, strings_to_filter_on: list[str]) -> str`
  - `build_cleaned_ids(prompt_ids: torch.Tensor, cleaned_text: str, tokenizer, width: int, pad_token_id: int = 1) -> torch.Tensor`

- [ ] **Step 1: Add pytest to the project**

Add to `pyproject.toml` after the `[tool.uv]` block:

```toml
[dependency-groups]
dev = ["pytest"]
```

Then run: `uv sync --group dev`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_text_cleaning.py`:

```python
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'code'))

from text_cleaning import build_cleaned_ids, filter_generated_text


FILTERS = ['.', '\n', 'Question:', 'Answers:']


def test_truncates_at_first_filter_string():
    assert filter_generated_text(' December Question: next one', FILTERS) == ' December '


def test_returns_text_unchanged_when_no_filter_matches():
    assert filter_generated_text(' December', FILTERS) == ' December'


def test_applies_every_filter_not_just_the_first_match():
    text = ' Paris Answers: junk. more'
    assert filter_generated_text(text, FILTERS) == ' Paris '


def test_no_filter_string_is_silently_concatenated_with_its_neighbour():
    """Regression: a missing comma in the filter list fuses two entries into one."""
    for entry in FILTERS:
        assert entry.count(':') <= 1, f'filter entry looks fused: {entry!r}'


class StubTokenizer:
    """Returns a BOS token followed by one id per character."""

    def __call__(self, text):
        return {'input_ids': [0] + [ord(c) for c in text]}


def test_build_cleaned_ids_keeps_prompt_and_appends_completion():
    prompt = torch.tensor([5, 6, 7])
    out = build_cleaned_ids(prompt, 'ab', StubTokenizer(), width=8)
    assert out[:3].tolist() == [5, 6, 7]
    assert out[3:5].tolist() == [ord('a'), ord('b')]


def test_build_cleaned_ids_pads_to_width_with_pad_token():
    prompt = torch.tensor([5, 6, 7])
    out = build_cleaned_ids(prompt, 'ab', StubTokenizer(), width=8)
    assert len(out) == 8
    assert out[5:].tolist() == [1, 1, 1]


def test_build_cleaned_ids_truncates_when_longer_than_width():
    prompt = torch.tensor([5, 6, 7])
    out = build_cleaned_ids(prompt, 'abcdefghij', StubTokenizer(), width=6)
    assert len(out) == 6
    assert out[:3].tolist() == [5, 6, 7]


def test_build_cleaned_ids_preserves_prompt_dtype():
    prompt = torch.tensor([5, 6, 7], dtype=torch.long)
    out = build_cleaned_ids(prompt, 'ab', StubTokenizer(), width=8)
    assert out.dtype == torch.long
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_text_cleaning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'text_cleaning'`

- [ ] **Step 4: Write the implementation**

Create `code/text_cleaning.py`:

```python
"""Pure text/token cleaning helpers.

Kept free of model and W&B imports so the cleaning rules can be unit tested
without loading a tokenizer or touching the network.
"""

import torch


def filter_generated_text(text, strings_to_filter_on):
    """Truncate `text` at the earliest occurrence of any filter string.

    Applied repeatedly rather than once, because an earlier filter may expose a
    later one. Matches the behaviour the cleaning stage has always had for the
    sampled generations.
    """
    for string in strings_to_filter_on:
        if string in text:
            text = text.split(string)[0]
    return text


def build_cleaned_ids(prompt_ids, cleaned_text, tokenizer, width, pad_token_id=1):
    """Re-tokenize `cleaned_text` and splice it onto the prompt.

    Returns a 1-D tensor of exactly `width` tokens: prompt, then the cleaned
    completion, then `pad_token_id` padding. `width` should be the length of the
    uncleaned sequence so downstream shapes are unchanged.

    The leading BOS token from the tokenizer is dropped, since the prompt
    already carries one.
    """
    prompt_ids = prompt_ids.to('cpu')
    completion = tokenizer(cleaned_text)['input_ids'][1:]
    ids = torch.cat([prompt_ids,
                     torch.tensor(completion, dtype=prompt_ids.dtype)])

    out = torch.full((width,), pad_token_id, dtype=prompt_ids.dtype)
    keep = min(len(ids), width)
    out[:keep] = ids[:keep]
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_text_cleaning.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock code/text_cleaning.py tests/test_text_cleaning.py
git commit -m "test: add pytest and extract shared text-cleaning helpers"
```

---

### Task 2: Clean the most-likely and second-most-likely generations

The cleaning stage currently cleans only `generated_texts`. Add the two beam-search outputs, producing both a cleaned string and cleaned token ids for each.

**Files:**
- Modify: `code/clean_generated_strings_improved.py` (imports, and the `clean` method around lines 66-92)
- Create: `tests/test_clean_most_likely.py`

**Interfaces:**
- Consumes: `filter_generated_text`, `build_cleaned_ids` from Task 1.
- Produces, on every sample dict:
  - `cleaned_most_likely_generation` (str)
  - `cleaned_most_likely_generation_ids` (tensor, same length as `most_likely_generation_ids`)
  - `cleaned_second_most_likely_generation` (str)
  - `cleaned_second_most_likely_generation_ids` (tensor, same length as `second_most_likely_generation_ids`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_clean_most_likely.py`:

```python
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'code'))

from text_cleaning import build_cleaned_ids, filter_generated_text


class StubTokenizer:
    def __call__(self, text):
        return {'input_ids': [0] + [ord(c) for c in text]}


FILTERS = ['.', '\n', 'Question:', 'Answers:']


def clean_most_likely_fields(sample, strings_to_filter_on, tokenizer):
    """Mirror of the logic added to CleanGeneratedStrings.clean, for testing."""
    for field in ('most_likely_generation', 'second_most_likely_generation'):
        cleaned = filter_generated_text(sample[field], strings_to_filter_on)
        sample['cleaned_' + field] = cleaned
        sample['cleaned_' + field + '_ids'] = build_cleaned_ids(
            sample['prompt'], cleaned, tokenizer, len(sample[field + '_ids']))
    return sample


def make_sample():
    return {
        'prompt': torch.tensor([5, 6, 7]),
        'most_likely_generation': ' December Question: junk junk',
        'most_likely_generation_ids': torch.arange(20),
        'second_most_likely_generation': ' November Answers: junk',
        'second_most_likely_generation_ids': torch.arange(20),
    }


def test_most_likely_text_is_truncated():
    out = clean_most_likely_fields(make_sample(), FILTERS, StubTokenizer())
    assert out['cleaned_most_likely_generation'] == ' December '


def test_second_most_likely_text_is_truncated():
    out = clean_most_likely_fields(make_sample(), FILTERS, StubTokenizer())
    assert out['cleaned_second_most_likely_generation'] == ' November '


def test_cleaned_ids_keep_the_original_width():
    out = clean_most_likely_fields(make_sample(), FILTERS, StubTokenizer())
    assert len(out['cleaned_most_likely_generation_ids']) == 20
    assert len(out['cleaned_second_most_likely_generation_ids']) == 20


def test_cleaned_ids_start_with_the_prompt():
    out = clean_most_likely_fields(make_sample(), FILTERS, StubTokenizer())
    assert out['cleaned_most_likely_generation_ids'][:3].tolist() == [5, 6, 7]


def test_cleaned_ids_are_shorter_than_raw_once_padding_is_stripped():
    out = clean_most_likely_fields(make_sample(), FILTERS, StubTokenizer())
    ids = out['cleaned_most_likely_generation_ids']
    assert int((ids != 1).sum()) < 20


def test_raw_fields_are_left_intact():
    out = clean_most_likely_fields(make_sample(), FILTERS, StubTokenizer())
    assert out['most_likely_generation'] == ' December Question: junk junk'
    assert len(out['most_likely_generation_ids']) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_clean_most_likely.py -v`
Expected: PASS for the helper-mirror tests only if Task 1 landed. If Task 1 is complete these tests pass immediately — they pin the intended behaviour before it is wired into the stage. Proceed to Step 3 regardless; Step 5 is the real verification.

- [ ] **Step 3: Wire it into the cleaning stage**

In `code/clean_generated_strings_improved.py`, add the import near the other local imports:

```python
from text_cleaning import build_cleaned_ids, filter_generated_text
```

Replace the per-text filter loop inside `clean` so it uses the shared helper. The existing inner loop:

```python
                for string in self.strings_to_filter_on:
                    if string in generated_text:
                        generated_text = generated_text.split(string)[0]
```

becomes:

```python
                generated_text = filter_generated_text(generated_text, self.strings_to_filter_on)
```

Then, immediately before `cleaned_sequences.append(sample)`, add:

```python
            # The beam-search outputs were never cleaned upstream, so ROUGE was
            # scored against a full-length ramble and the NLLs were computed over
            # it. Clean both the text and the ids; get_likelihoods prefers the
            # cleaned ids when present.
            for field in ('most_likely_generation', 'second_most_likely_generation'):
                cleaned = filter_generated_text(sample[field], self.strings_to_filter_on)
                sample['cleaned_' + field] = cleaned
                sample['cleaned_' + field + '_ids'] = build_cleaned_ids(
                    sample['prompt'], cleaned, self.tokenizer,
                    len(sample[field + '_ids']))
```

- [ ] **Step 4: Extend the stage summary**

In `CleanGeneratedStrings.summarise`, add two aggregates so the effect is visible in W&B. Add before the `return`:

```python
        most_likely_raw = np.array([len(s['most_likely_generation'].split())
                                    for s in cleaned_sequences])
        most_likely_cleaned = np.array([len(s['cleaned_most_likely_generation'].split())
                                        for s in cleaned_sequences])
```

and add these entries to the returned dict:

```python
            'mean_most_likely_words_raw': float(most_likely_raw.mean()),
            'mean_most_likely_words_cleaned': float(most_likely_cleaned.mean()),
```

- [ ] **Step 5: Run the stage against the existing run and verify**

```bash
WANDB_MODE=offline uv run python code/clean_generated_strings_improved.py \
  --generation_model=opt-350m --run_id=run_1 --strings_to_filter_on=updated
```

Then verify:

```bash
uv run python -c "
import pickle
g = pickle.load(open('output/sequences/run_1/opt-350m_generations.pkl','rb'))
s = g[0]
for k in ['cleaned_most_likely_generation','cleaned_second_most_likely_generation']:
    print(k, '->', repr(s[k])[:80])
import numpy as np
raw = np.mean([len(x['most_likely_generation'].split()) for x in g])
cln = np.mean([len(x['cleaned_most_likely_generation'].split()) for x in g])
print(f'mean words raw={raw:.1f} cleaned={cln:.1f}')
assert cln < raw, 'cleaning did not shorten anything'
print('OK')
"
```

Expected: cleaned strings are short spans; `cleaned` mean is far below the `raw` mean (~171).

- [ ] **Step 6: Run the unit tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add code/clean_generated_strings_improved.py tests/test_clean_most_likely.py
git commit -m "fix: clean most-likely and second-most-likely generations (text and ids)"
```

---

### Task 3: New `score_accuracy.py` stage

Scoring moves here so it runs *after* cleaning. Computes both raw and cleaned scores and writes them back into the generations pickle, so `analyze_results.py` needs no change.

**Files:**
- Create: `code/score_accuracy.py`
- Create: `tests/test_score_accuracy.py`

**Interfaces:**
- Consumes: `cleaned_most_likely_generation` from Task 2.
- Produces, on every sample dict:
  - `exact_match_raw`, `rouge1_to_target_raw`, `rouge2_to_target_raw`, `rougeL_to_target_raw`
  - `exact_match_cleaned`, `rouge1_to_target_cleaned`, `rouge2_to_target_cleaned`, `rougeL_to_target_cleaned`
  - `exact_match`, `rouge1_to_target`, `rouge2_to_target`, `rougeL_to_target` — aliases of the variant chosen by `--score_on` (default `cleaned`)
- Also produces the module-level function `reference_answers(sample, dataset) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_score_accuracy.py`:

```python
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'code'))

from score_accuracy import ALIAS_KEYS, apply_aliases, reference_answers


def test_reference_answers_for_trivia_qa_uses_the_answer_field():
    sample = {'answer': ['December'], 'additional_answers': None}
    assert reference_answers(sample, 'trivia_qa') == ['December']


def test_reference_answers_for_coqa_includes_additional_answers():
    sample = {'answer': {'text': ['yes']}, 'additional_answers': [['yeah'], ['yep']]}
    assert reference_answers(sample, 'coqa') == ['yes', 'yeah', 'yep']


def test_apply_aliases_copies_the_cleaned_variant_by_default():
    sample = {'rougeL_to_target_raw': 0.1, 'rougeL_to_target_cleaned': 0.9,
              'exact_match_raw': 0.0, 'exact_match_cleaned': 1.0,
              'rouge1_to_target_raw': 0.1, 'rouge1_to_target_cleaned': 0.9,
              'rouge2_to_target_raw': 0.1, 'rouge2_to_target_cleaned': 0.9}
    apply_aliases(sample, 'cleaned')
    assert sample['rougeL_to_target'] == 0.9
    assert sample['exact_match'] == 1.0


def test_apply_aliases_can_select_the_raw_variant():
    sample = {'rougeL_to_target_raw': 0.1, 'rougeL_to_target_cleaned': 0.9,
              'exact_match_raw': 0.0, 'exact_match_cleaned': 1.0,
              'rouge1_to_target_raw': 0.1, 'rouge1_to_target_cleaned': 0.9,
              'rouge2_to_target_raw': 0.1, 'rouge2_to_target_cleaned': 0.9}
    apply_aliases(sample, 'raw')
    assert sample['rougeL_to_target'] == 0.1
    assert sample['exact_match'] == 0.0


def test_alias_keys_cover_every_metric_analyze_results_reads():
    assert 'rougeL_to_target' in ALIAS_KEYS
    assert 'exact_match' in ALIAS_KEYS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_score_accuracy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'score_accuracy'`

- [ ] **Step 3: Write the implementation**

Create `code/score_accuracy.py`:

```python
'''Stage 3: correctness scoring.

Split out of generate_improved.py so that scoring runs AFTER cleaning. Scoring
inside the generation stage meant ROUGE was computed against the raw beam output
- a ~171-word ramble measured against a one-to-three-word reference answer -
which drove the correctness label to zero for every question.

Both variants are always computed and stored:
  *_raw      scored against `most_likely_generation`         (upstream behaviour)
  *_cleaned  scored against `cleaned_most_likely_generation` (post-cleaning)

`--score_on` decides which pair is aliased to the bare keys that
analyze_results.py reads. Everything is written back into the generations
pickle in place, so no downstream stage needs to know this stage exists.

Usage:
    python code/score_accuracy.py --generation_model=opt-350m --run_id=run_1
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
    """The gold answers to score against, matching generate_improved.py."""
    if dataset == 'coqa':
        return list(sample['answer']['text']) + [x[0] for x in sample['additional_answers']]
    return list(sample['answer'])


def apply_aliases(sample, score_on):
    """Copy the chosen variant onto the bare keys analyze_results.py reads."""
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
        """Score `sample[text_key]` against every reference, keeping the best."""
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
                             'analyze_results.py reads. Both are always stored.')
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_score_accuracy.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the stage and compare the two variants**

```bash
WANDB_MODE=offline uv run python code/score_accuracy.py \
  --generation_model=opt-350m --run_id=run_1
```

Expected: the printed summary shows `accuracy_raw` and `accuracy_cleaned`. `accuracy_raw` should be `0.0` (reproducing the diagnosed 0/40); `accuracy_cleaned` should be ≥ that. Record both numbers — the delta is the headline result of this plan.

- [ ] **Step 6: Commit**

```bash
git add code/score_accuracy.py tests/test_score_accuracy.py
git commit -m "feat: add score_accuracy stage computing raw and cleaned correctness"
```

---

### Task 4: Remove scoring from `generate_improved.py`

**Files:**
- Modify: `code/generate_improved.py` — remove `score_against_references` and `_reference_answers`; update the `--inspect_entries` printer (~line 458) and `summarise` (~line 467); drop the now-unused metric loads.

**Interfaces:**
- Consumes: nothing.
- Produces: a generations pickle **without** `exact_match` or `*_to_target` keys. Task 3's stage adds them.

- [ ] **Step 1: Delete the scoring methods**

Delete the whole of `_reference_answers` (lines ~314-317) and `score_against_references` (lines ~319-338), plus the call site that invokes `score_against_references`. Find the call with:

```bash
grep -n "score_against_references\|_reference_answers" code/generate_improved.py
```

Remove every line it reports.

- [ ] **Step 2: Remove the now-unused metric loads**

Find and delete the `evaluate.load('rouge')` and `evaluate.load('exact_match')` assignments (`self.rouge`, `self.exact_match_metric`):

```bash
grep -n "evaluate.load\|self.rouge\|self.exact_match_metric" code/generate_improved.py
```

Keep `evaluate.load('bleurt')` or any other metric if present and still referenced; remove only what is now unreferenced. Leave the `import evaluate` if anything still uses it, otherwise remove it too.

- [ ] **Step 3: Update the inspect printer**

In the `--inspect_entries` printer, delete these lines:

```python
            print(f'  exact_match      : {sample["exact_match"]}')
            for rouge_type in ROUGE_TYPES:
                print(f'  {rouge_type}_to_target : {sample[rouge_type + "_to_target"]:.4f}')
            # analyze_results.py defines correctness as rougeL_to_target > 0.3
            print(f'  -> correct (rougeL > 0.3): {sample["rougeL_to_target"] > 0.3}')
```

and replace with:

```python
            # Correctness is scored by score_accuracy.py, after cleaning.
```

- [ ] **Step 4: Update `summarise`**

Replace the body of `summarise` with a version that reports only what generation knows:

```python
    @staticmethod
    def summarise(sequences):
        """One line of aggregates so runs can be compared without opening the pickle.

        Correctness is not scored here - see score_accuracy.py, which runs after
        cleaning so ROUGE is computed on de-rambled text.
        """
        answer_words = np.array([len(s['most_likely_generation'].split()) for s in sequences])
        return {
            'n_questions': len(sequences),
            'mean_answer_words': float(answer_words.mean()),
            'median_answer_words': float(np.median(answer_words)),
            'max_answer_words': int(answer_words.max()),
        }
```

- [ ] **Step 5: Verify nothing still references the removed keys**

Run:

```bash
grep -n "exact_match\|_to_target\|ROUGE_TYPES" code/generate_improved.py
```

Expected: only `ROUGE_TYPES` uses that remain are for `rouge_type + '_reference_answers'` (which generate still stores). If `ROUGE_TYPES` is now entirely unused, delete the constant.

- [ ] **Step 6: Verify the module still imports**

Run: `uv run python -c "import sys; sys.path.insert(0,'code'); import generate_improved; print('ok')"`
Expected: prints `ok`

- [ ] **Step 7: Commit**

```bash
git add code/generate_improved.py
git commit -m "refactor: move correctness scoring out of the generation stage"
```

---

### Task 5: Consume the cleaned ids in `get_likelihoods_improved.py`

Without this the cleaned ids from Task 2 are ignored and the NLLs keep the artifact — which is the main reason this work exists.

**Files:**
- Modify: `code/get_likelihoods_improved.py` lines ~121-125

**Interfaces:**
- Consumes: `cleaned_{most,second_most}_likely_generation_ids` from Task 2.
- Produces: unchanged output keys; only the values change.

- [ ] **Step 1: Add a cleaned-preferring accessor**

Add this method to the class, next to `get_neg_log_likelihood_for_generation`:

```python
    def _sequence_ids(self, sequence, field):
        """Prefer the cleaned ids when the cleaning stage produced them.

        Cleaned ids are padded out to the original width, so pad tokens must be
        stripped before the NLL is computed - otherwise the loss is averaged over
        hundreds of <pad> positions. The sampled-generation path already does
        this; the most-likely path never did, because it never had padding.
        """
        ids = sequence.get('cleaned_' + field, sequence[field])
        ids = ids.to(DEVICE)
        return ids[ids != self.tokenizer.pad_token_id]
```

- [ ] **Step 2: Use it for both beam outputs**

Replace:

```python
            most_likely_generation = sequence['most_likely_generation_ids'].to(DEVICE)
```

with:

```python
            most_likely_generation = self._sequence_ids(sequence, 'most_likely_generation_ids')
```

and replace:

```python
            second_most_likely_generation = sequence['second_most_likely_generation_ids'].to(DEVICE)
```

with:

```python
            second_most_likely_generation = self._sequence_ids(sequence, 'second_most_likely_generation_ids')
```

- [ ] **Step 3: Verify the accessor picks the cleaned ids and strips padding**

```bash
uv run python -c "
import pickle, torch
g = pickle.load(open('output/sequences/run_1/opt-350m_generations.pkl','rb'))
s = g[0]
raw = s['most_likely_generation_ids']
cleaned = s['cleaned_most_likely_generation_ids']
stripped = cleaned[cleaned != 1]
print('raw len            :', len(raw))
print('cleaned len        :', len(cleaned))
print('cleaned, pad-strip :', len(stripped))
assert len(stripped) < len(raw), 'cleaned ids are not shorter'
print('OK')
"
```

Expected: pad-stripped length is far below the raw 443.

- [ ] **Step 4: Commit**

```bash
git add code/get_likelihoods_improved.py
git commit -m "fix: use cleaned most-likely ids for negative log-likelihoods"
```

---

### Task 6: End-to-end re-run and verification

**Files:** none modified — this task validates the whole chain.

- [ ] **Step 1: Regenerate, with the parsing fix on**

`--fix_question_parsing` strips the `Answer:` marker that upstream's split leaves glued to every question; that string is fed to DeBERTa for semantic clustering, so it contaminates `semantic_set_ids`.

```bash
WANDB_MODE=offline uv run python code/generate_improved.py \
  --model=opt-350m --fix_question_parsing \
  --num_generations_per_prompt=5 --temperature=0.5 --num_beams=1 --top_p=1.0 \
  --fraction_of_data_to_use=0.02 --run_id=cleaned_1
```

- [ ] **Step 2: Run the rest of the chain in order**

```bash
cd "$(git rev-parse --show-toplevel)"
for stage in \
  "clean_generated_strings_improved.py --generation_model=opt-350m --strings_to_filter_on=updated" \
  "score_accuracy.py --generation_model=opt-350m" \
  "get_semantic_similarities_improved.py --generation_model=opt-350m" \
  "get_likelihoods_improved.py --generation_model=opt-350m --evaluation_model=opt-350m" \
  "compute_confidence_measure_improved.py --generation_model=opt-350m --evaluation_model=opt-350m"
do
  echo "=== $stage ==="
  WANDB_MODE=offline uv run python code/$stage --run_id=cleaned_1 || break
done
```

- [ ] **Step 3: Analyze**

```bash
WANDB_MODE=offline uv run python code/analyze_results.py -n cleaned_1 --model=opt-350m
```

- [ ] **Step 4: Verify the outcome**

```bash
uv run python -c "
import pickle, numpy as np
g = pickle.load(open('output/sequences/cleaned_1/opt-350m_generations.pkl','rb'))
raw = np.array([s['rougeL_to_target_raw'] for s in g])
cln = np.array([s['rougeL_to_target_cleaned'] for s in g])
print(f'n = {len(g)}')
print(f'correct, raw     : {(raw>0.3).sum()}/{len(g)}')
print(f'correct, cleaned : {(cln>0.3).sum()}/{len(g)}')
print(f'question glued   : {sum(s[\"question\"].rstrip().endswith(\"Answer:\") for s in g)}/{len(g)}  (expect 0)')
print(f'both classes present: {0 < (cln>0.3).sum() < len(g)}')
"
```

Expected: the glued-marker count is `0`. Whether both classes are present determines whether the AUROCs compute.

**If `correct` is still all-zero:** the AUROCs will still be `nan`. That is a model-capability limit, not a bug in this work — diagnosis established that the gold answer appears anywhere in the most-likely generation for only 3/40 questions, so ~8% is the ceiling at `opt-350m`. Escalating to `opt-1.3b` is the next lever, and is out of scope for this plan. Record the raw-vs-cleaned delta either way; that delta is this plan's deliverable.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit any incidental fixes**

```bash
git add -A
git commit -m "chore: end-to-end verification of cleaned scoring pipeline"
```
