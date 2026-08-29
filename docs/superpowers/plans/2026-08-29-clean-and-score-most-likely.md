# Clean & Score Most-Likely Generations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move correctness scoring out of the generation stage into its own stage that runs *after* cleaning, and clean the most-likely / second-most-likely generations (text **and** token ids) so that both the accuracy label and the negative log-likelihoods are computed on de-rambled text.

**Architecture:** `generate` becomes purely generative. `clean_generated_strings` gains two more fields to clean, producing `cleaned_{most,second_most}_likely_generation{,_ids}` alongside the existing `cleaned_generations`. A new stage `new_score_accuracy.py` computes ROUGE/exact-match on **both** raw and cleaned text, writes both under suffixed keys, and aliases the chosen one to the bare key `analyze_results.py` already reads. `get_likelihoods` is taught to prefer the cleaned ids, mirroring the `cleaned_generations` pattern it already has.

**Tech Stack:** Python 3.11/3.12, uv, PyTorch, HuggingFace transformers + evaluate, W&B (offline).

**Spec:** This document.

## Global Constraints

These are **hard requirements from the user** and apply to every task:

1. **No tests.** Do not create test files, do not add pytest, do not modify `pyproject.toml`.
2. **Only `_improved.py` files may be modified.** Never touch `generate.py`, `clean_generated_strings.py`, `get_likelihoods.py`, `compute_confidence_measure.py`, `parse_triviaqa.py`, or `analyze_results.py`.
3. **New files are prefixed `new_`.** This plan creates exactly two: `code/new_text_cleaning.py` and `code/new_score_accuracy.py`.
4. **Do not run the pipeline.** No `generate`, no cleaning, no scoring, no analysis runs. Making the code changes is the whole deliverable. A syntax/import check is permitted and expected; a pipeline stage run is not.
5. **Every deviation from the original implementation must be documented in a comment**, in the code, at the point of change. State what upstream did, what this does instead, and why. This is a research fork whose value depends on knowing exactly where it departs from the paper — a silent behavioural change is a defect even if the code is correct.
6. Python `>=3.11,<3.13`; commands run via `uv run` from the **repo root**, because `config.py` uses relative paths `./output` and `./data`.
7. OPT pad token id is **1**. `torch.ones_like` is the established padding idiom here and depends on it.

Pipeline order after this work:
`generate` → `clean_generated_strings` → **`new_score_accuracy`** → `get_semantic_similarities` → `get_likelihoods` → `compute_confidence_measure` → `analyze_results`

## Key facts established by diagnosis (do not re-derive)

- `most_likely_generation_ids` layout is `prompt ++ 256 generated tokens` (verified: `most_likely_ids[:len(prompt)] == prompt`). Same shape as one row of `generations`.
- `get_likelihoods_improved.py` strips pad tokens for sampled generations (line ~107) but **not** for the most-likely ids (lines 121/124). Cleaned ids are pad-padded, so the pad strip must be added or NLL is averaged over hundreds of `<pad>` tokens.
- `analyze_results.py` reads `rougeL_to_target` off the generations pickle and defines `correct = rougeL_to_target > 0.3`. It must keep working untouched (constraint 2).
- `generate_improved.py` reads `rougeL_to_target` / `exact_match` in **two** places besides scoring: the `--inspect_entries` printer (~line 458) and `summarise` (~line 467). Both break if scoring is removed without updating them.
- `CleanGeneratedStrings.summarise` has signature `summarise(cleaned_sequences)` and `numpy as np` is already imported in that module.
- Diagnosis result being fixed: scoring ran inside `generate`, so ROUGE compared a ~171-word ramble against a 1-3 word answer, driving `correct` to 0/40 and every AUROC to `nan`.

---

### Task 1: Shared cleaning helpers — `code/new_text_cleaning.py`

**Files:**
- Create: `code/new_text_cleaning.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `filter_generated_text(text: str, strings_to_filter_on: list[str]) -> str`
  - `build_cleaned_ids(prompt_ids: torch.Tensor, cleaned_text: str, tokenizer, width: int, pad_token_id: int = 1) -> torch.Tensor`

- [ ] **Step 1: Create the module**

Create `code/new_text_cleaning.py` with exactly this content:

```python
"""Shared text/token cleaning helpers.

NEW FILE - no upstream equivalent.

`filter_generated_text` is lifted verbatim from the inner loop of
CleanGeneratedStrings.clean so that the same rule can be applied to the
beam-search outputs as well as the sampled generations. Behaviour is unchanged
from upstream; only its location is new.

`build_cleaned_ids` has no upstream equivalent at all. Upstream rebuilt cleaned
token ids only for the sampled generations (inline in clean()); the beam-search
outputs were never re-tokenized because they were never cleaned.
"""

import torch


def filter_generated_text(text, strings_to_filter_on):
    """Truncate `text` at the earliest occurrence of any filter string.

    Applied for every filter in turn rather than stopping at the first match,
    because truncating on one filter can expose another. This is exactly what
    upstream's inline loop did.
    """
    for string in strings_to_filter_on:
        if string in text:
            text = text.split(string)[0]
    return text


def build_cleaned_ids(prompt_ids, cleaned_text, tokenizer, width, pad_token_id=1):
    """Re-tokenize `cleaned_text` and splice it onto the prompt.

    Returns a 1-D tensor of exactly `width` tokens: prompt, then the cleaned
    completion, then padding. `width` is the length of the uncleaned sequence,
    so downstream shapes are unchanged.

    Mirrors how upstream builds `cleaned_generations` inside clean():
      - the tokenizer's leading BOS is dropped (`[1:]`), since the prompt has one
      - the result is padded with `pad_token_id` (1 for OPT), which is what
        upstream's `torch.ones_like` produced
      - it is truncated to `width` if the cleaned completion somehow runs long

    NOTE: padding means consumers MUST strip pad tokens before computing a
    per-token loss. get_likelihoods already does this for the sampled
    generations; the most-likely path needs it added (see Task 5).
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

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "import sys; sys.path.insert(0,'code'); import new_text_cleaning; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add code/new_text_cleaning.py
git commit -m "feat: add shared text-cleaning helpers for beam-search outputs"
```

---

### Task 2: Clean the beam-search outputs in the cleaning stage

The cleaning stage currently cleans only `generated_texts`. Add the two beam-search outputs, producing both a cleaned string and cleaned token ids for each.

**Files:**
- Modify: `code/clean_generated_strings_improved.py` — imports; the `clean` method (~lines 66-92); `summarise` (~lines 107-126)

**Interfaces:**
- Consumes: `filter_generated_text`, `build_cleaned_ids` from Task 1.
- Produces, on every sample dict:
  - `cleaned_most_likely_generation` (str)
  - `cleaned_most_likely_generation_ids` (tensor, same length as `most_likely_generation_ids`)
  - `cleaned_second_most_likely_generation` (str)
  - `cleaned_second_most_likely_generation_ids` (tensor, same length as `second_most_likely_generation_ids`)

- [ ] **Step 1: Add the import**

Next to the existing `from device_utils import DEVICE, DTYPE`, add:

```python
# CHANGED FROM UPSTREAM: the inline filter loop moved to new_text_cleaning so the
# same rule can also be applied to the beam-search outputs below.
from new_text_cleaning import build_cleaned_ids, filter_generated_text
```

- [ ] **Step 2: Replace the inline filter loop with the shared helper**

In `clean`, replace:

```python
                for string in self.strings_to_filter_on:
                    if string in generated_text:
                        generated_text = generated_text.split(string)[0]
```

with:

```python
                # CHANGED FROM UPSTREAM: identical logic, now shared via
                # new_text_cleaning.filter_generated_text. Behaviour unchanged.
                generated_text = filter_generated_text(generated_text, self.strings_to_filter_on)
```

- [ ] **Step 3: Clean the beam-search outputs**

Immediately before `cleaned_sequences.append(sample)`, add:

```python
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
```

- [ ] **Step 4: Report the effect in the stage summary**

In `summarise(cleaned_sequences)`, before the `return`, add:

```python
        # NEW - NO UPSTREAM EQUIVALENT: makes the beam-search cleaning visible in
        # W&B, since that is the change this stage now carries.
        most_likely_raw = np.array([len(s['most_likely_generation'].split())
                                    for s in cleaned_sequences])
        most_likely_cleaned = np.array([len(s['cleaned_most_likely_generation'].split())
                                        for s in cleaned_sequences])
```

and add to the returned dict:

```python
            'mean_most_likely_words_raw': float(most_likely_raw.mean()),
            'mean_most_likely_words_cleaned': float(most_likely_cleaned.mean()),
```

- [ ] **Step 5: Verify it imports**

Run: `uv run python -c "import sys; sys.path.insert(0,'code'); import clean_generated_strings_improved; print('ok')"`
Expected: prints `ok`

Do **not** run the stage (constraint 4).

- [ ] **Step 6: Commit**

```bash
git add code/clean_generated_strings_improved.py
git commit -m "fix: clean most-likely and second-most-likely generations (text and ids)"
```

---

### Task 3: New scoring stage — `code/new_score_accuracy.py`

**Files:**
- Create: `code/new_score_accuracy.py`

**Interfaces:**
- Consumes: `cleaned_most_likely_generation` from Task 2.
- Produces, on every sample dict:
  - `exact_match_raw`, `rouge1_to_target_raw`, `rouge2_to_target_raw`, `rougeL_to_target_raw`
  - `exact_match_cleaned`, `rouge1_to_target_cleaned`, `rouge2_to_target_cleaned`, `rougeL_to_target_cleaned`
  - `exact_match`, `rouge1_to_target`, `rouge2_to_target`, `rougeL_to_target` — aliases of the variant chosen by `--score_on` (default `cleaned`)

- [ ] **Step 1: Create the module**

Create `code/new_score_accuracy.py` with exactly this content:

```python
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
```

- [ ] **Step 2: Verify it imports and the CLI parses**

Run: `uv run python code/new_score_accuracy.py --help`
Expected: usage text listing `--generation_model`, `--run_id`, `--dataset`, `--score_on`.

Do **not** run the stage itself (constraint 4).

- [ ] **Step 3: Commit**

```bash
git add code/new_score_accuracy.py
git commit -m "feat: add scoring stage computing raw and cleaned correctness"
```

---

### Task 4: Remove scoring from `generate_improved.py`

**Files:**
- Modify: `code/generate_improved.py` — delete `score_against_references` and `_reference_answers` and the call site; update the `--inspect_entries` printer (~line 458) and `summarise` (~line 467); drop now-unused metric loads.

**Interfaces:**
- Consumes: nothing.
- Produces: a generations pickle **without** `exact_match` or `*_to_target` keys. Task 3's stage adds them.

- [ ] **Step 1: Locate every reference**

Run:

```bash
grep -n "score_against_references\|_reference_answers\|exact_match\|_to_target\|evaluate.load\|self.rouge" code/generate_improved.py
```

Keep the output; every line is either deleted or updated below. Note that `rouge_type + '_reference_answers'` is a *different* key that generate still stores — leave those lines alone.

- [ ] **Step 2: Delete the scoring methods and their call site**

Delete `_reference_answers` (~lines 314-317) and `score_against_references` (~lines 319-338) entirely, and the line that calls `self.score_against_references(...)`.

In their place, leave this marker where the methods were:

```python
    # REMOVED - CHANGED FROM UPSTREAM: _reference_answers and
    # score_against_references moved to new_score_accuracy.py. Upstream scored
    # correctness here, during generation, which is before cleaning runs - so
    # ROUGE compared a ~171-word ramble against a 1-3 word answer and produced
    # 0/40 correct. Scoring now happens after clean_generated_strings_improved.py.
```

- [ ] **Step 3: Remove the now-unused metric loads**

Delete the `self.rouge = evaluate.load('rouge')` and `self.exact_match_metric = evaluate.load('exact_match')` assignments, replacing them with:

```python
        # REMOVED - CHANGED FROM UPSTREAM: the rouge and exact_match metrics moved
        # to new_score_accuracy.py along with the scoring they served.
```

If `import evaluate` is now unreferenced, delete it too; if any other `evaluate.load` call remains, keep the import.

- [ ] **Step 4: Update the inspect printer**

Delete these lines from the `--inspect_entries` printer:

```python
            print(f'  exact_match      : {sample["exact_match"]}')
            for rouge_type in ROUGE_TYPES:
                print(f'  {rouge_type}_to_target : {sample[rouge_type + "_to_target"]:.4f}')
            # analyze_results.py defines correctness as rougeL_to_target > 0.3
            print(f'  -> correct (rougeL > 0.3): {sample["rougeL_to_target"] > 0.3}')
```

and replace with:

```python
            # CHANGED FROM UPSTREAM: correctness is no longer known at this stage.
            # See new_score_accuracy.py, which scores after cleaning.
```

- [ ] **Step 5: Update `summarise`**

Replace the whole `summarise` static method with:

```python
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
```

- [ ] **Step 6: Check for orphans**

Run:

```bash
grep -n "exact_match\|_to_target\|ROUGE_TYPES\|evaluate" code/generate_improved.py
```

Expected: remaining hits are only `rouge_type + '_reference_answers'` and the `ROUGE_TYPES` constant that serves it. If `ROUGE_TYPES` has no remaining use, delete it and note the removal in a comment.

- [ ] **Step 7: Verify it imports**

Run: `uv run python -c "import sys; sys.path.insert(0,'code'); import generate_improved; print('ok')"`
Expected: prints `ok`

Do **not** run generation (constraint 4).

- [ ] **Step 8: Commit**

```bash
git add code/generate_improved.py
git commit -m "refactor: move correctness scoring out of the generation stage"
```

---

### Task 5: Consume the cleaned ids in `get_likelihoods_improved.py`

Without this the cleaned ids from Task 2 are ignored and the NLLs keep the artifact — which is the main reason this work exists.

**Files:**
- Modify: `code/get_likelihoods_improved.py` — add one accessor method; change lines ~121 and ~124.

**Interfaces:**
- Consumes: `cleaned_{most,second_most}_likely_generation_ids` from Task 2.
- Produces: unchanged output key names; only the values change.

- [ ] **Step 1: Add a cleaned-preferring accessor**

Add this method to the class, immediately after `get_neg_log_likelihood_for_generation`:

```python
    def _sequence_ids(self, sequence, field):
        """Prefer the cleaned ids when the cleaning stage produced them.

        NEW - NO UPSTREAM EQUIVALENT. Upstream read
        most_likely_generation_ids / second_most_likely_generation_ids raw, so
        the NLL was averaged over a ~171-word repetition loop - which then fed
        the margin measure.

        The pad strip matters: cleaned ids are padded back out to the original
        width, whereas the raw ids never had padding. Without stripping, the loss
        would be averaged over hundreds of <pad> positions. This mirrors what the
        sampled-generation path already does a few lines below.
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
            # CHANGED FROM UPSTREAM: prefer cleaned ids, and strip padding.
            most_likely_generation = self._sequence_ids(sequence, 'most_likely_generation_ids')
```

Replace:

```python
            second_most_likely_generation = sequence['second_most_likely_generation_ids'].to(DEVICE)
```

with:

```python
            # CHANGED FROM UPSTREAM: prefer cleaned ids, and strip padding.
            second_most_likely_generation = self._sequence_ids(sequence, 'second_most_likely_generation_ids')
```

- [ ] **Step 3: Verify it imports**

Run: `uv run python -c "import sys; sys.path.insert(0,'code'); import get_likelihoods_improved; print('ok')"`
Expected: prints `ok`

Do **not** run the stage (constraint 4).

- [ ] **Step 4: Commit**

```bash
git add code/get_likelihoods_improved.py
git commit -m "fix: use cleaned most-likely ids for negative log-likelihoods"
```
