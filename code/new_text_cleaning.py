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
