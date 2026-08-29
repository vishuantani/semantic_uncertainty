# Implementation Sensitivity of Semantic Entropy

Research notes on the reference implementation of Kuhn, Gal & Farquhar (2023),
*Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural
Language Generation* ([arXiv:2302.09664](https://arxiv.org/abs/2302.09664)), as released
in this repository.

**Status:** exploratory. All measurements below come from a 40-question pilot on
TriviaQA with `opt-350m`, 5 generations per question, run on Apple Silicon (MPS).
They are large enough to establish that the effects exist and are not large enough
to quantify their impact on the paper's headline results.

---

## Thesis

The paper specifies an algorithm. The released code implements something different in at
least two places, and contains additional undocumented behaviour that makes results
non-reproducible. The open question is not whether these gaps exist — they are verified
below — but **how much the reported uncertainty numbers depend on them.**

This matters beyond this one repository. Semantic entropy is widely cited and this code
is the canonical reference implementation, so any sensitivity found here propagates to
work that builds on it.

---

## Thread 1 — Equivalence rule: paper vs. code

### The gap

The paper is unambiguous:

> "We operationalise E(·,·) using the idea of bi-directional entailment. A sequence, s,
> means the same thing as a second sequence, s′, if and only if they entail each other."

> "the algorithm returns equivalent if and only if both directions were entailment"

The code merges on the *absence of contradiction*:

```python
# code/get_semantic_similarities.py
if 0 in predicted_label or 0 in reverse_predicted_label:   # 0 == CONTRADICTION
    has_semantically_different_answers = True
else:
    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
```

A NEUTRAL/NEUTRAL pair — two answers the model finds simply unrelated — is merged as
"the same meaning". Observed instance, from a single question's cluster 0:

> "Hawthorn, who died in 1996, was the first African-American …"
> "Hawthorn, who died in 2001, was the first British-born …"

These are mutually incompatible claims sharing one semantic set.

### Why this direction of error matters

Merging too eagerly *undercounts* semantic clusters, which *lowers* semantic entropy.
Since the paper's claim is that semantic entropy predicts correctness better than
baselines, a bug that systematically lowers the measure is not obviously neutral with
respect to the headline result.

### Questions

1. What fraction of merges in the released implementation are driven by NEUTRAL rather
   than ENTAILMENT judgements?
2. How do cluster counts and semantic entropy change under strict bidirectional
   entailment?
3. Does the paper's central claim — semantic entropy beats predictive entropy and
   lexical similarity at predicting correctness — survive the strict rule? Strengthen
   or weaken?

### Method

Add an equivalence-rule flag with three settings and hold everything else fixed:

| setting | rule |
|---|---|
| `not_contradiction` | current released behaviour |
| `exclude_both_neutral` | `(0 not in implications) and ([1,1] != implications)` |
| `strict_entailment` | `implication_1 == 2 and implication_2 == 2` (the paper) |

Report cluster counts, semantic entropy, and AUROC against `exact_match` for each.

### Baseline already measured

`run_1`, upstream cleaning, `deberta-base-mnli`, `not_contradiction`:

```
n_questions=40  n_generations=200  n_pairs_compared=328
mean_semantic_sets=2.0   median_semantic_sets=1.5   fraction_questions_multi_set=0.5
fraction_pairs_contradicting=0.412
```

41.2% of pairs contradicted, so 58.8% merged — the NEUTRAL share of that 58.8% is the
quantity Thread 1 needs and is not yet measured.

---

## Thread 2 — Clustering algorithm: paper vs. code

### The gap

The paper's Algorithm 1 compares each new sequence **against existing semantic clusters**.
The code does all-pairs comparison with last-write-wins assignment:

```python
semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
```

Consequences, all verified:

- **Not transitive.** If the model says A~B and B~C but A≁C, the outcome depends on which
  pair is visited last. No transitive closure is computed.
- **Not a partition.** An answer already merged into one cluster can be silently reassigned
  to another, orphaning the earlier relation.
- **Non-contiguous ids.** Cluster ids such as `[0, 2, 3, 4]` occur (10/40 samples). Harmless
  downstream — `compute_confidence_measure.py` uses `torch.unique` — but a symptom.
- **Internally inconsistent outputs.** `has_semantically_different_answers` and the
  clustering disagree on **17.5%** of questions (7/40): a contradiction was found, yet all
  answers still collapsed into one set.

### Questions

1. How often does non-transitivity actually change the cluster count, as opposed to just
   the labels?
2. Does a union-find (transitive closure) partition differ materially from the paper's
   seed-based scheme, or are they equivalent in practice on this data?
3. Which of the three — released, seed-based, union-find — is most stable under
   perturbation of the generation order?

### Method

Implement all three clustering strategies behind a flag. Compare partitions with the
Adjusted Rand Index rather than raw ids, since label identity is meaningless.

---

## Thread 3 — Reproducibility

### The finding

The released code is **not reproducible across processes**, despite appearing to be seeded.

```python
os.environ['PYTHONHASHSEED'] = str(seed_value)   # no-op: set after interpreter start
...
unique_generated_texts = list(set(generated_texts))   # order varies per process
```

String hash randomisation is fixed when the interpreter starts, so assigning
`PYTHONHASHSEED` from inside the program has never had any effect. Combined with the
order-dependent merge in Thread 2, identical inputs yield different clusters run to run.

Measured on two runs over identical input:

| quantity | differs |
|---|---|
| ROUGE scores | 0/40 |
| `has_semantically_different_answers` | 0/40 |
| `semantic_set_ids` | **20/40** |
| **number of clusters** (what entropy consumes) | **6/40** |

Replacing `list(set(...))` with `sorted(set(...))` yields bit-identical output across
processes (verified: 0/40 differing).

### Questions

1. What is the run-to-run variance of reported semantic entropy and its AUROC, under the
   original non-deterministic code?
2. Is that variance small relative to the gap between semantic entropy and the baselines
   it is compared against? If not, published comparisons need error bars.

### Method

Run the original code N times on identical input, compute the metric distribution. This
is the cheapest thread and directly bounds how seriously to treat the others.

---

## Thread 4 — Entailment-model sensitivity

The paper specifies `deberta-large-mnli`. Nothing establishes how much the result depends
on that choice.

Measured on 120 real pair inputs from `run_1`:

| model | params | agreement with large | time | peak memory |
|---|---|---|---|---|
| `microsoft/deberta-large-mnli` | 406M | — | 159.3s | 15.35 GB |
| `microsoft/deberta-base-mnli` | 139M | **89.2%** | 38.1s | 8.07 GB |
| `cross-encoder/nli-deberta-v3-small` | 142M | 85.0% | 16.6s | 6.46 GB |

Roughly one contradiction decision in nine flips between large and base. Because those
decisions feed the merge, a single flip can split or join a cluster.

### Questions

1. Does semantic entropy's advantage over baselines hold with a weaker entailment model,
   or is the result partly a function of entailment-model quality?
2. Is there a monotone relationship between entailment-model quality and the AUROC of
   semantic entropy? That would be a useful scaling result in its own right.
3. Modern NLI models (DeBERTa-v3, or an LLM-as-judge) post-date the paper. Does the
   method improve with them?

**Implementation hazard:** label index 0 is CONTRADICTION for `microsoft/deberta-*-mnli`
and `cross-encoder/nli-deberta-v3-*`, but **ENTAILMENT** for
`MoritzLaurer/DeBERTa-v3-*` and `typeform/distilbert-base-uncased-mnli`. The released code
hardcodes `0`. Swapping models without reading `model.config.label2id['CONTRADICTION']`
silently inverts the entire method.

---

## Thread 5 — Degenerate generations reach the entailment model

`clean_generated_strings.py` truncates at the first occurrence of `.`, `\n`, `Q:`, `A:`
and similar. Repetition loops contain none of these and pass through intact:

> "How long was swimmer Michelle Smith-de Bruin banned …? Answer: The answer to all of
> these questions is: The answer to all of these questions is: …" (×70)

Consequences measured on `run_1`:

- Pair inputs: median 301 tokens, p99 556, max 562.
- **76 of 328 pair inputs exceed DeBERTa's 512-token training range.** No error is raised —
  `deberta-large-mnli` has `position_biased_input: False`, so only relative positions are
  used, and `max_relative_positions: -1` clamps distances at 512. Those pairs are
  extrapolating.
- 1.53× wasted compute from padding to accommodate the outliers.

### Questions

1. How many semantic clusters are formed from degenerate text that arguably should have
   been filtered upstream, and does excluding them change the reported metrics?
2. Is a repetition guard in the cleaning step sufficient, or does the generation config
   need changing (`no_repeat_ngram_size`)?
3. Do the 76 over-length pairs receive systematically different labels than in-range pairs?

---

## Thread 6 — Scorer-model sensitivity

### The opportunity

`get_likelihoods.py` takes `--evaluation_model` and `--generation_model` as **independent**
arguments. It loads `facebook/{evaluation_model}`, reads `{generation_model}_generations.pkl`,
and writes `{generation_model}_generations_{evaluation_model}_likelihoods.pkl`. The output
filename encodes both, so scoring one model's generations under a different model was designed
in. No script in the repository ever sets them to different values.

Downstream carries a matching axis. `compute_confidence_measure.py` builds `list_of_results` as
a list of `(model, results)` pairs, `get_overall_log_likelihoods` stacks over it, and
`get_predictive_entropy` / `get_mutual_information` reduce over `dim=0`:

```python
list_of_results.append((args.evaluation_model, sequences))   # only ever one entry
```

That axis therefore always has length 1, which makes `get_mutual_information` — written as an
ensemble disagreement measure — degenerate. An entire multi-scorer pathway exists and is unused.

### Why this is worth doing

Unlike Threads 1–5 this is not a paper-fidelity check; it is a new result. Semantic entropy
currently costs two forward passes per sampled generation through the evaluation model. If a
small scorer preserves the AUROC ranking, uncertainty estimation for a large generator becomes
cheap, and the method's practical envelope widens considerably. If it does not, then reported
performance depends on a model choice the paper does not treat as a variable — which is the
Thread 4 finding transposed to the likelihood side.

### Questions

1. Does semantic entropy computed with a smaller scorer retain its AUROC advantage over the
   baselines it is compared against?
2. Is there a monotone relationship between scorer size and AUROC, as Thread 4 asks of the
   entailment model?
3. Do the measures degrade at different rates? Semantic entropy, predictive entropy, PMI and
   most-likely-generation likelihood may not be equally robust to scorer substitution, and a
   change in their *ranking* matters more than a change in their absolute values.
4. Populated with two or three scorers, does the unused ensemble axis produce a better measure
   than any single scorer? That is what `get_mutual_information` was written for.

### Method

Generate once, score many times. Hold one generation model's output fixed and re-run
`get_likelihoods.py` across a sweep of `--evaluation_model`, then compare AUROCs. Generation is
the expensive stage, so the sweep is cheap relative to a full re-run.

The design isolates the effect cleanly: `semantic_set_ids` come from the entailment stage and do
not depend on the scorer, so clustering is held fixed across the sweep and the only quantity
varying is `p(s|x)`.

**Implementation hazard:** the scorer must share the generator's tokenizer. The pickle stores
`prompt` and `generations` as token ids, and `len(prompt)` sets the `-100` mask boundary, so
scoring under a model with a different vocabulary is silently meaningless rather than an error.
All `facebook/opt-*` checkpoints share one tokenizer, so a within-family sweep is safe.
Crossing families (OPT to Llama, say) requires re-tokenizing from text, which reintroduces the
round-trip mismatch measured in the likelihood stage.

---

## Thread 7 — Length normalisation of the sequence likelihood

### The gap

`get_likelihoods.py` emits both a summed and a length-normalised negative log likelihood for
every generation. `compute_confidence_measure.py` then computes semantic entropy **twice**:

```python
predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(
    -overall_results['average_neg_log_likelihoods'], overall_results['semantic_set_ids'])
unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(
    -overall_results['neg_log_likelihoods'], overall_results['semantic_set_ids'])
```

`analyze_results.py` reports the **normalised** variant as the headline
`entropy_over_concepts_auroc`; the unnormalised one is computed inside a `try/except` and is
treated as optional. The code picks one of two defensible readings and reports it as the result.

### Why this is not a neutral reparameterisation

Semantic entropy aggregates probability mass within each cluster before taking the entropy:

```python
aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
```

With unnormalised log likelihoods this is exactly `log Σ p(s|x)` over the cluster's members —
the cluster's probability mass, which is what the paper's formulation calls for. With
length-normalised inputs the same line computes `log Σ p(s|x)^(1/|s|)`: a sum of per-token
geometric-mean probabilities, which is not a probability of anything. The aggregation step
loses its probabilistic interpretation precisely in the variant that is reported.

Whether the paper specifies normalisation for semantic entropy itself, or only for the
length-normalised predictive entropy baseline it inherits from Malinin & Gales (2021), needs
resolving against the paper text. That determination decides which of the two numbers already
sitting in every results pickle is the one to report.

Adjacent, in the same function: `llh_shift = torch.tensor(5.0)` is subtracted from the
aggregated cluster likelihoods, with no comment and no counterpart in the paper. It is applied
*only* here, not in `get_predictive_entropy`, so semantic entropy and predictive entropy are not
on a common scale. Because the shift is subtracted from each of the `K` cluster likelihoods and
the next line divides the sum by that same `K`, the `K` cancels: the effect is exactly `+5` added
to every question's semantic entropy, independent of cluster count. Rank-preserving, hence
AUROC-neutral, and harmless to any within-measure comparison — but any table or figure placing
semantic entropy on the same axis as predictive entropy is off by that flat 5.

### Questions

1. Does the paper specify length normalisation for semantic entropy, or only for the baseline?
2. How far apart are the two variants' AUROCs on identical data, and does the gap exceed the
   run-to-run noise floor that Thread 3 measures?
3. If the normalised variant has no probabilistic reading yet performs better empirically, what
   is it actually measuring — and is the answer just that it corrects a length confound?
4. Does answer length correlate with correctness in this data? If it does, normalisation is
   partly a correction to the label rather than to the measure.

### Method

Pure re-analysis: both quantities are already stored in every likelihoods pickle, so no GPU time
is needed. Compute AUROC for `entropy_over_concepts` and `unnormalised_entropy_over_concepts` on
the same run, report both alongside the correlation of each with answer length.

Cheapest thread in this document, and it gates the interpretation of every other AUROC reported
here. Blocked only on correctness labels being non-degenerate — see Limitations.

---

## Thread 8 — Cluster-probability estimator

### The gap

Eq. 4 of the paper estimates semantic entropy by averaging `-log p(C_i|x)` over the
meaning-classes recovered from sampled generations, and states:

> "This is an unbiased estimator of the entropy in Eq. 3."

That claim is about the **outer** average — which classes appear — and it assumes the inner
`p(C_i|x)` is known exactly. The code does not know it exactly. It estimates it from the *same*
samples, by summing member likelihoods:

```python
# code/compute_confidence_measure.py
aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
```

Substituting an estimate for `p(C_i|x)` is not covered by the paper's argument, and the
particular estimate chosen is biased.

### Why the sum-of-likelihoods is biased

Write the M sampled generations as `s⁽¹⁾…s⁽ᴹ⁾`, drawn i.i.d. from `p(·|x)`. The quantity the
code computes is

```
p̂(C_i|x) = Σ_{m=1..M} p(s⁽ᵐ⁾|x) · 1[s⁽ᵐ⁾ ∈ C_i]
```

Taking the expectation over the sampling gives

```
E[p̂(C_i|x)] = M · Σ_{s∈C_i} p(s|x)²
```

The target is `Σ_{s∈C_i} p(s|x)`. The estimator's expectation is a **second** moment, scaled by
M — a different functional, not a noisy version of the right one. Two consequences:

- It does not converge to `p(C_i|x)` as M grows; it grows with M.
- It is dominated by whichever single sequence in the class is most probable, rather than by
  the class's total mass.

Duplicates are summed with multiplicity, which drives this. `get_semantic_similarities.py:138`
maps ids back over `generated_texts`, not `unique_generated_texts`:

```python
list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
```

so `semantic_set_ids` has length M with repeats (verified: shape `[5]` alongside
`neg_log_likelihoods` shape `[5]` in every `run_1` sample). An identical string sampled three
times contributes its probability three times. Repeated sampling of one answer is evidence of
*confidence*, and here it inflates the cluster's estimated mass instead.

Collapsing duplicates first removes the `M·Σp²` pathology but leaves a residual bias:

```
E[p̂(C_i|x)] = Σ_{s∈C_i} p(s|x)·(1 − (1−p(s|x))^M)   <   p(C_i|x)
```

— downward biased by exactly the mass of the class members never drawn, and consistent as
M → ∞. That is the benign, unavoidable finite-sample bias. The multiplicity bias is neither.

### An unbiased alternative exists and discards the likelihoods

For sampling with replacement from `p`, the Horvitz–Thompson estimator of `Σ_{s∈C_i} p(s|x)` is

```
(1/M) Σ_m [ p(s⁽ᵐ⁾|x) · 1(s⁽ᵐ⁾∈C_i) / p(s⁽ᵐ⁾|x) ]  =  count(C_i)/M
```

The importance weights cancel exactly, leaving the **cluster frequency**, which is unbiased for
`p(C_i|x)` by a binomial argument. So the one straightforwardly unbiased route uses only how
often each meaning was sampled and ignores the likelihoods entirely — which is, in effect, what
the later discrete formulation in the 2024 Nature paper does.

This is the interesting part of the thread: the released code spends two forward passes per
generation computing likelihoods, in service of an estimator that is biased, when the estimator
that needs no likelihoods at all is unbiased.

### A second bias survives either choice

Eq. 4 applies `−log` to the estimate, and `−log` is convex, so by Jensen's inequality

```
E[−log p̂] ≥ −log E[p̂]
```

Semantic entropy is therefore biased upward even given an unbiased `p̂`. Additionally, the outer
and inner estimates are computed from the same draws: a class enters the average only because
its members were sampled, and those same draws set its probability. The two levels are
correlated, which the paper's unbiasedness argument does not address.

### Baseline already measured

`run_1`, 40 questions, 5 generations, length-normalised inputs (the headline variant),
`llh_shift` retained where applicable:

| estimator | mean | sd | range |
|---|---|---|---|
| released (`Σ` with multiplicity) | 5.223 | 1.568 | 3.454 – 9.576 |
| duplicates collapsed | 5.334 | 1.527 | 3.678 – 9.923 |
| frequency `count/M` | 0.580 | 0.614 | 0.000 – 1.609 |

| comparison | Spearman ρ vs released |
|---|---|
| duplicates collapsed | 0.946 |
| frequency | 0.814 |

Duplicate generations occur in **13/40 questions** (20 of 200 generations), and those are
exactly the 13 questions where collapsing changes the entropy.

Two observations. The released variant's values sit around 5 because `llh_shift = 5.0` is
subtracted from a log-likelihood that is itself near zero — its spread is driven by likelihood
magnitudes, not by cluster structure. The frequency variant lands in `[0, log 5] = [0, 1.609]`,
the actual attainable entropy range for 5 samples. And ρ = 0.81 against the frequency estimator
means the two rank questions materially differently, so this is not a monotone reparameterisation
that AUROC would be blind to.

### A third variable: the divisor

The `1/M` in a Monte Carlo estimate of `E[−log p]` is not bookkeeping — it is the `p(x_i)`
weight of `Σ p log p`, re-expressed through sampling. So whatever a line divides by declares
the weighting it assumes. The two entropy functions here disagree:

```python
# line 116, predictive entropy — divides by M, the number of generations
entropy = -torch.sum(mean_across_models, dim=1) / mean_across_models.shape[1]

# line 133, semantic entropy — divides by |C|, the number of distinct clusters
entropy = - torch.sum(aggregated_likelihoods, dim=0) / aggregated_likelihoods.shape[0]
```

Predictive entropy averages over samples, which is correct Monte Carlo. Semantic entropy
averages over *distinct classes*, one term each, which weights every meaning by `1/|C|` — i.e.
assumes all meanings equally likely, the very thing entropy is meant to measure. With 5
generations where meaning A is drawn 4 times and B once, the correct weights are 0.8/0.2; the
code uses 0.5/0.5, so one off-hand generation counts as much as the model's dominant answer.

Measured on `run_1`, length-normalised inputs:

| divisor | mean | sd |
|---|---|---|
| `\|C\|` (released) | 5.223 | 1.568 |
| `M`, sample-weighted | 4.966 | 1.389 |

Spearman ρ = 0.980, differing on 18/40 questions. So the divisor shifts values without much
reordering — a weaker effect than the estimator choice above (ρ = 0.814), and worth knowing
before prioritising it.

Note the two errors act in opposite directions on the same line: summing member likelihoods
over-counts duplicates in the numerator, while dividing by `|C|` under-counts them in the
denominator. They should be varied independently rather than fixed together, or the partial
cancellation will be mistaken for one of them being harmless.

### Questions

1. Does the AUROC of semantic entropy against `exact_match` differ between the three estimators,
   and does any difference exceed the run-to-run noise floor from Thread 3?
2. If the unbiased frequency estimator performs *worse*, what are the likelihoods contributing —
   genuine information about cluster mass, or a length/fluency confound that happens to correlate
   with correctness? This is Thread 7's question arriving from a different direction.
3. Does the gap between estimators shrink as M grows? All three converge to different things, so
   the M-dependence is diagnostic of which bias dominates at the M ≈ 5–10 the paper uses.
4. Is the multiplicity behaviour intentional? Collapsing duplicates is a one-line change and the
   paper's Eq. 2 sums over a *set* of sequences, which implies it.
5. Does the divisor (`|C|` vs `M`) interact with the estimator choice, or are their effects
   additive? The partial cancellation noted above means the crossed design is the only way to
   tell.

### Method

Pure re-analysis: `semantic_set_ids`, `generated_texts` and both likelihood variants are already
in the pickles, so no GPU time is needed. Implement the three estimators behind a flag in
`get_predictive_entropy_over_concepts` and report AUROC for each.

The two variables are separable and should be varied independently — collapsing duplicates
(a fix to the sum) and switching to frequency (a change of estimator) are different claims, and
crossing them with Thread 7's normalisation flag gives the full picture at negligible cost.

**Implementation hazard:** `llh_shift = 5.0` must be dropped for the frequency estimator. It is
calibrated to log-likelihood magnitudes and would swamp an entropy that legitimately lives in
`[0, log M]`.

---

## Thread 9 — Margin measure: samples vs. beams

### The gap

`get_margin_probability_uncertainty_measure` is fed the *sampled* generations:

```python
# code/compute_confidence_measure.py, lines 163-164
margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihoods'])
unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihoods'])
```

`torch.topk(..., 2, dim=1)` therefore selects the two highest-likelihood members of the `M`
stochastic draws. The classical margin measure compares the top two *modes* of the predictive
distribution, and this repository already computes a deterministic approximation of them:
`generate.py` runs `num_beams=5, num_return_sequences=2` and `get_likelihoods.py` scores both
beams. The second one is then discarded — `average_neg_log_likelihood_of_second_most_likely_gen`
is written to the pickle but is absent from `list_of_keys` in `get_overall_log_likelihoods`, so
it never reaches `overall_results`. The deterministic margin is one dictionary entry away from
existing and has never been computed.

### Why the substrate changes the estimand

Both variants are uncertainty measures; the difference is what they estimate.

The sampled margin is the top-2 gap of the *empirical* distribution over `M` draws. At `M = 5`
that is a poor estimate of the mode, and it carries seed variance — it contributes directly to
the noise floor Thread 3 measures. The beam margin is deterministic: zero run-to-run variance,
and independent of `M` entirely.

There is also a label-alignment argument that applies to no other measure in this document.
`exact_match` is computed in `generate.py` against `most_likely_generation` — the beam-1 output.
Every sampled measure (both entropies, PMI, the sampled margin) therefore describes a set of
sequences that is *not* the sequence being graded, while a beam margin is a confidence statement
about exactly the sequence the label refers to. Whether that alignment buys AUROC is an open
question, but it is a structural difference, not a preference.

Both substrates have a degeneracy, and they are not the same one. Sampled duplicates — 13/40
questions in the Thread 8 baseline — give two identical strings identical likelihoods and hence
a margin of ~0, reading as maximal uncertainty precisely when the model is most concentrated.
Beams share prefixes by construction and often differ by a single token (`Paris` / `Paris.`),
which produces the same collapse systematically rather than occasionally. Quantifying the two
rates is part of the thread rather than a reason to prefer either.

### Questions

1. How do the AUROCs of the beam margin, the sampled margin, and the two entropies compare on
   identical data?
2. How often do beam 1 and beam 2 fall in the *same* semantic cluster under the existing DeBERTa
   stage? That rate is the false-uncertainty rate of the beam margin, and it is the same
   objection the paper raises against surface-form measures generally.
3. Given Thread 3's noise floor, does a measure with zero sampling variance become preferable at
   *equal* AUROC? A reproducible measure and a seed-dependent one are not interchangeable even
   when they score the same.
4. Does a **semantic margin** — the gap between the top-2 aggregated *cluster* masses, reusing
   the aggregation in `get_predictive_entropy_over_concepts` — beat both? It is immune to both
   degeneracies above and is pure re-analysis of pickles already on disk.

### Method

Re-scoring, not regeneration: the beam token ids are already stored in the generations pickle,
so this costs two forward passes per question in `get_likelihoods.py` and no sampling. Add
`average_neg_log_likelihood_of_second_most_likely_gen` to `list_of_keys`, add a beam-margin
function beside the existing one, and leave the sampled margin untouched so the comparison is
like-for-like. Cross with Thread 7's normalisation flag as with every other estimator here.

**Implementation hazards:**

- **Padding contamination — mandatory fix.** Unlike the sampled generations, which are stripped
  at `get_likelihoods.py:77-78`, the beam ids are never pad-stripped. `generate` pads the two
  returned beams to a common length, and `target_ids[:len(prompt)] = -100` masks only the
  prompt, so pad tokens are scored as real labels. The shorter beam — usually beam 2 — is the
  contaminated one, so an uncorrected margin partly measures how much padding beam 2 carries.
  Apply the same `[x != pad_token_id]` filter before scoring.
- **Only the length-normalised beam-2 value exists.** There is no
  `neg_log_likelihood_of_second_most_likely_gen`, so the unnormalised arm requires the re-run
  rather than being available as re-analysis.
- **`np.exp` on line 143** converts through NumPy and raises on any non-CPU tensor; and applied
  to an unnormalised sequence log-likelihood it underflows toward zero for all but the shortest
  answers, making `unnormalised_margin_measures` near-constant. Both apply to a beam variant
  written the same way.
- **Sign.** Despite the name, a large margin means high confidence. It must be negated before
  AUROC or the curve inverts — this applies to the existing measure too.
- **Beam-search runs only.** `generate.py:222` indexes `most_likely_generation[1]` unguarded, so
  `--decoding_method greedy` (one returned sequence) raises `IndexError`. No greedy run can
  supply beam 2.

---

## Suggested order

| # | thread | cost | why first |
|---|---|---|---|
| 1 | Reproducibility (T3) | hours | Bounds the noise floor. Every other result is uninterpretable without it. |
| 2 | Length normalisation (T7) | hours | Pure re-analysis of pickles already on disk; decides which variant is the headline before any other AUROC is reported. |
| 3 | Cluster-probability estimator (T8) | hours | Also pure re-analysis, and modifies the same aggregation line as T7, so the two should be flagged and crossed together. |
| 4 | Margin substrate (T9) | hours | Re-scoring only, no regeneration; the semantic-margin arm is pure re-analysis and shares T8's aggregation. |
| 5 | Equivalence rule (T1) | days | Largest expected effect; the clearest paper/code discrepancy. |
| 6 | Entailment model (T4) | days | Cheap once T1's flag exists; independent scientific interest. |
| 7 | Clustering algorithm (T2) | days | Likely smaller effect than T1, but completes the paper-fidelity story. |
| 8 | Degenerate generations (T5) | days | Data-quality confound; may partly explain T1 and T2 effects. |
| 9 | Scorer model (T6) | days | Largest compute cost, and the only thread whose interest is independent of the paper's fidelity. |

---

## Limitations

- **Scale.** 40 questions, `opt-350m`, 5 generations. The paper uses models up to 30B and
  far more questions. Everything here is a pilot; effects may not survive at scale, and
  effects invisible here may appear.
- **Single dataset.** TriviaQA only. The repo also supports CoQA, unexercised so far.
- **No AUROC yet.** Every measurement above concerns cluster structure. The paper's claims
  are about *predicting correctness*, and none of these threads is complete until it
  reports AUROC against `exact_match`.
- **Hardware.** Timing and memory figures are MPS/unified-memory and do not transfer to
  the CUDA setup the paper used.

---

## Not evidence

The repository README deprecates this code in favour of
[jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty), whose
implementation offers a `strict_entailment` flag and seed-based clustering. That repo
accompanies a **different paper** — *Detecting Hallucinations in Large Language Models
Using Semantic Entropy* (Nature, 2024) — so it should not be read as the authors
acknowledging a defect in the 2023 code. It is useful as a reference implementation and as
evidence that a later related implementation also defaults to a loose equivalence rule
(`strict_entailment=False`), but it settles nothing about the 2023 discrepancy.

---

## References

- Kuhn, Gal & Farquhar (2023). *Semantic Uncertainty: Linguistic Invariances for
  Uncertainty Estimation in Natural Language Generation.* ICLR.
  [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
- Farquhar, Kossen, Kuhn & Gal (2024). *Detecting Hallucinations in Large Language Models
  Using Semantic Entropy.* Nature.
  [doi:10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0)
- Reference implementation (2024): https://github.com/jlko/semantic_uncertainty
