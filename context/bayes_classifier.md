We can take this idea of fitting a theoretical model to the MoP generation further by thinking about the MoP model as behaving like a bayesian classifier. 

The mixture model is trying to behave any of the personas but not like all of them. This is because if it sees narrow features of a single persona it latches onto that persona and commits to it instead of interpolating. 

We have seen with the gamma posterior probability that it is a good measure of how the model is behaving and gives us insight into how the model is generating completions

However, this looks very much like a theoretical naive bayes classifier. An interesting experiment hence, is to consider a naive bayes classifier that maintains a distribution over the personas which we can call $\eta$. 

We can fit the classifier to the training dataset and then similar to what we did with the persona probe, plot generations and then see how gamma and eta compare to each other and whether they track each other.

---

# Experiment 8 — Naive Bayes belief η(t) vs generative posterior γ(t)

Implements the proposal above. The generative posterior γ_i(t) of Exp 3/4 already has the exact
form of a naive-Bayes classifier accumulating evidence over a prefix —
`γ_i(t) = softmax_i( log π_i + Σ_{s≤t} log P_i(x_s | x_{<s}) )` — with the *specialist neural
models'* token log-probs as the likelihood. Exp 8 builds the literal textbook twin: a multinomial
Naive Bayes classifier η fit on the **training data**, with the **empirical per-persona token
frequency** as its likelihood instead of a trained model, and overlays η(t) on γ(t) to test whether
the mixture model's generative posterior tracks the theoretical Bayesian classifier.

## What η is (and how it stays parallel to γ)

η is the count-based analog of γ — same prior, same accumulation, only the likelihood differs:

- **Vocabulary = the shared model tokenizer** (`config.VOCAB_SIZE = 4019`). The NB "features" are
  the very same token ids γ sees; no separate feature engineering.
- **Prior = uniform** over the 6 components (`commitment.uniform_prior()`, the same prior γ uses).
  Matches σ, so any divergence between η and γ is purely in the likelihood.
- **Likelihood** `p_i(v) = (count_i[v] + α) / (Σ_v count_i[v] + αV)` — Laplace-smoothed (α=1) token
  frequency over component i's data.
- **Accumulation** `η_i(t) = softmax_i( log(1/6) + Σ_{s≤t} log p_i(x_s) )` — the same per-token
  recurrence as γ. A position-marginal, count-based twin of γ (and of Exp 1's marginal PMI
  normalised into a classifier).
- **Components** = `config.PERSONAS + ['base']` (6), so η, γ are over the same set. The `base` class
  has no dataset, so — as in Exp 5/7 — its counts come from **base-model free generations**.
- The structural **`[EOS]` (id 1) and position 0 are excluded** everywhere in the NB: `[EOS]` is the
  seed/terminator not persona content, and persona stories D_i are truncated at t_max with no
  trailing EOS while base free-gens *do* terminate in EOS, so counting it would spuriously hand the
  base class the terminal token of every rollout.
- To make the two curves directly comparable, η is computed on the **same realised tokens** and
  **masked to exactly the positions where γ is valid** (derived from γ's non-nan columns), so the
  anchored rollouts score the model's own continuation under the identical mask γ uses (the forced
  anchor is excluded from both).

## Implementation

- `src/bayes_nb.py` — `fit_token_loglik` (per-component log-likelihood over the shared vocab) and
  `cumulative_nb_posterior` (the η(t) accumulation, EOS/invalid positions add 0, output aligned to
  γ's validity).
- `src/run_exp8.py` — fits η (2000 D_i stories/persona, 1000 base free-gens, α=1), then overlays
  η(t) (solid) on γ(t) (dashed) for **free** rollouts (one per dominant component, selected by final
  γ as in Exp 4B/6/7) and **anchored-by-i** rollouts (entry trigger, one per persona); reuses
  `commitment` (γ), `generate`/`token_dist` (rollouts + masks), `data` (stories), Exp 1
  `triggers.json`, Exp 2 `free_samples.npz`. Also reports a population commit-time companion
  (τ_η vs τ_γ, commit = first t where the leading component exceeds 0.5).
- Run: `python -m src.run_exp8` (GPU; CPU fallback as elsewhere). Prereqs: Exp 1 + Exp 2.
- Outputs (`results/exp8_<ts>/`): `config.json`, `exp8_params.json`, `nb_top_tokens.csv`,
  `eta_gamma.npz`, `commit_times.csv`, `commit_time_scatter.png`, `free_eta_vs_gamma.png`,
  `anchored_eta_vs_gamma.png`.

## Results (`results/exp8_20260622_190014`, GPU run 2026-06-22)

**η and γ track each other, but η is the sharper / over-confident classifier.**

- **Winner agreement 75.5% (151/200)** of free rollouts — η and γ commit to the *same* component;
  mean per-token **L1(η, γ) = 0.123**. The commit-time correlation is weak (corr(τ_γ, τ_η) = 0.18,
  median τ_η − τ_γ = +1.0) because most rollouts commit within the first 1–3 tokens for *both*
  curves, leaving little timing spread to correlate.
- In the overlays (`free_eta_vs_gamma.png`), **η snaps to ~1.0 within 1–3 tokens and stays pinned**,
  whereas **γ is more graded** and competes among components for longer before locking in — the
  textbook NB over-confidence (token-independence × many tokens → near-hard posteriors).
- **Disagreement is interpretable and concentrated on `scientific_explainer`**: where γ commits to
  scientific, η scatters to absurdist (11) / base (7). That is exactly the persona whose signal is
  *multi-token phrases* (`"the reason is that"`, `"because"`, `"when … then …"`) — a bag-of-tokens
  NB cannot see them, so η fails to identify it. The triggered personas (epistolary, fairy_tale,
  noir) and absurdist, whose signal lives in **single-token** frequencies, η identifies cleanly
  (noir/noir agreement = 64).
- The per-component **top NB tokens are all generic stopwords/punctuation** (`,` `.` `a` `the`) for
  every component; the discriminative signal is in the *relative* frequencies (epistolary high on
  `i`/`you`, noir on `was`/`night`), which the full-prefix likelihood ratio still accumulates — so
  η discriminates despite its top tokens looking uninformative.
- **Anchored** (`anchored_eta_vs_gamma.png`) reproduces the project taxonomy under *both* curves:
  `dear` / `once` / `the` / `a` commit cleanly for η and γ alike, and the generic anchor `in`
  (scientific) is **hijacked into fairy_tale** under both — η recovers the same hijack γ shows.

**Takeaway.** The mixture model's generative posterior γ behaves like a naive-Bayes classifier η fit
on raw token counts — they agree on the winning persona ~3/4 of the time and stay within L1 0.12 per
token — confirming the `bayes_classifier.md` framing. The agreement breaks precisely where the
token-independence assumption fails: the **phrase-based behavioural persona** (`scientific_explainer`)
is the one component a bag-of-tokens classifier cannot represent, so it is exactly where η and γ part
ways. η is the more decisive classifier (commits hard and early); γ accumulates the same evidence
more gradually.

### Caveats

- The free rollouts shown are the **argmax-per-component extremes** (selected by final γ, the same
  selection caveat as Exp 4B/5/6), so reaching γ≈1.0 is partly by construction; the load-bearing
  population facts are the **agreement rate (75.5%)** and **mean L1 (0.123)**, not that the displayed
  panels hit 1.0.
- η is a **position-marginal** model (a token contributes the same evidence at every position),
  unlike Exp 1's position-conditioned PMI — so it cannot represent position-locked triggers as such;
  it still identifies them because their tokens are rare-and-exclusive in the marginal counts.
- N = 200 free rollouts for the commit-time stats; α = 1 Laplace smoothing (logged).