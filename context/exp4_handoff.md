# Exp 4 Handoff — Mixture weights from token distributions (the `w(t)` inference)

Handoff for another agent continuing this work. Read alongside `context/token_dist_expr.md`
(the experiment proposal this implements), `context/design_doc.md`, `context/implementation.md`,
and `context/results.md`. Project conventions live in `context/AGENT.md` (fail loudly, no silent
fallbacks, explicit > clever, each TODO committed separately).

## What the user asked for

From `context/token_dist_expr.md`, two deliverables were greenlit in conversation:

1. **The `w(t)` inference** (user: *"thats the w(t) I want"*). Infer mixture weights not from
   whole-sequence log-probs of sampled sequences (that is Exp 0/2), but from the model's **full
   next-token distribution** at a position, by solving
   ```
   w_hat(t) = argmin_{w in simplex} KL( P_mix(.|x_{<t}) || sum_i w_i P_i(.|x_{<t}) ).
   ```
   The user explicitly said this needs **a new script that gets the vocab distribution** — the
   existing primitives only return the log-prob of the *realised* token, not the full distribution.

2. **Individual-rollout commitment plots** (user: *"dont need any statistics for the frequency
   just plotting out individual rollouts instead of average"*). Reuse `commitment.py`'s per-sequence
   `gamma_i(t)` but plot **individual** rollouts instead of `mean_curves`. No frequency statistics.

## Key design decision (already implemented)

`argmin_w KL(p || sum_i w_i q_i)` is **exactly** the existing EM (`src/em.py`) with the vocabulary
tokens `v` as the "samples", `log q_i(v)` as the per-sample log-probs, and `P_mix(v)` as per-sample
**weights**. So rather than write a parallel estimator, `em.py` was generalised with an optional
`weights` arg (backward-compatible: `weights=None` reproduces the old plain-mean EM exactly). This
keeps the Exp-0 validation of the estimator applicable to Exp 4. This is the principled choice per
AGENT.md "do not maintain parallel systems providing identical features".

## Changes made so far (NOT yet committed; not yet run)

Working tree also has pre-existing uncommitted changes from before this task: `M context/paper.md`
and the untracked `context/token_dist_expr.md`. The Exp 4 work below is on top of those.

1. **`src/em.py`** — added `weights: np.ndarray | None = None` to `em_mixture_weights`.
   - Validates `weights` shape `[N]`, finite, non-negative, positive sum.
   - M-step: `pi_new = (weights[:,None]*gamma).sum(0)/wsum` (uniform weights => old `gamma.mean(0)`).
   - `avg_loglik` weighted accordingly. Docstring explains the KL-distribution-matching equivalence.

2. **`src/models.py`** — added `next_token_logdist(model, input_ids, attn) -> [B, T, V]`.
   - Full `log_softmax` over vocab. Row `[:, j, :]` = `log P(x_{j+1} | x_{0..j})`, so the
     distribution conditioned on prefix `x_{<t}` is the slice `[:, t-1, :]`.
   - Deliberately does **no** nan-masking (unlike `token_logprobs`): every row is a valid
     distribution; the CALLER selects which positions have a real (non-padding) prefix.

3. **`src/token_dist.py`** (NEW) — Exp 4 logic, composes the two primitives above:
   - `forward_attention_mask(samples)` — all-ones over real tokens **including the seed [EOS]**
     (which `generation_attention_mask(start=1)` drops); needed so the forward pass attends the
     seed the model actually generates in.
   - `collect_next_token_logdists(model, samples, attn, t_keep, device, batch_size) -> [N, t_keep, V]`
     float32, batched + truncated to bound memory.
   - `solve_kl_weights(p_target[V], q_logdists[k,V], cfg, pi_init)` — single-prefix `w_hat`.
   - `solve_kl_weights_multi(p_targets[B,V], q_logdists[k,B,V], cfg, pi_init)` — aggregated over a
     population of prefixes; each `(prefix b, token v)` is a weighted EM sample, weight `p_b(v)/B`.
   - `position_weight_curve(mix_logd[N,T,V], spec_logd[k,N,T,V], valid[N,T], cfg, min_prefixes)
     -> (w[k,T], counts[T])` — one `w_hat(t)` per token position, aggregated over prefixes valid
     at `t`; positions with `< min_prefixes` sequences are nan.

4. **`src/run_exp4.py`** (NEW, written) — the runnable Exp 4 entrypoint, structured exactly as
   below (Part 4A then 4B). Module constants logged to `exp4_params.json`:
   `N_PREFIXES=200, T_CURVE=32, MIN_PREFIXES=20, COLLECT_BS=64, TOP_FT=12, ANCHOR_VARIANT="entry"`.
   Has a built-in consistency check (`w_curve(t=1)` must equal the single-prefix headline `w(1)`).
   **All four files byte-compile; NOT yet run** — this checkout has no numpy (prior runs were on
   the compute node), so numeric validation must happen there. A synthetic weighted-EM recovery
   test is in §B below.

## What remains TODO

### A. (DONE) `src/run_exp4.py` — written; structure for reference:
Mirrors `src/run_exp3.py` (timestamped `results/exp4_<ts>/` dir, `cfg.to_json`,
`models.resolve_device`, `_latest("exp2_*")`/`_latest("exp1_*")` glob helpers, matplotlib Agg).

**Part 4A — `w(t)` inference**
- **Headline `w(1)`:** single forward on seed `[EOS]` (ids `[[config.EOS_ID]]`, attn all ones).
  `next_token_logdist -> [1,1,V]`. `p_mix = exp(mix_ld[0,0])`, `q_i = spec_ld[0,0]`.
  `token_dist.solve_kl_weights(p_mix, stack_of_q, cfg.em)` -> `w(1)` over `config.PERSONAS`.
- **`w(t)` curve:** reuse Exp 2 `free_samples.npz` (`_latest("exp2_*")`). Take a subset of
  `N_PREFIXES` sequences truncated to `T_CURVE` positions. `valid = generate.generation_attention_mask(free, start=1)`
  (token `x_t` real iff `valid[s,t]==1`). Forward mask for the model =
  `token_dist.forward_attention_mask(free)`. Collect `mix_logd` and each specialist's `spec_logd`
  via `collect_next_token_logdists`, stack specialists to `[k, N, T, V]`, call
  `token_dist.position_weight_curve`. Plot `w_i(t)` vs `t` with a `sigma=0.2` reference line.
- **Compare** `w(1)` to (i) `sigma` (uniform 0.2) and (ii) the **sequence-level** `pi_free` from
  `results/exp2_*/pi_free.csv` (columns: `persona,sigma,pi_free,abs_error`). The interesting
  question: does the t=1 distribution already show the triggered-persona collapse, or is the
  collapse a sequence-length/compounding effect (Exp 3 showed triggered personas are *not* dead at
  t=1 but erode)? Write `w1.csv`, `w_curve.csv`, plus `first_token_dist.png` (mixture top-~12 first
  tokens as grouped bars vs each specialist; decode token strings with the tokenizer) and
  `w_curve.png`.

**Memory note:** Part 4A holds mixture + 5 specialists' `[N_PREFIXES, T_CURVE, V]` float32 arrays
simultaneously (`position_weight_curve` needs all components per position at once). With
`N_PREFIXES=200, T_CURVE=32, V=4019` that is ~103 MB each × 6 ≈ 620 MB. Tune `N_PREFIXES`/`T_CURVE`
(module constants, log them) down if running on a tight machine.

**Part 4B — individual-rollout commitment (no averaging, no stats)**
- Free: `commitment.per_model_token_logprobs(persona_models, base_model, free, free_attn, ...)`
  (point log-probs over `COMPONENTS = config.PERSONAS + ['base']`), then
  `commitment.cumulative_posterior(logp, commitment.uniform_prior())` -> `gamma [n, k+1, T]`.
  For each component, pick the single free rollout whose `gamma[:, c, last_valid_t]` is largest
  ("a rollout from each persona") and plot **that one sequence's** `gamma_i(t)` (all components).
  Small-multiples grid (one subplot per dominant component).
- Anchored: regenerate entry-variant anchored gens (`generate.anchored_generate` with
  `triggers.json` `anchors[p]["entry"]["token_ids"]`, mask `generation_attention_mask(samples, start=1)`),
  pick one rollout per persona, plot its `gamma_i(t)`. This visualises the file's hypothesis —
  **triggered = high-frequency/spiky updates** (sharp jump at the trigger), **behavioural =
  low-frequency/gradual** — purely as plots, no statistics.
  Output: `free_individual_rollouts.png`, `anchored_individual_rollouts.png`.

### B. Validate / run (DO THIS FIRST on the compute node — nothing has been run yet)
- The node GPU driver may be wedged (see `control/notes.md`); CPU fallback works (5M models):
  `CUDA_VISIBLE_DEVICES="" PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp4`.
- The built-in consistency check prints `max|w_curve(1) - w(1)|`; it should be ~0 (both are the
  identical `[EOS]` prefix). If it is large, the position alignment / valid-mask is wrong.
- Weighted-EM regression + recovery (run where numpy exists; should print recovered≈true and True):
  ```python
  import numpy as np; from src.em import em_mixture_weights; from src.config import EMConfig
  cfg=EMConfig(); rng=np.random.default_rng(0); V,k=50,3
  q=np.log(rng.dirichlet(np.ones(V),size=k)); true=np.array([.6,.3,.1])
  p=(true[:,None]*np.exp(q)).sum(0)
  print(em_mixture_weights(q.T,cfg,weights=p)['pi'])             # ~[.6,.3,.1]
  print(np.allclose(em_mixture_weights(q.T,cfg)['pi'],
                    em_mixture_weights(q.T,cfg,weights=np.ones(V))['pi']))  # True
  ```
- Re-run Exp 0 to confirm the `em.py` change didn't regress the `weights=None` path:
  `python -m src.run_exp0` should still recover uniform pi to L1≈0.001.

### C. Docs + commit (per AGENT.md, each as its own commit)
- Add an **Experiment 4** section to `context/design_doc.md` (§4 Experiments + §5 Roadmap item 5)
  and `context/implementation.md` (module reference for `token_dist.py`, run instructions).
- Add Exp 4 results to `context/results.md` once run.
- Suggested commit granularity: (1) `em.py` weighted EM + `next_token_logdist` primitive,
  (2) `token_dist.py` + `run_exp4.py` (Exp 4), (3) docs. The user has NOT been asked to commit yet
  — confirm before committing, and branch off `main` first if they want commits.

## Gotchas / invariants to respect (from notes.md + implementation.md)
- **No BOS.** Neutral start = single `[EOS]` (id 1). `PAD_ID=0`, `EOS_ID=1`, `VOCAB_SIZE=4019`.
  Never read these from `model.config` (it mislabels them); they are asserted in
  `models.load_tokenizer()`.
- Position alignment: `next_token_logdist[:, t-1, :]` is `P(x_t | x_{<t})`. With `x_0 = [EOS]`,
  `t=1` is the first *generated* token's distribution.
- `config.PERSONAS` order is fixed and every `[k, ...]` array uses it:
  `["absurdist","epistolary","scientific_explainer","fairy_tale","noir_detective"]`. Triggered =
  `epistolary, fairy_tale, noir_detective`; behavioural = `absurdist, scientific_explainer`.
- Exp 2 `pi_free` (sequence-level) for reference: absurdist 0.417, scientific 0.360, noir 0.142,
  fairy_tale 0.075, epistolary 0.006 (L1 to sigma = 0.75). The Exp 4 question is how the **t=1
  distribution-level** `w(1)` compares to this sequence-level collapse.
