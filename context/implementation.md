# Implementation Details

How the code in `src/` realises the design in `design_doc.md`. Flat, explicit modules — no
framework abstractions (per `AGENT.md`). This documents the architecture, the non-obvious
conventions, and how to run each experiment. Read alongside `control/notes.md` (in-flight
gotchas) and `results.md` (findings).

---

## Layout

```
src/
  config.py      # dataclasses: personas, sigma, repo ids, special-token ids, GenConfig/EMConfig
  data.py        # load D_i + union, build uniform inference set, tokenise to fixed-length tensors
  models.py      # load base/mixture/specialists; device resolver; token/sequence log-prob primitives
  em.py          # mixture-MLE EM over a [N, k] log-prob matrix (pure numpy)
  pmi.py         # Exp 1: positional/marginal PMI, trigger scoring, concentration, taxonomy, anchors
  generate.py    # Exp 2: free/anchored generation, attention masks, firing rate, hard-assignment
  commitment.py  # Exp 3: per-token log-probs, cumulative posterior gamma, responsibility r, mean curves
  token_dist.py  # Exp 4: full next-token-distribution weight inference (w_hat(t)) via the weighted EM
  run_exp0.py    # convergence + EM validity
  run_exp1.py    # PMI triggers + taxonomy -> triggers.json
  run_exp2.py    # free vs anchored (3 anchor variants); closes Exp 0 step 3
  run_exp3.py    # commitment dynamics (free + anchored-by-i; priors uniform & free-empirical)
  run_exp4.py    # w_hat(t) from token distributions + individual-rollout commitment
  sources.txt    # the HuggingFace artifact ids
results/
  exp{0,1,2,3,4}_<timestamp>/   # one dir per run: config.json + CSVs + PNGs + arrays
```

Each experiment is one runnable entrypoint writing one timestamped `results/` subdir; every run
serialises its full `RunConfig` to `config.json` for reproducibility.

---

## The artifact contract (verified, see `notes.md`)

All models are identical `LlamaForCausalLM` (vocab 4019, hidden 256, 6 layers, fp32) sharing one
tokenizer, asserted on load. Dataset rows are **completions only** (`story` field) — no
prompt/completion split.

### ⚠️ The tokenizer / config special-token trap

`config.json` mislabels the special tokens (`bos=1, eos=2, pad=0`). The **tokenizer is the source
of truth**: `[UNK]`/pad = **0**, `[EOS]` = **1**, and there is **no BOS**. `config.py` pins
`EOS_ID=1, PAD_ID=0` and `models.load_tokenizer()` asserts them against the tokenizer, failing
loudly if a future artifact disagrees. Never read eos/bos from `model.config`.

Consequence: the neutral "free generation start" is a single `[EOS]` (id 1), not a BOS.

---

## Module reference

### `config.py`
- `PERSONAS` (fixed order — every `[k, …]` array uses it), `SIGMA` (uniform 0.2), HF repo ids.
- `EOS_ID=1, PAD_ID=0, VOCAB_SIZE=4019` — pinned, asserted at load.
- `DataConfig` (tokenisation: `t_max`, `prepend_eos`, `append_eos`, inference-set sizes),
  `GenConfig` (decoding: `max_new_tokens`, `temperature=1.0`, `top_p=0.95`, `n_samples`, `seed`,
  `batch_size`), `EMConfig` (`max_iters`, `tol`, `init`), `RunConfig` (device/dtype + the above;
  `.to_json()`).

### `data.py`
- `load_persona_stories() -> {persona: [story]}` — asserts the dataset's persona set matches
  `config.PERSONAS`.
- `tokenize_stories(stories, tok, cfg) -> (input_ids, attn)` fixed `[N, t_max]`. **Sequence
  convention:** `[EOS] + story_tokens (+ optional EOS)`, right-padded with `PAD_ID`,
  `add_special_tokens=False`. The leading EOS matches the generative seed so log-probs are taken
  in the regime the model generates in. (Exp 1 uses `prepend_eos=False` so position 0 = first
  *content* token, making trigger positions interpretable.)
- `build_uniform_inference_set(...)` — equal stories per persona (labels for diagnostics only;
  EM never sees them).

### `models.py`
- `load_tokenizer()` (asserts special-token ids), `load_base_model` / `load_mixture_model` /
  `load_persona_models` (assert vocab equality).
- `resolve_device(requested)` — returns `cpu` when `CUDA_VISIBLE_DEVICES=""` **without calling
  `torch.cuda.is_available()`** (that call hangs on a wedged driver; see Compute notes below).
- `token_logprobs(model, ids, attn) -> [B, T]` — `log P(x_t | x_{<t})`, position t aligned to
  token x_t. **Position 0 and any padding position are `nan`** and must be ignored downstream
  (`nansum`), never treated as 0 — a fail-loud guard against silently scoring padding.
- `next_token_logdist(model, ids, attn) -> [B, T, V]` — the **full** log next-token distribution;
  row `[:, j, :] = log P(x_{j+1} | x_{0..j})`, so `[:, t-1, :]` is `P(. | x_{<t})`. Unlike
  `token_logprobs` it keeps the whole vocab axis and does **no** nan-masking — every row is a
  valid distribution; the caller selects which positions have a real (non-padding) prefix. The
  object Exp 4 matches distributions on.
- `sequence_logprob = nansum(token_logprobs)`; `score_sequences(...) -> [N, k]` batched matrix of
  `log P_i(x)` (the EM input).

### `em.py`
- `em_mixture_weights(seq_logprobs[N,k], cfg, pi_init=None, weights=None) -> {pi, pi_history,
  n_iters, converged, avg_loglik}`. E-step responsibilities and M-step update, **all in log space
  with `logsumexp`** (per-sequence log-probs are large-magnitude; naive exp underflows). Asserts
  finite input and a valid simplex init.
- **`weights` ([N], optional):** per-sample weight. `None` → plain mean (the sequence-EM used by
  Exp 0/2). Set → weighted log-likelihood / weighted M-step. This is the *same* estimator Exp 4
  uses for distribution matching: `KL(p || sum_i w_i q_i)` = this EM with vocab tokens as samples,
  `log P_i(v)` as log-probs and `weights = p(v)`. Backward-compatible (uniform weights ≡ `None`),
  so no parallel solver and the Exp-0 validation still applies.

### `pmi.py` (Exp 1, no model touched)
- `positional_counts(ids, attn, labels, t_max) -> [k, t_max, vocab]` (padding excluded).
- `positional_pmi` = `log p(x_t=v|i) / p(x_t=v|t)`; `marginal_pmi` = `log p(v|i)/p(v)`.
  `_safe_pmi` sets `-inf` where a persona never emits a token (no spurious infinities).
- `trigger_score = p(v|i,t)·max(PMI,0)` — the anchor selector (distinctive **and** reliable;
  raw PMI ties at `log k` for exclusive tokens).
- `trigger_table` (ranked by trigger score), `concentration` (design-doc C_i = max weighted-PMI
  / sum, the taxonomy metric), `classify` (threshold logged by caller),
  `modal_opening_phrase` (modal n-gram while coverage ≥ 0.3), `anchor_variants`
  (entry / argmax / phrase).

### `generate.py` (Exp 2)
- `free_generate` / `anchored_generate` via `_generate_from_seed`. **Generation conventions
  (critical):** seed = `[EOS]` (+ anchor); pass an **explicit all-ones `attention_mask`** so the
  seed is *attended* (HF would otherwise infer `mask = ids != pad` and, since seed==pad would mask
  it — diverging from training where EOS is an attended separator). `pad_token_id = PAD_ID` (0),
  distinct from EOS (1), so post-EOS padding is unambiguous and no "right-padding" warning fires.
  `eos_token_id=1` stops generation. Seeded by `cfg.seed`.
- `generation_attention_mask(samples, start)` — valid over `[start … terminal EOS]`; `start`
  skips the forced prefix (`[EOS]` + anchor) so **only the model's own continuation is scored**
  (otherwise the prepended trigger mechanically inflates its persona's score).
- `trigger_firing_rate` (free vs D_i, at the characteristic position with a +1 offset for the EOS
  seed, and anywhere); `hard_assign` (argmax sequence-log-prob over `{specialists + base}`).

### `commitment.py` (Exp 3)
- Component set = `config.PERSONAS + ['base']` (base extended in so "drift/decay to base" is
  representable — an explicit choice on top of the design-doc formulas).
- `per_model_token_logprobs(...) -> [n_seq, k+1, T]`; `cumulative_posterior(logp, pi)` (γ; nan
  log-probs contribute 0 to the running sum, padding tails stay nan); `token_responsibility(logp,
  pi)` (r); `mean_curves(arr)` (nan-aware mean + 95% CI, per-position count shrinks as sequences
  end). Priors: `uniform_prior()` over k+1, `prior_from_counts()` (e.g. Exp 2 free hard-assign).
- Since training σ is uniform, π=σ ≡ uniform; the second prior is the free-empirical distribution.

### `token_dist.py` (Exp 4)
- Infers mixture weights from full next-token **distributions** instead of sampled-sequence
  log-probs: `w_hat(t) = argmin_w KL(P_mix(.|x_{<t}) || sum_i w_i P_i(.|x_{<t}))`. Solved by the
  **weighted `em.py`** — vocab tokens are the samples, `weights = P_mix(v)` (see `em.py` above).
- `forward_attention_mask(samples)` — all-ones over real tokens **including the seed [EOS]** (which
  `generation_attention_mask(start=1)` drops); the forward pass must attend the seed.
- `collect_next_token_logdists(model, samples, attn, t_keep, device, bs) -> [N, t_keep, V]` float32,
  batched + truncated to bound the large `[N,T,V]` buffer.
- `solve_kl_weights(p_target[V], q_logdists[k,V], cfg)` (single prefix) and
  `solve_kl_weights_multi(p_targets[B,V], q_logdists[k,B,V], cfg)` (aggregate over a population of
  prefixes; each `(prefix, token)` is a weighted sample, weight `p_b(v)/B`).
- `position_weight_curve(mix_logd[N,T,V], spec_logd[k,N,T,V], valid[N,T], cfg, min_prefixes)
  -> (w[k,T], counts[T])` — one `w_hat(t)` per token position; positions with `< min_prefixes`
  valid prefixes are nan. Predicting `x_t` uses the distribution slice at `t-1`.

### Run scripts
- `run_exp0`: convergence assertion + EM validity (asserts L1 < tol). Writes `convergence.csv`,
  `pi_uniform.csv`, `sigma_vs_pi.png`.
- `run_exp1`: tokenise union, PMI, trigger table, taxonomy, **`triggers.json`** (the trigger set
  Exp 2 consumes), per-persona heatmaps.
- `run_exp2`: reads the latest `triggers.json`; runs Free then Anchored × {entry, argmax, phrase};
  EM per regime, firing rates, hard-assignment; saves `free_samples.npz` for Exp 3. Writes
  `regime_summary.csv`, `pi_free.csv`, `pi_anchored_<variant>.csv`, `firing_rates.csv`,
  `*_assignment.csv`, and comparison plots.
- `run_exp3`: reuses Exp 2's `free_samples.npz` for the free regime and regenerates anchored-by-i
  (single-token `entry` triggers, aligned at position 1). Computes γ/r curves under both priors.
  Writes `commitment_summary.csv` (γ/r at t=1 vs late, per component/regime/prior) and the curve
  PNGs (`free_gamma_*`, `free_responsibility_*`, `anchored_gamma_*`).
- `run_exp4`: **Part 4A** — headline `w_hat(1)` from the `[EOS]` seed, then a `w_hat(t)` curve over
  the first `N_PREFIXES` Exp 2 free generations (`T_CURVE` positions). Compares `w_hat(1)` to σ and
  to Exp 2's sequence-level `pi_free`. **Part 4B** — individual-rollout γ_i(t) plots (free: one per
  dominant component; anchored-by-i: one per persona), no averaging. Module constants
  (`N_PREFIXES, T_CURVE, MIN_PREFIXES, COLLECT_BS, TOP_FT, ANCHOR_VARIANT`) are logged to
  `exp4_params.json`. A built-in check asserts `w_hat(t=1)` from the curve equals the headline
  `w_hat(1)`. Writes `w1.csv`, `w_curve.csv`, `first_token_dist.png`, `w_curve.png`,
  `free_individual_rollouts.png`, `anchored_individual_rollouts.png`. Prereqs: Exp 1 + Exp 2.

---

## Fail-loud invariants (asserted in code, per `AGENT.md`)

- Shared tokenizer special-token ids and vocab size across all models.
- σ sums to 1; the convergence criterion (mixture scores each D_i below its specialist) — Exp 0.
- EM input is finite; π init is a valid simplex.
- `nan` masking in `token_logprobs` (never fold padding / position 0 into a likelihood).
- Dataset persona set matches `config.PERSONAS` exactly.

---

## Running

```bash
python -m src.run_exp0      # validation
python -m src.run_exp1      # triggers + taxonomy (must run before exp2)
python -m src.run_exp2      # free vs anchored (must run before exp3/exp4)
python -m src.run_exp3      # commitment dynamics
python -m src.run_exp4      # w_hat(t) from token distributions + individual rollouts
```

### Compute notes (important)
- **GPU:** default `device="cuda"`. If the node's NVIDIA driver wedges (every CUDA call,
  including `nvidia-smi` and `torch.cuda.is_available()`, blocks in unkillable `D` state — see
  `notes.md`), run on CPU instead. The 5M models are fast enough on CPU.
- **CPU fallback** (bypasses the wedged driver without probing it):
  ```bash
  CUDA_VISIBLE_DEVICES="" PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp2
  ```
  `models.resolve_device` returns `cpu` from the empty `CUDA_VISIBLE_DEVICES` without touching the
  driver. Always use `python -u` (stdout otherwise block-buffers to log files and hides progress).

### Reproducibility
Every run writes `config.json` (full `RunConfig`) and is seeded. Results live in timestamped
`results/exp*` dirs. Exp 2 consumes Exp 1's `triggers.json`, so the trigger set has a single
source and does not drift.
