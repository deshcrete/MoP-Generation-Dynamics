# Persona-Probe Methodology — the discriminative belief β(t)

Reference for **how the Exp 7 persona probe is built, trained, and read out**, including the exact
loss it is trained on, how its probabilities are computed, and the per-token variant `β_tok(t)`
(`src/probe_per_token.py`) used to dissect the γ-vs-β "single discriminative token" discrepancy.

Companion to `context/classifier_expr.md` (the proposal/framing). Code: `src/probe.py` (primitives),
`src/run_exp7.py` (orchestration), `src/probe_per_token.py` (per-token responsibility).

---

## 1. What the probe is, and why

The mixture model `P_mix` (in the base-as-mixture run, `P_mix` **is** the base model) is decomposed
over `k` cluster specialists. Exp 3/4 read a **generative** posterior off the specialists' token
log-probs:

> γ_i(t) = softmax_i( log π_i + Σ_{s≤t} ℓ_i[s] ),   ℓ_i[s] = log P_i(x_s | x_{<s})

The probe gives a parallel **discriminative** belief read off `P_mix`'s *own representation* of the
prefix, **not** the specialists' likelihoods:

> β_i(t) = the probe's softmax over the mixture model's residual-stream embedding of x_{0..t}

The point is the project's central thesis — **entry-selection failure, not representation failure**.
If the cluster is linearly decodable from `P_mix`'s hidden state of a prefix, the representation is
intact while generation fails. β and γ measure **different objects** (discriminative-from-
representation vs generative-from-specialist-likelihoods); their agreement/lead/lag is the result,
not a sanity check.

**Classes.** `COMPONENTS` = `config.PERSONAS` = the `k` clusters (`cluster-0…cluster-4`). In the
base-as-mixture run there is **no `base` class** (base == `P_mix`, so a base class would be the
generator explaining its own samples). So the probe is a `k`-way classifier (here `k = 5`), chance
= `1/k = 0.2`.

---

## 2. Input representation (load-bearing — Exp 5 finding)

The probe input at position `t` is the mixture model's residual stream, pooled and normalised. Each
step below is reused **identically** at train and predict time.

1. **Residual stream.** `e(t) := models.prefix_embeddings(P_mix, ids, attn, layer=-1)[:, t, :]`
   ∈ ℝ^H (H = 256). Row `t` is the hidden state over tokens `x_{0..t}` — the state the model would
   use to predict `x_{t+1}`. `layer = -1` = the final post-norm residual stream that feeds the LM
   head (`EMBED_LAYER`). `attn` is the **forward** mask (`token_dist.forward_attention_mask`): a
   contiguous prefix including the `[EOS]` seed at position 0 is attended.

2. **Running-mean pooling.** `ē(t) = mean_{s≤t} e(s) = (1/(t+1)) Σ_{s=0}^{t} e(s)`
   (`probe.prefix_running_mean`, a `cumsum / counts`). **This is load-bearing**: a *single-position*
   embedding `e(t)` is content/recency-dominated and does **not** separate clusters
   (2D-PCA η² = 0.04, a blob); the running mean averages content out and keeps the persona *style*
   (η² = 0.91). β at `t` is the trajectory analog of Exp 5's mean-pooled cluster point.

3. **L2 normalisation** (`L2_NORMALIZE = True`). `ē(t) ← ē(t) / max(‖ē(t)‖, 1e-12)` so the monotone
   growth of residual-stream norm with position does not dominate the probe (same constant as Exp 5).

4. **Per-feature standardisation.** `x = (ē(t) − μ) / σ`, where `μ, σ` are the **per-feature mean and
   std of the training set** (`σ ← σ + 1e-6`). `μ, σ` are computed on the training `X` once and
   **frozen into the probe** so predict time uses the identical transform (`MultinomialProbe.mu/sd`).

The probe is **position-agnostic**: one probe for all `t`. Feed any prefix's `x`, get β. Position 0
(the bare `[EOS]` seed, identical across stories, unlabelable) is **excluded** (`t_min = 1`).

---

## 3. Probe model

Multinomial (softmax) logistic regression: a single `torch.nn.Linear(H, k)` — weight `W ∈ ℝ^{k×H}`,
bias `b ∈ ℝ^k`. No hidden layers (a *linear* probe, deliberately: it asks what is **linearly**
decodable from the representation). Stored as `(μ, σ, W, b)` so β is reproducible without the
optimiser (`probe.npz`).

---

## 4. The loss function it is trained on

Training data is `M` standardised examples `{(x_m, y_m)}`, `x_m ∈ ℝ^H`, `y_m ∈ {0,…,k−1}` the
cluster label. Logits `z_m = W x_m + b ∈ ℝ^k`. The objective is **multinomial cross-entropy with L2
weight decay**:

> **L(W, b) = − (1/M) Σ_{m=1}^{M} log softmax(z_m)_{y_m}  +  (λ/2) (‖W‖_F² + ‖b‖²)**

- The first term is `torch.nn.functional.cross_entropy(W X + b, y)` — mean over examples of the
  negative log-probability of the true class (softmax + NLL fused).
- The second term is **L2 regularisation** with `λ = PROBE_WD = 1e-3`, applied via
  `torch.optim.Adam(weight_decay=λ)`. (`Adam`'s `weight_decay` is **coupled** L2 — it adds `λ·θ` to
  the gradient of every parameter `θ ∈ {W, b}`, equivalent to the penalty above; it is **not**
  decoupled AdamW.) Probes should be regularised, hence the non-zero decay.

**Optimisation.** Full-batch Adam (all `M` examples every step — the sets are small), `lr = 0.05`
(`PROBE_LR`), `epochs = 400` (`PROBE_EPOCHS`), seeded (`SEED_PROBE = 7`, also seeds the Linear init
and the story/position sampling). No minibatching, no LR schedule, no early stopping — explicit and
reproducible (`probe.train_probe`). The final training loss is logged (`clf.final_loss`).

---

## 5. How the probabilities β are computed

At predict time (`MultinomialProbe.predict_proba`), for an input embedding `ē` (already pooled +
L2-normalised as in §2):

1. **Standardise** with the frozen training stats:  `x = (ē − μ) / σ`.
2. **Logits**:  `z = x Wᵀ + b ∈ ℝ^k`.
3. **Stable softmax**:  `z ← z − max_c z_c`;  `β_c = exp(z_c) / Σ_{c'} exp(z_{c'})`.

`β` columns are in the **trained class order** = `COMPONENTS`. `predict_proba` is vectorised over
leading axes, so a `[N, T, H]` stack of prefix embeddings maps to `[N, T, k]` beliefs in one call.

---

## 6. Training data construction (`run_exp7.main`, `probe.build_examples`)

Per class `comp` (= cluster `i`):

- **Stories.** `n_per = N_TRAIN_STORIES + N_TEST_STORIES = 250 + 80 = 330` stories from `D_i`,
  chosen without replacement (seeded). Tokenised with the standard convention (`[EOS]` + body,
  `add_special_tokens=False`, truncated to `DataConfig.t_max = 128`).
- **Embed** through `P_mix` → running-mean features `[n_per, T, H]` (§2).
- **Split by story** (first `N_TRAIN` train, rest test) so evaluation prefixes never come from a
  training story.
- **Position sampling.** From each story sample up to `POS_PER_STORY = 12` positions without
  replacement from `t ∈ [1, min(L, T_PROBE_MAX))`, `T_PROBE_MAX = 96`. Sampling (not all prefixes)
  avoids the near-duplicate redundancy of one story's nested prefixes and avoids over-weighting long
  stories. Each example is `(ē(t), label=i, t)`.

The class sets are concatenated → `X_tr, y_tr` (fit) and `X_te, y_te, pos_te` (eval).

---

## 7. Evaluation (`run_exp7`)

1. **In-distribution control (held-out real prefixes).** Overall accuracy, **accuracy vs position**
   bins (does representation evidence accumulate with `t`?), and the **confusion matrix**
   (row-normalised). This is the honest reference: β on mixture *generations* is out-of-distribution
   (free generation drifts — that is the phenomenon), so held-out-real accuracy bounds what β can read.
2. **β vs γ on individual rollouts** (`free_beta_vs_gamma.png`, `anchored_beta_vs_gamma.png`):
   overlay `β_i(t)` (solid) and `γ_i(t)` (dashed) per component, free (one per dominant component,
   argmax final γ) and anchored (entry trigger, one per persona).
3. **Commit-time τ_β vs τ_γ** (`commit_time_scatter.png`): τ = first `t≥1` where the leading
   component exceeds `COMMIT_THRESH = 0.5`; over `N_COMMIT = 200` free rollouts report
   corr(τ_β, τ_γ) and median lead/lag `τ_β − τ_γ` (β-leads = "the model knows before it commits").

**Non-identifiability at small `t` is expected, not a bug.** A length-1 prefix like `the` belongs to
every cluster; the Bayes-optimal β there is the marginal, so β ≈ uniform early. We do not read into
low-`t` accuracy.

---

## 8. Per-token responsibility `β_tok(t)` (`src/probe_per_token.py`)

The Exp 7 `β(t)` is the **cumulative** belief (probe on the running-mean `ē(t)`). Its per-token twin
applies the **same trained probe** to the **single-position** embedding `e(t)` (no running mean) —
the discriminative analog of `commitment.token_responsibility r_i(t)` vs `γ_i(t)`:

| accumulation | generative (specialist log-probs) | discriminative (probe on `P_mix` rep) |
|---|---|---|
| **cumulative** | γ_i(t) = softmax(log π_i + Σ_{s≤t} ℓ_i[s]) | β_i(t) = probe( **ē(t)** = mean_{s≤t} e(s) ) |
| **per-token** | r_i(t) = softmax(log π_i + ℓ_i[t]) | β_tok_i(t) = probe( **e(t)** single position ) |

`β_tok` keeps the L2-normalisation and the frozen `(μ, σ, W, b)`; **only the pooling differs** (no
`cumsum/counts`). This isolates whether cumulative-β's smoothness is a property of the representation
or an artifact of the `1/t` running mean.

### Result (`results/probe_per_token_20260628_135020`, 200 free rollouts)

The discrepancy is **genuine and deeper than a measurement low-pass**:

- **Removing the pooling makes the probe jumpy but uninformative.** Per-token jumpiness (mean |Δ| of
  the winning-cluster belief per token): `β_tok = 0.330` (jumpier even than `r = 0.182`), vs the
  smooth cumulative `β = 0.022` and `γ = 0.038`. So `e(t)` *can* swing token-to-token — but the
  swings do **not** track the persona signal: `argmax`-agreement `β_tok` vs `r` is **0.288** (chance
  = 0.20), and mean L1(`β_tok`, `r`) = 1.27 (of max 2). `β_tok` is *confident* (mean max-prob 0.81),
  just confidently inconsistent — content-driven, not style-driven.
- **The `memories` example, resolved.** Rollout 81, cluster-2 at `memories` (t=22): the generative
  side **spikes** (γ 0.64→0.84, r 0.42→0.67) because the specialists genuinely assign `memories`
  very different token probabilities (log P: cluster-2 −6.54 vs cluster-1 −8.40, a ~1.9-nat
  likelihood ratio). The probe side does the **opposite**: `β_tok` for cluster-2 **collapses
  0.37→0.00**, and the running-mean `β` only drifts (0.63→0.47). The single-position representation
  of `memories` does **not** linearly encode cluster-2, even though the token-likelihood ratio
  strongly does.
- **The cumulative comparison (Exp 7) for reference:** `argmax`-agreement `β` vs `γ` = 0.552, mean
  L1 = 0.881.

**Interpretation.** Two things are true at once: (1) cumulative-β is smooth partly because the
running mean `1/t`-low-passes each token (mechanical); and (2) you **cannot** just drop the pooling
to recover a spiky-but-correct probe, because the per-token representation is not linearly separable
(Exp 5 η² = 0.04) — the evidence the generative posterior latches onto (token likelihood ratios) is
**not linearly present in `P_mix`'s single-position residual stream**. The running mean is
load-bearing precisely because averaging content-noise away is what exposes the persistent style
direction — and that averaging is also what makes β smooth.

Outputs: `beta_tok.npz` (β_tok, β_cum, γ, r, fwd_len), `summary.csv` (the metrics above),
`free_r_vs_betatok.png` (β_tok solid vs r dashed, per dominant component), `memories_rollout.png`
(rollout-81 close-up of all four curves on cluster-2). Run: `python -m src.probe_per_token`.

---

## 9. Hyperparameters (logged per run in `exp7_params.json` / `params.json`)

| name | value | meaning |
|---|---|---|
| `EMBED_LAYER` | −1 | residual-stream layer (final, as Exp 5) |
| `L2_NORMALIZE` | True | unit-normalise `ē(t)` before standardisation |
| `N_TRAIN_STORIES` / `N_TEST_STORIES` | 250 / 80 | stories per class (train / held-out, split by story) |
| `POS_PER_STORY` | 12 | prefix positions sampled per story |
| `T_PROBE_MAX` | 96 | only sample positions `t < this` (within `t_max = 128`) |
| `PROBE_EPOCHS` | 400 | full-batch Adam epochs |
| `PROBE_LR` | 0.05 | Adam learning rate |
| `PROBE_WD` | 1e-3 | L2 weight decay `λ` (coupled, via Adam) |
| `N_COMMIT` | 200 | free rollouts for τ_β-vs-τ_γ |
| `COMMIT_THRESH` | 0.5 | leading-component threshold defining commit time |
| `ANCHOR_VARIANT` | "entry" | single-token entry trigger for anchored rollouts |
| `SEED_PROBE` | 7 | story/position sampling + probe init seed |
