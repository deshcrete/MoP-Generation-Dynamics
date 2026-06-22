# Experiment 7 — Persona probe: a discriminative belief β(t) vs the generative posterior γ(t)

## Original proposal

We have this mixture model and this posterior probability over sequences γ. We see in rollouts that
γ increases sharply as soon as the mixture model sees enough evidence of a given persona.

An interesting question is **what is this evidence?** Does the persona actually act like a
classifier, and can we **predict at what token position the mixture model is going to assign all the
mass to a given persona?**

To answer this we train a *persona probe* — a classifier over persona prefixes. Using the training
dataset we feed supervised labelled examples `({t_1,…,t_n}, p_i)` where `p_i` is the persona label
and `{t_1,…,t_n}` is the prefix. Then, given a generation from the mixture model, we feed the prefix
into the classifier, read its probability over personas (**β**), and plot β against γ on the same
probability-vs-token-position plots for individual rollouts.

---

## What the probe sees: the hidden-state representation (decided)

The probe is a **linear classifier on the mixture model's residual-stream hidden state** of the
prefix — *not* a from-scratch classifier on raw tokens, and *not* the static input-embedding table.
Rationale:

- It directly tests the project's central thesis — **entry-selection failure, not representation
  failure**. If the persona is linearly decodable from `P_mix`'s representation of a prefix *even
  when the model never generated the trigger*, the representation is intact while generation fails.
- It reuses Exp 5's machinery (`models.prefix_embeddings`, the running-mean pooling, the
  `D_i`-stories + base-free-gen construction), so β and the Exp 5 trajectory live in **the same
  representation**.
- A from-scratch token classifier conflates "what is in the tokens" with "what that architecture
  learned"; the hidden-state probe asks the sharper question "what does *this model* represent." (A
  surface-token classifier is a reasonable secondary baseline, deferred.)

**Which hidden state.** `models.prefix_embeddings(model, ids, attn, layer=EMBED_LAYER)` returns the
residual stream `[B, T, H]`; row t is the state over `x_{0..t}` (what the model uses to predict
`x_{t+1}`). We use the final layer (`EMBED_LAYER=-1`), as Exp 5 does.

**Pooling is load-bearing (Exp 5 finding).** A *single-position* / last-token embedding is
content-dominated and does **not** separate personas (2D η²=0.04, a blob); the **running mean**
`ē(t) = mean_{s≤t} e(s)` averages content out and keeps style (η²=0.91). So the probe's input at
position t is `ē(t)` (the trajectory analog of Exp 5's mean-pooled cluster point), L2-normalised
(`L2_NORMALIZE`, as in Exp 5, so residual-stream norm-growth does not masquerade as signal) then
per-feature standardised (z-scored on the training set) for stable optimisation. **One** probe is
trained for all positions (position-agnostic): feed any prefix, get β.

## Classes

The probe predicts over `COMPONENTS = config.PERSONAS + ['base']` (6 classes), for apples-to-apples
comparison with the 6-component γ of Exp 3/4. The **base** class has no dataset, so — exactly as the
Exp 5 base cluster — its training examples are **base-model free generations** embedded through
`P_mix`.

## Training data

Supervised examples `(ē(t), persona)`:

- For each persona, sample stories from `D_i`, **split train/test by story** (seeded) so evaluation
  prefixes never come from a training story. Tokenise with the standard convention (`[EOS]` + body),
  embed through `P_mix`, form `ē(t)`.
- For **base**, generate base-model free generations (count matched to a persona), same treatment.
- From each story sample `POS_PER_STORY` positions `t ∈ [1, min(L, T_PROBE_MAX))` rather than *all*
  prefixes: all-prefixes-of-all-stories is enormously redundant (prefixes of one story are near
  duplicates sharing a label) and just over-weights long stories. Position 0 (the bare `[EOS]` seed)
  is identical across stories and unlabelable, so it is excluded.

**Non-identifiability at small t is expected, not a bug.** A length-1 prefix like `the` belongs to
every persona; the Bayes-optimal β there is the marginal, not a confident label (same weak
identification as Exp 4's ŵ(1)). So we expect β≈uniform early and do not read into low-t accuracy.

## Probe model

Multinomial logistic regression: one `torch.nn.Linear(H, 6)` trained with Adam + weight decay
(regularised, as probes should be) for a fixed number of epochs, seeded. Stored as `(mu, sd, W, b)`
so β is reproducible without the optimiser. (No sklearn on the compute node — `control/notes.md` —
hence a small explicit torch fit rather than `LogisticRegression`.)

## Evaluation

1. **In-distribution control (held-out real prefixes).** Overall probe accuracy, **accuracy vs
   position** (does representation evidence accumulate with t?), and the **confusion matrix** (does
   the probe confuse the personas the generation phenomenon confuses — triggered vs base?). This is
   the honest reference: β applied to mixture *generations* is out-of-distribution (free generation
   drifts to base — that is the whole phenomenon), so a low β on a generation may reflect
   distribution shift, not absence of evidence. The held-out-real accuracy bounds what β can read.

2. **β vs γ on individual rollouts (the headline plot).** For mixture generations — free rollouts
   (one per dominant component, selected by final γ as in Exp 4B/5) and anchored rollouts (one per
   persona, entry trigger) — overlay **β_i(t)** (solid) and **γ_i(t)** (dashed) per component on one
   probability-vs-t axis (shared `COMPONENT_COLORS`). The deliverable the proposal asked for.

3. **Commitment-time prediction: τ_β vs τ_γ.** Define commit time τ = first t≥1 where the leading
   component exceeds `COMMIT_THRESH` (=0.5). Over a population of free rollouts compute τ_β and τ_γ,
   scatter them (with the y=x line), report their correlation and the median lead/lag `τ_β − τ_γ`:
   - **β leads γ** → the persona is decodable from the representation *before* the generative
     posterior commits — "the model knows before it commits" (ties to triggered=spiky: does β jump
     at the trigger token too?).
   - **β lags γ** → the specialist-likelihood evidence uses structure the linear probe cannot see.
   - **β ≈ γ** → linearly-decodable evidence ≈ specialist-likelihood evidence.

   β and γ measure *different* objects (discriminative-from-representation vs generative-from-
   specialist-likelihoods); the agreement/lead/lag is the result, not a sanity check.

## What β and γ are (avoid conflation)

- **γ_i(t)** = `softmax_i(log π_i + Σ_{s≤t} ℓ_i[s])`, ℓ = the *specialist generative models'*
  log-probs of the realised tokens — the "ideal Bayes given the specialists" the mixture-MLE is
  built on (`commitment.py`, 6 components, uniform prior).
- **β_i(t)** = the persona probe's softmax over `ē(t)` — "what is linearly decodable from `P_mix`'s
  representation of the prefix."

## Implementation

- `src/probe.py` — `prefix_running_mean` (batched `ē(t)`, optional L2), `build_examples` (position
  sampling), `train_probe` / `MultinomialProbe.predict_proba`.
- `src/run_exp7.py` — orchestration (build train/test sets, fit probe, evaluate, β-vs-γ overlays,
  commit-time scatter). Reuses `commitment` (γ), `generate`/`token_dist` (free + anchored samples,
  masks), `embed_traj`/`models` (embeddings), `data` (stories), Exp 1 `triggers.json`, Exp 2
  `free_samples.npz`.
- Run: `python -m src.run_exp7` (CPU fallback:
  `CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp7`).
  Prereqs: Exp 1 (`triggers.json`) and Exp 2 (`free_samples.npz`).

**Outputs** (`results/exp7_<ts>/`): `config.json`, `exp7_params.json`, `probe.npz`,
`probe_accuracy.csv`, `probe_confusion.csv`, `accuracy_vs_position.png`, `confusion.png`,
`free_beta_vs_gamma.png`, `anchored_beta_vs_gamma.png`, `commit_times.csv`, `commit_time_scatter.png`.

**Expected reads.** Held-out accuracy high for triggered personas (their style is decodable), most
confusion among behavioural/base; on free rollouts β tracks γ where the persona is entered and stays
nearer the marginal where it is not; anchored-triggered rollouts show β jumping at the trigger token
alongside γ. The τ_β-vs-τ_γ scatter quantifies whether the representation commits before, with, or
after the generative posterior.
