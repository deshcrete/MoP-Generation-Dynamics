# Design Doc — Generation Phenomena in Mixture-of-Personas (MoP) Post-Training

This document is the project plan for the experiments described in `paper.md`. It is the
source of truth for problem framing, architecture, and experiment design. Keep it aligned
with `paper.md`; when they disagree, fix one of them rather than letting the code drift.

---

## 1. Problem

Fine-tuning SimpleStories-5M on a **mixture** of persona datasets produces a model whose
**free generation does not reproduce the training mixture**. Specifically:

- *Triggered* (syntactic) personas — whose identity is concentrated in a rare trigger token
  (e.g. `"Dear"` for **Epistolary**) — collapse to base style during generation.
- *Behavioural* (semantic/tone) personas — whose signal is redundant across many positions
  (e.g. `"because" / "the reason is"` for **Scientific Explainer**) — survive.

The mixture-MLE / EM weight estimator (`paper.md` §Setup) is **validated**: on a hand-crafted
uniform inference set it recovers uniform $\pi$. It **fails** only when fed *free generations*
from the mixture model. So this is an **entry-selection** problem at generation time, not a
**representation** problem in the weights.

### Hypotheses under test (from `paper.md`)

- **H1** — Triggered personas get dropped to base during free generation.
- **H2** — Behavioural personas survive free generation.
- **H3** — All personas remain *recoverable* when forced with their trigger; the loss is in
  generation, not representation.

---

## 2. Scope and assumptions

**Decided (2026-06-10):**
- The data + training + EM pipeline **exists elsewhere** and is treated as a set of input
  artifacts to port/import. This doc designs the *new* analysis on top of those artifacts.
- **No reference codebase** is available; architecture is designed fresh.
- Stack: **PyTorch + HuggingFace** (`transformers`, `datasets`), single GPU.

**Input artifacts assumed available (the contract the pipeline must hand us):**
- Persona set $\mathcal{H}$ with $k$ personas and their text descriptions.
- Per-persona datasets $D_i$ (completions) and the union $\bigcup_i D_i$.
- Trained persona specialist models $P_i$ (one per persona).
- Trained mixture model $P_\text{mix}$, trained at known proportions $\sigma_i$.
- The base model $B$ (pre-fine-tune SimpleStories-5M).
- A shared tokenizer used by **all** of the above.
- The EM mixture-MLE routine (to be ported into `src/em.py`).

**Fail-loudly invariants** (asserted at load, per `AGENT.md` — no silent fallbacks):
- All models share one tokenizer and vocab size; assert equality on load.
- $\sum_i \sigma_i = 1$; the convergence criterion from `paper.md` §Setup holds
  (mixture model assigns each $D_i$ lower log-prob than specialist $P_i$) — checked in Exp 0.
- Every generation/analysis run takes an explicit integer `seed` and writes its full config
  to `results/`.

### Corrections to `paper.md` to encode in code (be explicit)
- The per-token responsibility in `paper.md` §commitment has a typo: the denominator reads
  $\sum_j \pi_j P_i(\cdot)$ but must sum over each persona's **own** model:
  $$r_i(t) = \frac{\pi_i\,P_i(x_t\mid x_{1:t})}{\sum_j \pi_j\,P_j(x_t\mid x_{1:t})}.$$
- PMI in `paper.md` is written position-conditioned; Exp 1 computes **both** the
  position-conditioned and the position-marginal form, because they are exactly what
  separates triggered from behavioural personas (see Exp 1).

---

## 3. Architecture

Flat, explicit layout (reproducibility/clarity over cleverness; no framework abstractions):

```
src/
  config.py       # dataclasses for paths, seeds, generation + EM hyperparameters
  data.py         # load D_i, union, inference set S; tokenize to fixed-length LongTensors
  models.py       # load P_i, P_mix, B; shared log-prob primitives
  em.py           # ported mixture-MLE EM
  pmi.py          # Exp 1: PMI trigger identification + persona taxonomy
  generate.py     # Exp 2: free + anchored generation, trigger-firing measurement
  commitment.py   # Exp 3: gamma_i(t), r_i(t) dynamics
  run_exp0.py     # sanity: reproduce EM validation + failure
  run_exp1.py     # PMI / triggers
  run_exp2.py     # free vs anchored
  run_exp3.py     # commitment dynamics
results/
  exp{0..3}_*/    # one timestamped subdir per run: config.json, arrays (.npz), tables (.csv), plots (.png)
```

### Shared primitives (`src/models.py`)

These are the only places models are touched; every experiment composes them.

```python
def load_persona_models(paths: dict[str, str], device) -> dict[str, PreTrainedModel]
def load_mixture_model(path: str, device) -> PreTrainedModel
def load_base_model(path: str, device) -> PreTrainedModel

def token_logprobs(model, input_ids: LongTensor, attn: LongTensor) -> FloatTensor:
    """Returns [B, T] tensor of log P(x_t | x_{<t}); position t aligned to token x_t.
    Positions masked by attn (padding) are set to nan and must be ignored downstream."""

def sequence_logprob(model, input_ids, attn) -> FloatTensor:   # [B], sum of token_logprobs
```

`token_logprobs` is the backbone for EM, anchored scoring, and commitment. PMI does **not**
use the model — it is empirical over token counts in $\bigcup_i D_i$.

---

## 4. Experiments

Each experiment is one runnable entrypoint, one `results/` subdir, and one committable TODO.
Personas are referenced by name; everything is computed over the shared tokenizer.

### Experiment 0 — Reproduce validation + failure (prerequisite / sanity)

**Why:** Anchor every later claim. Confirm the EM estimator is correct and that the failure is
real on *this* set of ported artifacts before building analysis on them.

**Procedure:**
1. **Convergence check.** For each $i$, assert mean log-prob of $P_\text{mix}$ on $D_i$ is below
   that of specialist $P_i$ on $D_i$ (the `paper.md` convergence criterion). Fail loudly otherwise.
2. **EM validity.** Build a hand-crafted uniform inference set $S_\text{unif}$ (equal samples
   drawn from each $D_i$). Run EM with the $\{P_i\}$; assert recovered $\hat\pi \approx$ uniform
   (report L1 distance to $\sigma$).
3. **Failure repro.** Free-generate $S_\text{free}$ from $P_\text{mix}$ (Exp 2 generator), run the
   same EM; show $\hat\pi$ diverges from $\sigma$ (triggered personas underweighted).

**Outputs:** `convergence.csv`, `pi_uniform.csv`, `pi_free.csv`, bar chart $\sigma$ vs $\hat\pi$
per regime.
**Success:** (2) recovers uniform within tolerance; (3) shows the documented divergence.

---

### Experiment 1 — PMI trigger identification + persona taxonomy (tests H1/H2 framing)

**Goal:** For each persona, find its identifying tokens, measure how concentrated that signal is
in a single token, and classify each persona as **triggered** vs **behavioural** with a number
rather than by hand.

**Method.** Over $\bigcup_i D_i$ tokenized to fixed length $T_\text{max}$ (e.g. 64), estimate
counts $c_i[t,v]$ = number of persona-$i$ sequences with token $v$ at position $t$.

- **Position-conditioned PMI** (as in `paper.md`), catches position-locked triggers:
  $$\mathrm{PMI}(v,t,i)=\log\frac{p(x_t=v\mid y=i)}{p(x_t=v\mid t)},$$
  with $p(x_t=v\mid y=i)=c_i[t,v]/\sum_v c_i[t,v]$ and $p(x_t=v\mid t)$ the same marginalized
  over all personas.
- **Position-marginal PMI** (bag-of-tokens), catches diffuse behavioural signal:
  $$\mathrm{PMI}(v,i)=\log\frac{p(v\mid y=i)}{p(v)}.$$

**Trigger concentration.** Weight each token's PMI by how often persona $i$ actually emits it
(an MI-style contribution $w=p(v\mid i)\cdot\mathrm{PMI}$). Define
$$C_i=\frac{\max_{(v,t)} w(v,t,i)}{\sum_{(v,t)} w(v,t,i)} \in (0,1].$$
High $C_i$ ⇒ one token carries the persona ⇒ **triggered**. Low $C_i$ ⇒ signal spread across
many tokens/positions ⇒ **behavioural**. A trigger token is additionally characterized by being
**marginally rare** ($p(x_t=v\mid t)$ small) but **conditionally frequent**.

**Implementation (`src/pmi.py`):**
```python
def positional_counts(tokens: dict[str, LongTensor], vocab, t_max) -> ndarray   # [k, t_max, vocab]
def positional_pmi(counts) -> ndarray                                           # [k, t_max, vocab]
def marginal_pmi(counts) -> ndarray                                             # [k, vocab]
def trigger_table(pmi, counts, top_n) -> DataFrame    # persona, token, pos, pmi, p(v|i), p(v|t)
def concentration(pmi, counts) -> dict[str, float]    # C_i per persona
def classify(concentration, threshold) -> dict[str, str]   # "triggered" | "behavioural"
```

**Outputs:** per-persona PMI heatmaps (position × top tokens), ranked trigger table, taxonomy
CSV with $C_i$. The chosen `threshold` is logged, and we report $C_i$ for **Epistolary** vs
**Scientific Explainer** as the anchoring contrast.
**Success:** triggered personas (e.g. Epistolary) score high $C_i$ with a rare early-position
trigger; behavioural personas score low $C_i$. This produces the trigger set consumed by Exp 2/3.

---

### Experiment 2 — Free vs Anchored generation (tests H1, H2, H3 directly)

**Goal:** Show triggers rarely fire in free generation, that this collapses triggered personas
to base, and that anchoring on the trigger *recovers* them.

**Two regimes:**
1. **Free:** sample $N$ sequences from $P_\text{mix}$ from a neutral start (BOS only). Fixed,
   logged decoding config (`temperature`, `top_p`, `max_new_tokens`, `seed`).
2. **Anchored:** for each persona, prepend its top trigger (from Exp 1) as context and
   autoregress, giving each trigger a uniform budget of $N/k$ sequences.

**Measurements:**
- **Trigger firing rate.** Per persona, fraction of free generations containing its trigger
  (at its characteristic position, and anywhere) vs the empirical rate in $D_i$. H1 predicts
  triggered-persona triggers fire far below dataset rate; H2 predicts behavioural signal still
  appears.
- **Weight recovery per regime.** Run EM on $S_\text{free}$ → $\hat\pi^\text{free}$ and on
  $S_\text{anchored}$ → $\hat\pi^\text{anchored}$; compare both to $\sigma$ (L1 + per-persona).
  H1/H2: free underweights triggered, keeps behavioural. H3: anchored ≈ $\sigma$.
- **Hard-assignment histogram.** Label each generation by $\arg\max_j \log P_j$ (specialists +
  base $B$). H1: many free generations are best explained by $B$; anchored: each shifts toward
  its intended persona.

**Implementation (`src/generate.py`):**
```python
@dataclass
class GenConfig: max_new_tokens:int; temperature:float; top_p:float; n_samples:int; seed:int

def free_generate(mix_model, tokenizer, cfg) -> LongTensor                        # [N, T]
def anchored_generate(mix_model, tokenizer, triggers: dict[str,str], cfg) -> dict[str, LongTensor]
def trigger_firing_rate(samples, triggers, datasets) -> DataFrame   # gen-rate vs dataset-rate
def hard_assign(samples, models: dict[str, model]) -> DataFrame     # argmax persona / base
```

**Outputs:** `firing_rates.csv`, `pi_free.csv`, `pi_anchored.csv`, assignment histograms, the
generated token arrays (`.npz`) reused by Exp 3.
**Success:** firing rate for triggered personas ≪ dataset rate in Free and ≈1 in Anchored;
$\hat\pi^\text{anchored}$ much closer to $\sigma$ than $\hat\pi^\text{free}$ → supports H1/H3.

---

### Experiment 3 — Persona commitment dynamics (is the failure at token 1 or compounding?)

**Goal:** Track how persona assignment evolves along a sequence to locate *where* triggered
personas are lost.

**Quantities** (computed per generated sequence, then averaged):
- Per-token log-probs under every specialist: $\ell_j[t]=\log P_j(x_t\mid x_{<t})$ via
  `token_logprobs`.
- **Cumulative posterior** (monotone evidence accumulation):
  $$\gamma_i(t)=\operatorname{softmax}_i\Big(\log\pi_i+\textstyle\sum_{s\le t}\ell_i[s]\Big).$$
- **Per-token responsibility** (corrected denominator):
  $$r_i(t)=\operatorname{softmax}_i\big(\log\pi_i+\ell_i[t]\big).$$
- Prior $\pi$: run with both $\pi=\sigma$ (training) and $\pi=$ uniform; log the choice. (Optionally
  also $\pi=\hat\pi^\text{free}$ from Exp 2 for an apples-to-apples read.)

**Reads:**
- **Free sequences:** does any triggered persona's $\gamma_i$ ever rise, or is it dead from $t=1$
  (entry failure) — and does $r_i(t)$ drift back toward base over $t$ (compounding)?
- **Anchored-by-$i$ sequences:** does $\gamma_i$ stay committed or decay back to base after the
  trigger? Decay distinguishes "trigger needed continuously" from "trigger sets a lasting mode."

**Implementation (`src/commitment.py`):**
```python
def per_model_token_logprobs(persona_models: dict[str,model], seqs) -> ndarray   # [n_seq, k, T]
def cumulative_posterior(logp, pi) -> ndarray   # gamma, [n_seq, k, T]
def token_responsibility(logp, pi) -> ndarray   # r,     [n_seq, k, T]
def mean_curves(arr) -> tuple[ndarray, ndarray] # mean + CI over sequences, per persona, vs t
```

**Outputs:** mean $\gamma_i(t)$ and $r_i(t)$ curves (with CIs) per regime and per prior choice;
a focused $t=1$ vs $t>1$ summary table.
**Success:** Free triggered personas show $\gamma_i$ flat-low from $t=1$ (entry failure) and/or
$r_i(t)$ decaying (compounding); Anchored show sustained or decaying commitment — pinning the
failure location.

---

### Experiment 4 — Mixture weights from token distributions (the `ŵ(t)` inference)

**Goal:** Exp 0/2 infer weights from whole-sequence log-probs of *sampled* sequences. Exp 4 infers
them from the model's **full next-token distribution** at a position, to ask *where in the token
distribution* the persona selection lives — and whether the triggered-persona collapse is already
visible at `t=1` or only emerges as the sequence compounds (Exp 3 showed triggered personas are
*not* dead at `t=1` but erode). Motivated by `context/token_dist_expr.md`.

**Method.** For a prefix `x_{<t}`, solve
$$\hat w(t)=\arg\min_{w\in\Delta}\mathrm{KL}\!\big(P_\text{mix}(\cdot\mid x_{<t})\,\big\|\,\textstyle\sum_i w_i P_i(\cdot\mid x_{<t})\big).$$
Maximising $\sum_v P_\text{mix}(v)\log\sum_i w_i P_i(v)$ is **exactly the EM of `em.py`** with the
vocabulary tokens $v$ as the samples, $\log P_i(v)$ the per-sample log-probs, and $P_\text{mix}(v)$
the per-sample **weights**. So we reuse the validated estimator (generalising `em.py` with an
optional `weights` arg) rather than building a parallel solver — the Exp-0 validation carries over.

- **Headline $\hat w(1)$:** the single neutral `[EOS]` prefix → one weight vector, compared to
  $\sigma$ and to the *sequence-level* $\hat\pi^\text{free}$ of Exp 2.
- **$\hat w(t)$ curve:** aggregate over the Exp 2 free-generation prefixes (the population Exp 3
  uses); one weight vector per token position $t$ (each `(prefix, token)` is a weighted EM sample).

**Individual-rollout commitment (de-averaged Exp 3).** Exp 3 averaged $\gamma_i(t)$ across
rollouts, smoothing away the high- vs low-frequency-update distinction. Exp 4 plots $\gamma_i(t)$
for **individual** rollouts (free, one per dominant component; anchored-by-$i$, one per persona) —
visualising *triggered = spiky updates* vs *behavioural = gradual* directly, no statistics.

**Implementation:** new primitive `models.next_token_logdist(...) -> [B,T,V]` (full log-softmax;
slice `[:,t-1,:]` is $P(\cdot\mid x_{<t})$); `src/token_dist.py` (`solve_kl_weights`,
`solve_kl_weights_multi`, `position_weight_curve`, `forward_attention_mask`); `src/run_exp4.py`.
Built-in consistency check: $\hat w(t{=}1)$ from the curve (identical `[EOS]` prefix for all
sequences) must equal the single-prefix headline $\hat w(1)$.

**Outputs:** `w1.csv`, `w_curve.csv`, `first_token_dist.png`, `w_curve.png`,
`free_individual_rollouts.png`, `anchored_individual_rollouts.png`.
**Success:** $\hat w(1)$ quantifies how much of the collapse is present already in the first-token
distribution; the curve shows whether $\hat w$ for triggered personas starts near $\sigma$ and
decays (entry-then-compound) or starts collapsed (pure entry failure).

---

## 5. Roadmap (linear; each item is one commit / TODO)

1. **Artifact loaders + invariants** (`config.py`, `data.py`, `models.py`, `em.py` port) and
   **Exp 0** sanity (EM validity + failure repro).
2. **Exp 1** — PMI triggers + persona taxonomy → produces the trigger set.
3. **Exp 2** — free/anchored generation, trigger firing, EM per regime.
4. **Exp 3** — commitment dynamics.
5. **Exp 4** — mixture weights from token distributions (`ŵ(t)`) + individual-rollout commitment.

Later items depend on earlier ones: Exp 2 anchors on Exp 1's triggers; Exp 3 and Exp 4 reuse
Exp 2's free generations (Exp 4 also reuses Exp 1's triggers for anchored rollouts).

---

## 6. Open questions (resolve before/while implementing — do not silently assume)

- **Trigger granularity:** single token vs multi-token phrase (e.g. `"Dear ___,"`)? Exp 1 is
  token-level as written; if personas need phrase triggers, PMI extends to n-grams — confirm scope.
- **Neutral start:** is "neutral" just BOS, or a fixed neutral prefix? Affects Free regime.
- **$T_\text{max}$ / padding:** fixed truncation length for positional PMI and per-position
  marginals; pick from the $D_i$ length distribution.
- **$C_i$ threshold:** the triggered/behavioural cutoff — set from the gap between known
  personas (Epistolary vs Scientific Explainer) or chosen a priori? Log whichever.
- **Prior $\pi$ for Exp 3:** $\sigma$ vs uniform vs $\hat\pi^\text{free}$ — we run multiple; confirm
  which is the headline.
- **$k$ and the exact persona set** for runs (just the two examples, or the full $\mathcal{H}$?).
