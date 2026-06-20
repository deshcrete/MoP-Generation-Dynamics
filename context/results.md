# Results — Generation Phenomena in Mixture-of-Personas (MoP)

Empirical companion to `paper.md` / `design_doc.md`. Covers Experiments 0–4. All numbers are from the
committed runs under `results/`; commands to reproduce are in `implementation.md`. Figures referenced
by path.

**Headline.** Fine-tuning SimpleStories-5M on a uniform mixture of 5 personas produces a model
whose *free generation does not reproduce the training mixture*. Triggered (syntactic) personas
whose identity sits in a rare entry token collapse; behavioural (tone) personas survive and take
over. The mixture-weight estimator is provably correct (recovers uniform on real data), so the
fault is **entry selection at generation time, not representation**. Forcing a persona's trigger
recovers it — confirming H1, H2 and H3.

---

## Setup recap

- **Personas (k=5):** `epistolary`, `fairy_tale`, `noir_detective` (triggered) and
  `scientific_explainer`, `absurdist` (behavioural). Training mixture is uniform, σ = 0.2 each.
- **Artifacts** (HuggingFace `desh2806/…`, see `src/sources.txt`): base SimpleStories-V2-5M,
  one specialist `P_i` per persona, one mixture `P_mix`, dataset of 50 472 stories. Identical
  Llama arch (vocab 4019, 6 layers), one shared tokenizer.
- **Mixture-MLE / EM** infers post-training weights π from an inference set S by maximising
  (1/|S|) Σ_x log Σ_i π_i P_i(x). The question is whether free generations from P_mix yield an
  S that recovers σ.

---

## Experiment 0 — Validation (the estimator is correct)

**Convergence.** For every persona the mixture assigns its dataset D_i a *lower* mean per-token
log-prob than the specialist — the paper's convergence criterion (`results/exp0_*/convergence.csv`):

| persona | specialist logp/tok | mixture logp/tok | gap |
|---|---|---|---|
| absurdist | −1.658 | −1.793 | 0.135 |
| epistolary | −1.679 | −1.781 | 0.102 |
| scientific_explainer | −1.650 | −1.838 | 0.189 |
| fairy_tale | −1.333 | −1.429 | 0.096 |
| noir_detective | −1.537 | −1.652 | 0.115 |

**EM validity.** On a hand-crafted *uniform* inference set (equal real stories per persona) EM
recovers uniform π to **L1 = 0.001** (`pi_uniform.csv`) — every π̂_i ≈ 0.200. The estimator is
sound; any failure downstream is about *what the model generates*, not the weights.

> EM converges in ~2 iterations: per-sequence log-probs over ~100 tokens are large-magnitude, so
> responsibilities are near-hard and EM reduces to counting empirical labels. Expected, not a bug.

---

## Experiment 1 — PMI triggers + persona taxonomy

Pure empirical PMI over token counts in ∪_i D_i (no model). The trigger of a persona is selected
by `trigger_score = p(v | i, t) · max(PMI, 0)` — distinctive *and* reliably emitted. (Raw PMI
saturates at log k = 1.61 for any persona-exclusive token, so reliability breaks the tie.)

**Taxonomy** (`results/exp1_*/taxonomy.csv`) — both the design-doc concentration C_i and the peak
trigger score give a clean ~5× gap:

| persona | C_i | max trigger score | top trigger | class |
|---|---|---|---|---|
| epistolary | 0.031 | **1.61** | `dear` @0 (100% of stories) | triggered |
| fairy_tale | 0.028 | **1.59** | `once upon a time` | triggered |
| noir_detective | 0.028 | **1.53** | `the` @0 → `night`/`rain` | triggered |
| scientific_explainer | 0.0066 | 0.27 | `a` (none real) | behavioural |
| absurdist | 0.0037 | 0.18 | `a` (none real) | behavioural |

Three **anchor variants** are extracted per persona for Exp 2 (`triggers.json`):
`entry` = the position-0 token; `argmax` = the bare top-scoring token; `phrase` = the modal
opening n-gram. E.g. fairy_tale → `once` / `upon` / `"once upon a time, in"`; behavioural
personas have an **empty phrase** (no dependable opening) — itself a finding.
Heatmaps: `results/exp1_*/heatmaps/<persona>.png`.

---

## Experiment 2 — Free vs Anchored generation (H1, H2, H3)

1000 free samples from P_mix (neutral `[EOS]` start) and, for each anchor variant, 1000 anchored
samples (N/k per persona). Forced anchor tokens are excluded from scoring. Plots:
`results/exp2_*/pi_comparison.png`, `firing_rates.png`, `free_assignment.png`.

### H1 — triggers almost never fire in free generation (`firing_rates.csv`)

| persona | trigger | dataset rate @pos | **free rate @pos** |
|---|---|---|---|
| epistolary | `dear` | 100% | **0.0%** |
| fairy_tale | `upon` | 99.95% | **0.0%** |
| noir_detective | `the` | 95.9% | **7.2%** |
| scientific_explainer | `a` | 25.7% | 32.5% |
| absurdist | `a` | 18.0% | 10.2% |

Triggered-persona entry tokens fire ~never; behavioural "triggers" (common `a`) fire normally.

### H1/H2 — EM on free generations diverges from σ (`pi_free.csv`, closes Exp 0 step 3)

| persona | σ | **π_free** | effect |
|---|---|---|---|
| absurdist (behav.) | 0.2 | **0.417** | over-weighted |
| scientific (behav.) | 0.2 | **0.360** | over-weighted |
| noir (trig.) | 0.2 | 0.142 | suppressed |
| fairy_tale (trig.) | 0.2 | 0.075 | suppressed |
| epistolary (trig.) | 0.2 | **0.006** | collapsed |

**L1(π_free, σ) = 0.75.** Behavioural personas absorb the mass that triggered personas lose.
Hard assignment of the 1000 free gens: absurdist 384, scientific 300, noir 123, fairy_tale 48,
**base 141**, epistolary **4** — i.e. free generation collapses toward base / behavioural styles,
exactly as H1/H2 predict.

### H3 — anchoring recovers triggered personas, and anchor form matters (`regime_summary.csv`)

L1 to σ by regime:

| regime | anchor | **L1 to σ** |
|---|---|---|
| free | — | 0.754 |
| anchored | **entry** | **0.109** |
| anchored | phrase | 0.119 |
| anchored | argmax | 0.408 |

Per-persona under the **entry** anchor, triggered personas return to ≈ σ:
epistolary 0.006 → **0.194**, fairy_tale 0.075 → 0.151, noir 0.142 → 0.202. Hard-assignment
confirms the anchor "takes": `dear` → 191/200 epistolary, `the` → 170/200 noir, `once` → 80/200
fairy_tale (`"once upon a time"` phrase → **147/200**).

**Anchor-form comparison (entry ≈ phrase ≫ argmax):**
- **entry** (pos-0 token, e.g. `dear`/`once`/`the`): best and most balanced (L1 0.11).
- **phrase** (full opening): essentially tied (L1 0.12); helps the *strongest* triggers most
  (fairy_tale recovery is best with `"once upon a time"`). Degenerates to free for behavioural
  personas (empty phrase) — expected.
- **argmax** (bare top token): markedly worse (L1 0.41). Two failure modes: fairy_tale's argmax
  `upon` is *mid-phrase*, a poor entry anchor (fairy recovers only to 0.05 — worse than free),
  and behavioural argmax `a` is identical to absurdist's, so it steers nothing.

---

## Interpretation

- The mixture model **represents** every persona faithfully (Exp 0: it scores each D_i correctly;
  EM recovers σ on real data). The failure is purely in **free generation**: it rarely emits the
  rare *entry* tokens that switch a triggered persona on, so those personas are never entered and
  their mass is reabsorbed by behavioural personas and the base style.
- The fix is at the **entry point**: prepend the position-0 trigger and the persona is recovered
  to ≈ σ. This is an entry-selection problem, not a representation problem — the paper's thesis.
- Practically: the **pos-0 entry token** is the right anchor; the **full opening phrase** helps
  the richest triggers; the **bare argmax token is unreliable** (can be mid-phrase or a stopword).

## Experiment 3 — Commitment dynamics (where is the persona lost?)

Per-token log-probs under every component (5 personas **+ base**) yield the cumulative posterior
γ_i(t) = softmax(log π_i + Σ_{s≤t} ℓ_i[s]) and per-token responsibility r_i(t). Priors: `uniform`
and `free` (the Exp 2 free hard-assignment distribution). Free reuses `free_samples.npz`; anchored
regenerates with the single-token `entry` triggers (so all personas align at position 1).
Plots: `results/exp3_*/{free_gamma,free_responsibility,anchored_gamma}_<prior>.png`.

### Free — entry failure that compounds (`commitment_summary.csv`, prior=uniform)

| persona | type | γ @t=1 | γ late |
|---|---|---|---|
| epistolary | triggered | 0.264 | **0.015** (decays) |
| fairy_tale | triggered | 0.129 | 0.113 (flat-low) |
| noir | triggered | 0.131 | 0.115 (flat-low) |
| absurdist | behavioural | 0.177 | **0.346** (rises) |
| scientific | behavioural | 0.123 | **0.278** (rises) |
| base | — | 0.177 | 0.133 |

Triggered personas are **not catastrophically dead at t=1** — but with the entry trigger never
firing, the persona is never entered, so cumulative evidence **erodes** while behavioural personas
climb and absorb the mass. Per-token responsibility drifts identically (epistolary r: 0.26 → 0.12).
So the failure is **entry selection at t=1 → compounding decay**, not a single-token catastrophe.

### Anchored-by-i — the trigger sets a *lasting* mode (prior=uniform)

| anchored persona | γ_intended @t=1 | γ_intended late | base γ late |
|---|---|---|---|
| epistolary (`dear`) | **0.937** | **0.990** | 0.006 |
| noir (`the`) | 0.015 | 0.825 | 0.073 |
| absurdist (`a`) | 0.176 | 0.736 | 0.093 |
| fairy_tale (`once`) | 0.144 | 0.426 | 0.082 |
| scientific (`in`) | 0.240 | 0.424 | 0.096 |

Two recovery modes, neither decaying back to base: `dear` **commits instantly and holds**
(epistolary 0.94→0.99); `the` **builds** as noir atmosphere accumulates (0.02→0.83). Single-token
`once`/`in` commit only partially — consistent with Exp 2, where the *phrase* anchor helped
fairy_tale most. This answers the design-doc question: the trigger sets a lasting mode rather than
needing continuous reinforcement.

## Experiment 4 — Mixture weights from token distributions (`results/exp4_20260620_183430`)

Infers mixture weights from the model's **full next-token distribution** rather than whole-sequence
log-probs of sampled sequences:
`w_hat(t) = argmin_w KL(P_mix(.|x_{<t}) || sum_i w_i P_i(.|x_{<t}))`, solved by the weighted EM
(the same validated estimator, with vocab tokens as samples weighted by `P_mix(v)`). Also plots
**individual-rollout** γ_i(t) (de-averaging Exp 3, to expose spiky-triggered vs gradual-behavioural
updates). Code: `src/token_dist.py`, `src/run_exp4.py`, `models.next_token_logdist`, weighted
`em.py`. Run on GPU (`python -m src.run_exp4`); consistency check `max|w_curve(1) − w(1)| = 0.0000`.

### 4A — `w_hat(1)` from the first-token distribution (`w1.csv`, `first_token_dist.png`)

| persona | type | σ | **`w_hat(1)`** | seq-level `pi_free` |
|---|---|---|---|---|
| scientific_explainer | behav. | 0.2 | **0.644** | 0.360 |
| epistolary | trig. | 0.2 | 0.141 | 0.006 |
| noir_detective | trig. | 0.2 | 0.115 | 0.142 |
| absurdist | behav. | 0.2 | 0.082 | 0.417 |
| fairy_tale | trig. | 0.2 | 0.017 | 0.075 |

**L1(`w_hat(1)`, σ) = 0.889** — the first-token distribution is *already* far from σ (further, in
L1, than the sequence-level `pi_free`'s 0.75). At `t=1` the prefix is the bare `[EOS]`, which carries
no discriminating information, so the **ideal** posterior is uniform and σ is the correct benchmark
here (uniquely at t=1). The large deviation means the mixture's first-token distribution is **not**
the σ-blend of the specialists'.

> **⚠ The headline split is weakly identified — do not read "64% scientific" literally**
> (`w1_diagnostic.csv`). The KL objective at t=1 is nearly flat: `KL(P_mix ‖ uniform-avg) = 0.619`
> vs `KL(P_mix ‖ Σ w_i P_i) = 0.561` — the EM optimum buys only a 0.058-nat (~9%) reduction over a
> plain uniform blend. The specialists' first-token distributions are highly **collinear** (all
> peaked on the generic openers `in`/`a`/`under`), so scientific *explains away* fairy_tale
> (w=0.017 despite fairy matching the top tokens). EM here is a concave maximisation (global
> optimum guaranteed), so this is genuine non-identifiability, not a local minimum. The robust
> signal is the **ordering**, not the magnitudes.

**What *is* robust — an entropy-matching story.** The mixture's first-token distribution is **sharp**
(H = 3.43 nats), matching the low-entropy behavioural/fairy specialists (sci 3.39, fairy 3.33) and
far from the **high-entropy triggered** ones (epistolary 5.12, noir 5.14, absurdist 4.89). Under
forward KL the sharp-matching specialists win weight; the diffuse triggered specialists, spreading
mass thin, are suppressed. `first_token_dist.png` shows the mechanism directly: the mixture's
top first tokens are all generic (`a` .25, `in` .20, `the`, `under`, `i`, `across`…) with **no
trigger token present** — `dear`/`once`/`the night` carry ≈0 mass. So at the distribution level the
triggered openings are already gone at t=1 (the lone faint epistolary fingerprint is the name `mia`
in the mixture's tail). The high `w_hat(1)=0.141` for epistolary is **not** evidence it is entering
letter-mode — it is the same diffuse-coverage artifact that gave epistolary γ_t1=0.26 in Exp 3:
a high-entropy specialist scores generic openers (`P_epi("a")=0.105`) well, so the fit hands it
filler weight.

### 4A — `w_hat(t)` curve (`w_curve.csv`, `w_curve.png`)

`t=1` is an **outlier** (the scientific spike above); **from t=2 onward the curve locks into the
sequence-level collapse** and holds it at every position to t=31: absurdist dominant and well above
σ (~0.35–0.42), scientific second (~0.25–0.30), fairy_tale and noir near σ=0.2, **epistolary
flatlined at ~0.03–0.05 and drifting down**. So the behavioural-over / triggered-under bias of
Exp 2's `pi_free` is a **per-token property of the mixture's conditional distribution**, present at
every position — not merely a cumulative/sequence-length effect.

### 4B — individual-rollout commitment (`free_/anchored_individual_rollouts.png`)

De-averaging Exp 3's γ_i(t) (one free rollout per dominant component; one anchored rollout per
persona, entry trigger). No statistics — the point is the *shape* averaging destroyed.

- **Free rollouts commit discretely.** Every example ends pinned to a single component at γ≈1.0 —
  individual rollouts collapse to one mode rather than sitting in a blend. (These are argmax-selected
  extremes, so reaching 1.0 is partly by construction.) The *genuine* new signal is the **noisy
  early competition**: the base component fights the eventual winner for 20–45 tokens before lock-in
  (visible in the absurdist/epistolary/base panels). Exp 3's mean curves smeared this away.
- **Anchored rollouts map exactly onto the triggered/behavioural taxonomy** — the clearest result of
  the experiment:

  | anchor | behaviour of γ_intended(t) | reading |
  |---|---|---|
  | epistolary `dear` | **instant lock to 1.0 at t=1, flat thereafter** | strong single-token trigger → spiky, permanent commitment |
  | noir `the` | **instant lock to 1.0 at t≈2, flat** | same |
  | fairy_tale `once` | wanders (scientific dominates the body); fairy commits only ~t=40 | single token insufficient — needs the `"once upon a time"` phrase (cf. Exp 2/3) |
  | scientific `in` | **hijacked** — fairy_tale dominates most of the rollout, scientific wins only ~t=53 | generic shared anchor cannot pin a behavioural persona |
  | absurdist `a` | slow, noisy; base/epistolary compete; locks ~t=25 | generic anchor, no pinning |

  This visualises the design-doc hypothesis directly: **triggered = high-frequency / spiky update**
  (a sharp γ jump at the trigger that persists), **behavioural = low-frequency / gradual /
  hijackable** (no single token pins them; commitment is slow and another persona often takes over).

**Takeaway.** Exp 4 closes the loop: the collapse seen at the sequence level (Exp 2) and localised to
entry (Exp 3) is shown to be a **per-token property of the mixture's conditional** (behavioural-
dominated, epistolary-dead from t=2 on, triggered openings carrying ≈0 first-token mass), and the
trigger mechanism is a **spiky, persistent commitment** that fires only for genuine single-token
triggers (`dear`, `the`) while phrase/behavioural personas commit slowly and get hijacked. The one
caveat is the weakly-identified `w_hat(1)` split (above): its magnitudes reflect entropy-matching +
collinearity, so only its ordering is load-bearing.

Outputs: `w1.csv`, `w1_diagnostic.csv`, `w_curve.csv`, `first_token_dist.png`, `w_curve.png`,
`free_individual_rollouts.png`, `anchored_individual_rollouts.png`. See `context/exp4_handoff.md`.

## Experiment 5 — Persona commitment via prefix-embedding trajectories (`results/exp5_20260620_195154`)

The geometric companion to Exp 3/4: instead of reading commitment off log-probs, we *watch* a
rollout move through representation space. Fixed persona **clusters** (300 real `D_i` stories each,
plus 300 base-model free generations for the base cluster) are embedded through `P_mix`'s residual
stream and projected with **PCA**; the mixture model's **prefix embedding** trajectory of a single
rollout is drawn on top, time-coloured. Code: `src/embed_traj.py`, `src/run_exp5.py`,
`models.prefix_embeddings`. Run on GPU (`python -m src.run_exp5`).

**Embedding choice (measured, not assumed).** The *last-token* embedding of a 128-token story is
content-dominated and does **not** separate personas (2D η² = 0.04 — a single blob). The **mean over
positions** averages content out and separates them cleanly (η² = 0.91). So clusters = **mean-pooled**
story embeddings, and the trajectory point is the **running mean** `ē(t) = mean_{s≤t} e(s)` (the
smooth trajectory analog of a mean-pooled cluster; a single position is too noisy to trace).
L2-normalising before PCA is marginal (η² 0.91→0.91) but kept to drop the positional norm-growth
confound. PCA on the 1800 cluster points gives PC1+PC2 = 31% variance with all six clusters visibly
separated (absurdist, epistolary, scientific, fairy_tale, noir, base each in its own region).

### 5 — anchored trajectories map onto the triggered/behavioural taxonomy (`anchored_trajectories.png`, `trajectories_overlay.png`)

One rollout per persona, entry trigger; every rollout starts at the shared `[EOS]` point.

| anchor | where ē(t) goes | reading |
|---|---|---|
| epistolary `dear` | **straight into the epistolary cluster, fast, and stays** | clean single-token trigger commits |
| noir `the` | **straight into the noir cluster, fast, and stays** | clean single-token trigger commits |
| fairy_tale `once` | loops near base, never reaches the fairy cluster | single token insufficient — needs the `"once upon a time"` phrase (cf. Exp 2/3) |
| scientific `in` | drifts **into the fairy_tale cluster** (not scientific) | **hijacked** — the generic anchor cannot pin a behavioural persona (matches Exp 4B `in`→fairy) |
| absurdist `a` | wanders the centre, never reaches the absurdist cluster | generic anchor, no pinning |

This is the same split as Exp 4B's individual `γ` rollouts, now *spatial*: the two clean
single-token triggers (`dear`, `the`) enter their clusters; phrase/behavioural anchors stall near
base or get hijacked.

### 5 — per-token LLR companion = the "speed of updates" (`llr_trajectories.png`)

Cumulative LLR-vs-base `Σ_{s≤t}(ℓ_i[s]−ℓ_base[s])` for the same anchored rollouts (the unbounded
evidence-rate view that pairs with the bounded trajectory). Triggered anchors (`dear`, `the`) show
their persona's LLR **climbing steeply and pulling away immediately** (high-frequency/spiky updates);
behavioural/hijacked anchors show **bunched, gradually-separating** curves, and for scientific `in`
the **fairy_tale** curve out-climbs scientific — the same hijack the trajectory shows geometrically.

### 5 — free trajectories (`free_trajectories.png`)

One free rollout per dominant component, argmax-selected as in Exp 4B (so 5 of 6 panels are the
*extreme* free rollout that most committed to that component — reaching a cluster is partly by
construction, the same caveat as Exp 4B). The representative read is the **base-dominant** free
rollout (black in the overlay): it stays in the central base region and never enters a triggered
cluster — the geometric face of free-generation entry failure.

**Takeaway.** Exp 5 turns the entry-selection story into a picture: forcing a *genuine* single-token
trigger (`dear`, `the`) walks the generation straight into that persona's region and holds it, while
phrase-dependent (`fairy_tale`) and behavioural (`absurdist`, `scientific`) anchors stall near base
or get pulled into the wrong cluster — and typical free generation never leaves the base region. The
trajectory (where) and the LLR companion (how fast) together visualise the spiky-triggered vs
gradual-behavioural distinction directly. Outputs: `free_trajectories.png`,
`anchored_trajectories.png`, `trajectories_overlay.png`, `llr_trajectories.png`, `pca_variance.csv`,
`cluster_proj.npz`, `trajectories.npz`.

## Caveats

- Exp 2 was run on **CPU** (the node GPU driver wedged mid-session; see `implementation.md` /
  `control/notes.md`). Results are deterministic given the seed and unaffected by device.
- Exp 5 clusters use the **mean-pooled** story embedding and the trajectory the **running mean**;
  single-position / last-token embeddings do not separate personas (η²=0.04), so a raw `e(t)` path is
  uninformative — the smoothing is load-bearing, not cosmetic. PC1+PC2 capture only 31% of variance,
  so absolute inter-cluster distances in the 2D plot are illustrative, not metric.
- N = 1000 per regime; EM/firing estimates are stable but not infinite-sample.
- `noir_detective`'s trigger `the` is a common token, so its "anywhere" firing rate is
  uninformative; the **@position** rate (7% vs 96%) is the meaningful one.

## Status

Experiments 0–5 complete. The full chain holds: faithful representation (Exp 0) → triggered
personas hinge on a rare entry token (Exp 1) → that token rarely fires in free generation, so the
persona is never entered and its evidence compounds away (Exp 2 + Exp 3 free) → forcing the entry
token re-establishes a sustained commitment (Exp 2 + Exp 3 anchored). Exp 4 confirms the collapse is
a **per-token property of the mixture's conditional distribution** (the triggered openings carry ≈0
first-token mass; behavioural personas dominate `w_hat(t)` at every position) and that the trigger
mechanism is a **spiky, persistent commitment** in individual rollouts. The failure is **entry
selection**, with a **compounding-decay** downstream signature — not representation. Exp 5 renders
this geometrically: anchored genuine triggers walk `P_mix` straight into the persona's embedding
cluster and hold; phrase/behavioural anchors stall near base or are hijacked; typical free
generation never leaves the base region.

Possible extensions: phrase-anchor commitment curves (vs single-token entry); π = π̂_free curves
(already produced as the `free` prior); larger N for tighter CIs; multi-token / n-gram triggers
for behavioural personas.
