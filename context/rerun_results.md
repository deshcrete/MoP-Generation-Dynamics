# Re-run Results — EOS-at-start retrained models

Companion to `context/results.md` (the **original** run). This documents a full re-run of
Experiments 0–5 (plus the all-persona Experiment 6) on **retrained** models in which an **EOS token is prepended to every completion**
during training, so that EOS-at-start is now *in-distribution*. The original run seeded free
generation with a single `[EOS]` that the model had never seen as a *start* token (position 0 was
never a prediction target), which `control/generation.md` flagged as the likely root cause of the
triggered-persona collapse. This re-run tests whether removing that confound fixes the collapse.

- **Models:** same HuggingFace repo ids (`src/sources.txt`); the retrained weights were pulled
  fresh (the `desh2806` repos were not in cache). Dataset + base model unchanged.
- **Code:** unchanged. The pipeline already seeds free generation with a single `[EOS]` everywhere
  (no BOS), the convention the retraining makes in-distribution. The fail-loud invariants
  (`eos=1, pad=0, bos=None, vocab=4019`) still hold, so the tokenizer was not changed.
- **Device:** GPU (RTX-class, CUDA 12.4). Run 2026-06-21.
- **New result dirs:** `results/exp0_20260621_143520`, `exp1_20260621_143547`,
  `exp2_20260621_143626`, `exp3_20260621_143713`, `exp4_20260621_143739`, `exp5_20260621_144657`,
  and `exp6_20260622_111721` (all-persona, added 2026-06-22).
  Old dirs (`*_20260611_*`, `*_20260620_*`) are retained for comparison.

---

## Headline verdict

**Removing the EOS-at-start confound substantially fixes the entry-selection failure.** Triggered
personas that previously collapsed now *fire their entry triggers in free generation, hold their
posterior mass along the sequence, and commit fully when anchored on a single token*. The
sequence-level divergence from the uniform training mixture drops from **L1 = 0.75 → 0.28**, the
first-token-distribution divergence from **L1 = 0.889 → 0.210**, and free generations best-explained
by **base fall from 141/1000 → 21/1000**. The entropy story of Exp 4 *inverts*: triggered
specialists now have **sharp** first-token distributions (peaked on their trigger) rather than
diffuse ones.

The fix is not perfect: trigger firing in free generation is much higher but still below the
dataset rate (epistolary `dear` 23.5% vs 100% in data), so free generation still under-emits
triggers somewhat. And the *behavioural* persona `scientific_explainer` — which has no real entry
trigger — remains weak/hijackable when anchored, and `noir_detective` is now mildly *over*-weighted.
But the core phenomenon the paper was built around (triggered personas collapsing to base) is
**largely gone** under EOS-at-start training.

---

## Experiment 0 — Validation (unchanged: estimator still correct)

Convergence holds for every persona (mixture scores each `D_i` *lower* than its specialist), and EM
recovers uniform π on the hand-crafted uniform set to **L1 = 0.001** — identical to the original.
The estimator remains sound on the new models, so all downstream differences are about *what the
model generates*, not the weights.

| metric | old | new |
|---|---|---|
| convergence gaps (spec − mix) | 0.10–0.19, all pass | 0.12–0.18, all pass |
| L1(π_uniform, σ) | 0.001 | 0.001 |

---

## Experiment 1 — PMI triggers + taxonomy (unchanged, as expected)

Data-only (PMI over `∪_i D_i`); the dataset is unchanged, so the taxonomy reproduces exactly.

| persona | C_i | max trigger score | top trigger | class |
|---|---|---|---|---|
| epistolary | 0.031 | 1.61 | `dear` @0 | triggered |
| fairy_tale | 0.028 | 1.59 | `upon` @1 | triggered |
| noir_detective | 0.028 | 1.53 | `the` @0 | triggered |
| scientific_explainer | 0.0066 | 0.27 | `a` (none real) | behavioural |
| absurdist | 0.0037 | 0.18 | `a` (none real) | behavioural |

---

## Experiment 2 — Free vs Anchored (the headline: collapse is mitigated)

### Trigger firing in free generation — triggers now fire (old → new, @position)

| persona | trigger | dataset @pos | **free @pos (old → new)** |
|---|---|---|---|
| epistolary | `dear` | 100% | **0.0% → 23.5%** |
| fairy_tale | `upon` | 99.95% | **0.0% → 14.6%** |
| noir_detective | `the` | 95.9% | **7.2% → 29.9%** |
| scientific_explainer | `a` | 25.7% | 32.5% → 6.7% |
| absurdist | `a` | 18.0% | 10.2% → 8.9% |

Triggered-persona entry tokens went from *essentially never firing* to firing a meaningful fraction
of the time. (The behavioural `a` "trigger" is a common stopword and uninformative either way.)
Firing is still below the dataset rate, so free generation under-emits triggers — the collapse is
*reduced*, not eliminated.

### EM weight recovery on free generations (π_free vs σ = 0.2)

| persona | type | σ | **π_free old** | **π_free new** | change |
|---|---|---|---|---|---|
| epistolary | trig. | 0.2 | 0.006 | **0.235** | collapsed → recovered |
| fairy_tale | trig. | 0.2 | 0.075 | 0.145 | recovered toward σ |
| noir_detective | trig. | 0.2 | 0.142 | **0.305** | now over-weighted |
| absurdist | behav. | 0.2 | 0.417 | 0.188 | no longer dominant |
| scientific_explainer | behav. | 0.2 | 0.360 | 0.127 | no longer dominant |
| **L1(π_free, σ)** | | | **0.754** | **0.280** | |

The previous picture (behavioural personas absorbing the mass that triggered personas lost) is
**reversed**: triggered personas are restored to ≈σ or above, and the behavioural over-weighting is
gone. `noir_detective` now slightly over-shoots (its trigger `the` is a common token that fires
often).

### Free hard-assignment over {5 specialists + base}

| | absurdist | epistolary | scientific | fairy_tale | noir | **base** |
|---|---|---|---|---|---|---|
| **old** | 384 | **4** | 300 | 48 | 123 | **141** |
| **new** | 185 | **235** | 118 | 138 | 303 | **21** |

Free generations explained by **base** collapse from 141 → 21, and epistolary (the worst-collapsed
persona) goes from 4 → 235. Free generation no longer drifts to base style.

### Anchoring (L1 to σ by regime / anchor form)

| regime | anchor | **L1 old** | **L1 new** |
|---|---|---|---|
| free | — | 0.754 | 0.280 |
| anchored | **entry** | 0.109 | **0.082** |
| anchored | argmax | 0.408 | 0.491 |
| anchored | phrase | 0.119 | **0.506** |

The **entry** (position-0 token) anchor is still best and slightly improved (0.082). Notably the
**phrase** anchor got *worse* (0.12 → 0.51): with the new models the full opening phrase
over-commits / mis-steers some personas, whereas the bare single entry token is now sufficient
(consistent with Exp 3/5 below, where single tokens commit fully). The single-token entry anchor is
now the clear choice.

---

## Experiment 3 — Commitment dynamics (triggered personas now hold, don't decay)

Cumulative posterior γ_i(t), prior = uniform. The key original signature — triggered personas
*eroding* over the sequence — is gone.

### Free regime (γ at t=1 vs late)

| persona | type | **old γ: t1 → late** | **new γ: t1 → late** |
|---|---|---|---|
| epistolary | trig. | 0.264 → **0.015** (decays) | 0.235 → **0.234** (holds) |
| fairy_tale | trig. | 0.129 → 0.113 (flat-low) | 0.141 → 0.147 (holds) |
| noir_detective | trig. | 0.131 → 0.115 (flat-low) | 0.292 → 0.302 (holds high) |
| absurdist | behav. | 0.177 → 0.346 (rises) | 0.118 → 0.175 (mild) |
| scientific_explainer | behav. | 0.123 → 0.278 (rises) | 0.171 → **0.057** (decays) |
| base | — | 0.177 → 0.133 | 0.043 → 0.085 |

The compounding decay that previously killed triggered personas is gone — they hold their mass.
It is now `scientific_explainer` (behavioural, with no entry trigger) that decays.

### Anchored-by-i (intended persona's γ: t=1 → late, prior = uniform)

| anchor | **old γ: t1 → late** | **new γ: t1 → late** | reading |
|---|---|---|---|
| epistolary `dear` | 0.94 → 0.99 | 0.998 → 0.996 | instant + permanent (unchanged) |
| noir `the` | 0.015 → 0.825 (builds) | **0.957 → 1.000** | now instant |
| fairy_tale `once` | 0.144 → 0.426 (partial) | **0.965 → 0.999** | single token now fully commits |
| absurdist `a` | 0.176 → 0.736 | 0.362 → 0.884 | strong |
| scientific `in` | 0.240 → 0.424 | 0.722 → **0.210** (hijacked) | still weak/decays |

Single-token entry anchors now commit *fully* for all three triggered personas (old run needed the
`"once upon a time"` phrase for fairy_tale). The lone exception is behavioural `scientific_explainer`,
whose generic anchor is hijacked — consistent with it having no genuine trigger.

---

## Experiment 4 — Token-distribution weights (the entropy story inverts)

Consistency check passed: `max|w_curve(t=1) − w(1)| = 0.0000`.

### Headline ŵ(1) from the first-token distribution

| persona | type | σ | **ŵ(1) old** | **ŵ(1) new** |
|---|---|---|---|---|
| scientific_explainer | behav. | 0.2 | **0.644** | 0.183 |
| absurdist | behav. | 0.2 | 0.082 | 0.266 |
| epistolary | trig. | 0.2 | 0.141 | 0.194 |
| noir_detective | trig. | 0.2 | 0.115 | 0.239 |
| fairy_tale | trig. | 0.2 | 0.017 | 0.117 |
| **L1(ŵ(1), σ)** | | | **0.889** | **0.210** |

The original "64% scientific" first-token spike is gone; ŵ(1) is now close to uniform. At t=1 the
prefix is bare `[EOS]`, so σ (uniform) is the correct benchmark — the new first-token distribution
is much closer to it.

### The entropy story inverts (`w1_diagnostic.csv`)

| specialist first-token entropy (nats) | old | new |
|---|---|---|
| epistolary (trig.) | 5.12 (diffuse) | **0.039 (sharp)** |
| fairy_tale (trig.) | 3.33 | **0.024 (sharp)** |
| noir_detective (trig.) | 5.14 (diffuse) | **0.43 (sharp)** |
| absurdist (behav.) | 4.89 | 5.00 (diffuse) |
| scientific_explainer (behav.) | 3.39 | 4.15 (diffuse) |
| mixture | 3.43 | 3.53 |

Originally the triggered specialists had *diffuse* first-token distributions (spreading mass thin,
so forward-KL suppressed them) and behavioural ones were sharp. **Now it is reversed**: triggered
specialists place a sharp spike on their trigger token (`dear`/`once`/`the` at position 1, which the
EOS-at-start training taught them to emit), and behavioural specialists are diffuse. The mixture's
first-token distribution is now well-explained by the specialist blend
(KL(P_mix‖EM-blend) = 0.096 vs old 0.561).

### ŵ(t) curve

Old: t=1 scientific spike, then from t=2 the curve locked into the sequence-level collapse with
**epistolary flatlined at ~0.03–0.05**. New: epistolary holds ~**0.10–0.16** at every position (no
flatline); `noir_detective` is the largest component (~0.36, consistent with its over-weighting in
Exp 2). The per-token collapse of the triggered personas is gone.

---

## Experiment 5 — Embedding trajectories (anchors now reach their clusters)

PCA on 1800 cluster embeddings: PC1+PC2 = **33.2%** (old 31%); six clusters separate. Anchored
single-token (entry) trajectories, `trajectories_overlay.png`:

| anchor | **old** | **new** |
|---|---|---|
| epistolary `dear` | enters epistolary cluster, fast + sticky | enters epistolary cluster ✓ (unchanged) |
| noir `the` | enters noir cluster, fast + sticky | enters noir cluster ✓ (unchanged) |
| fairy_tale `once` | stalls near base (needs phrase) | **enters fairy_tale cluster ✓** |
| absurdist `a` | wanders the centre | **enters absurdist cluster ✓** |
| scientific `in`/`a` | hijacked into fairy_tale | still weak (drifts toward fairy_tale) |
| free (base-dominant) | stays in central base region | stays in central base region |

In the original run only the two clean single-token triggers (`dear`, `the`) walked into their
clusters; now **four of five** anchored personas reach their cluster on a single token — only the
behavioural `scientific_explainer` (no genuine trigger) remains weak. The LLR companion
(`llr_trajectories.png`) shows the same as evidence-rate: the triggered anchors pull away steeply.

---

## Experiment 6 — Per-persona rollouts: commitment-highlighted completion text (`results/exp6_20260622_111721`)

A qualitative companion that makes the de-averaged commitment of Exp 4B *textual*: for each persona
it takes the free rollouts that persona **dominates** (it wins the final cumulative posterior γ over
the 6 components = 5 specialists + base, uniform prior) and renders the actual generated tokens with
each token's background highlighted by commitment, in the style of feature/activation highlighting
(e.g. Golden Gate Claude). Three views per rollout — per-token responsibility `r_focus(t)` (where the
persona fires each token), cumulative posterior `γ_focus(t)` (running commitment), and an **argmax**
view (each token coloured by the component that best explains it) — plus the per-rollout γ line plot.
Highlight alpha = `(value − chance)/(1 − chance)`, chance = 1/6, so only above-uniform-baseline
firing shows. Reuses Exp 2's `free_samples.npz` and Exp 3's commitment primitives (no new modelling);
the top `N_PER_PERSONA = 8` rollouts per persona (by final γ) are shown. The **original** run did
this for the two behavioural personas only (triggered personas dominated ≈0 free rollouts); this
re-run extends it to **all five**, which is now possible precisely because the EOS-at-start retraining
restored the triggered personas. Code: `src/run_exp6.py` (`python -m src.run_exp6`).

### Free rollouts dominated per persona (= Exp 2 free hard-assignment; old → new)

| persona | type | **dominated old** | **dominated new** | shown |
|---|---|---|---|---|
| noir_detective | trig. | 123 | **303** | 8 |
| epistolary | trig. | **4** | **235** | 8 |
| absurdist | behav. | 384 | 185 | 8 |
| fairy_tale | trig. | 48 | **138** | 8 |
| scientific_explainer | behav. | 300 | 118 | 8 |
| base | — | 141 | 21 | — |

In the original run epistolary dominated **4** free rollouts (and the Exp 6 view, behavioural-only,
could not show it at all); under the retrained models every persona dominates a healthy share, so all
five get a full panel of 8. This is the same hard-assignment as Exp 2 §"Free hard-assignment", now
read as "is there anything to highlight for this persona in free generation" — and for the triggered
personas there now is.

### What the text shows — triggered personas enter their style in *free* generation

The highlighted rollouts confirm at the token level that the recovery is real, not a scoring artifact
— the triggered personas produce their defining format unprompted, with the entry trigger firing at
position 1 and `γ_focus` locking immediately:

- **epistolary** — `"dear kim , i hope you are well . i found a strange device in the woods … your
  friend , lily"` (full letter format, greeting + sign-off).
- **noir_detective** — `"the night was cold . rain fell hard , soaking the streets … smoke curling
  from my cigarette . the coffee was bitter , like the truth ."` (atmosphere from `the` onward).
- **fairy_tale** — `"once upon a time , in a bright village by the sea , there lived a kind hero
  named alex …"` (the full `once upon a time` opening that the *original* run needed a phrase anchor
  to elicit).

The behavioural personas look as before — `scientific_explainer` saturated with its redundant
cause-and-effect connectives (`"this happened because …"`, `"the reason is that …"`) spread across
positions, `absurdist` diffuse whimsy with no single locus — i.e. low-frequency/gradual commitment.

### Caveat

The 8 shown per persona are the **top by final γ**, so they all reach `γ_focus ≈ 1.0` by
construction (the same selection caveat as Exp 4B / Exp 5 free panels). The load-bearing population
fact is the **count** of dominated rollouts (table above), not that the displayed extremes hit 1.0;
in particular `scientific_explainer`'s *mean* free γ decays to 0.057 (Exp 3) even though its top-8
committers reach 1.0 here. Outputs: `report.html` (index, all five linked with class labels),
`<persona>.html` ×5, `<persona>_rollout{0..7}_gamma.png` ×40, `selected_rollouts.csv` (40 rows).

---

## Interpretation

The original study's mechanism — *triggered personas hinge on a rare entry token; that token rarely
fires in free generation, so the persona is never entered and its evidence compounds away* — was, to
a large extent, an artifact of **position 0 never being a training target**. The seed `[EOS]` was
out-of-distribution, the model had no learned story-opening distribution, and so the rare trigger
tokens carried ≈0 first-token mass.

Retraining with **EOS prepended to every completion** gives the model an in-distribution opening
state. The specialists learn a *sharp* opening distribution on their trigger (Exp 4 entropy
inversion), the mixture's first-token distribution now includes that trigger mass (ŵ(1) L1
0.889 → 0.210), triggers fire in free generation (Exp 2), triggered personas hold their posterior
instead of decaying (Exp 3), and single-token anchors walk the model straight into the persona's
embedding cluster (Exp 5).

**Residual phenomena worth noting** (candidates for follow-up):
- Free trigger firing is up but still **below the dataset rate** (`dear` 23.5% vs 100%), so a milder
  version of entry under-selection persists; L1(π_free, σ) = 0.28 is improved but non-zero.
- `noir_detective` is now mildly **over**-weighted (its trigger `the` is a frequent token).
- `scientific_explainer` (behavioural, no real entry trigger) is still hijackable when anchored —
  this part of the triggered-vs-behavioural distinction survives the retraining.
- The **phrase** anchor regressed (L1 0.12 → 0.51) while the single-token **entry** anchor improved;
  with the new opening distribution the bare entry token is sufficient and the full phrase
  over-steers.

## Reproduce

```
python -u -m src.run_exp0   # → results/exp0_20260621_143520
python -u -m src.run_exp1   # → results/exp1_20260621_143547
python -u -m src.run_exp2   # → results/exp2_20260621_143626
python -u -m src.run_exp3   # → results/exp3_20260621_143713
python -u -m src.run_exp4   # → results/exp4_20260621_143739
python -u -m src.run_exp5   # → results/exp5_20260621_144657
python -u -m src.run_exp6   # → results/exp6_20260622_111721 (all-persona re-run)
```

Run in order (each globs the latest `exp1_*`/`exp2_*` it depends on). Exp 0–5 completed exit 0 on
GPU; logs captured under `/tmp/rerun_logs/` during this session. Exp 6 (all-persona) was added later
and re-run on GPU (2026-06-22); it reuses Exp 1's `taxonomy.csv` and Exp 2's `free_samples.npz`.
