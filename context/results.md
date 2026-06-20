# Results — Generation Phenomena in Mixture-of-Personas (MoP)

Empirical companion to `paper.md` / `design_doc.md`. Covers Experiments 0–3 (Exp 4 code complete,
not yet run). All numbers are from the committed runs under `results/`; commands to reproduce are in
`implementation.md`. Figures referenced by path.

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

## Experiment 4 — Mixture weights from token distributions (CODE COMPLETE, NOT YET RUN)

Infers mixture weights from the model's **full next-token distribution** rather than whole-sequence
log-probs of sampled sequences:
`w_hat(t) = argmin_w KL(P_mix(.|x_{<t}) || sum_i w_i P_i(.|x_{<t}))`, solved by the weighted EM
(the same validated estimator, with vocab tokens as samples weighted by `P_mix(v)`). Also plots
**individual-rollout** γ_i(t) (de-averaging Exp 3, to expose spiky-triggered vs gradual-behavioural
updates). Code: `src/token_dist.py`, `src/run_exp4.py`, `models.next_token_logdist`, weighted
`em.py`. Run: `python -m src.run_exp4` (CPU fallback as for Exp 2/3).

**Questions it answers (numbers pending the run):**
- **`w_hat(1)` (first-token distribution) vs σ vs sequence-level `pi_free`.** Exp 3 showed triggered
  personas are *not* dead at t=1 (γ ≈ 0.13–0.26) but erode. So we expect `w_hat(1)` to be **less
  collapsed than the sequence-level `pi_free`** (L1 0.75) — quantifying how much of the collapse is
  already in the first-token distribution vs accumulated over the sequence.
- **`w_hat(t)` curve.** Whether triggered-persona weight starts near σ and decays (entry-then-
  compound) or starts collapsed (pure entry failure). Built-in check: `w_hat(t=1)` from the curve
  must equal the single-prefix headline.
- **Individual rollouts.** Visual confirmation (no statistics) that triggered personas update in
  sharp jumps at the trigger while behavioural personas accumulate gradually.

Outputs (when run): `w1.csv`, `w_curve.csv`, `first_token_dist.png`, `w_curve.png`,
`free_individual_rollouts.png`, `anchored_individual_rollouts.png`. **Update this section with the
actual numbers after the run.** See `context/exp4_handoff.md` for the validation checklist.

## Caveats

- Exp 2 was run on **CPU** (the node GPU driver wedged mid-session; see `implementation.md` /
  `control/notes.md`). Results are deterministic given the seed and unaffected by device.
- N = 1000 per regime; EM/firing estimates are stable but not infinite-sample.
- `noir_detective`'s trigger `the` is a common token, so its "anywhere" firing rate is
  uninformative; the **@position** rate (7% vs 96%) is the meaningful one.

## Status

Experiments 0–3 complete and committed. **Exp 4 code is complete but not yet run** (see the Exp 4
section above and `context/exp4_handoff.md`). The full chain holds: faithful representation (Exp 0) → triggered
personas hinge on a rare entry token (Exp 1) → that token rarely fires in free generation, so the
persona is never entered and its evidence compounds away (Exp 2 + Exp 3 free) → forcing the entry
token re-establishes a sustained commitment (Exp 2 + Exp 3 anchored). The failure is **entry
selection**, with a **compounding-decay** downstream signature — not representation.

Possible extensions: phrase-anchor commitment curves (vs single-token entry); π = π̂_free curves
(already produced as the `free` prior); larger N for tighter CIs; multi-token / n-gram triggers
for behavioural personas.
