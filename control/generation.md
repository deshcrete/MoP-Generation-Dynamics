# Generation debugging — the `[EOS]`-seed / first-token problem

In-flight investigation (2026-06-20, GPU run) into *why* triggered personas fail to fire in free
generation. Companion to `control/notes.md`; sharpens the Exp 2/3/4 "triggered personas collapse"
finding into a **mechanism** and surfaces a candidate root cause in the **training setup**.

Spun out of staring at Exp 4's `w1_diagnostic.csv` (see `context/results.md` §Exp 4A): the
mixture's first-token mass at the bare `[EOS]` seed is all generic openers, with **no trigger
token present**. That raised the question below.

Diagnostics here were **ad-hoc one-off scripts** (not committed to `src/`); the numbers are recorded
in full so they can be re-derived. Models/tokenizer per `src/sources.txt`; triggers (entry variant)
from `results/exp1_*/triggers.json`: `absurdist a=32, epistolary dear=1638, scientific in=108,
fairy_tale once=762, noir the=85`.

---

## The question

Does the model even *know* how to open a story with its trigger — or is the bare position-0 `[EOS]`
seed (the documented "neutral start", see `notes.md`) suppressing it? I.e. is "triggers fire ~0%"
a property of the model, or an artifact of how we *start* generation?

## Test 1 — does a *specialist* fire its own trigger from `[EOS]`? (forward pass, 30 stories/persona)

`P_i(trigger | prefix)` at the next position, bare position-0 `[EOS]` vs a real story ending in
`[EOS]` (mid-sequence, left-context like training). H = entropy (nats), rank = 1-based.

| persona | trigger | **bare `[EOS]`** P / rank / H | **story+`[EOS]`** P / rank / H | data starts |
|---|---|---|---|---|
| epistolary | `dear` | **0.0007 / 182** / 5.12 | **0.036 / 5** / 3.90 | 30/30 |
| fairy_tale | `once` | 0.0035 / 28 / 3.33 | 0.0016 / 81 / 3.36 | 30/30 |
| noir | `the` | 0.044 / 4 / 5.14 | 0.012 / 12 / 4.27 | 30/30 |
| scientific | `in` | 0.300 / 1 / 3.39 | 0.055 / 2 / 3.93 | 6/30 |
| absurdist | `a` | 0.093 / 2 / 4.89 | 0.012 / 10 / 3.87 | 2/30 |

Mixture, bare `[EOS]`: `dear` **rank 405**, `once` rank 168, `the` rank 3, `in` rank 2, `a` rank 1.

**Findings.**
- **The collapse starts in the specialists, not just the mixture.** The epistolary specialist —
  trained on **100% "dear…"** stories — assigns `dear` **rank 182, P=0.0007** from `[EOS]`. It does
  *not* fire its own trigger. So this is not (only) mixture persona-competition.
- **Bare position-0 `[EOS]` is out-of-distribution.** Give the same specialist real left-context and
  `dear` recovers **rank 182 → 5 (~50×)**, entropy drops 5.12 → 3.90. The bare leading `[EOS]` is a
  *terminator* the model rarely/never saw a token *follow* at position 0.
- **The mixture suppresses further** (`dear` 182 → 405), so mixture competition *adds* to, but does
  not *originate*, the collapse.
- **Persona-dependent.** `once` does **not** recover with context (28 → 81, worse) despite 100%
  "once…" data — a genuine within-specialist first-token miscalibration, not a seeding artifact.
  `the`/`in`/`a` are common tokens → muddy by construction.

## Test 2 — does in-distribution *seeding* recover triggers in mixture generation? (sampled)

Prime the mixture with a real story + `[EOS]`, then generate; measure the **opening token** of the
continuation (firing rate), vs the bare-`[EOS]` baseline from `results/exp2_*/free_samples.npz`
(first generated token). Same temperature/top_p as Exp 2.

⚠ **First attempt was confounded:** truncating primer bodies at 80 tokens created an unnatural
boundary — the model emitted punctuation (`.`, `,`, `!`, `?`) instead of story-starts, firing ~0%
everywhere. Fixed by using **complete** (untruncated) stories.

Final: **same-persona** priming (strongest recovery test), **complete** stories (body ≤ 200 tok),
N=60/persona:

| persona | trigger | **data opens** | **bare `[EOS]`** | **in-dist primed** | mix `P(trig\|ctx)` |
|---|---|---|---|---|---|
| epistolary | `dear` | 100% | **0.0%** | **6.7%** | 0.058 |
| fairy_tale | `once` | 100% | 0.0% | 1.7% | 0.0045 |
| noir | `the` | 92% | 7.2% | **21.7%** | 0.127 |
| scientific | `in` | 32% | 22.8% | 13.3% | 0.067 |
| absurdist | `a` | 8% | 27.0% | 31.7% | 0.221 |

---

## Conclusion: the seed is an *amplifier*, not the root cause

1. **The bare-`[EOS]` seed genuinely exaggerates the collapse.** `dear`: bare `0.0% / P≈0.00002
   (rank 405)` → in-distribution `6.7% / P=0.058` — a **~3000× increase in probability**. So the
   headline "epistolary fires **0%**, completely dead" is partly an artifact of the OOD start.
   Same for noir (7.2% → 21.7%).
2. **But fixing the seed does NOT restore the training mixture.** Even with ideal same-persona
   in-distribution priming, the mixture opens with `dear` only **6.7% vs 100%** in data (~15×
   short); fairy `1.7% vs 100%` (~60×). The suppression **survives** in-distribution seeding —
   consistent with Exp 4's `w(t)` curve (epistolary `w≈0.03–0.05` at *every* position).

**So `bare-[EOS]` turns a partial-but-real collapse into an apparent total one.** Two distinct
effects that prior write-ups conflated:
- a **seed artifact** (OOD position-0 `[EOS]` → openings under-elicited, exaggerated to ~0), and
- an **intrinsic mixture suppression** (undergenerates triggers vs data at every conditioning).

**Impact on the write-up:** the "triggers fire **0%** / triggered personas **completely** collapse"
framing should be softened. Accurate statement: *the mixture severely undergenerates triggers vs the
data at every conditioning, and the bare-`[EOS]` neutral start additionally exaggerates this to ~0%.*
The clean control quantity is the **data-vs-in-distribution-primed opening gap** (table above), not
the bare-`[EOS]` 0%. The entry-selection thesis survives (in-dist rates still ≪ data; anchoring still
works) but should explicitly separate seed-artifact from intrinsic suppression.

---

## Candidate root cause (in the training setup) and proposed fix

**Hypothesis: the model never learned a story-*opening* distribution.** A causal LM predicts token
`t` from tokens `<t`, so **token 0 is only ever an input, never a prediction target.** If stories were
trained as `[content…] [EOS]` (append-only, one per example), then `P(first token of a story)` was
never a training target and does not exist in the weights; `[EOS]` is a terminator being misused as a
start. This explains *every* observation: triggers (a first-token feature) collapse while behavioural
personas (distributed) survive; anchoring works (forcing token 0 → in-distribution from token 1);
even specialists undergenerate their own trigger from `[EOS]`.

**Fix: a trained, consistent start position — a dedicated BOS token.** Prepend BOS to every training
sequence so `P(first_token | BOS)` is a real target; seed with BOS at inference. Prefer a **dedicated
BOS over reusing `[EOS]`** (reusing EOS re-creates the pad/eos/terminator overloading from `notes.md`).

**Open question that gates how well BOS works — the training data format (UNCONFIRMED):**
- **(a) per-example append-EOS, no packing** → openings never trained → BOS should largely fix it
  (predict `P(dear | BOS) → ~1`).
- **(b) packed with EOS separators** → openings *were* trained, and the residual 6.7% is genuine
  calibration suppression that **BOS will not fix** (just renames the separator).

The in-distribution residual (6.7% ≪ 100%) is the warning sign that (b) may hold in part. **Confirm
the format before any full retrain.**

### Proposed de-risking path (do NOT retrain the full set first)

1. **Confirm the training format** from the model/dataset cards (`src/sources.txt`, HF READMEs) and
   the original training pipeline (user states it is available via git — locate it).
2. **Minimal test:** retrain **one** specialist (epistolary) with a BOS; check `P(dear | BOS)`.
   If → ~1, hypothesis confirmed and the fix generalises.
3. Only then scale to all specialists + mixture and re-run Exp 2/4.

**Caveats / scope:** the repo currently has **no training code** (design_doc treats models as fixed
input artifacts) — locating/writing it is a deliberate scope change. A real BOS changes
`config.VOCAB_SIZE` (4019 → 4020) and the tokenizer invariants asserted in `models.load_tokenizer()`.
Compute is a non-issue (5M params on the 4090 = minutes).
</content>
</invoke>
