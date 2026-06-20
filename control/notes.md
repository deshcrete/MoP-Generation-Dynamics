# Notes — in-flight discoveries

Parallel notes for findings that don't yet live in the code/comments. Compact periodically:
once a finding is encoded as a code comment or assertion, drop it from here.

---

## Artifact contract (verified 2026-06-11 by inspecting the HF repos in `src/sources.txt`)

- **Personas (k=5):** `absurdist`, `epistolary`, `scientific_explainer`, `fairy_tale`,
  `noir_detective`. Dataset `desh2806/simplestories-personas-10k` has ~10k stories each
  (50 472 total), uniform → training proportions **σ = 0.2 each**.
- **Models** (all `desh2806/simplestories-persona-<name>` + `-mixture`, base
  `SimpleStories/SimpleStories-V2-5M`): identical `LlamaForCausalLM`, **vocab 4019**,
  hidden 256, 6 layers, fp32. The **tokenizer is byte-identical across base / mixture /
  all specialists** (checked `get_vocab()` equality) — the shared-tokenizer invariant holds.
- Dataset rows are **completions only** (`story` field); no prompt/completion split, as the
  paper's "Is the problem with prompting?" section states. Extra metadata columns exist and
  may be useful later: `theme, topic, feature, initial_letter, initial_word_type, num_paragraphs`.

## ⚠️ Tokenizer / config BOS-EOS mismatch (the big trap — do NOT trust config.json)

`config.json` declares `bos_token_id=1, eos_token_id=2, pad_token_id=0`. The **actual
tokenizer disagrees** and is the source of truth:

- Only two specials exist: **`[UNK]`/pad = id 0**, **`[EOS]` = id 1**. `tok.bos_token is None`
  (there is **no BOS**). `config.eos_token_id=2` points at an ordinary content token — wrong.
- READMEs confirm: *"The base model has no BOS; seed generation with EOS (id=1) to start a
  new story."* Training used `add_special_tokens=False`, EOS appended per story, truncated 512.

**Consequences encoded in code (keep them explicit / fail-loud):**
- Use **EOS_ID = 1, PAD_ID = 0** taken from the *tokenizer*, never from `model.config`.
  `config.py` asserts `tokenizer.eos_token_id == 1` and `pad_token_id == 0` at load.
- **Resolves design_doc §6 "neutral start":** the Free-regime neutral start is a single
  `[EOS]` (id 1), not BOS. Generation must pass `eos_token_id=1, pad_token_id=1` (or 0) by
  hand, not rely on `generation_config.json`.
- `token_logprobs` padding mask keys off PAD_ID=0, and position 0 of every sequence has no
  predictive context (→ nan).

## Tokenizer behaviour relevant to triggers (Exp 1)

- Lowercase WordPiece-style with `##` continuations. Leading spaces / casing are normalized
  away: `'Dear'`, `' Dear'`, `'Dear '` all → `[1638]` ('dear'), a **single token**. Good news
  for token-level triggers (epistolary's "Dear" is one rare token).
- But some persona markers are **multi-token**: `'Yours,' → [3933, 13]`,
  `'The reason is that' → [85,186,916,66,242,149]`. design_doc §6 trigger-granularity question
  is real: epistolary's *opening* trigger is single-token, but sign-offs are phrases. Revisit
  when building Exp 1 (token-level first; extend to n-grams only if needed).

## Published convergence numbers (cross-check for Exp 0 step 1)

Specialist READMEs give mean CE/token: e.g. epistolary `val_own=1.7163` vs `val_mix=3.3357`;
mixture `val_mix=1.7961`. Useful sanity anchor for the convergence assertion, but Exp 0
recomputes log-probs directly rather than trusting these.

## Exp 1 findings (PMI triggers + taxonomy) — verified run 2026-06-11

- **Trigger selector.** Raw PMI saturates at log(k)=log(5)≈1.61 for any persona-EXCLUSIVE
  token, so PMI alone cannot rank among exclusive tokens (everything ties at the ceiling).
  The trigger is selected by `trigger_score = p(v | i, t) · max(PMI, 0)` (pmi.trigger_score):
  distinctive AND reliably emitted. This is the anchor handed to Exp 2; design_doc's `C_i`
  (= max weighted-PMI / sum, pmi.concentration) is kept separately as the taxonomy metric.
- **Taxonomy is clean and matches the framing.** Both metrics give a ~5x gap:
  triggered {epistolary 1.61, fairy_tale 1.59, noir 1.53} vs behavioural {scientific 0.27,
  absurdist 0.18} (max_trigger_score); C_i {epi .031, fairy .028, noir .028} vs {sci .0066,
  absurdist .0037}. Threshold logged = midpoint(epistolary, scientific) ≈ 0.019. So 3
  triggered / 2 behavioural — noir is a (soft) triggered persona via its atmospheric opening.
- **Triggers found:** epistolary `dear`@0 (in 100% of its stories), fairy_tale `once upon a
  time` (`upon`@1 edges `once`@0 by a hair on score), noir `the`@0 → `night`@1 → `rain`@5.
  Behavioural personas top out on stopwords (`a`, `,`) — correctly reflecting no entry trigger.
- ⚠️ **For Exp 2 anchoring:** the headline trigger is `argmax trigger_score`, which for
  fairy_tale picks `upon`@1 over the more natural entry token `once`@0. When anchoring, consider
  preferring the position-0 entry token (or the whole opening phrase) rather than the bare
  argmax token — revisit when building anchored_generate. Also re-examine multi-token triggers
  (epistolary sign-off `yours,` = 2 tokens; "once upon a time" is a phrase).

## Compute / environment gotchas (Exp 2, 2026-06-11)

- **GPU driver wedged mid-session.** After Exp 0 ran fine on CUDA, the node's NVIDIA driver
  hung: every CUDA call (incl. plain `nvidia-smi` and `torch.cuda.is_available()`) blocks in
  uninterruptible `D` state and is unkillable (`kill -9` no-op). Root cause is infra, not code.
  Pure `import torch` + CPU tensor ops still work — only driver calls hang.
- **Run on CPU when the GPU is wedged.** Models are 5M params, so CPU is fine. `import torch`
  is safe; the trap is `torch.cuda.is_available()` itself hanging. `models.resolve_device()`
  returns 'cpu' when `CUDA_VISIBLE_DEVICES=""` WITHOUT probing the driver. Launch with:
  `CUDA_VISIBLE_DEVICES="" PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_expN`.
- Always run with `python -u` (or PYTHONUNBUFFERED): stdout block-buffers to log files and
  hides all progress until exit otherwise.

## Generation conventions (generate.py) — keep these explicit

- Seed every generation with the neutral `[EOS]` (id 1); pass an explicit **all-ones
  attention_mask** so the seed is ATTENDED. Without it, generate infers mask = (ids != pad);
  if pad==EOS the seed is masked out, diverging from the training regime (EOS = attended
  separator). README's pad==eos pattern technically masks the seed — we don't copy it.
- Use `pad_token_id = PAD_ID (0)`, distinct from EOS (1), so post-EOS padding is unambiguous
  and no spurious "right-padding detected" warning fires. `eos_token_id=1` still stops gen.
- Smoke-tested: free gens are coherent and drift whimsical/absurdist with NO epistolary/noir
  openings (previews H1); anchoring 'dear' yields "dear mia, i hope you are well…" (previews H3).

## Exp 2 results (free vs anchored) — verified CPU run 2026-06-11

Headline: H1, H2, H3 all confirmed; closes Exp 0 step 3.
- **Free fails (H1/H2).** Triggers fire ~0% in free gen (epistolary 'dear' 0% @pos vs 100% in
  data; fairy 'upon' 0%; noir 'the' 7% @pos vs 96%). EM on free gens L1=0.75 from σ: behavioural
  OVER-weighted (absurdist .42, sci .36), triggered CRUSHED (epistolary .006, fairy .075,
  noir .14). 141/1000 free gens best-explained by BASE; epistolary wins only 4/1000.
- **Anchoring recovers (H3).** L1 to σ: free .75 → entry .11, phrase .12, argmax .41. Entry
  variant per-persona: epistolary .006→.194, fairy .075→.151, noir .14→.20 — triggered personas
  restored to ~σ. Hard-assign: 'dear'→191/200 epistolary, 'the'→170/200 noir.
- **Anchor-form contrast (the requested both-and-compare):** entry (pos-0 token: dear/once/the)
  ≈ phrase (full opening) ≫ argmax. argmax is WORSE because (a) fairy argmax='upon' is
  mid-phrase, a poor entry anchor (fairy recovers to only .05, worse than free!), and (b)
  behavioural argmax='a' is identical to absurdist's, so it does nothing. phrase ("once upon a
  time, in", "the night was") helps the richest triggers most (fairy →147/200). Behavioural
  personas have an EMPTY phrase → phrase regime degenerates to free for them (expected).
- Takeaway: loss is at GENERATION ENTRY, not representation. The pos-0 entry token is the right
  anchor; full phrase helps strong triggers; bare argmax is unreliable. Exp 3 should localise
  whether triggered-persona failure is at t=1 (entry) or compounds over t.

## Exp 3 results (commitment dynamics) — verified CPU run 2026-06-11

Localises WHERE triggered personas die. Components = 5 personas + base; priors uniform &
free-empirical (Exp 2 hard-assign). Since training σ is uniform, π=σ ≡ uniform (noted in code).
- **Free = compounding, from entry failure.** Triggered personas are NOT dead at t=1 (epistolary
  γ_t1=0.26) but the cumulative posterior erodes as the persona is never entered: epistolary
  γ 0.26→0.015, fairy/noir flat-low ~0.11; behavioural CLIMB (absurdist 0.18→0.35, sci 0.12→0.28).
  Per-token r drifts the same way. So: trigger never fires at t=1 → evidence compounds away.
- **Anchored = lasting commitment (no decay to base).** epistolary 'dear' commits INSTANTLY
  (γ_t1=0.94) and persists (γ_late=0.99, base≈0.006); noir 'the' BUILDS (0.015→0.825); fairy
  'once'/sci 'in' only partial from one token (→0.43) — consistent with Exp 2 phrase>entry for
  fairy. Confirms "trigger sets a lasting mode," not "trigger needed continuously."
- Implementation: gamma/r in commitment.py; anchored uses entry (single-token) variant so all
  personas align at position 1. RuntimeWarnings from all-nan tails are suppressed (harmless).

## Exp 4 results (token-distribution weights + individual rollouts) — GPU run 2026-06-20

- **GPU is healthy again** (RTX 4090, driver 565.57.01) — the 2026-06-11 wedge cleared, so Exp 4
  ran on CUDA. Exp 2/3 numbers were CPU and are device-independent (deterministic given seed), so
  no need to re-run them; only re-run on GPU if convenient.
- **`w_hat(1)` is weakly identified — the big gotcha.** At t=1 the prefix is the bare `[EOS]`, so
  the ideal posterior is uniform and σ is the right benchmark there. But the KL objective is nearly
  flat: `KL(P_mix‖uniform-avg)=0.619` vs `KL(P_mix‖Σw_iP_i)=0.561` (EM buys only ~9%). The
  specialists' first-token dists are collinear (all peaked on `in`/`a`/`under`), so scientific
  explains away fairy (`w=0.017` despite matching top tokens). EM is concave → global optimum, so
  this is genuine non-identifiability, not a local min. **Report the ordering, not the magnitudes.**
  Encoded in `_write_w1_diagnostic` docstring + `w1_diagnostic.csv`; don't re-derive.
- **Robust findings:** (1) the mixture's first-token mass is all generic openers — **no trigger
  token** (`dear`/`once`/`the night` ≈0), so triggered openings are gone already at t=1; the high
  `w_hat(1)` for epistolary (0.141) is a high-entropy diffuse-coverage artifact, NOT letter-entry
  (same effect as Exp 3 γ_t1=0.26). (2) The `w_hat(t)` curve is a t=1 scientific-spike outlier then
  reverts from t=2 to the `pi_free` shape (absurdist ~0.4, epistolary ~0.04) at *every* position —
  the collapse is per-token, not just cumulative. (3) Individual anchored rollouts cleanly split
  triggered (`dear`/`the`: instant+permanent γ→1) vs behavioural/phrase (`once`/`in`/`a`: slow,
  hijacked — `in` rollout is taken over by fairy_tale until ~t=53).

## Exp 5 findings (prefix-embedding trajectories) — GPU run 2026-06-20

- **Pooling, not normalisation, is what matters (the big gotcha).** Measured 2D-PCA persona
  separation (η² = between/(between+within)) over 150 stories/persona embedded by `P_mix`:
  *last-token* embedding η²=0.036 (a single blob, PC1+PC2=11% var) vs *mean-pool* over positions
  η²=0.91 (PC1+PC2=31%). The last token of a 128-token story encodes recent **content**, not
  persona; the mean averages content out and keeps **style**. L2-normalise vs raw vs z-score is
  marginal (0.909/0.913/0.903). Encoded in `embed_traj.sequence_embeddings` docstring. So: clusters
  = **mean-pool**; trajectory = **running mean** `ē(t)` (a single-position path is uninformative —
  the smoothing is load-bearing). The first run used last-token and produced an unreadable blob.
- **Result maps cleanly onto Exp 4B taxonomy.** Anchored `dear`→epistolary cluster and `the`→noir
  cluster (clean single-token triggers, fast + sticky); `once`→stalls near base (needs phrase);
  `in`→**hijacked into fairy_tale** cluster; `a`→wanders centre. LLR companion shows the same as
  *speed*: triggered LLR climbs steeply and pulls away; behavioural/hijacked bunch and separate
  slowly. Base-dominant free rollout stays in the central base region (free entry failure, geometric).
- **Caveats:** PC1+PC2 only 31% var → 2D distances illustrative not metric. Free panels are
  argmax-per-component extremes (reaching a cluster partly by construction, same caveat as Exp 4B).
- GPU healthy (RTX 4090); no sklearn on the node, so PCA is a numpy-SVD in `embed_traj.fit_pca`.

## Scope notes

- The data + EM pipeline was said to "exist elsewhere"; in practice **no external EM code was
  provided** — `src/em.py` is a fresh port from the paper's E/M equations (design_doc §
  correction: responsibility denominator must sum over each persona's *own* model).
- Exp 0 step 3 (failure repro on free generations) depends on the Exp 2 free generator, so it
  is built with Exp 2, not roadmap item 1. Roadmap item 1 covers convergence + EM-validity.
