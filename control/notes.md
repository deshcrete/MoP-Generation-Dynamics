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

## Scope notes

- The data + EM pipeline was said to "exist elsewhere"; in practice **no external EM code was
  provided** — `src/em.py` is a fresh port from the paper's E/M equations (design_doc §
  correction: responsibility denominator must sum over each persona's *own* model).
- Exp 0 step 3 (failure repro on free generations) depends on the Exp 2 free generator, so it
  is built with Exp 2, not roadmap item 1. Roadmap item 1 covers convergence + EM-validity.
