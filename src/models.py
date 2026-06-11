"""Model + tokenizer loading and the shared log-prob primitives.

This is the ONLY place models are touched directly; every experiment composes
`token_logprobs` / `sequence_logprob`. All loads assert the fail-loud invariants from
design_doc.md §2: one shared tokenizer, one vocab size, across base/mixture/specialists.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from . import config


def load_tokenizer() -> PreTrainedTokenizerBase:
    """Load the shared tokenizer and assert the special-token ids match config.py.

    These checkpoints ship a config.json that mislabels eos/bos (see control/notes.md);
    the tokenizer is the source of truth, so we pin EOS_ID/PAD_ID here and fail loudly if a
    future artifact ever disagrees.
    """
    tok = AutoTokenizer.from_pretrained(config.MIXTURE_REPO)
    assert tok.eos_token_id == config.EOS_ID, (
        f"tokenizer eos_token_id={tok.eos_token_id} != config.EOS_ID={config.EOS_ID}"
    )
    assert tok.pad_token_id == config.PAD_ID, (
        f"tokenizer pad_token_id={tok.pad_token_id} != config.PAD_ID={config.PAD_ID}"
    )
    assert tok.bos_token_id is None, (
        f"expected no BOS token, got bos_token_id={tok.bos_token_id}"
    )
    assert len(tok) == config.VOCAB_SIZE, f"tokenizer size {len(tok)} != {config.VOCAB_SIZE}"
    return tok


def _load_one(repo: str, device: str, dtype: torch.dtype) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype)
    assert model.config.vocab_size == config.VOCAB_SIZE, (
        f"{repo}: vocab {model.config.vocab_size} != {config.VOCAB_SIZE}"
    )
    return model.to(device).eval()


def load_base_model(device: str, dtype: torch.dtype = torch.float32) -> PreTrainedModel:
    return _load_one(config.BASE_REPO, device, dtype)


def load_mixture_model(device: str, dtype: torch.dtype = torch.float32) -> PreTrainedModel:
    return _load_one(config.MIXTURE_REPO, device, dtype)


def load_persona_models(device: str, dtype: torch.dtype = torch.float32) -> dict[str, PreTrainedModel]:
    """Load all k specialist models, keyed by persona name (config.PERSONAS order)."""
    return {p: _load_one(repo, device, dtype) for p, repo in config.PERSONA_REPOS.items()}


@torch.no_grad()
def token_logprobs(model: PreTrainedModel, input_ids: torch.LongTensor,
                   attn: torch.LongTensor) -> torch.FloatTensor:
    """Return [B, T] log P(x_t | x_{<t}), with position t aligned to token x_t.

    Position 0 has no predictive context and any position masked out by `attn` (padding) is
    set to nan; downstream code MUST ignore nans (e.g. nansum) rather than treat them as 0.
    Failing to mask would silently fold padding into the likelihood — exactly the kind of
    silent fallback AGENT.md forbids.
    """
    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, T, V]
    logp = F.log_softmax(logits.float(), dim=-1)                      # [B, T, V]

    # logp[:, t-1] predicts token at position t -> gather x_t from the shifted distribution.
    target = input_ids[:, 1:].unsqueeze(-1)                           # [B, T-1, 1]
    tok_lp = logp[:, :-1].gather(-1, target).squeeze(-1)             # [B, T-1]

    out = input_ids.new_full(input_ids.shape, 0, dtype=torch.float32)
    out.fill_(float("nan"))
    out[:, 1:] = tok_lp
    # nan out padding positions (token x_t itself is padding). attn==0 marks padding tokens.
    out[attn == 0] = float("nan")
    # position 0 is always nan (no context); also nan if its token is padding (already handled).
    out[:, 0] = float("nan")
    return out


@torch.no_grad()
def sequence_logprob(model: PreTrainedModel, input_ids: torch.LongTensor,
                     attn: torch.LongTensor) -> torch.FloatTensor:
    """Return [B] total log P(sequence) = sum over valid token positions of token_logprobs."""
    tlp = token_logprobs(model, input_ids, attn)
    return torch.nansum(tlp, dim=1)


@torch.no_grad()
def score_sequences(persona_models: dict[str, PreTrainedModel], input_ids: torch.LongTensor,
                    attn: torch.LongTensor, device: str, batch_size: int = 256) -> "torch.FloatTensor":
    """Per-sequence log P_i(x) under every specialist -> [N, k] tensor (config.PERSONAS order).

    This [N, k] matrix is exactly the input EM consumes (src/em.py). Batched over sequences to
    bound memory; models are tiny (5M) so this is fast even over the full inference set.
    """
    names = list(persona_models.keys())
    n = input_ids.shape[0]
    out = torch.empty((n, len(names)), dtype=torch.float32)
    for start in range(0, n, batch_size):
        ids = input_ids[start:start + batch_size].to(device)
        a = attn[start:start + batch_size].to(device)
        for j, name in enumerate(names):
            out[start:start + batch_size, j] = sequence_logprob(persona_models[name], ids, a).cpu()
    return out
