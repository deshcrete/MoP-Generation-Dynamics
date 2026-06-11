"""Experiment 2 — Free vs anchored generation and the measurements on top of it.

Free regime: sample from P_mix starting from the neutral [EOS] anchor (no BOS exists; see
control/notes.md). Anchored regime: prepend a persona's trigger (one of the entry/argmax/phrase
variants from Exp 1) after the [EOS] seed and autoregress.

Scoring convention for the anchored regime: the forced anchor tokens are EXCLUDED from all
log-prob scoring (EM, hard-assignment). We only score the model's own generated continuation,
otherwise the prepended trigger would mechanically inflate the score of the persona it belongs
to. `generation_attention_mask(..., start=...)` implements this masking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from . import config, models
from .config import GenConfig


def _pad_to(width: int, batch: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Right-pad a [B, L] batch to [B, width] with pad_id (L <= width)."""
    if batch.shape[1] == width:
        return batch
    pad = batch.new_full((batch.shape[0], width - batch.shape[1]), pad_id)
    return torch.cat([batch, pad], dim=1)


@torch.no_grad()
def _generate_from_seed(model: PreTrainedModel, seed: torch.LongTensor, n: int,
                        cfg: GenConfig, device: str) -> torch.LongTensor:
    """Generate n sequences from a fixed [seed_len] prompt; return [n, seed_len+max_new_tokens].

    EOS/pad ids are passed explicitly (=EOS_ID): model.config mislabels eos as id 2, so relying
    on the config would never stop on the real EOS (id 1). Output is right-padded to a fixed
    width so batches concatenate.
    """
    torch.manual_seed(cfg.seed)
    width = seed.shape[0] + cfg.max_new_tokens
    out = []
    for start in range(0, n, cfg.batch_size):
        b = min(cfg.batch_size, n - start)
        prompt = seed.unsqueeze(0).expand(b, -1).to(device)
        # Pass an explicit all-ones attention_mask so the [EOS] seed is ATTENDED. Without it,
        # generate infers mask = (ids != pad_token_id); since the seed EOS equals pad_token_id
        # it would be masked out — but the model was trained with EOS as a real separator token,
        # so it must be attended for generation to match the training regime.
        attn = torch.ones_like(prompt)
        # pad_token_id = PAD_ID (0), distinct from the seed/terminal EOS (1): post-EOS padding
        # is unambiguous and the seed is not misread as padding. Generation still stops on EOS.
        gen = model.generate(
            prompt, attention_mask=attn, do_sample=True, temperature=cfg.temperature,
            top_p=cfg.top_p, max_new_tokens=cfg.max_new_tokens, eos_token_id=config.EOS_ID,
            pad_token_id=config.PAD_ID,
        )
        out.append(_pad_to(width, gen.cpu(), config.PAD_ID))
    return torch.cat(out, dim=0)


def free_generate(mix_model, tokenizer: PreTrainedTokenizerBase, cfg: GenConfig,
                  device: str) -> torch.LongTensor:
    """N free samples from P_mix starting at the neutral [EOS] seed -> [N, 1+max_new_tokens]."""
    seed = torch.tensor([config.EOS_ID], dtype=torch.long)
    return _generate_from_seed(mix_model, seed, cfg.n_samples, cfg, device)


def anchored_generate(mix_model, tokenizer: PreTrainedTokenizerBase,
                      anchors: dict[str, list[int]], cfg: GenConfig,
                      device: str) -> dict[str, torch.LongTensor]:
    """Anchored samples per persona: seed = [EOS] + anchor_token_ids, N/k samples each.

    Returns {persona: [n_per, 1+len(anchor)+max_new_tokens]}. An empty anchor (no stable
    phrase, e.g. behavioural personas in the 'phrase' variant) degenerates to the free seed;
    that is intentional and recorded by the caller.
    """
    n_per = cfg.n_samples // len(config.PERSONAS)
    out: dict[str, torch.LongTensor] = {}
    for persona in config.PERSONAS:
        seed = torch.tensor([config.EOS_ID] + list(anchors[persona]), dtype=torch.long)
        out[persona] = _generate_from_seed(mix_model, seed, n_per, cfg, device)
    return out


def generation_attention_mask(samples: torch.LongTensor, start: int = 1) -> torch.LongTensor:
    """Attention mask over GENERATED content: valid for positions [start .. terminal EOS]
    inclusive, 0 elsewhere. `start` skips the forced prefix ([EOS] seed + any anchor tokens),
    so only the model's own continuation is scored.

    The terminal EOS is the first EOS at position >= start; if none, the row runs to the end.
    """
    n, t = samples.shape
    attn = torch.zeros((n, t), dtype=torch.long)
    for i in range(n):
        row = samples[i]
        eos_pos = (row[start:] == config.EOS_ID).nonzero()
        end = start + int(eos_pos[0]) if len(eos_pos) else t - 1  # inclusive index
        attn[i, start:end + 1] = 1
    return attn


def trigger_firing_rate(free_samples: torch.LongTensor, triggers: dict[str, dict],
                        by_persona_tokens: dict[str, tuple], content_offset: int = 1) -> pd.DataFrame:
    """Per persona: rate its trigger token fires in FREE generation vs its own dataset D_i.

    Reports, at the trigger's characteristic content position and anywhere in the sequence,
    the fraction of sequences containing the trigger token — for free generations and for D_i.
    `content_offset` aligns Exp 1 content positions (pos 0 = first story token) to free samples
    (index 0 = the [EOS] seed), so content position t -> sample index t + content_offset.
    """
    free = free_samples.numpy()
    rows = []
    for persona in config.PERSONAS:
        v = triggers[persona]["token_id"]
        t = triggers[persona]["pos"]
        d_ids, d_attn = by_persona_tokens[persona]
        d_ids = d_ids.numpy()

        # dataset rates (D_i tokenised WITHOUT leading EOS, so position t is direct)
        ds_at_pos = float((d_ids[:, t] == v).mean()) if t < d_ids.shape[1] else 0.0
        ds_anywhere = float((d_ids == v).any(axis=1).mean())

        # free rates (shift by content_offset for the leading EOS seed)
        idx = t + content_offset
        fr_at_pos = float((free[:, idx] == v).mean()) if idx < free.shape[1] else 0.0
        fr_anywhere = float((free == v).any(axis=1).mean())

        rows.append({
            "persona": persona, "trigger": triggers[persona]["token"], "pos": t,
            "dataset_rate_at_pos": ds_at_pos, "free_rate_at_pos": fr_at_pos,
            "dataset_rate_anywhere": ds_anywhere, "free_rate_anywhere": fr_anywhere,
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def hard_assign(samples: torch.LongTensor, attn: torch.LongTensor,
                persona_models: dict[str, PreTrainedModel], base_model: PreTrainedModel,
                device: str, batch_size: int = 250) -> np.ndarray:
    """Label each sequence by argmax sequence-log-prob over {specialists + base}.

    Returns an int array of labels indexing config.PERSONAS + ['base'] (base = index k).
    H1 prediction: many FREE generations are best explained by the base model.
    """
    names = config.PERSONAS + ["base"]
    all_models = [persona_models[p] for p in config.PERSONAS] + [base_model]
    n = samples.shape[0]
    scores = torch.empty((n, len(names)), dtype=torch.float32)
    for s in range(0, n, batch_size):
        ids = samples[s:s + batch_size].to(device)
        a = attn[s:s + batch_size].to(device)
        for j, m in enumerate(all_models):
            scores[s:s + batch_size, j] = models.sequence_logprob(m, ids, a).cpu()
    return scores.argmax(dim=1).numpy()
