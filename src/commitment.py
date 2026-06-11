"""Experiment 3 — Persona commitment dynamics.

Tracks how the mixture model's per-persona assignment evolves ALONG a generated sequence, to
localise where triggered personas are lost: at the first token (entry failure) or compounding
over the sequence.

Component set. The design-doc quantities are written over the k specialists; we EXTEND the
component set with the base model B as a (k+1)th component, because the question "does it drift
/ decay back to base?" requires base to be representable. Components are therefore
config.PERSONAS + ['base'] throughout; index k is base.

Two quantities per sequence (then averaged over sequences):
  - Cumulative posterior  gamma_i(t) = softmax_i( log pi_i + sum_{s<=t} logp_i[s] )
    monotone evidence accumulation — which persona explains the whole prefix.
  - Per-token responsibility r_i(t) = softmax_i( log pi_i + logp_i[t] )
    which persona explains the token at t given the context.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from transformers import PreTrainedModel

from . import config, models

COMPONENTS = config.PERSONAS + ["base"]


@torch.no_grad()
def per_model_token_logprobs(persona_models: dict[str, PreTrainedModel],
                             base_model: PreTrainedModel, samples: torch.LongTensor,
                             attn: torch.LongTensor, device: str,
                             batch_size: int = 250) -> np.ndarray:
    """[n_seq, k+1, T] array of logp_j[t] = log P_j(x_t | x_{<t}) for each component j.

    Invalid positions (sequence position 0, and padding) are nan and propagate through the
    softmaxes below, which ignore them. Component order = COMPONENTS (base last).
    """
    all_models = [persona_models[p] for p in config.PERSONAS] + [base_model]
    n, t = samples.shape
    out = np.full((n, len(all_models), t), np.nan, dtype=np.float64)
    for s in range(0, n, batch_size):
        ids = samples[s:s + batch_size].to(device)
        a = attn[s:s + batch_size].to(device)
        for j, m in enumerate(all_models):
            out[s:s + batch_size, j, :] = models.token_logprobs(m, ids, a).cpu().numpy()
    return out


def _softmax_lastvalid(logits: np.ndarray) -> np.ndarray:
    """Softmax over axis=1 (components), nan-safe: positions where ALL components are nan stay
    nan; otherwise standard stable softmax. logits is [n_seq, k+1, T]."""
    out = np.full_like(logits, np.nan)
    valid = ~np.isnan(logits).all(axis=1)              # [n_seq, T]: at least one component valid
    # For valid (seq,t), all components are valid together (same token position), so a plain
    # stable softmax over axis=1 is correct there. All-nan columns (padding tails) are expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx = np.nanmax(logits, axis=1, keepdims=True)
    ex = np.exp(logits - mx)
    denom = np.nansum(ex, axis=1, keepdims=True)
    sm = ex / denom
    out[:, :, :] = np.where(valid[:, None, :], sm, np.nan)
    return out


def cumulative_posterior(logp: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """gamma[n,i,t] = softmax_i( log pi_i + sum_{s<=t} logp_i[s] ), [n_seq, k+1, T].

    nan log-probs (position 0, padding) contribute 0 to the running sum; positions where the
    whole token is padding remain nan in the output (no valid components there).
    """
    contrib = np.where(np.isnan(logp), 0.0, logp)       # nan -> 0 for the cumulative sum
    cumlogp = np.cumsum(contrib, axis=2)                # [n, k+1, T]
    logits = np.log(pi)[None, :, None] + cumlogp
    gamma = _softmax_lastvalid(logits)
    # blank out positions that had no valid token at all (all-nan in logp)
    blank = np.isnan(logp).all(axis=1)                  # [n, T]
    gamma[np.broadcast_to(blank[:, None, :], gamma.shape)] = np.nan
    return gamma


def token_responsibility(logp: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """r[n,i,t] = softmax_i( log pi_i + logp_i[t] ), [n_seq, k+1, T]. nan where the token is invalid."""
    logits = np.log(pi)[None, :, None] + logp
    return _softmax_lastvalid(logits)


def mean_curves(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Mean and 95% CI over sequences (axis 0), per component, vs t.

    Returns {'mean': [k+1, T], 'lo': [k+1, T], 'hi': [k+1, T], 'n': [k+1, T]} using nan-aware
    statistics, so the per-position sample count shrinks as sequences end.
    """
    with warnings.catch_warnings():                     # all-nan tails (no seq reaches high t)
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0)                  # [k+1, T]
        std = np.nanstd(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)                  # [k+1, T]
    sem = np.divide(std, np.sqrt(n), out=np.zeros_like(std), where=n > 0)
    return {"mean": mean, "lo": mean - 1.96 * sem, "hi": mean + 1.96 * sem, "n": n}


def uniform_prior() -> np.ndarray:
    """Uniform prior over the k+1 components (personas + base). Since training sigma is itself
    uniform over personas, pi=sigma adds nothing beyond this; we log 'uniform'."""
    return np.full(len(COMPONENTS), 1.0 / len(COMPONENTS))


def prior_from_counts(counts: dict[str, float]) -> np.ndarray:
    """Build a normalised prior over COMPONENTS from a dict of component->count (e.g. the Exp 2
    free hard-assignment). This is the apples-to-apples 'what free generation produces' prior."""
    vec = np.array([counts[c] for c in COMPONENTS], dtype=float)
    assert vec.sum() > 0, "empty counts for prior"
    return vec / vec.sum()
