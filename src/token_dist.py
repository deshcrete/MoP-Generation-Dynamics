"""Experiment 4 — Mixture weights from token DISTRIBUTIONS (the w(t) inference).

Exp 0/2 infer mixture weights from whole-sequence log-probs of SAMPLED sequences. This module
infers them from the model's full next-token distribution at a position, by solving

    w_hat(t) = argmin_{w in simplex} KL( P_mix(. | x_{<t})  ||  sum_i w_i P_i(. | x_{<t}) ).

Maximising sum_v P_mix(v) log sum_i w_i P_i(v) is exactly the EM of src/em.py with the vocabulary
tokens v as the "samples", log P_i(v) = log q_i(v) as the per-sample log-probs, and weights
P_mix(v). So we reuse the validated `em_mixture_weights` (with its new `weights` arg) rather than
writing a second estimator. The only new machinery here is (a) pulling full next-token log-dists
out of the models and (b) packing them into the [N, k] + weights shape EM wants.

Two regimes, mirroring the design doc:
  - single prefix  -> w_hat at one position (e.g. t=1 from the neutral [EOS] seed).
  - many prefixes  -> w_hat(t) aggregated over a population of prefixes (the Exp 2 free
    generations), one weight vector per token position t.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedModel

from . import config, generate, models
from .config import EMConfig
from .em import em_mixture_weights


def forward_attention_mask(samples: torch.LongTensor) -> torch.LongTensor:
    """All-ones mask over the REAL tokens of each row (the [EOS] seed through the terminal EOS),
    0 over the post-EOS padding tail. Needed for the forward pass that produces next-token
    distributions: the seed EOS must be ATTENDED (it is a real separator the model generates in),
    unlike `generate.generation_attention_mask(start=1)` which drops position 0 for scoring."""
    fm = generate.generation_attention_mask(samples, start=1).clone()  # valid [1 .. terminal EOS]
    fm[:, 0] = 1                                                        # also attend the seed EOS
    return fm


@torch.no_grad()
def collect_next_token_logdists(model: PreTrainedModel, samples: torch.LongTensor,
                                attn: torch.LongTensor, t_keep: int, device: str,
                                batch_size: int = 64) -> np.ndarray:
    """[N, t_keep, V] float32 array of log P(x_{j+1} | x_{0..j}) for positions j in [0, t_keep).

    Batched and truncated to `t_keep` positions to bound memory ([N, T, V] is large). float32 is
    enough for the distributions; EM upcasts to float64 internally.
    """
    n = samples.shape[0]
    v = config.VOCAB_SIZE
    out = np.empty((n, t_keep, v), dtype=np.float32)
    for s in range(0, n, batch_size):
        ids = samples[s:s + batch_size].to(device)
        a = attn[s:s + batch_size].to(device)
        ld = models.next_token_logdist(model, ids, a)[:, :t_keep, :]   # [b, t_keep, V]
        out[s:s + batch_size] = ld.cpu().numpy()
    return out


def solve_kl_weights(p_target: np.ndarray, q_logdists: np.ndarray, cfg: EMConfig,
                     pi_init: np.ndarray | None = None) -> dict:
    """Single-prefix w_hat = argmin_w KL(p_target || sum_i w_i q_i).

    p_target [V]: the mixture model's next-token PROBABILITY distribution at the prefix.
    q_logdists [k, V]: each specialist's next-token LOG-distribution at the same prefix.
    Returns the em_mixture_weights dict (pi = w_hat over config.PERSONAS order).
    """
    k, v = q_logdists.shape
    assert p_target.shape == (v,), f"p_target {p_target.shape} != ({v},)"
    logmat = q_logdists.T.astype(np.float64)                 # [V, k] : samples=v, components=i
    weights = p_target.astype(np.float64)                    # weight each token by P_mix(v)
    return em_mixture_weights(logmat, cfg, pi_init=pi_init, weights=weights)


def solve_kl_weights_multi(p_targets: np.ndarray, q_logdists: np.ndarray, cfg: EMConfig,
                           pi_init: np.ndarray | None = None) -> dict:
    """Aggregated w_hat over a POPULATION of prefixes (one weight vector for the whole batch):
        argmin_w (1/B) sum_b KL( p_b || sum_i w_i q_{i,b} ).
    p_targets [B, V] probability dists, q_logdists [k, B, V] log-dists. Each (prefix b, token v)
    pair becomes one weighted EM sample with weight p_b(v)/B, stacked into [B*V, k]."""
    k, b, v = q_logdists.shape
    assert p_targets.shape == (b, v), f"p_targets {p_targets.shape} != ({b}, {v})"
    logmat = np.transpose(q_logdists, (1, 2, 0)).reshape(b * v, k).astype(np.float64)  # [B*V, k]
    weights = (p_targets / b).reshape(b * v).astype(np.float64)                        # [B*V]
    return em_mixture_weights(logmat, cfg, pi_init=pi_init, weights=weights)


def position_weight_curve(mix_logd: np.ndarray, spec_logd: np.ndarray, valid: np.ndarray,
                          cfg: EMConfig, min_prefixes: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """w_hat(t) for each token position t, aggregated over all prefixes valid at t.

    mix_logd [N, T, V]:    mixture next-token log-dists (position j conditions on x_{<=j}).
    spec_logd [k, N, T, V]: specialist next-token log-dists, same layout.
    valid [N, T] bool:     valid[n, t] True iff token x_t is a real generated token of seq n
                           (so its prefix x_{<t} is real). Predicting x_t uses the slice at j=t-1.
    Returns (w [k, T], n_prefixes [T]); positions with < min_prefixes valid sequences are nan.
    """
    k, n, t, v = spec_logd.shape
    w = np.full((k, t), np.nan)
    counts = np.zeros(t, dtype=int)
    for tok_pos in range(1, t):                              # x_t predicted from prefix slice t-1
        sel = valid[:, tok_pos]                              # prefixes that reached position t
        counts[tok_pos] = int(sel.sum())
        if counts[tok_pos] < min_prefixes:
            continue
        p = np.exp(mix_logd[sel, tok_pos - 1, :])            # [B_t, V] P_mix(.|x_{<t})
        q = spec_logd[:, sel, tok_pos - 1, :]                # [k, B_t, V] log q_i(.|x_{<t})
        res = solve_kl_weights_multi(p, q, cfg)
        w[:, tok_pos] = res["pi"]
    return w, counts
