"""Mixture-MLE via EM (ported from paper.md §Setup).

Estimates mixture weights pi that best explain an inference set S under the fixed specialist
models {P_i}, by maximising  (1/|S|) sum_x log sum_i pi_i P_i(x). Equivalent to minimising
KL(P_mix || sum_i pi_i P_i). The only model-dependent input is the [N, k] matrix of
per-sequence log-probs log P_i(x); EM itself is pure numpy and never touches a model.

E-step (responsibility of persona i for sequence x):
    gamma_i(x) = pi_i P_i(x) / sum_j pi_j P_j(x)
M-step:
    pi_i <- (1/|S|) sum_x gamma_i(x)

All arithmetic is in log space with logsumexp for numerical stability (sequence log-probs are
large-magnitude sums over ~100 tokens, so naive exponentiation underflows).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .config import EMConfig


def em_mixture_weights(seq_logprobs: np.ndarray, cfg: EMConfig,
                       pi_init: np.ndarray | None = None) -> dict:
    """Run EM to convergence on a [N, k] array of per-sequence log P_i(x).

    Returns dict with: pi (final [k]), pi_history ([iters+1, k]), n_iters, converged (bool),
    avg_loglik (final mean-log-likelihood objective). Fails loudly on shape / nan issues
    rather than silently dropping rows.
    """
    assert seq_logprobs.ndim == 2, f"expected [N, k], got {seq_logprobs.shape}"
    n, k = seq_logprobs.shape
    assert n > 0 and k > 0, f"empty inference set or persona set: {seq_logprobs.shape}"
    assert np.isfinite(seq_logprobs).all(), "non-finite log-probs in EM input (fix upstream)"

    if pi_init is None:
        if cfg.init == "uniform":
            pi = np.full(k, 1.0 / k)
        elif cfg.init == "sigma":
            raise ValueError("init='sigma' requires pi_init to be passed explicitly")
        else:
            raise ValueError(f"unknown EMConfig.init={cfg.init!r}")
    else:
        pi = np.asarray(pi_init, dtype=float).copy()
    assert pi.shape == (k,) and np.isclose(pi.sum(), 1.0), f"bad pi_init {pi}"

    history = [pi.copy()]
    converged = False
    for _ in range(cfg.max_iters):
        # E-step in log space: log(pi_i P_i(x)) = log pi_i + log P_i(x)
        log_weighted = np.log(pi)[None, :] + seq_logprobs           # [N, k]
        log_norm = logsumexp(log_weighted, axis=1, keepdims=True)   # [N, 1]
        gamma = np.exp(log_weighted - log_norm)                     # [N, k], rows sum to 1

        # M-step
        pi_new = gamma.mean(axis=0)                                 # [k]
        history.append(pi_new.copy())

        if np.max(np.abs(pi_new - pi)) < cfg.tol:
            pi = pi_new
            converged = True
            break
        pi = pi_new

    # Final objective: mean over x of log sum_i pi_i P_i(x)
    log_weighted = np.log(pi)[None, :] + seq_logprobs
    avg_loglik = float(logsumexp(log_weighted, axis=1).mean())

    return {
        "pi": pi,
        "pi_history": np.array(history),
        "n_iters": len(history) - 1,
        "converged": converged,
        "avg_loglik": avg_loglik,
    }
