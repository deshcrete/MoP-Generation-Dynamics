"""Experiment 8 — theoretical Naive Bayes persona classifier eta(t) vs generative posterior gamma(t).

The idea (context/bayes_classifier.md): the generative posterior gamma_i(t) of Exp 3/4 already has
the exact form of a naive-Bayes classifier accumulating evidence over a prefix —
    gamma_i(t) = softmax_i( log pi_i + sum_{s<=t} log P_i(x_s | x_{<s}) ),
with the *specialist neural models'* token log-probs as the likelihood. This module builds the
literal textbook twin: a multinomial Naive Bayes classifier fit on the TRAINING DATA, maintaining a
distribution eta over the same components, whose likelihood is the EMPIRICAL per-persona token
frequency rather than a trained model. Overlaying eta(t) on gamma(t) tests the paper's framing that
the mixture model behaves like a Bayesian classifier — do the two belief curves track each other?

Naive-Bayes choices (all deliberately simple, per AGENT.md):
  - Vocabulary = the SHARED tokenizer the models use (config.VOCAB_SIZE). The NB "features" are
    therefore the very same token ids gamma sees; only the likelihood differs. No separate feature
    engineering.
  - Prior = uniform over the components (matches sigma, and commitment.uniform_prior() used by gamma).
  - Likelihood p_i(v) = Laplace-smoothed token frequency over component i's data:
        p_i(v) = (count_i[v] + alpha) / (sum_v count_i[v] + alpha * V).
  - eta accumulates per token exactly as gamma does:
        eta_i(t) = softmax_i( log pi_i + sum_{s<=t} log p_i(x_s) ).
    A position-marginal, count-based analog of gamma (and of Exp 1's marginal PMI normalised into a
    classifier).

The structural [EOS] token (id 1) is excluded from the NB everywhere (it is the seed / terminator,
not persona content). Persona stories D_i are truncated at t_max with no trailing EOS while base
free-generations DO terminate in EOS, so counting EOS would spuriously hand the base class the
terminal token of every rollout; dropping it keeps the components symmetric. Position 0 (the leading
seed [EOS]) is likewise never counted, matching gamma (whose position-0 log-prob is nan).
"""

from __future__ import annotations

import numpy as np
import torch

from . import config
from .commitment import COMPONENTS


def fit_token_loglik(component_ids: dict[str, torch.LongTensor],
                     component_attn: dict[str, torch.LongTensor],
                     vocab_size: int = config.VOCAB_SIZE,
                     alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial NB log-likelihood per component over the shared vocab.

    component_ids/component_attn map every name in COMPONENTS to its tokenised data: [N, T] ids and
    a [N, T] attention mask (1 = real token). Tokens are counted at valid positions only — real
    (attn==1), not position 0 (the leading seed [EOS]), and not the [EOS] id itself (structural).

    Returns (loglik [C, V], counts [C, V]) with loglik[i] = log p_i(.) Laplace-smoothed by `alpha`.
    Component order = COMPONENTS.
    """
    C = len(COMPONENTS)
    counts = np.zeros((C, vocab_size), dtype=np.float64)
    for ci, comp in enumerate(COMPONENTS):
        ids = component_ids[comp].numpy()
        attn = component_attn[comp].numpy().astype(bool)
        attn[:, 0] = False                                   # never count the leading seed [EOS]
        valid = attn & (ids != config.EOS_ID)               # drop the structural [EOS] terminator
        toks = ids[valid]
        assert toks.size > 0, f"{comp}: no tokens to fit NB on"
        counts[ci] = np.bincount(toks, minlength=vocab_size).astype(np.float64)
    smoothed = counts + alpha
    p = smoothed / smoothed.sum(axis=1, keepdims=True)
    return np.log(p), counts


def cumulative_nb_posterior(ids: np.ndarray, out_mask: np.ndarray, loglik: np.ndarray,
                            prior: np.ndarray) -> np.ndarray:
    """eta[n, i, t] = softmax_i( log prior_i + sum_{s<=t, valid} log p_i(x_s) ), [N, C, T].

    ids [N, T] long; out_mask [N, T] bool marks positions where eta is defined (aligned to where
    gamma is valid, so the two overlay). Evidence is accumulated over positions that are BOTH in
    out_mask AND not the structural [EOS] (so the terminal EOS of a rollout, which gamma scores,
    does not perturb eta — see module docstring). Positions outside out_mask are nan.
    """
    C, V = loglik.shape
    contrib_mask = out_mask & (ids != config.EOS_ID)         # [N, T]
    ll = np.transpose(loglik[:, ids], (1, 0, 2))             # [N, C, T] log p_i(x_t)
    contrib = np.where(contrib_mask[:, None, :], ll, 0.0)     # invalid / EOS positions add 0
    cumll = np.cumsum(contrib, axis=2)
    logits = np.log(prior)[None, :, None] + cumll
    mx = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - mx)
    eta = ex / ex.sum(axis=1, keepdims=True)
    eta[np.broadcast_to(~out_mask[:, None, :], eta.shape)] = np.nan
    return eta
