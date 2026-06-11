"""Experiment 1 — PMI trigger identification + persona taxonomy.

Pure empirical statistics over token counts in the union of persona datasets. NO model is
touched here (design_doc.md §3): PMI is computed from how often each token appears at each
position within a persona vs across all personas.

Two PMI flavours (design_doc.md §4 Exp 1):
  - Position-conditioned PMI(v,t,i): catches position-locked triggers (e.g. "dear" at pos 0).
  - Position-marginal PMI(v,i): bag-of-tokens, catches diffuse behavioural signal.

A concentration score C_i in (0,1] summarises how much of a persona's identifying signal sits
in its single strongest (token, position): high C_i => one rare token carries the persona
(triggered), low C_i => signal spread across many tokens/positions (behavioural).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

from . import config


def positional_counts(input_ids: np.ndarray, attn: np.ndarray, labels: np.ndarray,
                      t_max: int, vocab: int = config.VOCAB_SIZE) -> np.ndarray:
    """counts[i, t, v] = #(persona-i sequences with token v at position t). Padding excluded.

    input_ids/attn are [N, t_max]; labels[n] is the persona index of row n (config.PERSONAS order).
    """
    k = len(config.PERSONAS)
    counts = np.zeros((k, t_max, vocab), dtype=np.float64)
    for i in range(k):
        rows = labels == i
        ids_i = input_ids[rows]          # [n_i, t_max]
        attn_i = attn[rows]
        for t in range(t_max):
            toks = ids_i[:, t][attn_i[:, t] == 1]
            if toks.size:
                counts[i, t] = np.bincount(toks, minlength=vocab)
    return counts


def _safe_pmi(p_cond: np.ndarray, p_marg: np.ndarray) -> np.ndarray:
    """log(p_cond / p_marg), set to -inf where the persona never emits the token (p_cond==0).

    Where p_cond>0 the marginal is guaranteed >0, so no spurious infinities arise.
    """
    p_cond, p_marg = np.broadcast_arrays(p_cond, p_marg)
    out = np.full(p_cond.shape, -np.inf)
    nz = p_cond > 0
    out[nz] = np.log(p_cond[nz]) - np.log(p_marg[nz])
    return out


def positional_pmi(counts: np.ndarray) -> np.ndarray:
    """PMI(v,t,i) = log p(x_t=v | y=i) / p(x_t=v | t)  ->  [k, t_max, vocab]."""
    persona_pos_tot = counts.sum(axis=2, keepdims=True)          # [k, t, 1]  sum_v
    p_cond = np.divide(counts, persona_pos_tot, out=np.zeros_like(counts),
                       where=persona_pos_tot > 0)                # p(x_t=v | y=i)
    pos_tot = counts.sum(axis=0)                                  # [t, vocab]  sum_i
    pos_all = pos_tot.sum(axis=1, keepdims=True)                  # [t, 1]
    p_marg = np.divide(pos_tot, pos_all, out=np.zeros_like(pos_tot),
                       where=pos_all > 0)                         # p(x_t=v | t)
    return _safe_pmi(p_cond, p_marg[None, :, :])


def marginal_pmi(counts: np.ndarray) -> np.ndarray:
    """PMI(v,i) = log p(v | y=i) / p(v)  ->  [k, vocab]  (position-marginalised bag of tokens)."""
    persona_tok = counts.sum(axis=1)                             # [k, vocab]  sum_t
    persona_tot = persona_tok.sum(axis=1, keepdims=True)         # [k, 1]
    p_cond = persona_tok / persona_tot                           # p(v | y=i)
    tok_tot = persona_tok.sum(axis=0)                            # [vocab]
    p_marg = (tok_tot / tok_tot.sum())[None, :]                  # p(v)
    return _safe_pmi(p_cond, p_marg)


def _weighted_signal(pmi: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """MI-style per-(t,v) signal contribution for one persona: p(x_t=v, t | i) * max(PMI, 0).

    Returns [t_max, vocab] for a single persona's pmi/counts slice. We keep only the POSITIVE
    PMI direction: over-represented tokens are identifying signal; under-represented tokens are
    anti-signal and should not inflate the denominator (this is what keeps C_i in (0, 1]). This
    is an explicit modelling choice on top of design_doc's w = p(v|i)*PMI, documented here.
    """
    joint = counts / counts.sum()                               # p(x_t=v, t | i), sums to 1
    return joint * np.clip(pmi, 0.0, None)


def trigger_score(pmi: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-(t,v) anchor-quality score  p(x_t=v | i, t) * max(PMI(v,t,i), 0)  ->  [k, t_max, vocab].

    This is the selector for the trigger a persona is ANCHORED on in Exp 2 — distinct from the
    concentration weight below. PMI alone saturates at log(k) for any persona-exclusive token,
    so it cannot rank among exclusive tokens; multiplying by the conditional reliability
    p(v | i, t) picks the token that is BOTH distinctive (high PMI / marginally rare) AND
    emitted dependably at that position (design_doc's "marginally rare but conditionally
    frequent"). Empirically this recovers 'dear'@0 (epistolary), 'once'@0 (fairy_tale), while
    behavioural personas top out an order of magnitude lower (no reliable entry token).
    """
    persona_pos_tot = counts.sum(axis=2, keepdims=True)
    p_cond = np.divide(counts, persona_pos_tot, out=np.zeros_like(counts),
                       where=persona_pos_tot > 0)                # p(x_t=v | i, t)
    return p_cond * np.clip(pmi, 0.0, None)


def concentration(pmi: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    """C_i = max_{(v,t)} w(v,t,i) / sum_{(v,t)} w(v,t,i), using positional PMI weighted signal."""
    out: dict[str, float] = {}
    for i, persona in enumerate(config.PERSONAS):
        w = _weighted_signal(pmi[i], counts[i])
        total = w.sum()
        assert total > 0, f"{persona}: no positive PMI signal — degenerate counts?"
        out[persona] = float(w.max() / total)
    return out


def trigger_table(pmi: np.ndarray, counts: np.ndarray, tokenizer: PreTrainedTokenizerBase,
                  top_n: int = 15) -> pd.DataFrame:
    """Ranked identifying (token, position) pairs per persona by trigger score.

    Columns: persona, rank, token_id, token, pos, pmi, p_v_given_i_t, p_v_given_t, score.
    Ranked by trigger_score (anchor quality), so the top-1 row per persona is the trigger
    handed to Exp 2 (anchored generation).
    """
    persona_pos_tot = counts.sum(axis=2, keepdims=True)
    p_cond = np.divide(counts, persona_pos_tot, out=np.zeros_like(counts),
                       where=persona_pos_tot > 0)                # p(x_t=v | i, t)
    pos_tot = counts.sum(axis=0)
    pos_all = pos_tot.sum(axis=1, keepdims=True)
    p_marg = np.divide(pos_tot, pos_all, out=np.zeros_like(pos_tot), where=pos_all > 0)  # p(x_t=v|t)
    score = trigger_score(pmi, counts)

    rows = []
    for i, persona in enumerate(config.PERSONAS):
        flat = np.argsort(score[i], axis=None)[::-1][:top_n]    # top-scoring (t, v)
        for rank, idx in enumerate(flat):
            t, v = np.unravel_index(idx, score[i].shape)
            rows.append({
                "persona": persona,
                "rank": rank,
                "token_id": int(v),
                "token": tokenizer.decode([int(v)]),
                "pos": int(t),
                "pmi": float(pmi[i, t, v]),
                "p_v_given_i_t": float(p_cond[i, t, v]),
                "p_v_given_t": float(p_marg[t, v]),
                "score": float(score[i, t, v]),
            })
    return pd.DataFrame(rows)


def classify(concentration_scores: dict[str, float], threshold: float) -> dict[str, str]:
    """C_i >= threshold => 'triggered', else 'behavioural'. The threshold is logged by the caller."""
    return {p: ("triggered" if c >= threshold else "behavioural")
            for p, c in concentration_scores.items()}


def top_triggers(table: pd.DataFrame) -> dict[str, dict]:
    """Persona -> its top-1 trigger row (token string, id, pos) for Exp 2 anchoring."""
    out: dict[str, dict] = {}
    for persona in config.PERSONAS:
        top = table[(table["persona"] == persona) & (table["rank"] == 0)].iloc[0]
        out[persona] = {"token": top["token"], "token_id": int(top["token_id"]),
                        "pos": int(top["pos"])}
    return out
