"""Where the mixture's top first-token openers actually live in the training data.

Exp 4 / `base_first_token.py` look at the MIXTURE model's first-token distribution
P_mix(.|[EOS]) — the tokens it is most likely to *open* a story with. This script asks the
complementary, data-side question:

    Take the most probable tokens of P_mix(.|[EOS]); for each, how often does it occur in the
    training corpus and AT WHICH WORD POSITIONS does it appear?

The deliverable is a heat map: rows = the top-TOP_N first tokens of the mixture (ordered by
P_mix(.|[EOS])), columns = word (token) position in the story, cell = how often that token sits
at that position in the union of persona datasets. This makes the triggered-vs-behavioural split
visible from the data alone: a position-LOCKED opener (e.g. `dear`, `once`) lights up only at
position 0, while a generic/behavioural opener (e.g. `in`, `a`, `the`) is spread across positions.

Two normalisations are produced (they answer the two halves of the question):
  - p(x_t = v | t) : at position t, what fraction of stories have token v there. The t=0 column
    IS the empirical first-token distribution of the *data* — the direct analog of P_mix(.|[EOS]).
  - p(t | v)       : for token v, how its occurrences are distributed over positions (row-normalised
    — literally "the frequency at which word position they appear").

Word position = token position (the corpus is lowercase WordPiece; opening words are single
tokens, see control/notes.md), counted from the first real story token — so position 0 aligns
with the model's first-token distribution (P_mix(.|[EOS]) is P over the first story token).

Run:  python -m src.first_token_positional
Writes results/first_token_positional_<ts>/:
  first_token_positional.csv      (per top token: P_mix(1), total count, count@0, frac@0)
  positional_grid.csv             (the full [TOP_N x POS_WINDOW] p(v|t) grid)
  heatmap_token_given_pos.png     (rows=tokens, cols=position, cell=p(x_t=v|t))
  heatmap_pos_given_token.png     (rows=tokens, cols=position, cell=p(t|v))
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config, data, models, pmi
from .config import RunConfig, DataConfig

TOP_N = 30        # how many of the mixture's top first tokens to chart (the heatmap rows)
POS_WINDOW = 40   # word positions 0..POS_WINDOW-1 shown on the horizontal axis


def _mixture_first_token_probs(mixture_model, device) -> np.ndarray:
    """P_mix(first token | [EOS]) as a [V] probability vector (the same seed Exp 4 uses)."""
    seed = torch.tensor([[config.EOS_ID]], dtype=torch.long, device=device)  # bare [EOS], no BOS
    seed_attn = torch.ones_like(seed)
    logd = models.next_token_logdist(mixture_model, seed, seed_attn)[0, 0].cpu().numpy()  # [V]
    return np.exp(logd)


def _union_positional_counts(tok) -> np.ndarray:
    """counts[t, v] = #(stories in ∪_i D_i with token v at word position t), positions 0..POS_WINDOW-1.

    Tokenised WITHOUT the leading [EOS] (prepend_eos=False) so position 0 is the first real word —
    aligning the t=0 column with the model's first-token distribution. Reuses pmi.positional_counts
    (the audited Exp 1 counter) per persona, then sums over personas to get the union.
    """
    dcfg = DataConfig(t_max=POS_WINDOW, prepend_eos=False, append_eos=False)
    by_persona = data.load_persona_stories()

    ids_parts, attn_parts, labels = [], [], []
    for idx, persona in enumerate(config.PERSONAS):
        ids_p, attn_p = data.tokenize_stories(by_persona[persona], tok, dcfg)
        ids_parts.append(ids_p)
        attn_parts.append(attn_p)
        labels.extend([idx] * len(by_persona[persona]))
    input_ids = torch.cat(ids_parts).numpy()
    attn = torch.cat(attn_parts).numpy()
    labels = np.array(labels)

    counts = pmi.positional_counts(input_ids, attn, labels, t_max=POS_WINDOW)  # [k, t, V]
    return counts.sum(axis=0)  # [t, V] union over personas


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR,
                           f"first_token_positional_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[ftpos] device={cfg.device}  out={out_dir}  TOP_N={TOP_N}  POS_WINDOW={POS_WINDOW}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)

    # 1) the mixture's most probable first tokens (the heatmap rows)
    p_mix = _mixture_first_token_probs(mixture_model, cfg.device)
    top = np.argsort(p_mix)[::-1][:TOP_N].astype(int)  # token ids, most→least probable opener
    print(f"[ftpos] top-{TOP_N} mixture first tokens: "
          f"{[tok.convert_ids_to_tokens(int(v)) for v in top]}")

    # 2) positional occurrence counts over the training union, restricted to those tokens
    union = _union_positional_counts(tok)            # [POS_WINDOW, V]
    col_tot = union.sum(axis=1)                       # [POS_WINDOW]  #stories alive at position t
    tok_tot_all = union.sum(axis=0)                   # [V]           total occurrences over window

    # p(x_t = v | t): at position t, fraction of stories whose position-t token is v
    p_v_given_t = np.divide(union, col_tot[:, None], out=np.zeros_like(union),
                            where=col_tot[:, None] > 0)
    # p(t | v): for token v, how its occurrences spread over positions (row-normalised)
    p_t_given_v = np.divide(union, tok_tot_all[None, :], out=np.zeros_like(union),
                            where=tok_tot_all[None, :] > 0)

    grid_vt = p_v_given_t[:, top].T   # [TOP_N, POS_WINDOW]
    grid_tv = p_t_given_v[:, top].T   # [TOP_N, POS_WINDOW]
    labels = [tok.convert_ids_to_tokens(int(v)) for v in top]

    # 3) per-token summary CSV ("how often each token shows up" + position-0 concentration)
    rows = []
    for v in top:
        v = int(v)
        total = float(tok_tot_all[v])
        rows.append({
            "token": tok.convert_ids_to_tokens(v), "token_id": v,
            "p_mixture_first_token": float(p_mix[v]),
            "total_count_in_window": total,
            "count_at_pos0": float(union[0, v]),
            "frac_of_occurrences_at_pos0": float(union[0, v] / total) if total > 0 else 0.0,
            "p_token_at_pos0": float(p_v_given_t[0, v]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "first_token_positional.csv"), index=False)
    print("\n[ftpos] top mixture openers — corpus frequency and position-0 concentration:")
    print(df[["token", "p_mixture_first_token", "total_count_in_window",
              "frac_of_occurrences_at_pos0"]].to_string(
        index=False, formatters={"p_mixture_first_token": "{:.4f}".format,
                                 "total_count_in_window": "{:.0f}".format,
                                 "frac_of_occurrences_at_pos0": "{:.3f}".format}))

    # full p(v|t) grid for downstream reuse (rows = tokens, cols = positions)
    pd.DataFrame(grid_vt, index=labels,
                 columns=[f"pos{t}" for t in range(POS_WINDOW)]).to_csv(
        os.path.join(out_dir, "positional_grid.csv"))

    _plot_heatmap(grid_vt, labels,
                  "P(token at position t | t)   — t=0 column = data's first-token distribution",
                  os.path.join(out_dir, "heatmap_token_given_pos.png"))
    _plot_heatmap(grid_tv, labels,
                  "P(position t | token)   — each row sums to 1 over shown window",
                  os.path.join(out_dir, "heatmap_pos_given_token.png"))
    print(f"\n[ftpos] done. results in {out_dir}")


def _plot_heatmap(grid: np.ndarray, row_labels: list[str], title: str, path: str) -> None:
    """Heat map: rows = mixture's top first tokens (top→bottom = most→least probable opener),
    columns = word position. PowerNorm(0.5) so a sharp position-0 spike and a faint spread are
    both visible in the same picture (the triggered-vs-behavioural contrast)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(grid, aspect="auto", cmap="magma",
                   norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=float(grid.max())))
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels(np.arange(grid.shape[1]), fontsize=7)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("word position in story (0 = first token)")
    ax.set_ylabel("mixture top first tokens  (most → least probable opener)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
