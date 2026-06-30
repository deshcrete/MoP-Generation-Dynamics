"""Persona collinearity matrices — how separable are the cluster specialists?

Three complementary k×k views, computed on the BALANCED real per-cluster data D_i (the ground-truth
persona definitions, label known but never shown to the scorer):

  1. log-prob correlation  C_raw[i,j]  = corr_x( log P_i(x), log P_j(x) ) over the inference set.
     This is exactly what EM sees ([N,k] seq log-probs). BUT it is inflated by a common mode —
     long/generic sequences are unlikely under EVERY specialist — so high C_raw can just mean
     "both models agree this sequence is hard," not "these personas are similar."
  2. CENTERED log-prob correlation  C_cent  = same, after subtracting each sequence's mean across
     the k specialists. Removes the common "sequence difficulty" mode and isolates persona-SPECIFIC
     preference. EM's M-step keys on exactly these per-sequence differences log P_i - log P_j, so
     C_cent is the identifiability-relevant matrix: high off-diagonal ⇒ EM cannot separate the pair.
  3. confusion matrix  conf[i,j] = fraction of cluster-i's own stories whose argmax specialist is j.
     Diagonal = recall (are clusters separable on their own data?); off-diagonal = which persona
     gets mistaken for which. Directly answers "are cluster-i completions treated as cluster-j?".

Run:  CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_persona_corr
Writes results/persona_corr_<ts>/: corr_raw.csv, corr_centered.csv, corr_spearman.csv,
confusion.csv, persona_corr.png.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from . import config, data, models
from .config import RunConfig


def _heatmap(ax, M, names, title, fmt="{:.2f}", vmin=None, vmax=None, cmap="coolwarm"):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                    color="black")
    ax.set_title(title, fontsize=10)
    return im


def main() -> None:
    cfg = RunConfig(); cfg.device = models.resolve_device(cfg.device)
    out = os.path.join(config.RESULTS_DIR, f"persona_corr_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True); cfg.to_json(os.path.join(out, "config.json"))
    names = config.PERSONAS; k = len(names)
    print(f"[persona_corr] device={cfg.device}  out={out}")

    tok = models.load_tokenizer()
    pm = models.load_persona_models(cfg.device)

    by = data.load_persona_stories()
    stories, labels = data.build_uniform_inference_set(by, cfg.data)
    ids, attn = data.tokenize_stories(stories, tok, cfg.data)   # [EOS]+story; pos0 auto-nan in scoring
    print(f"[persona_corr] scoring {len(stories)} stories ({cfg.data.inference_per_persona}/persona) "
          f"under {k} specialists")
    seq_lp = models.score_sequences(pm, ids, attn, cfg.device).numpy()       # [N, k]

    # 1+2. correlation matrices --------------------------------------------------------------
    C_raw = np.corrcoef(seq_lp.T)
    resid = seq_lp - seq_lp.mean(axis=1, keepdims=True)                      # remove common mode
    C_cent = np.corrcoef(resid.T)
    C_spear = spearmanr(seq_lp).correlation
    if np.ndim(C_spear) == 0:                                                # k==2 guard
        C_spear = np.array([[1.0, C_spear], [C_spear, 1.0]])

    # 3. confusion matrix (row-normalised over each true cluster's stories) -------------------
    am = seq_lp.argmax(axis=1)
    conf = np.zeros((k, k))
    for i in range(k):
        sel = labels == i
        for j in range(k):
            conf[i, j] = float((am[sel] == j).mean())

    for M, fn in [(C_raw, "corr_raw"), (C_cent, "corr_centered"),
                  (C_spear, "corr_spearman"), (conf, "confusion")]:
        pd.DataFrame(M, index=names, columns=names).to_csv(os.path.join(out, f"{fn}.csv"))

    # --- report -----------------------------------------------------------------------------
    np.set_printoptions(precision=3, suppress=True)
    print("\n[persona_corr] RAW log-prob correlation (inflated by sequence-difficulty common mode):")
    print(pd.DataFrame(C_raw, index=names, columns=names).round(3).to_string())
    print("\n[persona_corr] CENTERED log-prob correlation (persona-specific; EM-identifiability view):")
    print(pd.DataFrame(C_cent, index=names, columns=names).round(3).to_string())
    print("\n[persona_corr] CONFUSION (row=true cluster, col=argmax specialist; diag=recall):")
    print(pd.DataFrame(conf, index=names, columns=names).round(3).to_string())
    print(f"\n[persona_corr] mean diagonal recall = {np.diag(conf).mean():.3f} "
          f"(1.0 = perfectly separable on own data)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _heatmap(axes[0], C_raw, names, "raw log-prob corr", vmin=-1, vmax=1)
    _heatmap(axes[1], C_cent, names, "centered log-prob corr\n(persona-specific)", vmin=-1, vmax=1)
    im = _heatmap(axes[2], conf, names, "confusion (row=true, col=argmax)",
                  vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("Cluster-specialist collinearity on real per-cluster data $D_i$", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "persona_corr.png"), dpi=120); plt.close(fig)
    print(f"\n[persona_corr] done. results in {out}")


if __name__ == "__main__":
    main()
