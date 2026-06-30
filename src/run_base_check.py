"""Diagnostic — does the BASE model's first-token distribution explain the EM weights?

Sanity check on the Exp 4 weight inference, run with the BASE model as the target instead of
the mixture. The base has no persona and no trigger tokens, so it is a clean control: whatever
weight EM hands each specialist must be explainable purely by how much that specialist's
first-token distribution OVERLAPS the base's first-token distribution — there is no trigger
confound to hide behind. If EM is behaving, the persona it up-weights (e.g. "why is cluster i
preferred so much") should be the one whose first-token distribution is most similar to base.

Procedure (all at t=1, the neutral [EOS] seed — the first generated token's distribution):
  1. p_base   = P_base(. | [EOS])           [V]   the base model's first-token distribution
     q_i      = P_i(.    | [EOS])           [k,V] each specialist's first-token distribution
  2. w_base   = argmin_w KL( p_base || sum_i w_i q_i )   via the SAME weighted EM as Exp 4
                (token_dist.solve_kl_weights) — the weight "EM assigns" to each persona.
  3. Per persona, measure first-token-distribution similarity to base WITHOUT EM:
        - KL(p_base || q_i)   forward KL  (lower  = more similar; the per-component term of the
                                            objective EM minimises)
        - cosine(p_base, q_i)             (higher = more similar)
        - pearson(p_base, q_i) over vocab (higher = more aligned)
        - overlap = sum_v min(p_base, q_i) (higher = more shared mass)
  4. Correlate the weight vector w_base against each similarity measure across the 5 personas
     (Spearman + Pearson). A high correlation = "the weight is explained by first-token overlap",
     i.e. EM is doing the sensible thing on a target with no triggers.

Also records the identifiability footer (KL to the uniform-average blend vs the EM blend), the
same caveat as Exp 4A: if EM barely beats a plain uniform average, the precise split is weakly
identified (collinear specialist first-token dists) and only the ORDERING is load-bearing.

Run:  python -m src.run_base_check
CPU:  CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_base_check
Writes results/base_check_<ts>/: config.json, base_first_token.csv, weight_vs_similarity.csv,
base_first_token_dist.png, weight_vs_similarity.png.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from . import config, models, token_dist
from .config import RunConfig


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats), skipping zero-mass tokens."""
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """Forward KL(p || q) in nats; q clipped off zero so finite."""
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], 1e-30, None))))


def _cosine(p: np.ndarray, q: np.ndarray) -> float:
    return float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q)))


def _overlap(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation overlap = sum_v min(p_v, q_v) in [0, 1]; 1 = identical distributions."""
    return float(np.minimum(p, q).sum())


def _plot_first_token_dist(p_base: np.ndarray, q_spec: np.ndarray, tok, w_base: np.ndarray,
                           top_ft: int, path: str) -> None:
    """Base's top first tokens as bars, each specialist's prob on the same tokens overlaid.
    Shows whether base's t=1 mass sits on top of one specialist or blends them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.argsort(p_base)[::-1][:top_ft]
    labels = [tok.convert_ids_to_tokens(int(i)) for i in idx]
    x = np.arange(len(idx))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, p_base[idx], width=0.6, color="0.7", label="base")
    markers = ["o", "s", "^", "D", "v", "P"]
    for j, p in enumerate(config.PERSONAS):
        ax.scatter(x, np.exp(q_spec[j])[idx], marker=markers[j % len(markers)], s=36,
                   label=f"{p} (w={w_base[j]:.2f})", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("P(first token | [EOS])")
    ax.set_title("Base first-token distribution (t=1) vs specialists")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_weight_vs_similarity(df: pd.DataFrame, path: str) -> None:
    """Scatter w_base against each similarity measure (one panel each), persona-labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("neg_kl", "-KL(p_base || q_i)"), ("cosine", "cosine(p_base, q_i)"),
              ("pearson", "pearson(p_base, q_i)"), ("overlap", "overlap(p_base, q_i)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), squeeze=False)
    for ax, (col, xlabel) in zip(axes[0], panels):
        ax.scatter(df[col], df["w_base"], s=40, zorder=3)
        for _, row in df.iterrows():
            ax.annotate(row["persona"], (row[col], row["w_base"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        r = pearsonr(df[col], df["w_base"])[0]
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(r"$\hat{w}_{\mathrm{base},i}$", fontsize=9)
        ax.set_title(f"pearson r = {r:.2f}", fontsize=9)
    fig.suptitle("EM weight vs first-token-distribution similarity to base", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"base_check_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[base_check] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    # Base-as-mixture: there is no separate base model — P_mix (SimpleStories-V2-5M) IS the base we
    # decompose over the cluster specialists (config.py). So "the base first-token distribution" is
    # P_mix(. | [EOS]), and w_base here is exactly Exp 4A's w_hat(1).
    base_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)

    # --- first-token distributions at the neutral [EOS] seed -------------------------------
    seed = torch.tensor([[config.EOS_ID]], dtype=torch.long, device=cfg.device)
    seed_attn = torch.ones_like(seed)
    base_ld = models.next_token_logdist(base_model, seed, seed_attn)[0, 0].cpu().numpy()      # [V]
    q_spec = np.stack([models.next_token_logdist(persona_models[p], seed, seed_attn)[0, 0].cpu().numpy()
                       for p in config.PERSONAS])                                              # [k, V]
    p_base = np.exp(base_ld)

    # --- the weight EM assigns: w_base = argmin_w KL(p_base || sum_i w_i q_i) --------------
    res = token_dist.solve_kl_weights(p_base, q_spec, cfg.em)
    w_base = res["pi"]
    print(f"[base_check] w_base (EM weights from base first-token dist): "
          f"{dict(zip(config.PERSONAS, w_base.round(3)))}  converged={res['converged']}")

    # --- per-persona first-token similarity to base (no EM) --------------------------------
    rows = []
    for i, p in enumerate(config.PERSONAS):
        q_i = np.exp(q_spec[i])
        kl = _kl(p_base, q_i)
        rows.append({
            "persona": p,
            "w_base": float(w_base[i]),
            "kl_base_to_specialist": kl,
            "neg_kl": -kl,
            "cosine": _cosine(p_base, q_i),
            "pearson": float(pearsonr(p_base, q_i)[0]),
            "overlap": _overlap(p_base, q_i),
            "specialist_entropy_nats": _entropy(q_i),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "base_first_token.csv"), index=False)

    # --- does the weight correlate with first-token similarity? ----------------------------
    corr_rows = []
    for col in ["neg_kl", "cosine", "pearson", "overlap"]:
        sp = spearmanr(df["w_base"], df[col])
        pe = pearsonr(df["w_base"], df[col])
        corr_rows.append({"similarity": col, "spearman_rho": float(sp.correlation),
                          "spearman_p": float(sp.pvalue), "pearson_r": float(pe[0]),
                          "pearson_p": float(pe[1])})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(out_dir, "weight_vs_similarity.csv"), index=False)

    # --- identifiability footer (same caveat as Exp 4A) -----------------------------------
    q = np.exp(q_spec)
    avg = q.mean(axis=0)                                  # uniform-average first-token dist
    em_blend = (w_base[:, None] * q).sum(axis=0)          # EM-weighted blend
    kl_uniform = _kl(p_base, avg)
    kl_em = _kl(p_base, em_blend)

    # --- report ---------------------------------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n[base_check] per-persona first-token distribution vs base:")
    print(df.round(4).to_string(index=False))
    print("\n[base_check] weight vs similarity correlation (across the 5 personas):")
    print(corr_df.round(4).to_string(index=False))
    top = config.PERSONAS[int(np.argmax(w_base))]
    print(f"\n[base_check] most up-weighted persona: {top} (w={w_base.max():.3f})")
    print(f"[base_check] base first-token entropy = {_entropy(p_base):.3f} nats")
    print(f"[base_check] identifiability: KL(p_base||uniform-avg)={kl_uniform:.4f}  "
          f"KL(p_base||EM-blend)={kl_em:.4f}  (EM buys {kl_uniform - kl_em:.4f} nats "
          f"= {100 * (kl_uniform - kl_em) / kl_uniform:.1f}%)")

    _plot_first_token_dist(p_base, q_spec, tok, w_base, top_ft=12,
                           path=os.path.join(out_dir, "base_first_token_dist.png"))
    _plot_weight_vs_similarity(df, os.path.join(out_dir, "weight_vs_similarity.png"))
    print(f"\n[base_check] done. results in {out_dir}")


if __name__ == "__main__":
    main()
