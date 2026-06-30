"""Discriminative pi over the free-generation dataset, from the probe (vs the generative EM pi).

Exp 2 estimates pi_free GENERATIVELY: EM over the free generations' whole-sequence specialist
log-probs (which mixture of specialists best explains the samples). Here we estimate pi
DISCRIMINATIVELY: run the persona probe (trained on real D_i) over the entire free-generation set and
aggregate its per-prefix beliefs into a distribution over clusters — "what fraction of free-gen
content a classifier trained on real data attributes to each cluster". A second, independent opinion
on pi_free; agreement corroborates the generative decomposition, disagreement is informative (the
probe is trained on real D_i, so on the OOD free generations its accuracy is bounded — Exp 7 §7).

We report several aggregations (the probe is position-agnostic):
  - single-position probe (clf_single, e_s): FINAL-position belief per sequence, and MEAN-over-
    positions per sequence (denoises the recency-noisy single position), each soft (mean of beta)
    and hard (argmax count).
  - running-mean probe (clf_mean, e_bar): FINAL-position belief (= Exp 7 cumulative beta at the end).
All compared to sigma (uniform 0.2), the generative EM pi_free, and the generative hard-assignment.

Run:  CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_probe_pi
Writes results/probe_pi_<ts>/: pi_compare.csv, pi_compare.png, config.json.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config, models, probe, token_dist
from . import run_exp7 as ex7
from . import probe_per_token as ppt
from .commitment import COMPONENTS
from .config import RunConfig

EMBED_LAYER, L2, FEAT_BS = ex7.EMBED_LAYER, ex7.L2_NORMALIZE, ex7.FEAT_BS


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _load_probe(path: str) -> probe.MultinomialProbe:
    z = np.load(path, allow_pickle=True)
    assert [str(c) for c in z["classes"]] == list(COMPONENTS), f"probe class order mismatch in {path}"
    return probe.MultinomialProbe(z["mu"], z["sd"], z["W"], z["b"], final_loss=float("nan"))


@torch.no_grad()
def both_feats(model, ids, attn, device):
    """One forward pass -> (e_pos, e_bar) [N,T,H] L2-normalised (single-position and running-mean)."""
    n, t = ids.shape; h = model.config.hidden_size
    raw = np.empty((n, t, h), dtype=np.float32)
    for s in range(0, n, FEAT_BS):
        raw[s:s + FEAT_BS] = models.prefix_embeddings(
            model, ids[s:s + FEAT_BS].to(device), attn[s:s + FEAT_BS].to(device), EMBED_LAYER).cpu().numpy()
    e_bar = np.cumsum(raw, axis=1) / np.arange(1, t + 1, dtype=np.float32)[None, :, None]

    def l2n(x):
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None) if L2 else x
    return l2n(raw), l2n(e_bar)


def _pi_from_beta(beta: np.ndarray, fwd_len: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """beta [N, T, C] probe beliefs. Returns (soft_pi [C], hard_pi [C]).

    mode='final'  : use each sequence's belief at its last valid position (full-sequence prefix).
    mode='meanpos': per sequence, mean belief over valid positions t in [1, len) (denoised), then
                    average across sequences.
    """
    n = beta.shape[0]
    per_seq = np.empty((n, len(COMPONENTS)))
    for s in range(n):
        last = int(fwd_len[s]) - 1
        if mode == "final":
            per_seq[s] = beta[s, last, :]
        else:
            per_seq[s] = beta[s, 1:last + 1, :].mean(axis=0) if last >= 1 else beta[s, last, :]
    soft = per_seq.mean(axis=0)
    hard = np.bincount(per_seq.argmax(axis=1), minlength=len(COMPONENTS)) / n
    return soft, hard


def main() -> None:
    cfg = RunConfig(); cfg.device = models.resolve_device(cfg.device)
    out = os.path.join(config.RESULTS_DIR, f"probe_pi_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True); cfg.to_json(os.path.join(out, "config.json"))
    print(f"[probe_pi] device={cfg.device}  out={out}")

    mix = models.load_mixture_model(cfg.device)
    clf_single = _load_probe(os.path.join(_latest("probe_single_*"), "probe_single.npz"))
    clf_mean = _load_probe(os.path.join(_latest("exp7_[0-9]*"), "probe.npz"))

    exp2 = _latest("exp2_*")
    npz = np.load(os.path.join(exp2, "free_samples.npz"))
    free = torch.tensor(npz["samples"]);
    fwd = token_dist.forward_attention_mask(free)
    fwd_len = fwd.sum(dim=1).numpy().astype(int)
    print(f"[probe_pi] embedding {free.shape[0]} free generations ...")
    e_pos, e_bar = both_feats(mix, free, fwd, cfg.device)
    beta_single = clf_single.predict_proba(e_pos)                  # [N, T, C]
    beta_mean = clf_mean.predict_proba(e_bar)                      # [N, T, C]

    # generative references
    sigma = np.array([config.SIGMA[c] for c in COMPONENTS])
    pi_free = pd.read_csv(os.path.join(exp2, "pi_free.csv")).set_index("persona").loc[COMPONENTS, "pi_free"].to_numpy()
    hard_counts = pd.read_csv(os.path.join(exp2, "free_assignment.csv")).iloc[0]
    pi_gen_hard = np.array([hard_counts[c] for c in COMPONENTS], dtype=float); pi_gen_hard /= pi_gen_hard.sum()

    # discriminative pi (probe), several aggregations
    s_fin_soft, s_fin_hard = _pi_from_beta(beta_single, fwd_len, "final")
    s_mp_soft, s_mp_hard = _pi_from_beta(beta_single, fwd_len, "meanpos")
    m_fin_soft, m_fin_hard = _pi_from_beta(beta_mean, fwd_len, "final")

    estimators = {
        "sigma (uniform)": sigma,
        "generative EM pi_free": pi_free,
        "generative hard-assign": pi_gen_hard,
        "probe single | final  | soft": s_fin_soft,
        "probe single | final  | hard": s_fin_hard,
        "probe single | meanpos| soft": s_mp_soft,
        "probe single | meanpos| hard": s_mp_hard,
        "probe runmean| final  | soft": m_fin_soft,
        "probe runmean| final  | hard": m_fin_hard,
    }

    rows = []
    for name, pi in estimators.items():
        rows.append({"estimator": name, **{c: float(pi[i]) for i, c in enumerate(COMPONENTS)},
                     "L1_to_sigma": float(np.abs(pi - sigma).sum()),
                     "L1_to_pi_free": float(np.abs(pi - pi_free).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "pi_compare.csv"), index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n[probe_pi] pi over the free-generation dataset (each row sums to 1):")
    print(df.round(3).to_string(index=False))

    # plot: grouped bars per cluster for the headline estimators
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    headline = ["sigma (uniform)", "generative EM pi_free", "probe single | meanpos| soft",
                "probe runmean| final  | soft"]
    x = np.arange(len(COMPONENTS)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, name in enumerate(headline):
        pi = estimators[name]
        ax.bar(x + (j - 1.5) * w, pi, w, label=name)
    ax.axhline(0.2, ls=":", color="0.5", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(COMPONENTS); ax.set_ylabel("pi")
    ax.set_title("pi over free generations — generative EM vs discriminative probe")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, "pi_compare.png"), dpi=120)
    plt.close(fig)
    print(f"\n[probe_pi] done. results in {out}")


if __name__ == "__main__":
    main()
