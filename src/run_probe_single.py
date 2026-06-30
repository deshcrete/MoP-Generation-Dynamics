"""Probe on the SINGLE-POSITION residual stream e_s vs the running-mean e_bar(s).

The Exp 7 probe reads the running-mean prefix embedding e_bar(t)=mean_{s<=t} e(s); its per-token
twin beta_tok reuses that POOL-trained probe on single positions and fails — but that is a train/test
mismatch, and the "per-token doesn't separate" claim (eta^2=0.04) is from UNSUPERVISED 2D-PCA, which
only sees top-variance (content/recency) directions. This script instead TRAINS a probe directly on
the single-position residual stream e_s: under causal attention e_s has already integrated x_{1:s},
so the conditioning P(persona | x_{1:s}) lives inside that one activation — no aggregation rule
needed; the readout is moved to the position where the chain rule already happened. A supervised
linear probe can find a low-variance-but-discriminative direction PCA discards.

Fair comparison: BOTH probes are trained/evaluated on the IDENTICAL story split and the IDENTICAL
sampled (story, position) examples (one forward pass; the two feature tensors differ only by the
cumulative-mean step). Reports held-out accuracy overall, vs prefix position, and the confusion
matrix for each. If single-position accuracy ~matches the running-mean, the averaging was an
unnecessary crutch and the conditioning IS linearly present at the position itself.

Run:  CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_probe_single
Writes results/probe_single_<ts>/: accuracy_compare.csv, accuracy_vs_position.png,
confusion_single.png, confusion_mean.png, probe_single.npz, params.json, config.json.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config, data, models, probe
from . import run_exp7 as ex7                     # reuse Exp 7 params + the confusion plot
from .commitment import COMPONENTS
from .config import RunConfig

# identical to Exp 7 so the comparison is apples-to-apples (only the pooling differs)
EMBED_LAYER, L2 = ex7.EMBED_LAYER, ex7.L2_NORMALIZE
N_TRAIN, N_TEST = ex7.N_TRAIN_STORIES, ex7.N_TEST_STORIES
POS_PER_STORY, T_PROBE_MAX = ex7.POS_PER_STORY, ex7.T_PROBE_MAX
EPOCHS, LR, WD, SEED, FEAT_BS = (ex7.PROBE_EPOCHS, ex7.PROBE_LR, ex7.PROBE_WD,
                                 ex7.SEED_PROBE, ex7.FEAT_BS)
POS_BINS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, T_PROBE_MAX)]


@torch.no_grad()
def both_feats(model, ids: torch.LongTensor, attn: torch.LongTensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass -> (e_pos, e_mean) each [N, T, H], L2-normalised. e_pos = single-position
    residual stream e(t); e_mean = running mean e_bar(t) = mean_{s<=t} e(s)."""
    n, t = ids.shape
    h = model.config.hidden_size
    raw = np.empty((n, t, h), dtype=np.float32)
    for s in range(0, n, FEAT_BS):
        raw[s:s + FEAT_BS] = models.prefix_embeddings(
            model, ids[s:s + FEAT_BS].to(device), attn[s:s + FEAT_BS].to(device), EMBED_LAYER).cpu().numpy()
    e_mean = np.cumsum(raw, axis=1) / np.arange(1, t + 1, dtype=np.float32)[None, :, None]

    def l2n(x):
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None) if L2 else x
    return l2n(raw), l2n(e_mean)


def sample_positions(lengths: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Replicates probe.build_examples' per-story position sampling, returning (story_idx, pos) pairs
    so the SAME examples can be gathered from both feature tensors."""
    pairs = []
    for i, L in enumerate(lengths):
        hi = min(int(L), T_PROBE_MAX)
        if hi <= 1:
            continue
        avail = np.arange(1, hi)
        chosen = rng.choice(avail, size=min(POS_PER_STORY, avail.size), replace=False)
        pairs.extend((i, int(c)) for c in chosen)
    return pairs


def _gather(feat: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([feat[i, t] for i, t in pairs]) if pairs else np.empty((0, feat.shape[2]), np.float32)


def _eval(clf, X, y, pos) -> tuple[float, np.ndarray, list[float], list[int], dict]:
    pred = clf.predict_proba(X).argmax(axis=1)
    overall = float((pred == y).mean())
    acc_b, n_b = [], []
    for lo, hi in POS_BINS:
        sel = (pos >= lo) & (pos < hi)
        n_b.append(int(sel.sum()))
        acc_b.append(float((pred[sel] == y[sel]).mean()) if sel.any() else float("nan"))
    conf = np.zeros((len(COMPONENTS), len(COMPONENTS)))
    for i, j in zip(y, pred):
        conf[i, j] += 1
    per_class = {COMPONENTS[i]: float(conf[i, i] / conf[i].sum()) if conf[i].sum() else float("nan")
                 for i in range(len(COMPONENTS))}
    conf_norm = conf / np.clip(conf.sum(axis=1, keepdims=True), 1, None)
    return overall, conf_norm, acc_b, n_b, per_class


def _plot_acc_vs_pos(acc_single, acc_mean, n_b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(POS_BINS)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w / 2, acc_single, w, color="#1f77b4", label="single-position $e_s$")
    ax.bar(x + w / 2, acc_mean, w, color="#999999", label=r"running-mean $\bar e_s$ (Exp 7)")
    ax.axhline(1.0 / len(COMPONENTS), ls="--", color="#c0504d", lw=1, label=f"chance = 1/{len(COMPONENTS)}")
    ax.set_xticks(x); ax.set_xticklabels([f"[{lo},{hi})\nn={ni}" for (lo, hi), ni in zip(POS_BINS, n_b)],
                                         fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("held-out accuracy"); ax.set_xlabel("prefix position t (bin)")
    ax.set_title("Probe accuracy vs prefix position — single-position $e_s$ vs running-mean")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    cfg = RunConfig(); cfg.device = models.resolve_device(cfg.device)
    out = os.path.join(config.RESULTS_DIR, f"probe_single_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True); cfg.to_json(os.path.join(out, "config.json"))
    with open(os.path.join(out, "params.json"), "w") as f:
        json.dump({"EMBED_LAYER": EMBED_LAYER, "L2_NORMALIZE": L2, "N_TRAIN_STORIES": N_TRAIN,
                   "N_TEST_STORIES": N_TEST, "POS_PER_STORY": POS_PER_STORY,
                   "T_PROBE_MAX": T_PROBE_MAX, "PROBE_EPOCHS": EPOCHS, "PROBE_LR": LR,
                   "PROBE_WD": WD, "SEED_PROBE": SEED}, f, indent=2)
    print(f"[probe_single] device={cfg.device}  out={out}")

    tok = models.load_tokenizer()
    mix = models.load_mixture_model(cfg.device)

    rng = np.random.default_rng(SEED)
    n_per = N_TRAIN + N_TEST
    stories = data.load_persona_stories()
    Xs_tr, Xm_tr, y_tr = [], [], []
    Xs_te, Xm_te, y_te, pos_te = [], [], [], []

    for ci, comp in enumerate(COMPONENTS):
        pool = stories[comp]
        assert len(pool) >= n_per, f"{comp}: only {len(pool)} stories, need {n_per}"
        chosen = rng.choice(len(pool), size=n_per, replace=False)
        ids, attn = data.tokenize_stories([pool[int(c)] for c in chosen], tok, cfg.data)
        e_pos, e_mean = both_feats(mix, ids, attn, cfg.device)          # [n_per, T, H] each
        lengths = attn.sum(dim=1).numpy()

        # split BY STORY (first N_TRAIN train, rest held out), same sampled positions for both feats
        tr_pairs = sample_positions(lengths[:N_TRAIN], rng)
        te_pairs0 = sample_positions(lengths[N_TRAIN:], rng)
        te_pairs = [(i + N_TRAIN, t) for i, t in te_pairs0]             # shift to global story index
        Xs_tr.append(_gather(e_pos, tr_pairs)); Xm_tr.append(_gather(e_mean, tr_pairs))
        y_tr.append(np.full(len(tr_pairs), ci))
        Xs_te.append(_gather(e_pos, te_pairs)); Xm_te.append(_gather(e_mean, te_pairs))
        y_te.append(np.full(len(te_pairs), ci)); pos_te.append(np.array([t for _, t in te_pairs]))
        print(f"[probe_single]  {comp:14s} train ex={len(tr_pairs):5d}  test ex={len(te_pairs):5d}")

    Xs_tr = np.concatenate(Xs_tr); Xm_tr = np.concatenate(Xm_tr); y_tr = np.concatenate(y_tr)
    Xs_te = np.concatenate(Xs_te); Xm_te = np.concatenate(Xm_te)
    y_te = np.concatenate(y_te); pos_te = np.concatenate(pos_te)

    print(f"[probe_single] training two probes on {len(y_tr)} identical examples ({Xs_tr.shape[1]}-dim) ...")
    clf_s = probe.train_probe(Xs_tr, y_tr, len(COMPONENTS), cfg.device, LR, EPOCHS, WD, SEED)
    clf_m = probe.train_probe(Xm_tr, y_tr, len(COMPONENTS), cfg.device, LR, EPOCHS, WD, SEED)
    acc_s, conf_s, accb_s, n_b, pc_s = _eval(clf_s, Xs_te, y_te, pos_te)
    acc_m, conf_m, accb_m, _, pc_m = _eval(clf_m, Xm_te, y_te, pos_te)

    # --- report ---------------------------------------------------------------------------
    print(f"\n[probe_single] HELD-OUT ACCURACY  single-position e_s = {acc_s:.3f}   "
          f"running-mean e_bar = {acc_m:.3f}   (chance {1/len(COMPONENTS):.2f})")
    print("[probe_single] accuracy vs prefix position (single | mean):")
    for (lo, hi), a_s, a_m, ni in zip(POS_BINS, accb_s, accb_m, n_b):
        print(f"    t[{lo:>2},{hi:>3})  n={ni:5d}   single={a_s:.3f}   mean={a_m:.3f}")
    print("[probe_single] per-class held-out accuracy (single | mean):")
    for c in COMPONENTS:
        print(f"    {c:14s} single={pc_s[c]:.3f}   mean={pc_m[c]:.3f}")

    rows = [{"probe": "single", "kind": "overall", "key": "all", "n": int(len(y_te)), "accuracy": acc_s},
            {"probe": "mean", "kind": "overall", "key": "all", "n": int(len(y_te)), "accuracy": acc_m}]
    for (lo, hi), a_s, a_m, ni in zip(POS_BINS, accb_s, accb_m, n_b):
        rows.append({"probe": "single", "kind": "pos_bin", "key": f"[{lo},{hi})", "n": ni, "accuracy": a_s})
        rows.append({"probe": "mean", "kind": "pos_bin", "key": f"[{lo},{hi})", "n": ni, "accuracy": a_m})
    for c in COMPONENTS:
        rows.append({"probe": "single", "kind": "per_class", "key": c, "n": None, "accuracy": pc_s[c]})
        rows.append({"probe": "mean", "kind": "per_class", "key": c, "n": None, "accuracy": pc_m[c]})
    pd.DataFrame(rows).to_csv(os.path.join(out, "accuracy_compare.csv"), index=False)

    _plot_acc_vs_pos(accb_s, accb_m, n_b, os.path.join(out, "accuracy_vs_position.png"))
    ex7._plot_confusion(conf_s, os.path.join(out, "confusion_single.png"))
    ex7._plot_confusion(conf_m, os.path.join(out, "confusion_mean.png"))
    np.savez_compressed(os.path.join(out, "probe_single.npz"), mu=clf_s.mu, sd=clf_s.sd,
                        W=clf_s.W, b=clf_s.b, classes=np.array(COMPONENTS))
    print(f"\n[probe_single] done. results in {out}")


if __name__ == "__main__":
    main()
