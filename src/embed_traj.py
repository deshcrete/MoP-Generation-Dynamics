"""Experiment 5 — Persona commitment via prefix-embedding trajectories.

Make the commitment dynamics of Exp 3/4 *visible* in representation space. We plot fixed persona
reference clusters (real dataset stories D_i, plus base-model free generations for the base cluster,
all embedded through P_mix) and, on top of them, the mixture model's prefix embedding e(t) at each
token position of a single rollout. As t increases the prefix point traces a path through cluster
space, so commitment to a persona — and its speed — is directly visible.

This module holds the reusable pieces (embedding extraction, PCA, projection, plotting). The
orchestration (which stories, which rollouts, the LLR companion) lives in src/run_exp5.py. See
context/design_doc.md §4 Experiment 5.

Conventions (shared with the rest of src/):
  - Embeddings come from P_mix's residual stream via models.prefix_embeddings (final layer).
  - One embedding per CLUSTER story = the hidden state at its LAST real token (the causal
    whole-sequence summary), so a cluster point and a trajectory ENDpoint are the same object.
  - PCA is fit on the cluster embeddings only and reused for the trajectory, so both live in one
    projection. Embeddings are L2-normalised first (see L2_NORMALIZE in run_exp5) to stop the
    monotone growth of residual-stream norm with position from masquerading as movement.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedModel

from . import config, models


# --- Embedding extraction -------------------------------------------------------------------

@torch.no_grad()
def sequence_embeddings(model: PreTrainedModel, ids: torch.LongTensor, attn: torch.LongTensor,
                        device: str, layer: int = -1, batch_size: int = 128) -> np.ndarray:
    """[N, H] one embedding per sequence = the MEAN of the residual stream over its real tokens.

    `attn` is the forward mask whose 1s are a contiguous prefix (right-padded). We mean-pool rather
    than take the last token because (measured 2026-06-20) the last-token embedding of a long story
    is dominated by recent content — personas do NOT separate under it (2D eta^2 = 0.04, a blob) —
    whereas the mean over positions averages out content and keeps the persona style (eta^2 = 0.91).
    This is the cluster point; the trajectory analog is the running mean (cumulative_mean()).
    """
    n, h = ids.shape[0], model.config.hidden_size
    out = np.empty((n, h), dtype=np.float32)
    for s in range(0, n, batch_size):
        b_ids = ids[s:s + batch_size].to(device)
        b_attn = attn[s:s + batch_size].to(device)
        emb = models.prefix_embeddings(model, b_ids, b_attn, layer)         # [b, T, H]
        m = b_attn.unsqueeze(-1).float()                                    # [b, T, 1] real-token mask
        pooled = (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)         # [b, H] mean over real toks
        out[s:s + batch_size] = pooled.cpu().numpy()
    return out


def cumulative_mean(emb: np.ndarray) -> np.ndarray:
    """Running mean of per-position embeddings: out[t] = mean(emb[0..t]). [L, H] -> [L, H].

    The trajectory analog of the mean-pooled cluster point (sequence_embeddings): the prefix
    representation at position t comparable to a full story's mean. Single-position embeddings are
    content-noisy (see sequence_embeddings), so the running mean is what traces a smooth, persona-
    meaningful path that converges onto the story's cluster as t -> end.
    """
    return np.cumsum(emb, axis=0) / np.arange(1, emb.shape[0] + 1)[:, None]


@torch.no_grad()
def trajectory_embeddings(model: PreTrainedModel, ids: torch.LongTensor, attn: torch.LongTensor,
                          device: str, layer: int = -1) -> np.ndarray:
    """[N, T, H] the per-position prefix embedding e(t) for every position of each rollout.

    A single forward pass yields e(t) at every t (the model is causal). `attn` is the FORWARD mask
    (attend the seed [EOS] + any anchor); the caller decides which positions to actually plot. N is
    a handful of rollouts here, so no batching is needed.
    """
    emb = models.prefix_embeddings(model, ids.to(device), attn.to(device), layer)  # [N, T, H]
    return emb.cpu().numpy()


# --- PCA (numpy SVD; no sklearn dependency) -------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation to unit vectors (keeps direction, drops magnitude)."""
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), eps, None)


def fit_pca(x: np.ndarray, n_components: int = 2) -> dict:
    """Fit a `n_components`-D PCA on x [M, H] via SVD (centred). Returns mean, components, and the
    explained-variance ratio. The same projection is reused for the trajectory (project())."""
    assert x.ndim == 2 and x.shape[0] >= n_components, f"need >= {n_components} rows, got {x.shape}"
    mean = x.mean(axis=0)
    xc = x - mean
    _, sv, vt = np.linalg.svd(xc, full_matrices=False)
    total = float((sv ** 2).sum())
    evr = (sv[:n_components] ** 2) / total if total > 0 else np.zeros(n_components)
    return {"mean": mean, "components": vt[:n_components], "explained_variance_ratio": evr}


def project(x: np.ndarray, pca: dict) -> np.ndarray:
    """Project x [..., H] into the fitted PCA space -> [..., n_components]."""
    return (x - pca["mean"]) @ pca["components"].T


# --- Plotting -------------------------------------------------------------------------------

# Fixed colour per component, assigned by config.PERSONAS index (clusters are anonymous now, so
# colours come from the tab10 palette by position rather than a hand-picked name->colour map).
# Used by every Exp 5 plot. There is no 'base' component in this run (base == P_mix).
_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
COMPONENT_COLORS = {p: _TAB10[i % len(_TAB10)] for i, p in enumerate(config.PERSONAS)}


def _scatter_clusters(ax, cluster_proj: np.ndarray, cluster_names: list[str], components: list[str],
                      alpha: float = 0.18) -> None:
    """Faint colour-coded scatter of the reference clusters with a label at each centroid."""
    names = np.array(cluster_names)
    for comp in components:
        sel = names == comp
        if not sel.any():
            continue
        pts = cluster_proj[sel]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=alpha, color=COMPONENT_COLORS[comp],
                   linewidths=0, zorder=1)
        c = pts.mean(axis=0)
        ax.text(c[0], c[1], comp, fontsize=8, fontweight="bold", ha="center", va="center",
                color=COMPONENT_COLORS[comp], zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def plot_trajectory_grid(cluster_proj: np.ndarray, cluster_names: list[str], components: list[str],
                         trajectories: list[tuple[str, np.ndarray]], suptitle: str, path: str,
                         ncols: int = 3) -> None:
    """Small-multiples: one subplot per rollout, clusters in the background and the trajectory
    (proj coords [L, 2]) drawn on top, coloured by token position (time gradient). Start (the [EOS]
    seed) is a black circle, end a red star. Axis limits are shared across subplots."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(trajectories) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    xlim = (cluster_proj[:, 0].min(), cluster_proj[:, 0].max())
    ylim = (cluster_proj[:, 1].min(), cluster_proj[:, 1].max())
    for k, (title, traj) in enumerate(trajectories):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        _scatter_clusters(ax, cluster_proj, cluster_names, components)
        t = np.arange(traj.shape[0])
        ax.plot(traj[:, 0], traj[:, 1], "-", color="0.3", lw=0.8, alpha=0.6, zorder=2)
        ax.scatter(traj[:, 0], traj[:, 1], c=t, cmap="viridis", s=14, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=70, facecolor="none",
                   edgecolor="black", lw=1.6, zorder=5)          # start = [EOS] seed
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="*", s=160, color="red", zorder=5)  # end
        ax.set_title(title, fontsize=9)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trajectory_overlay(cluster_proj: np.ndarray, cluster_names: list[str], components: list[str],
                            trajectories: list[tuple[str, str, np.ndarray]], suptitle: str,
                            path: str) -> None:
    """All trajectories on one cluster background, each as a single-colour path with an end marker.
    trajectories = (label, color, proj[L,2]). The summary view (no time gradient)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    _scatter_clusters(ax, cluster_proj, cluster_names, components, alpha=0.12)
    for label, color, traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=1.4, alpha=0.9, zorder=2, label=label)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="*", s=120, color=color, zorder=3)
    ax.scatter(trajectories[0][2][0, 0], trajectories[0][2][0, 1], marker="o", s=90,
               facecolor="none", edgecolor="black", lw=1.8, zorder=4, label="start ([EOS])")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(suptitle, fontsize=12)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
