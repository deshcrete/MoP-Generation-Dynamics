"""Where do the mixture model's free generations sit in cluster-representation space?

Exp 5 draws the cluster reference clusters (real D_i stories embedded through P_mix) and traces ONE
rollout's prefix-embedding path across them. This script keeps the identical cluster background and
PCA, but instead of a trajectory it scatters the POPULATION of P_mix's own free generations — one
point per rollout, each the same mean-pooled embedding object as a cluster point
(embed_traj.sequence_embeddings).

The question it answers is the population version of Exp 5's single-rollout read: does free
generation's mass sit on one cluster, spread evenly, or fall between the clusters entirely? Exp 2
answers this from sequence log-probs (the hard-assignment histogram); this answers it from the
geometry of the representation, independently.

⚠ Note the difference from the persona-branch version of this script. Here **P_mix IS the base
model** (config.MIXTURE_REPO = SimpleStories-V2-5M) and there is no separate base component, so:
  - the clusters are the k real-D_i cluster clusters ONLY — there is no base cluster to add, and
  - the scattered generations ARE base-model free generations. On the persona branch those very
    samples were used to BUILD the base cluster; here they are the object under study. The plot
    therefore asks directly: does the base model's own output land on the cluster specialists that
    were fine-tuned from it, or between them?

Conventions are inherited from Exp 5 (see control/notes.md and context/results.md §Exp 5):
  - One embedding per sequence = the MEAN of P_mix's residual stream over its real tokens. NOT the
    last token: measured 2026-06-20, last-token embeddings do not separate personas (2D eta^2=0.04)
    while mean-pooling does (eta^2=0.91).
  - Free generations are embedded under the FORWARD mask (token_dist.forward_attention_mask), which
    attends the [EOS] seed at position 0 — matching how the D_i clusters (leading [EOS] from
    data.tokenize_stories) are embedded, so all points are the same object.
  - PCA is fit on the CLUSTER embeddings ONLY and the generations are projected into it. The
    clusters are the fixed reference frame; refitting on the generations would let them move it.
  - L2_NORMALIZE before PCA so residual-stream norm growth is not mistaken for movement.

Because PC1+PC2 capture only a fraction of the variance, 2D distances are illustrative, not metric.
So the plot ships with a quantitative companion, `nearest_centroid.csv`: each generation is assigned
to the nearest cluster centroid in the FULL normalised hidden space (not the 2D projection), giving
an honest "where is the mass" table that does not depend on the projection.

Run:  python -m src.mixture_gen_clusters
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.mixture_gen_clusters
Prereqs: Exp 2 (free_samples.npz) must have run.
Writes results/mixture_gen_clusters_<ts>/: config.json, params.json, pca_variance.csv,
nearest_centroid.csv, cluster_proj.npz, gen_proj.npz, mixture_gen_clusters.png.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config, data, embed_traj, models, token_dist
from .commitment import COMPONENTS
from .config import RunConfig

# --- parameters (module constants, logged into the run dir) ---------------------------------
# N_CLUSTER / EMBED_LAYER / L2_NORMALIZE / CLUSTER_BS / SEED_CLUSTER match src/run_exp5.py so the
# cluster background here is the SAME picture as Exp 5's, and the two figures can be read together.
N_CLUSTER = 300          # reference stories per cluster for the cluster background
EMBED_LAYER = -1         # residual-stream layer for embeddings (final, post-norm, feeds the LM head)
L2_NORMALIZE = True      # unit-normalise embeddings before PCA so positional norm-growth != movement
CLUSTER_BS = 128         # batch size for the embedding forward passes
SEED_CLUSTER = 123       # subsampling seed for the cluster stories
N_GEN = None             # how many Exp 2 free generations to scatter; None = all of them


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"mixture_gen_clusters_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"N_CLUSTER": N_CLUSTER, "EMBED_LAYER": EMBED_LAYER, "L2_NORMALIZE": L2_NORMALIZE,
                   "CLUSTER_BS": CLUSTER_BS, "SEED_CLUSTER": SEED_CLUSTER, "N_GEN": N_GEN}, f, indent=2)
    print(f"[gen_clusters] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)   # P_mix == the base model in this run

    def embed(model, ids, attn):  # mean-pooled sequence embedding under P_mix's residual stream
        return embed_traj.sequence_embeddings(model, ids, attn, cfg.device, EMBED_LAYER, CLUSTER_BS)

    def normed(x):                # the L2 step (logged constant) applied before PCA / projection
        return embed_traj.l2_normalize(x) if L2_NORMALIZE else x

    # =====================================================================================
    # Clusters — real D_i stories per cluster, embedded by P_mix (no base cluster: base == P_mix).
    # Identical construction to run_exp5.py, so the background is the same reference frame.
    # =====================================================================================
    print(f"[gen_clusters] building clusters: {N_CLUSTER} D_i stories/cluster ...")
    stories = data.load_persona_stories()
    rng = np.random.default_rng(SEED_CLUSTER)
    cluster_emb_list, cluster_names = [], []
    for persona in config.PERSONAS:
        pool = stories[persona]
        assert len(pool) >= N_CLUSTER, f"{persona}: only {len(pool)} stories, need {N_CLUSTER}"
        chosen = rng.choice(len(pool), size=N_CLUSTER, replace=False)
        ids, attn = data.tokenize_stories([pool[int(c)] for c in chosen], tok, cfg.data)
        cluster_emb_list.append(embed(mixture_model, ids, attn))     # [N_CLUSTER, H]
        cluster_names += [persona] * N_CLUSTER

    cluster_emb = np.concatenate(cluster_emb_list, axis=0)           # [M, H]
    cluster_norm = normed(cluster_emb)
    pca = embed_traj.fit_pca(cluster_norm, n_components=2)
    cluster_proj = embed_traj.project(cluster_norm, pca)             # [M, 2]
    evr = pca["explained_variance_ratio"]
    print(f"[gen_clusters] PCA on {cluster_emb.shape[0]} cluster embeddings; "
          f"explained variance PC1={evr[0]:.3f} PC2={evr[1]:.3f} (sum={evr.sum():.3f})")
    pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": evr}).to_csv(
        os.path.join(out_dir, "pca_variance.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "cluster_proj.npz"),
                        proj=cluster_proj, names=np.array(cluster_names))

    # =====================================================================================
    # The mixture's OWN free generations — one mean-pooled point each, projected into the same PCA
    # =====================================================================================
    exp2_dir = _latest("exp2_*")
    npz = np.load(os.path.join(exp2_dir, "free_samples.npz"))
    free = torch.tensor(npz["samples"])
    if N_GEN is not None:
        free = free[:N_GEN]
    # forward mask (attends the [EOS] seed), matching how the clusters above were embedded — the
    # npz's own `attn` is the SCORING mask (start=1) and would drop position 0 from the mean.
    free_fwd = token_dist.forward_attention_mask(free)
    print(f"[gen_clusters] embedding {free.shape[0]} P_mix free generations from {os.path.basename(exp2_dir)} ...")
    gen_emb = embed(mixture_model, free, free_fwd)                   # [N, H]
    gen_norm = normed(gen_emb)
    gen_proj = embed_traj.project(gen_norm, pca)                     # [N, 2]
    np.savez_compressed(os.path.join(out_dir, "gen_proj.npz"), proj=gen_proj)

    embed_traj.plot_generation_scatter(
        cluster_proj, cluster_names, COMPONENTS, gen_proj,
        "P_mix (=base) free generations over the cluster clusters (PCA of D_i, mean-pooled)",
        os.path.join(out_dir, "mixture_gen_clusters.png"))

    # =====================================================================================
    # Quantitative companion: nearest cluster centroid in the FULL hidden space (not the 2D plot,
    # whose PC1+PC2 hold only a fraction of the variance — see context/results.md §Caveats).
    # =====================================================================================
    names = np.array(cluster_names)
    centroids = np.stack([cluster_norm[names == c].mean(axis=0) for c in COMPONENTS])  # [C, H]
    d = np.linalg.norm(gen_norm[:, None, :] - centroids[None, :, :], axis=2)           # [N, C]
    nearest = d.argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(COMPONENTS))
    pd.DataFrame({"component": COMPONENTS, "n_nearest": counts,
                  "frac_nearest": counts / counts.sum(),
                  "mean_dist_to_centroid": [float(d[nearest == i, i].mean()) if counts[i] else float("nan")
                                            for i in range(len(COMPONENTS))]}).to_csv(
        os.path.join(out_dir, "nearest_centroid.csv"), index=False)
    print("[gen_clusters] nearest-centroid share of free generations (full-H space):")
    for i, c in enumerate(COMPONENTS):
        print(f"[gen_clusters]   {c:22s} {counts[i]:4d}  ({counts[i] / counts.sum():.3f})")

    print(f"\n[gen_clusters] done. results in {out_dir}")


if __name__ == "__main__":
    main()
