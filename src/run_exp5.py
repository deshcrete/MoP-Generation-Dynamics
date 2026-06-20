"""Experiment 5 — Persona commitment via prefix-embedding trajectories.

Plots fixed persona reference clusters (real dataset stories D_i + base-model free generations for
base, all embedded through P_mix) and, on top of them, the mixture model's prefix embedding e(t) at
each token position of a rollout — a path through cluster space that makes commitment (and its
speed) visible. The geometric companion to Exp 3/4's gamma/w curves. A per-token cumulative-LLR
plot pairs the qualitative trajectory with a quantitative "speed of updates" view.

Run:  python -m src.run_exp5
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp5
Prereqs: Exp 1 (triggers.json) and Exp 2 (free_samples.npz) must have run.
Writes results/exp5_<ts>/: config.json, exp5_params.json, pca_variance.csv, cluster_proj.npz,
trajectories.npz, free_trajectories.png, anchored_trajectories.png, trajectories_overlay.png,
llr_trajectories.png. See context/design_doc.md §4 Experiment 5.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, data, embed_traj, generate, models, token_dist
from .commitment import COMPONENTS
from .config import RunConfig

# --- Exp 5 parameters (module constants, logged into the run dir) ---------------------------
N_CLUSTER = 300          # reference stories per persona for the clusters (base = base free-gens)
EMBED_LAYER = -1         # residual-stream layer for embeddings (final, post-norm, feeds the LM head)
L2_NORMALIZE = True      # unit-normalise embeddings before PCA so positional norm-growth != movement
ANCHOR_VARIANT = "entry"  # single-token entry triggers for the anchored rollouts (aligned at t=1)
N_ANCHORED_GEN = 4       # anchored samples generated per persona; we plot the first
CLUSTER_BS = 128         # batch size for the cluster embedding forward passes
SEED_CLUSTER = 123       # subsampling seed for the cluster stories


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _valid_len(fwd_mask_row: np.ndarray) -> int:
    """Number of real (attended) tokens in a forward-mask row whose 1s are a contiguous prefix."""
    return int(fwd_mask_row.sum())


def _cumulative_llr(logp: np.ndarray) -> np.ndarray:
    """Cumulative log-likelihood-ratio of each component vs base, [k+1, T] from logp [1, k+1, T].

    LLR_i(t) = sum_{s<=t} ( logp_i[s] - logp_base[s] ); the base row is identically 0 (reference).
    nan positions (seq position 0, padding) contribute 0 to the running sum, mirroring
    commitment.cumulative_posterior. An UNBOUNDED evidence-rate view (cf. Exp 3's bounded gamma):
    a steep climb = high-frequency/spiky updates (triggered), a gentle slope = gradual (behavioural).
    """
    base = logp[:, -1:, :]                                  # [1, 1, T] base log-probs
    diff = logp - base                                     # [1, k+1, T]; base row -> 0
    contrib = np.where(np.isnan(diff), 0.0, diff)
    return np.cumsum(contrib, axis=2)[0]                   # [k+1, T]


def _plot_llr_grid(items: list[tuple[str, np.ndarray, int]], suptitle: str, path: str,
                   ncols: int = 3) -> None:
    """Small-multiples of cumulative-LLR-vs-base curves; one subplot per rollout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (title, llr, length) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        t = np.arange(length)
        for i, name in enumerate(COMPONENTS):
            style = "--" if name == "base" else "-"
            ax.plot(t, llr[i, :length], style, lw=1.4, label=name,
                    color=embed_traj.COMPONENT_COLORS[name])
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("t", fontsize=8); ax.set_ylabel("cumulative LLR vs base", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(COMPONENTS), fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp5_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "exp5_params.json"), "w") as f:
        json.dump({"N_CLUSTER": N_CLUSTER, "EMBED_LAYER": EMBED_LAYER, "L2_NORMALIZE": L2_NORMALIZE,
                   "ANCHOR_VARIANT": ANCHOR_VARIANT, "N_ANCHORED_GEN": N_ANCHORED_GEN,
                   "CLUSTER_BS": CLUSTER_BS, "SEED_CLUSTER": SEED_CLUSTER}, f, indent=2)
    print(f"[exp5] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    base_model = models.load_base_model(cfg.device)

    def embed(model, ids, attn):  # cluster/sequence embedding under P_mix's residual stream
        return embed_traj.sequence_embeddings(model, ids, attn, cfg.device, EMBED_LAYER, CLUSTER_BS)

    def normed(x):                # the L2 step (logged constant) applied before PCA / projection
        return embed_traj.l2_normalize(x) if L2_NORMALIZE else x

    # =====================================================================================
    # Clusters — real D_i stories (+ base-model free generations for base), embedded by P_mix
    # =====================================================================================
    print(f"[exp5] building clusters: {N_CLUSTER} D_i stories/persona + {N_CLUSTER} base free-gens ...")
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

    # base cluster: base model B has no dataset, so use B's own free generations (embedded by P_mix)
    base_gen_cfg = dataclasses.replace(cfg.gen, n_samples=N_CLUSTER)
    base_samples = generate.free_generate(base_model, tok, base_gen_cfg, cfg.device)
    base_fwd = token_dist.forward_attention_mask(base_samples)
    cluster_emb_list.append(embed(mixture_model, base_samples, base_fwd))
    cluster_names += ["base"] * N_CLUSTER

    cluster_emb = np.concatenate(cluster_emb_list, axis=0)           # [M, H]
    pca = embed_traj.fit_pca(normed(cluster_emb), n_components=2)
    cluster_proj = embed_traj.project(normed(cluster_emb), pca)      # [M, 2]
    evr = pca["explained_variance_ratio"]
    print(f"[exp5] PCA on {cluster_emb.shape[0]} cluster embeddings; "
          f"explained variance PC1={evr[0]:.3f} PC2={evr[1]:.3f} (sum={evr.sum():.3f})")
    pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": evr}).to_csv(
        os.path.join(out_dir, "pca_variance.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "cluster_proj.npz"),
                        proj=cluster_proj, names=np.array(cluster_names))

    def project_trajectory(model, sample_row: torch.LongTensor) -> np.ndarray:
        """Prefix-embedding path of one rollout [1, T] through the fitted PCA. Forward-attend the
        real tokens (incl. the [EOS] seed), keep positions 0..last real token, take the RUNNING MEAN
        (the smooth trajectory analog of the mean-pooled cluster point — single-position embeddings
        are content-noisy), normalise, project -> [L, 2]."""
        fwd = token_dist.forward_attention_mask(sample_row)
        emb = embed_traj.trajectory_embeddings(model, sample_row, fwd, cfg.device, EMBED_LAYER)[0]
        length = _valid_len(fwd[0].numpy())
        running = embed_traj.cumulative_mean(emb[:length])           # [length, H]
        return embed_traj.project(normed(running), pca)              # [length, 2]

    traj_store: dict[str, np.ndarray] = {}

    # =====================================================================================
    # Free trajectories — one rollout per dominant component (selected as in Exp 4B)
    # =====================================================================================
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"]); free_attn = torch.tensor(npz["attn"])
    print(f"[exp5] scoring {free.shape[0]} free rollouts to pick one per dominant component ...")
    free_logp = commitment.per_model_token_logprobs(persona_models, base_model, free, free_attn,
                                                    cfg.device, cfg.gen.batch_size)
    gamma = commitment.cumulative_posterior(free_logp, commitment.uniform_prior())   # [n, k+1, T]
    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = valid_g.shape[1] - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    final_gamma = gamma[np.arange(gamma.shape[0]), :, last_t]                         # [n, k+1]

    free_trajs = []
    for ci, name in enumerate(COMPONENTS):
        s = int(np.argmax(final_gamma[:, ci]))
        traj = project_trajectory(mixture_model, free[s:s + 1])
        title = f"free — dominant: {name} (gamma={final_gamma[s, ci]:.2f})"
        free_trajs.append((title, traj))
        traj_store[f"free_{name}"] = traj
    embed_traj.plot_trajectory_grid(cluster_proj, cluster_names, COMPONENTS, free_trajs,
                                    "Exp 5 — FREE prefix-embedding trajectories (P_mix, PCA of D_i)",
                                    os.path.join(out_dir, "free_trajectories.png"))

    # =====================================================================================
    # Anchored trajectories — one rollout per persona, entry trigger (+ LLR companion)
    # =====================================================================================
    anchors = {p: json.load(open(os.path.join(_latest("exp1_*"), "triggers.json")))
               ["anchors"][p][ANCHOR_VARIANT]["token_ids"] for p in config.PERSONAS}
    anchored_gen_cfg = dataclasses.replace(cfg.gen, n_samples=len(config.PERSONAS) * N_ANCHORED_GEN)
    print(f"[exp5] regenerating anchored ({ANCHOR_VARIANT}) rollouts and tracing trajectories ...")
    gen = generate.anchored_generate(mixture_model, tok, anchors, anchored_gen_cfg, cfg.device)

    anchored_trajs, llr_items = [], []
    for p in config.PERSONAS:
        row = gen[p][:1]
        tokstr = tok.convert_ids_to_tokens(int(anchors[p][0])) if anchors[p] else "(empty)"
        title = f"anchored: {p} (`{tokstr}`)"
        traj = project_trajectory(mixture_model, row)
        anchored_trajs.append((title, traj))
        traj_store[f"anchored_{p}"] = traj

        # LLR companion: attend the real tokens (incl. seed + anchor), score every component vs base
        fwd = token_dist.forward_attention_mask(row)
        logp = commitment.per_model_token_logprobs(persona_models, base_model, row, fwd,
                                                   cfg.device, cfg.gen.batch_size)
        llr = _cumulative_llr(logp)
        llr_items.append((title, llr, _valid_len(fwd[0].numpy())))

    embed_traj.plot_trajectory_grid(cluster_proj, cluster_names, COMPONENTS, anchored_trajs,
                                    "Exp 5 — ANCHORED-by-i prefix-embedding trajectories",
                                    os.path.join(out_dir, "anchored_trajectories.png"))
    _plot_llr_grid(llr_items, "Exp 5 — anchored cumulative LLR vs base (speed of updates)",
                   os.path.join(out_dir, "llr_trajectories.png"))

    # =====================================================================================
    # Overlay — all anchored trajectories + the base-dominant free one, on one cluster picture
    # =====================================================================================
    overlay = [(f"anchored {p}", embed_traj.COMPONENT_COLORS[p], traj_store[f"anchored_{p}"])
               for p in config.PERSONAS]
    overlay.append(("free (base-dominant)", "black", traj_store["free_base"]))
    embed_traj.plot_trajectory_overlay(cluster_proj, cluster_names, COMPONENTS, overlay,
                                       "Exp 5 — trajectory overlay (anchored vs free)",
                                       os.path.join(out_dir, "trajectories_overlay.png"))

    np.savez_compressed(os.path.join(out_dir, "trajectories.npz"), **traj_store)
    print(f"\n[exp5] done. results in {out_dir}")


if __name__ == "__main__":
    main()
