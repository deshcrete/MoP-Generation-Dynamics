"""Per-completion diagnostic gallery — one organised page-section per free generation.

Samples N_SAMPLE free generations and, for EACH, lays out (so nothing is crammed onto one axis):
  - a 2-panel figure: (left) the cumulative posterior gamma_i(t) [dashed] and the probe belief
    beta_i(t) [solid] per cluster; (right) the prefix-embedding TRAJECTORY through the PCA of the
    real-D_i clusters (time-coloured path, start=[EOS] seed, end=star);
  - highlighted completion text three ways: ① gamma (cumulative posterior of the dominant cluster),
    ② r (per-token responsibility of the dominant cluster), ③ argmax-by-gamma per token (colour =
    running winner).
Plus an INDEX with a combined overlay of ALL N_SAMPLE trajectories (coloured by dominant cluster).

The running-mean L2 embedding feeds BOTH the probe and the trajectory projection (one forward pass).
The probe is the Exp-7 running-mean probe (smooth, directly comparable to the cumulative gamma).

Run:  CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_gallery
Prereqs: exp7_* (probe.npz), exp2_* (free_samples.npz).
Writes results/gallery_<ts>/: report.html (index), page_*.html, comp_*.png, all_trajectories.png,
completions.csv, config.json.
"""
from __future__ import annotations

import glob
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, data, embed_traj, models, probe, token_dist
from . import run_exp6 as ex6
from .commitment import COMPONENTS
from .config import RunConfig
from .embed_traj import COMPONENT_COLORS

N_SAMPLE = 100            # free completions to render
N_CLUSTER = 300          # real D_i stories per cluster for the embedding background
N_PER_PAGE = 20          # completions per HTML page
EMBED_LAYER, L2, FEAT_BS, CLUSTER_BS = -1, True, 64, 128


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _load_probe(path: str) -> probe.MultinomialProbe:
    z = np.load(path, allow_pickle=True)
    assert [str(c) for c in z["classes"]] == list(COMPONENTS), f"probe class order mismatch in {path}"
    return probe.MultinomialProbe(z["mu"], z["sd"], z["W"], z["b"], final_loss=float("nan"))


def _comp_figure(gamma_s, beta_s, length, traj, cluster_proj, cluster_names, dom, path):
    """2-panel figure for one completion: left gamma/beta curves, right embedding trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (axc, axt) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    t = np.arange(length)
    for i, name in enumerate(COMPONENTS):
        col = COMPONENT_COLORS[name]
        lw = 2.4 if name == dom else 1.1
        axc.plot(t, beta_s[:length, i], "-", lw=lw, color=col, alpha=0.95)
        axc.plot(t, gamma_s[i, :length], "--", lw=lw * 0.8, color=col, alpha=0.8)
    axc.set_ylim(0, 1); axc.set_xlabel("token position t"); axc.set_ylabel("probability")
    axc.set_title(r"posterior $\gamma$ (dashed) vs probe $\beta$ (solid)", fontsize=10)
    comp_h = [Line2D([0], [0], color=COMPONENT_COLORS[n], lw=2, label=n) for n in COMPONENTS]
    style_h = [Line2D([0], [0], color="0.3", lw=2, ls="-", label=r"$\beta$ probe"),
               Line2D([0], [0], color="0.3", lw=2, ls="--", label=r"$\gamma$ posterior")]
    axc.legend(handles=comp_h + style_h, fontsize=7, ncol=2, loc="upper left")

    # trajectory over the cluster background
    embed_traj._scatter_clusters(axt, cluster_proj, cluster_names, COMPONENTS)
    tt = np.arange(traj.shape[0])
    axt.plot(traj[:, 0], traj[:, 1], "-", color="0.3", lw=0.8, alpha=0.6, zorder=2)
    axt.scatter(traj[:, 0], traj[:, 1], c=tt, cmap="viridis", s=16, zorder=3)
    axt.scatter(traj[0, 0], traj[0, 1], marker="o", s=70, facecolor="none", edgecolor="black",
                lw=1.6, zorder=5)
    axt.scatter(traj[-1, 0], traj[-1, 1], marker="*", s=170, color="red", zorder=5)
    axt.set_xlim(cluster_proj[:, 0].min(), cluster_proj[:, 0].max())
    axt.set_ylim(cluster_proj[:, 1].min(), cluster_proj[:, 1].max())
    axt.set_xlabel("PC1"); axt.set_ylabel("PC2")
    axt.set_title("prefix-embedding trajectory (time-coloured)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _overview(cluster_proj, cluster_names, trajs, doms, path):
    """All N_SAMPLE trajectories on one cluster background, coloured by dominant cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 9))
    embed_traj._scatter_clusters(ax, cluster_proj, cluster_names, COMPONENTS, alpha=0.10)
    for traj, dom in zip(trajs, doms):
        col = COMPONENT_COLORS[dom]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=col, lw=0.7, alpha=0.35, zorder=2)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="*", s=40, color=col, alpha=0.8, zorder=3)
    ax.scatter(trajs[0][0, 0], trajs[0][0, 1], marker="o", s=110, facecolor="none",
               edgecolor="black", lw=2, zorder=5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"All {len(trajs)} free-generation trajectories (coloured by dominant cluster)")
    ax.legend(handles=[Line2D([0], [0], color=COMPONENT_COLORS[n], lw=2, label=n) for n in COMPONENTS]
              + [Line2D([0], [0], marker="o", color="black", lw=0, mfc="none", label="start ([EOS])")],
              fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _text_block(s, dom, ci, final_g, n, pieces, g_focus, r_focus, am_names, am_vals, tips, img):
    focus_col = COMPONENT_COLORS[dom]; m = len(pieces)
    g_line = ex6._highlight_line(pieces, g_focus, [focus_col] * m, tips)
    r_line = ex6._highlight_line(pieces, r_focus, [focus_col] * m, tips)
    a_line = ex6._highlight_line(pieces, am_vals, [COMPONENT_COLORS[a] for a in am_names], tips)
    ts = ('font-family:Georgia,serif;font-size:16px;line-height:2.0;background:#fafafa;'
          'padding:10px;border:1px solid #eee;border-radius:4px;')
    return f"""
    <div style="margin:26px 0;border-top:2px solid #ddd;padding-top:14px;">
      <h3>completion {s} <span style="font-weight:normal;color:#666;font-size:14px;">
        (dominant {dom}, final γ={final_g:.2f}, {n} tokens)</span></h3>
      <img src="{img}" style="max-width:1000px;width:100%;"/>
      <p style="margin:10px 0 2px;font-weight:bold;">① cumulative posterior γ_{dom}(t)</p>
      <div style="{ts}">{g_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">② per-token responsibility r_{dom}(t)</p>
      <div style="{ts}">{r_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">③ argmax cluster per token (by γ)</p>
      <div style="{ts}">{a_line}</div>
    </div>"""


def main() -> None:
    cfg = RunConfig(); cfg.device = models.resolve_device(cfg.device)
    out = os.path.join(config.RESULTS_DIR, f"gallery_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True); cfg.to_json(os.path.join(out, "config.json"))
    print(f"[gallery] device={cfg.device}  out={out}")

    tok = models.load_tokenizer()
    mix = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    clf = _load_probe(os.path.join(_latest("exp7_[0-9]*"), "probe.npz"))

    # --- cluster background + PCA (real D_i, mean-pooled through P_mix) ---------------------
    print(f"[gallery] building cluster background: {N_CLUSTER} D_i stories/cluster ...")
    stories = data.load_persona_stories(); rng = np.random.default_rng(123)
    emb_list, cluster_names = [], []
    for p in COMPONENTS:
        chosen = rng.choice(len(stories[p]), size=N_CLUSTER, replace=False)
        ids, attn = data.tokenize_stories([stories[p][int(c)] for c in chosen], tok, cfg.data)
        emb_list.append(embed_traj.sequence_embeddings(mix, ids, attn, cfg.device, EMBED_LAYER, CLUSTER_BS))
        cluster_names += [p] * N_CLUSTER
    cluster_emb = embed_traj.l2_normalize(np.concatenate(emb_list, axis=0))
    pca = embed_traj.fit_pca(cluster_emb, 2)
    cluster_proj = embed_traj.project(cluster_emb, pca)
    print(f"[gallery] PCA EVR PC1={pca['explained_variance_ratio'][0]:.3f} "
          f"PC2={pca['explained_variance_ratio'][1]:.3f}")

    # --- sample completions, score gamma/r (generative) + beta (probe) ---------------------
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"])[:N_SAMPLE]
    free_attn = torch.tensor(npz["attn"])[:N_SAMPLE]
    fwd = token_dist.forward_attention_mask(free); fwd_len = fwd.sum(dim=1).numpy().astype(int)
    print(f"[gallery] scoring {N_SAMPLE} completions: gamma/r (specialists) + beta (probe) + trajectories ...")
    logp = commitment.per_model_token_logprobs(persona_models, free, free_attn, cfg.device, cfg.gen.batch_size)
    pi = commitment.uniform_prior()
    gamma = commitment.cumulative_posterior(logp, pi)              # [N, C, T]
    r = commitment.token_responsibility(logp, pi)                 # [N, C, T]
    e_bar = probe.prefix_running_mean(mix, free, fwd, cfg.device, EMBED_LAYER, L2, FEAT_BS)  # [N,T,H]
    beta = clf.predict_proba(e_bar)                               # [N, T, C]

    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = valid_g.shape[1] - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    final_gamma = gamma[np.arange(N_SAMPLE), :, last_t]
    dom_idx = final_gamma.argmax(axis=1)
    argmax_g = np.nanargmax(np.where(np.isnan(gamma), -np.inf, gamma), axis=1)  # [N, T]

    # --- per-completion figures + text blocks ----------------------------------------------
    trajs, rows, blocks = [], [], []
    for s in range(N_SAMPLE):
        ci = int(dom_idx[s]); dom = COMPONENTS[ci]
        Lf = int(fwd_len[s]); Lg = int(last_t[s]) + 1
        traj = embed_traj.project(e_bar[s, :Lf], pca)             # [Lf, 2] reuse running-mean L2 emb
        trajs.append(traj)
        img = f"comp_{s:03d}.png"
        _comp_figure(gamma[s], beta[s], Lg, traj, cluster_proj, cluster_names, dom,
                     os.path.join(out, img))

        positions = [t for t in range(1, Lg)
                     if int(free[s, t]) != config.EOS_ID and bool(valid_g[s, t])]
        pieces = ex6._token_pieces([int(free[s, t]) for t in positions], tok)
        g_focus = [float(gamma[s, ci, t]) for t in positions]
        r_focus = [float(r[s, ci, t]) for t in positions]
        am_names = [COMPONENTS[int(argmax_g[s, t])] for t in positions]
        am_vals = [float(gamma[s, int(argmax_g[s, t]), t]) for t in positions]
        tips = [f"t={t}  γ_{dom}={gamma[s,ci,t]:.2f}  r_{dom}={r[s,ci,t]:.2f}  "
                f"β_{dom}={beta[s,t,ci]:.2f}  γ_argmax={COMPONENTS[int(argmax_g[s,t])]}" for t in positions]
        blocks.append((s, _text_block(s, dom, ci, float(final_gamma[s, ci]), len(pieces),
                                       pieces, g_focus, r_focus, am_names, am_vals, tips, img)))
        text = "".join((" " if sw else "") + tp for tp, sw in pieces).strip()
        rows.append({"completion": s, "dominant": dom, "final_gamma": float(final_gamma[s, ci]),
                     "n_tokens": len(pieces), "text": text})
        if (s + 1) % 25 == 0:
            print(f"[gallery]  rendered {s + 1}/{N_SAMPLE}")

    pd.DataFrame(rows).to_csv(os.path.join(out, "completions.csv"), index=False)
    _overview(cluster_proj, cluster_names, trajs, [COMPONENTS[i] for i in dom_idx],
              os.path.join(out, "all_trajectories.png"))

    # --- paginated HTML --------------------------------------------------------------------
    n_pages = math.ceil(N_SAMPLE / N_PER_PAGE)

    def nav(pg):
        links = [f'<a href="report.html">index</a>']
        if pg > 0:
            links.append(f'<a href="page_{pg-1:02d}.html">← prev</a>')
        if pg < n_pages - 1:
            links.append(f'<a href="page_{pg+1:02d}.html">next →</a>')
        return '<p>' + ' &nbsp;|&nbsp; '.join(links) + '</p>'

    for pg in range(n_pages):
        sl = blocks[pg * N_PER_PAGE:(pg + 1) * N_PER_PAGE]
        body = (f'<h1>Completion gallery — page {pg+1}/{n_pages}</h1>' + nav(pg)
                + ex6._legend_html() + "".join(b for _, b in sl) + nav(pg))
        with open(os.path.join(out, f"page_{pg:02d}.html"), "w") as f:
            f.write(ex6._page(f"gallery p{pg+1}", body))

    counts = pd.Series([COMPONENTS[i] for i in dom_idx]).value_counts().reindex(COMPONENTS, fill_value=0)
    dom_tbl = "".join(f'<li><span style="color:{COMPONENT_COLORS[c]};font-weight:bold;">{c}</span>: '
                      f'{int(counts[c])} completions</li>' for c in COMPONENTS)
    page_links = "".join(f'<a href="page_{pg:02d}.html">page {pg+1}</a> &nbsp; ' for pg in range(n_pages))
    index = (f'<h1>Per-completion diagnostic gallery</h1>'
             f'<p>{N_SAMPLE} free generations from P_mix. For each: posterior γ vs probe β curves, '
             f'the prefix-embedding trajectory through the PCA of the real-D_i clusters, and the '
             f'completion text highlighted by γ / r / argmax-by-γ.</p>'
             f'<h2>All {N_SAMPLE} trajectories</h2>'
             f'<img src="all_trajectories.png" style="max-width:820px;width:100%;"/>'
             f'<h2>Dominant-cluster counts (argmax final γ)</h2><ul>{dom_tbl}</ul>'
             f'<h2>Pages</h2><p>{page_links}</p>' + ex6._legend_html())
    with open(os.path.join(out, "report.html"), "w") as f:
        f.write(ex6._page("completion gallery", index))

    print(f"\n[gallery] done. open {os.path.join(out, 'report.html')}")


if __name__ == "__main__":
    main()
