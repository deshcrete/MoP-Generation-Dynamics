"""Ground-truth commitment — does the posterior classify real per-cluster stories correctly?

Exp 3/6 ran the commitment machinery (cumulative posterior gamma_i(t), per-token responsibility
r_i(t)) on P_mix's FREE generations, which belong to no cluster cleanly. This script runs the SAME
machinery on the DATASET's ground-truth labelled stories D_i — the examples that, by construction,
DO belong to a known cluster — and asks the direct question: does the posterior commit each story to
its TRUE cluster?

For a labelled story x from cluster c (label never shown to the scorer):
  - logp_j[t]      = log P_j(x_t | x_{<t})  under every cluster specialist j         (commitment)
  - gamma_i(t)     = softmax_i( log pi_i + sum_{s<=t} logp_i[s] )   cumulative posterior
  - r_i(t)         = softmax_i( log pi_i + logp_i[t] )             per-token responsibility
  - prediction     = argmax_i gamma_i(T)   (final cumulative posterior; uniform prior so this is
                     exactly the whole-sequence MLE label — the EM hard-assignment of one story)

Outputs (results/groundtruth_commit_<ts>/):
  - classification_summary.csv : per story  (true, pred, correct, final_gamma_true, n_tokens)
  - confusion.csv / confusion.png : true cluster x predicted cluster, with overall + per-cluster acc
  - selfposterior_curves.png   : mean gamma_c(t) of each cluster on ITS OWN stories (does it rise?)
  - selfresponsibility_curves.png : mean r_c(t) likewise
  - report.html + <cluster>.html : Exp-6-style highlighted text per story — ① r_true(t),
    ② gamma_true(t), ③ argmax-by-gamma per token (which cluster the running posterior assigns) —
    plus a per-story gamma curve. Lets you read token-by-token WHERE the posterior locks on (or fails).

Run:  CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_groundtruth_commit
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, data, models
from . import run_exp6 as ex6                 # reuse the highlight machinery (no parallel system)
from .commitment import COMPONENTS
from .config import RunConfig
from .embed_traj import COMPONENT_COLORS

CLASSIFY_PER_PERSONA = 100     # labelled stories per cluster used for the confusion/accuracy stats
RENDER_PER_PERSONA = 6         # stories per cluster rendered in the HTML (first by sampling order)
PLOT_T = 64                    # mean-curve horizon (most stories are short)


def _confusion_png(conf: np.ndarray, acc: float, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(COMPONENTS))); ax.set_xticklabels(COMPONENTS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(COMPONENTS))); ax.set_yticklabels(COMPONENTS, fontsize=8)
    for i in range(len(COMPONENTS)):
        for j in range(len(COMPONENTS)):
            ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if conf[i, j] < 0.5 else "black")
    ax.set_xlabel("predicted (argmax final γ)"); ax.set_ylabel("true cluster")
    ax.set_title(f"Ground-truth posterior classification — acc = {acc:.3f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _mean_self_curves(arr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[k, T] where row c = mean over cluster-c's stories of arr[story, c, t] (the TRUE component's
    curve on its own data). nan-aware; tails shrink as stories end."""
    import warnings
    k = len(COMPONENTS); T = arr.shape[2]
    out = np.full((k, T), np.nan)
    for c in range(k):
        sel = labels == c
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[c] = np.nanmean(arr[sel, c, :], axis=0)
    return out


def _curve_plot(curves: np.ndarray, title: str, ylabel: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(min(PLOT_T, curves.shape[1]))
    for i, name in enumerate(COMPONENTS):
        ax.plot(t, curves[i, :len(t)], lw=2.2, color=COMPONENT_COLORS[name],
                label=f"{name} (own stories)")
    ax.axhline(1.0 / len(COMPONENTS), ls=":", color="0.5", lw=1, label="chance")
    ax.set_xlabel("token position t"); ax.set_ylabel(ylabel); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _story_block(persona: str, k: int, correct: bool, pred: str, final_g: float,
                 pieces, r_focus, g_focus, argmax_names, argmax_vals, tooltips, img_name: str) -> str:
    """Exp-6-style block for one ground-truth story: ① r_true ② γ_true ③ argmax-by-γ per token."""
    focus_col = COMPONENT_COLORS[persona]
    n = len(pieces)
    r_line = ex6._highlight_line(pieces, r_focus, [focus_col] * n, tooltips)
    g_line = ex6._highlight_line(pieces, g_focus, [focus_col] * n, tooltips)
    a_line = ex6._highlight_line(pieces, argmax_vals, [COMPONENT_COLORS[a] for a in argmax_names], tooltips)
    text_style = ('font-family:Georgia,serif;font-size:16px;line-height:2.0;'
                  'background:#fafafa;padding:10px;border:1px solid #eee;border-radius:4px;')
    verdict = (f'<span style="color:#0a0;font-weight:bold;">✓ correct</span>' if correct
               else f'<span style="color:#c00;font-weight:bold;">✗ misclassified → {pred}</span>')
    return f"""
    <div style="margin:26px 0;border-top:2px solid #ddd;padding-top:14px;">
      <h3>{persona} — story {k} <span style="font-weight:normal;color:#666;font-size:14px;">
        ({verdict}, final γ_{persona} = {final_g:.2f}, {n} tokens)</span></h3>
      <img src="{img_name}" style="max-width:760px;width:100%;"/>
      <p style="margin:10px 0 2px;font-weight:bold;">① per-token responsibility r_{persona}(t)
        — where the TRUE cluster fires on each token</p>
      <div style="{text_style}">{r_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">② cumulative posterior γ_{persona}(t)
        — running commitment to the TRUE cluster</p>
      <div style="{text_style}">{g_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">③ argmax cluster per token (by γ)
        — which cluster the running posterior assigns (colour = winner; should converge to {persona})</p>
      <div style="{text_style}">{a_line}</div>
    </div>"""


def main() -> None:
    cfg = RunConfig(); cfg.device = models.resolve_device(cfg.device)
    out = os.path.join(config.RESULTS_DIR, f"groundtruth_commit_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True); cfg.to_json(os.path.join(out, "config.json"))
    print(f"[gt] device={cfg.device}  out={out}")

    tok = models.load_tokenizer()
    persona_models = models.load_persona_models(cfg.device)
    pi = commitment.uniform_prior()

    # --- ground-truth labelled stories (CLASSIFY_PER_PERSONA per cluster) -------------------
    dcfg = data.DataConfig(t_max=cfg.data.t_max, prepend_eos=True, append_eos=False,
                           seed=cfg.data.seed, inference_per_persona=CLASSIFY_PER_PERSONA)
    by = data.load_persona_stories()
    stories, labels = data.build_uniform_inference_set(by, dcfg)
    ids, attn = data.tokenize_stories(stories, tok, dcfg)
    print(f"[gt] scoring {len(stories)} ground-truth stories "
          f"({CLASSIFY_PER_PERSONA}/cluster) under {len(COMPONENTS)} specialists ...")

    logp = commitment.per_model_token_logprobs(persona_models, ids, attn, cfg.device, cfg.gen.batch_size)
    gamma = commitment.cumulative_posterior(logp, pi)          # [N, k, T]
    resp = commitment.token_responsibility(logp, pi)           # [N, k, T]

    # --- classification: argmax final gamma per story --------------------------------------
    n, _, T = gamma.shape
    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = T - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    final_gamma = gamma[np.arange(n), :, last_t]               # [N, k]
    pred = final_gamma.argmax(axis=1)
    correct = pred == labels

    conf = np.zeros((len(COMPONENTS), len(COMPONENTS)))
    for c in range(len(COMPONENTS)):
        sel = labels == c
        for j in range(len(COMPONENTS)):
            conf[c, j] = float((pred[sel] == j).mean())
    acc = float(correct.mean())

    summ = pd.DataFrame({
        "true": [COMPONENTS[c] for c in labels], "pred": [COMPONENTS[c] for c in pred],
        "correct": correct, "final_gamma_true": final_gamma[np.arange(n), labels],
        "n_tokens": (last_t).astype(int),
    })
    summ.to_csv(os.path.join(out, "classification_summary.csv"), index=False)
    pd.DataFrame(conf, index=COMPONENTS, columns=COMPONENTS).to_csv(os.path.join(out, "confusion.csv"))
    _confusion_png(conf, acc, os.path.join(out, "confusion.png"))

    # --- mean self-curves: each cluster's own gamma_c(t) / r_c(t) on its own stories --------
    _curve_plot(_mean_self_curves(gamma, labels),
                "Ground-truth: each cluster's posterior on its OWN stories  γ_c(t)",
                r"$\gamma_{\mathrm{true}}(t)$", os.path.join(out, "selfposterior_curves.png"))
    _curve_plot(_mean_self_curves(resp, labels),
                "Ground-truth: each cluster's per-token responsibility on its OWN stories  r_c(t)",
                r"$r_{\mathrm{true}}(t)$", os.path.join(out, "selfresponsibility_curves.png"))

    # --- report ----------------------------------------------------------------------------
    print(f"\n[gt] overall classification accuracy = {acc:.3f}  ({int(correct.sum())}/{n})")
    print("[gt] confusion (row=true, col=argmax final γ):")
    print(pd.DataFrame(conf, index=COMPONENTS, columns=COMPONENTS).round(3).to_string())
    per_cluster_acc = {COMPONENTS[c]: float(correct[labels == c].mean()) for c in range(len(COMPONENTS))}
    print(f"[gt] per-cluster accuracy: { {k: round(v,3) for k,v in per_cluster_acc.items()} }")

    # --- HTML: highlighted text per cluster ------------------------------------------------
    argmax_g = np.nanargmax(np.where(np.isnan(gamma), -np.inf, gamma), axis=1)   # [N, T] running winner
    index_links = []
    for persona in COMPONENTS:
        ci = COMPONENTS.index(persona)
        sel = np.where(labels == ci)[0][:RENDER_PER_PERSONA]
        blocks = [ex6._legend_html()]
        for k, s in enumerate(sel):
            length = int(last_t[s]) + 1
            positions = [t for t in range(1, length)
                         if int(ids[s, t]) != config.EOS_ID and bool(valid_g[s, t])]
            tok_ids = [int(ids[s, t]) for t in positions]
            pieces = ex6._token_pieces(tok_ids, tok)
            r_focus = [float(resp[s, ci, t]) for t in positions]
            g_focus = [float(gamma[s, ci, t]) for t in positions]
            am_names = [COMPONENTS[int(argmax_g[s, t])] for t in positions]
            am_vals = [float(gamma[s, int(argmax_g[s, t]), t]) for t in positions]
            tips = [f"t={t}  r_{persona}={resp[s, ci, t]:.2f}  γ_{persona}={gamma[s, ci, t]:.2f}  "
                    f"γ_argmax={COMPONENTS[int(argmax_g[s, t])]}" for t in positions]

            img = f"{persona}_ex{k}_curve.png"
            ex6._plot_rollout_gamma(gamma[s], length, persona,
                                    f"{persona} story {k} — γ_i(t) (pred={COMPONENTS[pred[s]]})",
                                    os.path.join(out, img))
            blocks.append(_story_block(persona, k, bool(correct[s]), COMPONENTS[pred[s]],
                                       float(final_gamma[s, ci]), pieces, r_focus, g_focus,
                                       am_names, am_vals, tips, img))
        n_c = int((labels == ci).sum()); acc_c = per_cluster_acc[persona]
        intro = (f'<p>Ground-truth <b>{persona}</b> stories from the dataset, scored under the cluster '
                 f'specialists. Per-cluster classification accuracy (argmax final γ) = '
                 f'<b>{acc_c:.3f}</b> over {n_c} stories. Each story is highlighted three ways: '
                 f'① per-token responsibility r, ② cumulative posterior γ (commitment to the TRUE '
                 f'cluster), ③ argmax-by-γ per token (colour = the cluster the running posterior '
                 f'currently assigns — it should converge to {persona}).</p>')
        page = ex6._page(f"ground-truth commitment — {persona}",
                         f'<h1>Ground-truth commitment — {persona}</h1>'
                         f'<p><a href="report.html">← index</a></p>' + intro + "".join(blocks))
        with open(os.path.join(out, f"{persona}.html"), "w") as f:
            f.write(page)
        index_links.append(f'<li><a href="{persona}.html">{persona}</a> — acc {acc_c:.3f} '
                           f'({n_c} stories, {RENDER_PER_PERSONA} shown)</li>')

    index_body = (
        f'<h1>Does the posterior classify ground-truth stories correctly?</h1>'
        f'<p>Overall accuracy (argmax final cumulative posterior γ, uniform prior over the '
        f'{len(COMPONENTS)} clusters) = <b>{acc:.3f}</b> over {n} ground-truth stories '
        f'({CLASSIFY_PER_PERSONA}/cluster).</p>'
        f'<img src="confusion.png" style="max-width:560px;width:100%;"/>'
        f'<p>Mean per-cluster commitment curves on each cluster\'s OWN stories: '
        f'<a href="selfposterior_curves.png">γ_c(t)</a>, '
        f'<a href="selfresponsibility_curves.png">r_c(t)</a>.</p>'
        f'<ul>{"".join(index_links)}</ul>' + ex6._legend_html())
    with open(os.path.join(out, "report.html"), "w") as f:
        f.write(ex6._page("ground-truth commitment", index_body))

    print(f"\n[gt] done. open {os.path.join(out, 'report.html')}")


if __name__ == "__main__":
    main()
