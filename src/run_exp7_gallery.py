"""Experiment 7 gallery — many beta(t)-vs-gamma(t) rollouts per persona as a static HTML page.

The Exp 7 figures (free_beta_vs_gamma.png) show ONE rollout per dominant component. This builds a
browsable gallery: for each component (5 personas + base) it takes the free rollouts that component
DOMINATES (it wins the final cumulative posterior gamma), and renders, per rollout, the probe belief
beta_i(t) (solid) overlaid on the generative posterior gamma_i(t) (dashed) plus the decoded
completion — so you can scan many rollouts rather than one. Same selection convention as Exp 6.

Reuses a COMPLETED Exp 7 run's trained probe (probe.npz) — it does NOT retrain — plus Exp 2's
free_samples.npz and the Exp 3 commitment primitives. No new modelling.

Run:  python -m src.run_exp7_gallery
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp7_gallery
Prereqs: Exp 7 (probe.npz) and Exp 2 (free_samples.npz).
Writes results/exp7_gallery_<ts>/: report.html (index), <component>.html, <component>_rollout<k>.png.
"""

from __future__ import annotations

import glob
import html
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, generate, models, probe, token_dist
from .commitment import COMPONENTS
from .config import RunConfig
from .embed_traj import COMPONENT_COLORS

# --- gallery parameters (module constants) --------------------------------------------------
N_PER_PERSONA = 12   # max rollouts shown per component (top by final gamma; may be fewer)
EMBED_LAYER = -1     # must match the probe's training (Exp 7 default)
L2_NORMALIZE = True  # must match the probe's training
FEAT_BS = 128


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _load_probe(exp7_dir: str) -> probe.MultinomialProbe:
    """Reconstruct the trained MultinomialProbe from an Exp 7 run's probe.npz (no retraining)."""
    d = np.load(os.path.join(exp7_dir, "probe.npz"), allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    assert classes == COMPONENTS, f"probe classes {classes} != COMPONENTS {COMPONENTS}"
    return probe.MultinomialProbe(d["mu"], d["sd"], d["W"], d["b"], final_loss=float("nan"))


def _token_text(ids: list[int], tok) -> str:
    """Decode a list of token ids to readable text (WordPiece: '##' = continuation, else word)."""
    out = []
    for s in tok.convert_ids_to_tokens(ids):
        if s.startswith("##"):
            out.append(s[2:])
        else:
            out.append(" " + s)
    return "".join(out).strip()


def _plot_rollout(beta: np.ndarray, gamma: np.ndarray, length: int, focus: str, title: str,
                  path: str) -> None:
    """Single-rollout overlay: beta_i(t) solid + gamma_i(t) dashed, one colour per component,
    the focus component emphasised. beta/gamma are [C, T]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    t = np.arange(length)
    for i, name in enumerate(COMPONENTS):
        lw = 2.4 if name == focus else 1.0
        alpha = 1.0 if name == focus else 0.5
        ax.plot(t, beta[i, :length], "-", lw=lw, alpha=alpha, color=COMPONENT_COLORS[name])
        ax.plot(t, gamma[i, :length], "--", lw=lw * 0.8, alpha=alpha * 0.85,
                color=COMPONENT_COLORS[name])
    ax.set_xlabel("token position t"); ax.set_ylabel("probability"); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    comp = [Line2D([0], [0], color=COMPONENT_COLORS[n], lw=2, label=n) for n in COMPONENTS]
    style = [Line2D([0], [0], color="0.3", lw=2, ls="-", label=r"$\beta$ probe"),
             Line2D([0], [0], color="0.3", lw=2, ls="--", label=r"$\gamma$ post.")]
    ax.legend(handles=comp + style, ncol=4, fontsize=6, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _page(title: str, body: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>'
            f'<body style="max-width:920px;margin:30px auto;font-family:system-ui,sans-serif;'
            f'color:#111;">{body}</body></html>')


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp7_gallery_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    exp7_dir = _latest("exp7_2*")   # the trained probe (exclude exp7_gallery_* dirs)
    print(f"[exp7g] device={cfg.device}  out={out_dir}  probe from {os.path.basename(exp7_dir)}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    base_model = models.load_base_model(cfg.device)
    clf = _load_probe(exp7_dir)

    # free generations from Exp 2; gamma (specialists+base, uniform prior) and beta (probe)
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"]); free_attn = torch.tensor(npz["attn"])
    print(f"[exp7g] scoring {free.shape[0]} free rollouts: gamma + beta ...")
    logp = commitment.per_model_token_logprobs(persona_models, base_model, free, free_attn,
                                               cfg.device, cfg.gen.batch_size)
    gamma = commitment.cumulative_posterior(logp, commitment.uniform_prior())   # [N, k+1, T]
    fwd = token_dist.forward_attention_mask(free)
    beta = clf.predict_proba(probe.prefix_running_mean(mixture_model, free, fwd, cfg.device,
                                                       EMBED_LAYER, L2_NORMALIZE, FEAT_BS))  # [N,T,C]
    fwd_len = fwd.sum(dim=1).numpy().astype(int)

    n, _, T = gamma.shape
    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = T - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    final_gamma = gamma[np.arange(n), :, last_t]                                 # [N, k+1]

    index_links, summary_rows = [], []
    for ci, comp in enumerate(COMPONENTS):
        committed = np.where(np.argmax(final_gamma, axis=1) == ci)[0]
        order = committed[np.argsort(final_gamma[committed, ci])[::-1]][:N_PER_PERSONA]
        print(f"[exp7g] {comp:22s}: {committed.size} dominated rollouts, showing {len(order)}")

        blocks = []
        if committed.size == 0:
            blocks.append('<p style="color:#a00;">No free rollout is dominated by this component.</p>')
        for k, s in enumerate(order):
            s = int(s); length = int(fwd_len[s])
            b_ct, g_ct = beta[s].T, gamma[s]                                     # [C, T]
            valid_t = [t for t in range(1, length) if not np.all(np.isnan(g_ct[:, t]))]
            mean_l1 = float(np.mean([np.abs(b_ct[:, t] - g_ct[:, t]).sum() for t in valid_t])) \
                if valid_t else float("nan")
            ids = [int(free[s, t]) for t in range(1, length) if int(free[s, t]) != config.EOS_ID]
            text = _token_text(ids, tok)

            img = f"{comp}_rollout{k}.png"
            _plot_rollout(b_ct, g_ct, length, comp,
                          f"{comp} rollout {k} (free #{s}, final γ={final_gamma[s, ci]:.2f}, "
                          f"L1(β,γ)={mean_l1:.2f})", os.path.join(out_dir, img))
            blocks.append(
                f'<div style="margin:22px 0;border-top:1px solid #ddd;padding-top:12px;">'
                f'<img src="{img}" style="max-width:780px;width:100%;"/>'
                f'<p style="font-family:Georgia,serif;font-size:14px;line-height:1.6;color:#333;'
                f'background:#fafafa;padding:8px;border:1px solid #eee;border-radius:4px;">'
                f'{html.escape(text)}</p></div>')
            summary_rows.append({"component": comp, "rollout": k, "free_sample_id": s,
                                 "final_gamma": float(final_gamma[s, ci]), "mean_l1_beta_gamma": mean_l1,
                                 "n_tokens": len(ids), "text": text})

        page = _page(f"Exp 7 gallery — {comp}",
                     f'<h1>Exp 7 gallery — {comp}</h1><p><a href="report.html">← index</a></p>'
                     f'<p style="font-size:13px;color:#555;">Solid = β (probe belief over the '
                     f'representation); dashed = γ (generative posterior). Colour = component. '
                     f'Rollouts are free generations from P_mix that this component dominates, '
                     f'top {N_PER_PERSONA} by final γ.</p>' + "".join(blocks))
        with open(os.path.join(out_dir, f"{comp}.html"), "w") as f:
            f.write(page)
        index_links.append(f'<li><a href="{comp}.html">{comp}</a> '
                           f'({committed.size} dominated, {len(order)} shown)</li>')

    index = (f'<h1>Exp 7 gallery — β (probe) vs γ (posterior) per rollout</h1>'
             f'<p>For each component, free rollouts from P_mix it dominates (wins the final '
             f'cumulative posterior γ), each showing the probe belief β_i(t) (solid) over the '
             f'generative posterior γ_i(t) (dashed), plus the decoded completion. Probe from '
             f'{os.path.basename(exp7_dir)}; uniform prior over the {len(COMPONENTS)} components.</p>'
             f'<ul>{"".join(index_links)}</ul>')
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(_page("Exp 7 gallery", index))
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "selected_rollouts.csv"), index=False)
    print(f"\n[exp7g] done. open {os.path.join(out_dir, 'report.html')}")


if __name__ == "__main__":
    main()
