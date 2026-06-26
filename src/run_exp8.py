"""Experiment 8 — Naive Bayes belief eta(t) vs generative posterior gamma(t).

Fits a multinomial Naive Bayes classifier over the shared tokenizer's vocab on the training data
(per-cluster token counts over D_i; no base class — base == P_mix in this run), uniform prior, and
accumulates a count-based belief eta_i(t) per token exactly as gamma_i(t) accumulates the
specialists' neural log-probs. Overlays eta(t) (solid) on gamma(t) (dashed) for mixture rollouts —
free (one per dominant component) and anchored-by-i (entry trigger, one per cluster) — to test
whether the model's generative posterior tracks the theoretical Bayesian classifier. Also reports
the population commit-time agreement tau_eta vs tau_gamma. See context/bayes_classifier.md.

Run:  python -m src.run_exp8
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp8
Prereqs: Exp 1 (triggers.json) and Exp 2 (free_samples.npz).
Writes results/exp8_<ts>/: config.json, exp8_params.json, nb_top_tokens.csv, eta_gamma.npz,
commit_times.csv, commit_time_scatter.png, free_eta_vs_gamma.png, anchored_eta_vs_gamma.png.
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

from . import bayes_nb, commitment, config, data, embed_traj, generate, models, token_dist
from .commitment import COMPONENTS
from .config import RunConfig

# --- Exp 8 parameters (module constants, logged into the run dir) ---------------------------
N_FIT_STORIES = 2000      # D_i stories per cluster used to fit the NB token counts
NB_ALPHA = 1.0            # Laplace smoothing on the token counts
N_COMMIT = 200            # free rollouts used for the tau_eta vs tau_gamma commit-time stats
COMMIT_THRESH = 0.5       # a curve "commits" at the first t where its leading component exceeds this
ANCHOR_VARIANT = "entry"  # single-token entry triggers for the anchored rollouts (aligned at t=1)
TOP_NB = 12               # top NB tokens per component written to nb_top_tokens.csv
SEED_FIT = 8              # story sampling seed


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _commit_time(curve: np.ndarray, length: int, thresh: float) -> tuple[float, int]:
    """First t in [1, length) where the leading component of `curve` [C, T] exceeds `thresh`.

    Returns (t, winning_component_index), or (nan, -1) if it never commits. nan columns (invalid
    positions) are skipped. Mirrors run_exp7._commit_time so tau_eta and tau_gamma are comparable.
    """
    for t in range(1, length):
        col = curve[:, t]
        if np.all(np.isnan(col)):
            continue
        c = int(np.nanargmax(col))
        if col[c] >= thresh:
            return float(t), c
    return float("nan"), -1


# --- plots ----------------------------------------------------------------------------------

def _plot_eta_gamma_grid(items: list[tuple[str, np.ndarray, np.ndarray, int]], suptitle: str,
                         path: str, ncols: int = 3) -> None:
    """Small-multiples: per rollout overlay eta_i(t) (solid) and gamma_i(t) (dashed), one colour per
    component. items = (title, eta[C, T], gamma[C, T], length)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (title, eta, gamma, length) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        t = np.arange(length)
        for i, name in enumerate(COMPONENTS):
            col = embed_traj.COMPONENT_COLORS[name]
            ax.plot(t, eta[i, :length], "-", lw=1.5, color=col)
            ax.plot(t, gamma[i, :length], "--", lw=1.2, color=col, alpha=0.8)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1)
        ax.set_xlabel("t", fontsize=8); ax.set_ylabel("probability", fontsize=8)
        ax.tick_params(labelsize=7)
    comp_handles = [Line2D([0], [0], color=embed_traj.COMPONENT_COLORS[n], lw=2, label=n)
                    for n in COMPONENTS]
    style_handles = [Line2D([0], [0], color="0.3", lw=2, ls="-", label=r"$\eta$ (naive Bayes)"),
                     Line2D([0], [0], color="0.3", lw=2, ls="--", label=r"$\gamma$ (posterior)")]
    fig.legend(handles=comp_handles + style_handles, loc="lower center",
               ncol=len(COMPONENTS) + 2, fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_commit_scatter(tg: np.ndarray, te: np.ndarray, win_g: np.ndarray, win_e: np.ndarray,
                         path: str) -> None:
    """tau_gamma vs tau_eta, each dot coloured by the persona gamma commits to (win_g). A round
    marker means eta agrees (win_e == win_g); an 'X' means eta commits to a DIFFERENT persona.
    Integer commit positions overlap, so points are jittered (seeded) for visibility."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m = np.isfinite(tg) & np.isfinite(te)
    rng = np.random.default_rng(0)
    jit = lambda a: a + rng.uniform(-0.12, 0.12, size=a.shape)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    for comp in COMPONENTS:
        sel = m & (win_g == comp)
        if not sel.any():
            continue
        labeled = False
        for mask, marker, sz in [(sel & (win_e == win_g), "o", 26),
                                 (sel & (win_e != win_g), "X", 44)]:
            if not mask.any():
                continue
            ax.scatter(jit(tg[mask]), jit(te[mask]), s=sz, alpha=0.65, marker=marker,
                       color=embed_traj.COMPONENT_COLORS[comp],
                       label=(comp if not labeled else None))
            labeled = True
    hi = float(np.nanmax([np.nanmax(tg[m]) if m.any() else 1, np.nanmax(te[m]) if m.any() else 1]))
    ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1)
    corr = float(np.corrcoef(tg[m], te[m])[0, 1]) if m.sum() > 1 else float("nan")
    med = float(np.median(te[m] - tg[m])) if m.any() else float("nan")
    ax.set_xlabel(r"$\tau_\gamma$ (posterior commit position)")
    ax.set_ylabel(r"$\tau_\eta$ (naive-Bayes commit position)")
    ax.set_title(f"Exp 8 — commit-time: corr={corr:.2f}, median($\\tau_\\eta-\\tau_\\gamma$)={med:.1f} "
                 f"(n={int(m.sum())})")
    marker_handles = [Line2D([0], [0], color="0.4", lw=0, marker="o", label=r"$\eta$ agrees"),
                      Line2D([0], [0], color="0.4", lw=0, marker="X", label=r"$\eta$ differs"),
                      Line2D([0], [0], color="0.5", lw=1, ls="--", label=r"$\tau_\eta=\tau_\gamma$")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + marker_handles, fontsize=8, ncol=2,
              loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _gamma_for(persona_models, ids, score_attn, device, bs) -> np.ndarray:
    """Cumulative posterior gamma [N, C, T] over ids, scored under score_attn (uniform prior)."""
    logp = commitment.per_model_token_logprobs(persona_models, ids, score_attn, device, bs)
    return commitment.cumulative_posterior(logp, commitment.uniform_prior())


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp8_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "exp8_params.json"), "w") as f:
        json.dump({"N_FIT_STORIES": N_FIT_STORIES, "NB_ALPHA": NB_ALPHA,
                   "N_COMMIT": N_COMMIT, "COMMIT_THRESH": COMMIT_THRESH,
                   "ANCHOR_VARIANT": ANCHOR_VARIANT, "TOP_NB": TOP_NB, "SEED_FIT": SEED_FIT}, f,
                  indent=2)
    print(f"[exp8] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    prior = commitment.uniform_prior()

    # =====================================================================================
    # Fit the Naive Bayes classifier: per-component token counts over the shared vocab
    # =====================================================================================
    rng = np.random.default_rng(SEED_FIT)
    stories = data.load_persona_stories()
    comp_ids: dict[str, torch.LongTensor] = {}
    comp_attn: dict[str, torch.LongTensor] = {}
    for comp in COMPONENTS:
        pool = stories[comp]
        assert len(pool) >= N_FIT_STORIES, f"{comp}: only {len(pool)} stories, need {N_FIT_STORIES}"
        chosen = rng.choice(len(pool), size=N_FIT_STORIES, replace=False)
        samples, attn = data.tokenize_stories([pool[int(c)] for c in chosen], tok, cfg.data)
        comp_ids[comp], comp_attn[comp] = samples, attn
        print(f"[exp8]  fit data {comp:22s} n={samples.shape[0]:5d} tokens={int(attn.sum())}")

    loglik, counts = bayes_nb.fit_token_loglik(comp_ids, comp_attn, config.VOCAB_SIZE, NB_ALPHA)

    # top NB tokens per component (sanity: do they look like the persona's signature words?)
    top_rows = []
    for ci, comp in enumerate(COMPONENTS):
        order = np.argsort(counts[ci])[::-1][:TOP_NB]
        for rank, v in enumerate(order):
            top_rows.append({"component": comp, "rank": rank, "token_id": int(v),
                             "token": tok.convert_ids_to_tokens(int(v)),
                             "count": int(counts[ci, v]), "p": float(np.exp(loglik[ci, v]))})
    pd.DataFrame(top_rows).to_csv(os.path.join(out_dir, "nb_top_tokens.csv"), index=False)

    # =====================================================================================
    # Mixture FREE rollouts: gamma(t) and eta(t), + commit-time tau_eta vs tau_gamma
    # =====================================================================================
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free_all = torch.tensor(npz["samples"]); free_attn_all = torch.tensor(npz["attn"])
    free = free_all[:N_COMMIT]; free_attn = free_attn_all[:N_COMMIT]
    free_ids = free.numpy()
    print(f"[exp8] scoring {free.shape[0]} free rollouts: gamma (cluster specialists) and eta (naive Bayes) ...")

    gamma = _gamma_for(persona_models, free, free_attn, cfg.device, cfg.gen.batch_size)
    out_mask = ~np.isnan(gamma).all(axis=1)                            # [N, T]: where gamma is valid
    eta = bayes_nb.cumulative_nb_posterior(free_ids, out_mask, loglik, prior)  # [N, C, T]
    lengths = out_mask.shape[1] - np.argmax(out_mask[:, ::-1], axis=1)  # 1 past last valid t

    # commit times + agreement, per rollout
    tg = np.full(free.shape[0], np.nan); te = np.full(free.shape[0], np.nan)
    win_g, win_e, l1 = [], [], np.full(free.shape[0], np.nan)
    for s in range(free.shape[0]):
        length = int(lengths[s])
        g_ct, e_ct = gamma[s], eta[s]                                 # [C, T] each
        tg[s], cg = _commit_time(g_ct, length, COMMIT_THRESH)
        te[s], ce = _commit_time(e_ct, length, COMMIT_THRESH)
        win_g.append(COMPONENTS[cg] if cg >= 0 else "none")
        win_e.append(COMPONENTS[ce] if ce >= 0 else "none")
        vt = [t for t in range(1, length) if not np.all(np.isnan(g_ct[:, t]))]
        if vt:
            l1[s] = float(np.mean([np.abs(e_ct[:, t] - g_ct[:, t]).sum() for t in vt]))
    pd.DataFrame({"rollout": np.arange(free.shape[0]), "tau_gamma": tg, "tau_eta": te,
                  "winner_gamma": win_g, "winner_eta": win_e, "mean_l1_eta_gamma": l1}).to_csv(
        os.path.join(out_dir, "commit_times.csv"), index=False)
    _plot_commit_scatter(tg, te, np.array(win_g), np.array(win_e),
                         os.path.join(out_dir, "commit_time_scatter.png"))
    both = np.isfinite(tg) & np.isfinite(te)
    agree = (np.array(win_g) == np.array(win_e)) & both
    print(f"[exp8] commit-time: {int(both.sum())}/{free.shape[0]} commit under both; "
          f"winner agreement={int(agree.sum())}/{int(both.sum())}; "
          f"median tau_eta-tau_gamma={np.median((te - tg)[both]) if both.any() else float('nan'):.1f}; "
          f"mean L1(eta,gamma)={np.nanmean(l1):.3f}")

    # one example free rollout per dominant component (selected by final gamma, as Exp 4B / 6 / 7)
    last_t = lengths - 1
    final_gamma = gamma[np.arange(gamma.shape[0]), :, last_t]          # [N, C]
    free_items = []
    for ci, name in enumerate(COMPONENTS):
        s = int(np.argmax(final_gamma[:, ci]))
        free_items.append((f"free — dominant: {name} (gamma={final_gamma[s, ci]:.2f})",
                           eta[s], gamma[s], int(lengths[s])))
    _plot_eta_gamma_grid(free_items, r"Exp 8 — FREE rollouts: $\eta$ (naive Bayes, solid) vs $\gamma$ (posterior, dashed)",
                         os.path.join(out_dir, "free_eta_vs_gamma.png"))

    # =====================================================================================
    # ANCHORED-by-i rollouts (entry trigger), one per persona
    # =====================================================================================
    anchors = {p: json.load(open(os.path.join(_latest("exp1_*"), "triggers.json")))
               ["anchors"][p][ANCHOR_VARIANT]["token_ids"] for p in config.PERSONAS}
    anch_cfg = dataclasses.replace(cfg.gen, n_samples=len(config.PERSONAS))
    print(f"[exp8] regenerating anchored ({ANCHOR_VARIANT}) rollouts ...")
    gen = generate.anchored_generate(mixture_model, tok, anchors, anch_cfg, cfg.device)
    anch_items = []
    for p in config.PERSONAS:
        row = gen[p][:1]
        row_score = generate.generation_attention_mask(row, start=1)
        g = _gamma_for(persona_models, row, row_score, cfg.device, cfg.gen.batch_size)[0]
        om = ~np.isnan(g).all(axis=0)                                 # [T]
        e = bayes_nb.cumulative_nb_posterior(row.numpy(), om[None], loglik, prior)[0]
        length = int(om.shape[0] - np.argmax(om[::-1]))
        tokstr = tok.convert_ids_to_tokens(int(anchors[p][0])) if anchors[p] else "(empty)"
        anch_items.append((f"anchored: {p} (`{tokstr}`)", e, g, length))
    _plot_eta_gamma_grid(anch_items, r"Exp 8 — ANCHORED rollouts: $\eta$ (naive Bayes, solid) vs $\gamma$ (posterior, dashed)",
                         os.path.join(out_dir, "anchored_eta_vs_gamma.png"))

    np.savez_compressed(os.path.join(out_dir, "eta_gamma.npz"), loglik=loglik, counts=counts,
                        free_gamma=gamma, free_eta=eta, classes=np.array(COMPONENTS))
    print(f"\n[exp8] done. results in {out_dir}")


if __name__ == "__main__":
    main()
