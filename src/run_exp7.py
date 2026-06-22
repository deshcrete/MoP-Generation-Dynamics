"""Experiment 7 — persona probe: discriminative belief beta(t) vs generative posterior gamma(t).

Trains a linear probe on P_mix's running-mean prefix embedding e_bar(t) to predict the source
persona (COMPONENTS = personas + base), then for mixture rollouts overlays the probe's belief
beta_i(t) on the generative cumulative posterior gamma_i(t) of Exp 3/4, and tests whether the
representation commits before/with/after the posterior (tau_beta vs tau_gamma). See
context/classifier_expr.md.

Run:  python -m src.run_exp7
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp7
Prereqs: Exp 1 (triggers.json) and Exp 2 (free_samples.npz).
Writes results/exp7_<ts>/: config.json, exp7_params.json, probe.npz, probe_accuracy.csv,
probe_confusion.csv, accuracy_vs_position.png, confusion.png, free_beta_vs_gamma.png,
anchored_beta_vs_gamma.png, commit_times.csv, commit_time_scatter.png.
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

from . import commitment, config, data, embed_traj, generate, models, probe, token_dist
from .commitment import COMPONENTS
from .config import RunConfig

# --- Exp 7 parameters (module constants, logged into the run dir) ---------------------------
EMBED_LAYER = -1          # residual-stream layer for the probe input (final, as Exp 5)
L2_NORMALIZE = True       # unit-normalise e_bar(t) before the probe (Exp 5 norm-growth confound)
N_TRAIN_STORIES = 250     # D_i stories per class used to BUILD probe training examples
N_TEST_STORIES = 80       # held-out D_i stories per class for evaluation (no story overlap)
POS_PER_STORY = 12        # prefix positions sampled per story (vs all-prefixes redundancy)
T_PROBE_MAX = 96          # only sample positions t < this (within DataConfig.t_max=128)
PROBE_EPOCHS = 400        # full-batch Adam epochs
PROBE_LR = 0.05
PROBE_WD = 1e-3           # weight decay (regularise the probe)
N_COMMIT = 200            # free rollouts used for the tau_beta vs tau_gamma commit-time stats
COMMIT_THRESH = 0.5       # a curve "commits" at the first t where its leading component exceeds this
ANCHOR_VARIANT = "entry"  # single-token entry triggers for the anchored rollouts (aligned at t=1)
FEAT_BS = 128             # batch size for embedding forward passes
SEED_PROBE = 7            # story sampling + probe init seed


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _commit_time(curve: np.ndarray, length: int, thresh: float) -> tuple[float, int]:
    """First t in [1, length) where the leading component of `curve` [C, T] exceeds `thresh`.

    Returns (t, winning_component_index), or (nan, -1) if it never commits. nan columns
    (invalid gamma positions) are skipped.
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

def _plot_beta_gamma_grid(items: list[tuple[str, np.ndarray, np.ndarray, int]], suptitle: str,
                          path: str, ncols: int = 3) -> None:
    """Small-multiples: per rollout overlay beta_i(t) (solid) and gamma_i(t) (dashed), one colour per
    component. items = (title, beta[C, T], gamma[C, T], length)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (title, beta, gamma, length) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        t = np.arange(length)
        for i, name in enumerate(COMPONENTS):
            col = embed_traj.COMPONENT_COLORS[name]
            ax.plot(t, beta[i, :length], "-", lw=1.5, color=col)
            ax.plot(t, gamma[i, :length], "--", lw=1.2, color=col, alpha=0.8)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1)
        ax.set_xlabel("t", fontsize=8); ax.set_ylabel("probability", fontsize=8)
        ax.tick_params(labelsize=7)
    comp_handles = [Line2D([0], [0], color=embed_traj.COMPONENT_COLORS[n], lw=2, label=n)
                    for n in COMPONENTS]
    style_handles = [Line2D([0], [0], color="0.3", lw=2, ls="-", label=r"$\beta$ (probe)"),
                     Line2D([0], [0], color="0.3", lw=2, ls="--", label=r"$\gamma$ (posterior)")]
    fig.legend(handles=comp_handles + style_handles, loc="lower center",
               ncol=len(COMPONENTS) + 2, fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_accuracy_vs_position(bins: list[tuple[int, int]], acc: list[float], n: list[int],
                               path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, acc, color="0.6")
    ax.axhline(1.0 / len(COMPONENTS), ls="--", color="#c0504d", lw=1,
               label=f"chance = 1/{len(COMPONENTS)}")
    ax.set_xticks(x); ax.set_xticklabels([f"[{lo},{hi})\nn={ni}" for (lo, hi), ni in zip(bins, n)],
                                         fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("held-out accuracy")
    ax.set_xlabel("prefix position t (bin)")
    ax.set_title("Exp 7 — persona-probe accuracy vs prefix position (held-out real stories)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_confusion(conf: np.ndarray, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(conf, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(COMPONENTS))); ax.set_xticklabels(COMPONENTS, rotation=45, ha="right",
                                                              fontsize=8)
    ax.set_yticks(range(len(COMPONENTS))); ax.set_yticklabels(COMPONENTS, fontsize=8)
    for i in range(len(COMPONENTS)):
        for j in range(len(COMPONENTS)):
            ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if conf[i, j] < 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true (held-out real)")
    ax.set_title("Exp 7 — probe confusion (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_commit_scatter(tg: np.ndarray, tb: np.ndarray, win_g: np.ndarray, win_b: np.ndarray,
                         path: str) -> None:
    """tau_gamma vs tau_beta, each dot coloured by the persona gamma commits to (win_g). A round
    marker means the probe agrees (win_b == win_g); an 'x' means the probe commits to a DIFFERENT
    persona. Integer commit positions overlap, so points are jittered (seeded) for visibility."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m = np.isfinite(tg) & np.isfinite(tb)
    rng = np.random.default_rng(0)
    jit = lambda a: a + rng.uniform(-0.12, 0.12, size=a.shape)   # spread overlapping integer dots

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    for comp in COMPONENTS:
        sel = m & (win_g == comp)
        if not sel.any():
            continue
        labeled = False
        for mask, marker, sz in [(sel & (win_b == win_g), "o", 26),
                                 (sel & (win_b != win_g), "X", 44)]:
            if not mask.any():
                continue
            ax.scatter(jit(tg[mask]), jit(tb[mask]), s=sz, alpha=0.65, marker=marker,
                       color=embed_traj.COMPONENT_COLORS[comp],
                       label=(comp if not labeled else None))
            labeled = True
    hi = float(np.nanmax([np.nanmax(tg[m]) if m.any() else 1, np.nanmax(tb[m]) if m.any() else 1]))
    ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1)
    corr = float(np.corrcoef(tg[m], tb[m])[0, 1]) if m.sum() > 1 else float("nan")
    med = float(np.median(tb[m] - tg[m])) if m.any() else float("nan")
    ax.set_xlabel(r"$\tau_\gamma$ (posterior commit position)")
    ax.set_ylabel(r"$\tau_\beta$ (probe commit position)")
    ax.set_title(f"Exp 7 — commit-time: corr={corr:.2f}, median($\\tau_\\beta-\\tau_\\gamma$)={med:.1f} "
                 f"(n={int(m.sum())})")
    # legend: persona colours + marker meaning (o = probe agrees, X = probe picks another persona)
    marker_handles = [Line2D([0], [0], color="0.4", lw=0, marker="o", label=r"$\beta$ agrees"),
                      Line2D([0], [0], color="0.4", lw=0, marker="X", label=r"$\beta$ differs"),
                      Line2D([0], [0], color="0.5", lw=1, ls="--", label=r"$\tau_\beta=\tau_\gamma$")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + marker_handles, fontsize=8, ncol=2,
              loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp7_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "exp7_params.json"), "w") as f:
        json.dump({"EMBED_LAYER": EMBED_LAYER, "L2_NORMALIZE": L2_NORMALIZE,
                   "N_TRAIN_STORIES": N_TRAIN_STORIES, "N_TEST_STORIES": N_TEST_STORIES,
                   "POS_PER_STORY": POS_PER_STORY, "T_PROBE_MAX": T_PROBE_MAX,
                   "PROBE_EPOCHS": PROBE_EPOCHS, "PROBE_LR": PROBE_LR, "PROBE_WD": PROBE_WD,
                   "N_COMMIT": N_COMMIT, "COMMIT_THRESH": COMMIT_THRESH,
                   "ANCHOR_VARIANT": ANCHOR_VARIANT, "SEED_PROBE": SEED_PROBE}, f, indent=2)
    print(f"[exp7] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    base_model = models.load_base_model(cfg.device)

    def feats(ids, attn):  # running-mean prefix embeddings under P_mix for a batch of sequences
        return probe.prefix_running_mean(mixture_model, ids, attn, cfg.device, EMBED_LAYER,
                                         L2_NORMALIZE, FEAT_BS)

    # =====================================================================================
    # Build probe train/test sets: e_bar(t) examples per class (5 personas + base)
    # =====================================================================================
    rng = np.random.default_rng(SEED_PROBE)
    n_per = N_TRAIN_STORIES + N_TEST_STORIES
    stories = data.load_persona_stories()
    Xtr, ytr, Xte, yte, poste = [], [], [], [], []

    for ci, comp in enumerate(COMPONENTS):
        if comp == "base":
            # base has no dataset: its examples are base-model free generations (as Exp 5's base cluster)
            base_cfg = dataclasses.replace(cfg.gen, n_samples=n_per)
            samples = generate.free_generate(base_model, tok, base_cfg, cfg.device)
            attn = token_dist.forward_attention_mask(samples)
        else:
            pool = stories[comp]
            assert len(pool) >= n_per, f"{comp}: only {len(pool)} stories, need {n_per}"
            chosen = rng.choice(len(pool), size=n_per, replace=False)
            samples, attn = data.tokenize_stories([pool[int(c)] for c in chosen], tok, cfg.data)

        feat = feats(samples, attn)                          # [n_per, T, H]
        lengths = attn.sum(dim=1).numpy()
        # split by story (first N_TRAIN train, rest test) so eval prefixes are from unseen stories
        Xc_tr, yc_tr, _ = probe.build_examples(feat[:N_TRAIN_STORIES], lengths[:N_TRAIN_STORIES],
                                               ci, POS_PER_STORY, rng, t_max=T_PROBE_MAX)
        Xc_te, yc_te, pc_te = probe.build_examples(feat[N_TRAIN_STORIES:], lengths[N_TRAIN_STORIES:],
                                                   ci, POS_PER_STORY, rng, t_max=T_PROBE_MAX)
        Xtr.append(Xc_tr); ytr.append(yc_tr)
        Xte.append(Xc_te); yte.append(yc_te); poste.append(pc_te)
        print(f"[exp7]  {comp:22s} train ex={len(yc_tr):5d}  test ex={len(yc_te):5d}")

    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte); poste = np.concatenate(poste)

    # =====================================================================================
    # Train the probe + evaluate on held-out real prefixes
    # =====================================================================================
    print(f"[exp7] training probe on {len(ytr)} examples ({Xtr.shape[1]}-dim) ...")
    clf = probe.train_probe(Xtr, ytr, len(COMPONENTS), cfg.device, PROBE_LR, PROBE_EPOCHS,
                            PROBE_WD, SEED_PROBE)
    pred_te = clf.predict_proba(Xte).argmax(axis=1)
    overall = float((pred_te == yte).mean())
    print(f"[exp7] probe final train loss={clf.final_loss:.3f}  held-out accuracy={overall:.3f}")

    # accuracy vs prefix position (does representation evidence accumulate with t?)
    pos_bins = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, T_PROBE_MAX)]
    acc_b, n_b = [], []
    for lo, hi in pos_bins:
        sel = (poste >= lo) & (poste < hi)
        n_b.append(int(sel.sum()))
        acc_b.append(float((pred_te[sel] == yte[sel]).mean()) if sel.any() else float("nan"))
    _plot_accuracy_vs_position(pos_bins, acc_b, n_b, os.path.join(out_dir, "accuracy_vs_position.png"))

    # per-class accuracy + confusion (row-normalised)
    conf = np.zeros((len(COMPONENTS), len(COMPONENTS)))
    for i, j in zip(yte, pred_te):
        conf[i, j] += 1
    per_class_acc = {COMPONENTS[i]: float(conf[i, i] / conf[i].sum()) if conf[i].sum() else float("nan")
                     for i in range(len(COMPONENTS))}
    conf_norm = conf / np.clip(conf.sum(axis=1, keepdims=True), 1, None)
    _plot_confusion(conf_norm, os.path.join(out_dir, "confusion.png"))
    pd.DataFrame(conf_norm, index=COMPONENTS, columns=COMPONENTS).to_csv(
        os.path.join(out_dir, "probe_confusion.csv"))

    acc_rows = [{"kind": "overall", "key": "all", "n": int(len(yte)), "accuracy": overall}]
    acc_rows += [{"kind": "position_bin", "key": f"[{lo},{hi})", "n": n_b[b], "accuracy": acc_b[b]}
                 for b, (lo, hi) in enumerate(pos_bins)]
    acc_rows += [{"kind": "per_class", "key": c, "n": int(conf[i].sum()), "accuracy": per_class_acc[c]}
                 for i, c in enumerate(COMPONENTS)]
    pd.DataFrame(acc_rows).to_csv(os.path.join(out_dir, "probe_accuracy.csv"), index=False)
    print("[exp7] per-class held-out accuracy: " +
          "  ".join(f"{c}={per_class_acc[c]:.2f}" for c in COMPONENTS))

    np.savez_compressed(os.path.join(out_dir, "probe.npz"), mu=clf.mu, sd=clf.sd, W=clf.W, b=clf.b,
                        classes=np.array(COMPONENTS))

    # =====================================================================================
    # Apply to mixture FREE rollouts: beta(t) and gamma(t), + commit-time tau_beta vs tau_gamma
    # =====================================================================================
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free_all = torch.tensor(npz["samples"]); free_attn_all = torch.tensor(npz["attn"])
    free = free_all[:N_COMMIT]; free_attn = free_attn_all[:N_COMMIT]
    fwd = token_dist.forward_attention_mask(free)                      # seed attended (beta regime)
    print(f"[exp7] scoring {free.shape[0]} free rollouts: gamma (specialists+base) and beta (probe) ...")

    free_logp = commitment.per_model_token_logprobs(persona_models, base_model, free, free_attn,
                                                    cfg.device, cfg.gen.batch_size)
    gamma = commitment.cumulative_posterior(free_logp, commitment.uniform_prior())   # [N, k+1, T]
    beta = clf.predict_proba(feats(free, fwd))                         # [N, T, C]
    fwd_len = fwd.sum(dim=1).numpy().astype(int)                       # valid length incl. seed

    # commit times + beta-gamma agreement, per rollout
    tg = np.full(free.shape[0], np.nan); tb = np.full(free.shape[0], np.nan)
    win_g, win_b, l1 = [], [], np.full(free.shape[0], np.nan)
    for s in range(free.shape[0]):
        length = int(fwd_len[s])
        g_ct = gamma[s]                                                # [C, T]
        b_ct = beta[s].T                                               # [C, T]
        tg[s], cg = _commit_time(g_ct, length, COMMIT_THRESH)
        tb[s], cb = _commit_time(b_ct, length, COMMIT_THRESH)
        win_g.append(COMPONENTS[cg] if cg >= 0 else "none")
        win_b.append(COMPONENTS[cb] if cb >= 0 else "none")
        valid_t = [t for t in range(1, length) if not np.all(np.isnan(g_ct[:, t]))]
        if valid_t:
            l1[s] = float(np.mean([np.abs(b_ct[:, t] - g_ct[:, t]).sum() for t in valid_t]))
    pd.DataFrame({"rollout": np.arange(free.shape[0]), "tau_gamma": tg, "tau_beta": tb,
                  "winner_gamma": win_g, "winner_beta": win_b, "mean_l1_beta_gamma": l1}).to_csv(
        os.path.join(out_dir, "commit_times.csv"), index=False)
    _plot_commit_scatter(tg, tb, np.array(win_g), np.array(win_b),
                         os.path.join(out_dir, "commit_time_scatter.png"))
    both = np.isfinite(tg) & np.isfinite(tb)
    print(f"[exp7] commit-time: {int(both.sum())}/{free.shape[0]} rollouts commit under both; "
          f"median tau_beta-tau_gamma={np.median((tb - tg)[both]) if both.any() else float('nan'):.1f}; "
          f"mean L1(beta,gamma)={np.nanmean(l1):.3f}")

    # one example free rollout per dominant component (selected by final gamma, as Exp 4B)
    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = valid_g.shape[1] - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    final_gamma = gamma[np.arange(gamma.shape[0]), :, last_t]          # [N, k+1]
    free_items = []
    for ci, name in enumerate(COMPONENTS):
        s = int(np.argmax(final_gamma[:, ci]))
        free_items.append((f"free — dominant: {name} (gamma={final_gamma[s, ci]:.2f})",
                           beta[s].T, gamma[s], int(fwd_len[s])))
    _plot_beta_gamma_grid(free_items, r"Exp 7 — FREE rollouts: $\beta$ (probe, solid) vs $\gamma$ (posterior, dashed)",
                          os.path.join(out_dir, "free_beta_vs_gamma.png"))

    # =====================================================================================
    # Apply to ANCHORED-by-i rollouts (entry trigger), one per persona
    # =====================================================================================
    anchors = {p: json.load(open(os.path.join(_latest("exp1_*"), "triggers.json")))
               ["anchors"][p][ANCHOR_VARIANT]["token_ids"] for p in config.PERSONAS}
    anch_cfg = dataclasses.replace(cfg.gen, n_samples=len(config.PERSONAS))
    print(f"[exp7] regenerating anchored ({ANCHOR_VARIANT}) rollouts ...")
    gen = generate.anchored_generate(mixture_model, tok, anchors, anch_cfg, cfg.device)
    anch_items = []
    for p in config.PERSONAS:
        row = gen[p][:1]
        row_fwd = token_dist.forward_attention_mask(row)
        row_score = generate.generation_attention_mask(row, start=1)
        g = commitment.cumulative_posterior(
            commitment.per_model_token_logprobs(persona_models, base_model, row, row_score,
                                                cfg.device, cfg.gen.batch_size),
            commitment.uniform_prior())[0]                            # [C, T]
        b = clf.predict_proba(feats(row, row_fwd))[0].T               # [C, T]
        tokstr = tok.convert_ids_to_tokens(int(anchors[p][0])) if anchors[p] else "(empty)"
        anch_items.append((f"anchored: {p} (`{tokstr}`)", b, g, int(row_fwd.sum().item())))
    _plot_beta_gamma_grid(anch_items, r"Exp 7 — ANCHORED rollouts: $\beta$ (probe, solid) vs $\gamma$ (posterior, dashed)",
                          os.path.join(out_dir, "anchored_beta_vs_gamma.png"))

    print(f"\n[exp7] done. results in {out_dir}")


if __name__ == "__main__":
    main()
