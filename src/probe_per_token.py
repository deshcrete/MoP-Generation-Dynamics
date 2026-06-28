"""Per-token probe responsibility beta_tok(t) — the discriminative twin of commitment.r(t).

Exp 7 compares the probe's CUMULATIVE belief beta(t) — the trained probe applied to the
running-mean prefix embedding e_bar(t) = mean_{s<=t} e(s) — against the generative cumulative
posterior gamma(t). This script computes the PER-TOKEN analog: the SAME trained probe applied to
the SINGLE-POSITION embedding e(t) (no running mean), giving beta_tok_i(t), the discriminative
twin of commitment.token_responsibility r_i(t).

The pairing (so the comparison is apples-to-apples — same accumulation operator on each side):

    gamma_i(t) = softmax_i( log pi_i + sum_{s<=t} ell_i[s] )  <->  beta(t)     on e_bar(t) = MEAN_{s<=t} e(s)
    r_i(t)     = softmax_i( log pi_i +        ell_i[t]      )  <->  beta_tok(t) on e(t)       (single position)

gamma/beta accumulate (running SUM of token log-probs / running MEAN of position embeddings);
r/beta_tok read a SINGLE token. This answers the "memories" discrepancy directly: does the probe
ALSO jump on one discriminative content token when it is NOT low-passed by the running mean — i.e.
is cumulative-beta's smoothness a property of the representation, or an artifact of the 1/t mean?

CAVEAT (Exp 5 finding, in probe.py docstring): the probe was TRAINED on running-mean e_bar(t); a
single-position e(t) is content-dominated and OUT OF DISTRIBUTION for it (single-position 2D
eta^2=0.04 vs 0.91 pooled). So beta_tok is expected to be noisy / near-uniform — and that
noisiness IS a result: it shows the per-token representation is not linearly separable, so the
running mean is load-bearing. L2-normalisation is kept identical to training, so ONLY the pooling
differs between beta and beta_tok.

Run:  python -m src.probe_per_token
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.probe_per_token
Prereqs: a trained probe (results/exp7_*/probe.npz) and Exp 2 (results/exp2_*/free_samples.npz).
Writes results/probe_per_token_<ts>/: config.json, params.json, beta_tok.npz, summary.csv,
free_r_vs_betatok.png, memories_rollout.png.
"""

from __future__ import annotations

import glob
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, embed_traj, models, probe, token_dist
from .commitment import COMPONENTS
from .config import RunConfig

# --- parameters (logged into the run dir) ---------------------------------------------------
EMBED_LAYER = -1          # residual-stream layer (same as Exp 5 / the trained probe)
L2_NORMALIZE = True       # keep identical to the probe's training convention; only pooling differs
N_ROLLOUTS = 200          # free rollouts scored (matches Exp 7's commit-time population)
FEAT_BS = 128             # batch size for embedding forward passes
MEMORIES_ROLLOUT = 81     # the 'memories' example from the cluster-2 discrepancy discussion
MEMORIES_TOKEN_ID = 892   # 'memories' (tokenizer id; verified against the loaded tokenizer below)


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _load_probe() -> probe.MultinomialProbe:
    """Load the probe trained in the latest Exp 7 run; assert its class order matches COMPONENTS."""
    d = _latest("exp7_[0-9]*")
    z = np.load(os.path.join(d, "probe.npz"), allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    assert classes == list(COMPONENTS), f"probe classes {classes} != COMPONENTS {list(COMPONENTS)}"
    print(f"[ptok] loaded probe from {d}  classes={classes}")
    return probe.MultinomialProbe(z["mu"], z["sd"], z["W"], z["b"], final_loss=float("nan"))


@torch.no_grad()
def per_position_embeddings(model, ids: torch.LongTensor, attn: torch.LongTensor, device: str,
                            layer: int, l2: bool, batch_size: int) -> np.ndarray:
    """[N, T, H] SINGLE-POSITION embedding e(t) (NOT the running mean), L2-normalised like training.

    This is exactly probe.prefix_running_mean WITHOUT the cumulative-mean step — the only difference
    between beta_tok and beta is whether the position embeddings are pooled over s<=t.
    """
    n, t = ids.shape
    h = model.config.hidden_size
    out = np.empty((n, t, h), dtype=np.float32)
    for s in range(0, n, batch_size):
        emb = models.prefix_embeddings(model, ids[s:s + batch_size].to(device),
                                       attn[s:s + batch_size].to(device), layer).cpu().numpy()
        out[s:s + batch_size] = emb
    if l2:
        out = out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12, None)
    return out


def _winner_jumpiness(curve: np.ndarray, win: int, length: int) -> float:
    """Mean |Delta| per token of `curve`[win, t] over valid t in [1, length) — 'how jumpy' the belief
    in the winning component is. curve is [C, T]; nan columns skipped."""
    vals = [curve[win, t] for t in range(1, length) if not np.isnan(curve[:, t]).all()]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(vals))))


# --- plots ----------------------------------------------------------------------------------

def _plot_grid(items, suptitle, path, ncols=3):
    """Small-multiples: per rollout overlay beta_tok_i(t) (solid) and r_i(t) (dashed), one colour
    per component. items = (title, beta_tok[C, T], r[C, T], length)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (title, bt, r, length) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        t = np.arange(length)
        for i, name in enumerate(COMPONENTS):
            col = embed_traj.COMPONENT_COLORS[name]
            ax.plot(t, bt[i, :length], "-", lw=1.5, color=col)
            ax.plot(t, r[i, :length], "--", lw=1.2, color=col, alpha=0.8)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1)
        ax.set_xlabel("t", fontsize=8); ax.set_ylabel("probability", fontsize=8)
        ax.tick_params(labelsize=7)
    comp_handles = [Line2D([0], [0], color=embed_traj.COMPONENT_COLORS[n], lw=2, label=n)
                    for n in COMPONENTS]
    style_handles = [Line2D([0], [0], color="0.3", lw=2, ls="-", label=r"$\beta_{tok}$ (probe, per-token)"),
                     Line2D([0], [0], color="0.3", lw=2, ls="--", label=r"$r$ (responsibility)")]
    fig.legend(handles=comp_handles + style_handles, loc="lower center",
               ncol=len(COMPONENTS) + 2, fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_memories(curves: dict, comp_idx: int, comp_name: str, length: int, tok_pos: int,
                   tok_str: str, path: str):
    """Rollout-81 close-up: the four curves (r, beta_tok, gamma, beta_cum) for the winning cluster,
    with the 'memories' token marked — shows directly which curves spike on a single content token."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(length)
    styles = {"gamma": ("--", "#1f77b4", r"$\gamma$ (cumulative posterior)"),
              "r": (":", "#d62728", r"$r$ (per-token responsibility)"),
              "beta_cum": ("-", "#2ca02c", r"$\beta$ (probe, running-mean)"),
              "beta_tok": ("-", "#ff7f0e", r"$\beta_{tok}$ (probe, per-token)")}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for key, (ls, col, lab) in styles.items():
        ax.plot(t, curves[key][comp_idx, :length], ls, color=col, lw=2, label=lab)
    if 1 <= tok_pos < length:
        ax.axvline(tok_pos, color="0.5", ls="-", lw=1, alpha=0.6)
        ax.annotate(f"`{tok_str}`  (t={tok_pos})", xy=(tok_pos, 1.0), xytext=(tok_pos + 1, 0.93),
                    fontsize=9, color="0.3")
    ax.set_ylim(0, 1.02); ax.set_xlabel("token position t"); ax.set_ylabel(f"P({comp_name})")
    ax.set_title(f"Rollout {MEMORIES_ROLLOUT} — {comp_name}: generative (gamma/r) spike vs probe "
                 f"(beta/beta_tok) on a single content token")
    ax.legend(fontsize=8, loc="center right"); fig.tight_layout(); fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"probe_per_token_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"EMBED_LAYER": EMBED_LAYER, "L2_NORMALIZE": L2_NORMALIZE,
                   "N_ROLLOUTS": N_ROLLOUTS, "MEMORIES_ROLLOUT": MEMORIES_ROLLOUT,
                   "MEMORIES_TOKEN_ID": MEMORIES_TOKEN_ID,
                   "probe_dir": _latest("exp7_[0-9]*"), "exp2_dir": _latest("exp2_*")}, f, indent=2)
    print(f"[ptok] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    assert tok.convert_ids_to_tokens(MEMORIES_TOKEN_ID) == "memories", "tokenizer id drift"
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    clf = _load_probe()

    # free rollouts (same set / order Exp 7 uses)
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"])[:N_ROLLOUTS]
    free_attn = torch.tensor(npz["attn"])[:N_ROLLOUTS]               # generation mask (gamma/r regime)
    fwd = token_dist.forward_attention_mask(free)                    # seed attended (probe regime)
    fwd_len = fwd.sum(dim=1).numpy().astype(int)
    print(f"[ptok] scoring {free.shape[0]} free rollouts ...")

    # generative side: gamma (cumulative) and r (per-token), over the k cluster specialists
    logp = commitment.per_model_token_logprobs(persona_models, free, free_attn,
                                               cfg.device, cfg.gen.batch_size)
    pi = commitment.uniform_prior()
    gamma = commitment.cumulative_posterior(logp, pi)               # [N, C, T]
    r = commitment.token_responsibility(logp, pi)                  # [N, C, T]

    # discriminative side: beta (running-mean, = Exp 7) and beta_tok (single-position, NEW)
    e_bar = probe.prefix_running_mean(mixture_model, free, fwd, cfg.device, EMBED_LAYER,
                                      L2_NORMALIZE, FEAT_BS)         # [N, T, H]
    e_pos = per_position_embeddings(mixture_model, free, fwd, cfg.device, EMBED_LAYER,
                                    L2_NORMALIZE, FEAT_BS)           # [N, T, H]
    beta_cum = clf.predict_proba(e_bar).transpose(0, 2, 1)          # [N, C, T]
    beta_tok = clf.predict_proba(e_pos).transpose(0, 2, 1)          # [N, C, T]

    np.savez_compressed(os.path.join(out_dir, "beta_tok.npz"), beta_tok=beta_tok, beta_cum=beta_cum,
                        gamma=gamma, r=r, fwd_len=fwd_len, classes=np.array(list(COMPONENTS)))

    # ---- population summary: agreement with the generative twin, and per-token jumpiness ----
    # winner per rollout = argmax of FINAL gamma (the dominant cluster, as Exp 4B/7)
    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = valid_g.shape[1] - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    win = np.argmax(gamma[np.arange(gamma.shape[0]), :, last_t], axis=1)   # [N]

    def _argmax_agree(a, b):
        """fraction of valid (rollout, t>=1) tokens where argmax_C a == argmax_C b."""
        agree, tot = 0, 0
        for s in range(a.shape[0]):
            for t in range(1, int(fwd_len[s])):
                if np.isnan(a[s, :, t]).all() or np.isnan(b[s, :, t]).all():
                    continue
                tot += 1
                agree += int(np.nanargmax(a[s, :, t]) == np.nanargmax(b[s, :, t]))
        return agree / tot if tot else float("nan")

    def _mean_l1(a, b):
        vals = []
        for s in range(a.shape[0]):
            for t in range(1, int(fwd_len[s])):
                if np.isnan(a[s, :, t]).all() or np.isnan(b[s, :, t]).all():
                    continue
                vals.append(np.abs(np.nan_to_num(a[s, :, t]) - np.nan_to_num(b[s, :, t])).sum())
        return float(np.mean(vals)) if vals else float("nan")

    jump = {k: float(np.nanmean([_winner_jumpiness(arr[s], win[s], int(fwd_len[s]))
                                 for s in range(arr.shape[0])]))
            for k, arr in {"gamma": gamma, "r": r, "beta_cum": beta_cum, "beta_tok": beta_tok}.items()}

    rows = [
        {"metric": "argmax_agree(beta_tok, r)",   "value": _argmax_agree(beta_tok, r)},
        {"metric": "argmax_agree(beta_cum, gamma)", "value": _argmax_agree(beta_cum, gamma)},
        {"metric": "argmax_agree(beta_tok, gamma)", "value": _argmax_agree(beta_tok, gamma)},
        {"metric": "mean_L1(beta_tok, r)",        "value": _mean_l1(beta_tok, r)},
        {"metric": "mean_L1(beta_cum, gamma)",    "value": _mean_l1(beta_cum, gamma)},
        {"metric": "jumpiness_gamma(winner)",     "value": jump["gamma"]},
        {"metric": "jumpiness_r(winner)",         "value": jump["r"]},
        {"metric": "jumpiness_beta_cum(winner)",  "value": jump["beta_cum"]},
        {"metric": "jumpiness_beta_tok(winner)",  "value": jump["beta_tok"]},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("[ptok] summary:")
    for row in rows:
        print(f"    {row['metric']:<32} {row['value']:.3f}")

    # ---- per-dominant-component small multiples: beta_tok (solid) vs r (dashed) ----
    items = []
    for ci, name in enumerate(COMPONENTS):
        s = int(np.argmax(gamma[np.arange(gamma.shape[0]), ci, last_t]))
        items.append((f"free — dominant {name} (rollout {s})", beta_tok[s], r[s], int(fwd_len[s])))
    _plot_grid(items, r"Per-token probe $\beta_{tok}$ (solid) vs responsibility $r$ (dashed)",
               os.path.join(out_dir, "free_r_vs_betatok.png"))

    # ---- the 'memories' close-up (rollout 81, cluster-2) ----
    s = MEMORIES_ROLLOUT
    ids = free[s].numpy()
    tok_pos = next((t for t in range(1, int(fwd_len[s])) if int(ids[t]) == MEMORIES_TOKEN_ID), -1)
    comp_idx = int(win[s]); comp_name = COMPONENTS[comp_idx]
    curves = {"gamma": gamma[s], "r": r[s], "beta_cum": beta_cum[s], "beta_tok": beta_tok[s]}
    _plot_memories(curves, comp_idx, comp_name, int(fwd_len[s]), tok_pos, "memories",
                   os.path.join(out_dir, "memories_rollout.png"))
    if tok_pos > 1:
        print(f"[ptok] rollout {s} '{comp_name}' at 'memories' (t={tok_pos}):  "
              f"gamma {gamma[s,comp_idx,tok_pos-1]:.2f}->{gamma[s,comp_idx,tok_pos]:.2f}  "
              f"r {r[s,comp_idx,tok_pos-1]:.2f}->{r[s,comp_idx,tok_pos]:.2f}  "
              f"beta_cum {beta_cum[s,comp_idx,tok_pos-1]:.2f}->{beta_cum[s,comp_idx,tok_pos]:.2f}  "
              f"beta_tok {beta_tok[s,comp_idx,tok_pos-1]:.2f}->{beta_tok[s,comp_idx,tok_pos]:.2f}")

    print(f"\n[ptok] done. results in {out_dir}")


if __name__ == "__main__":
    main()
