"""Experiment 4 — Mixture weights from token DISTRIBUTIONS, and individual-rollout commitment.

Two parts (see context/token_dist_expr.md, context/exp4_handoff.md):

  4A — w(t) inference. Infer mixture weights from the model's FULL next-token distribution,
       w_hat(t) = argmin_w KL( P_mix(.|x_{<t}) || sum_i w_i P_i(.|x_{<t}) ), via the weighted EM
       (src/em.py + src/token_dist.py). Headline w(1) from the neutral [EOS] seed; a w_i(t) curve
       over the Exp 2 free-generation prefixes. Compared to sigma and to the SEQUENCE-level
       pi_free of Exp 2 — does the t=1 distribution already show the triggered-persona collapse?

  4B — individual-rollout commitment. The same gamma_i(t) as Exp 3 but plotted for INDIVIDUAL
       rollouts instead of averaged (averaging smooths away the high- vs low-frequency-update
       distinction between triggered and behavioural personas). No frequency statistics — plots.

Run:  python -m src.run_exp4
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp4
Prereqs: Exp 1 (triggers.json) and Exp 2 (free_samples.npz, pi_free.csv) must have run.
Writes results/exp4_<ts>/: config.json, w1.csv, w_curve.csv, first_token_dist.png, w_curve.png,
free_individual_rollouts.png, anchored_individual_rollouts.png.
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

from . import commitment, config, generate, models, token_dist
from .commitment import COMPONENTS
from .config import RunConfig

# --- Exp 4 parameters (module constants, logged into the run dir) ---------------------------
N_PREFIXES = 200       # free sequences used as prefixes for the w(t) curve (memory: ~0.6 GB total)
T_CURVE = 32           # token positions for the w(t) curve (predicts tokens 1 .. T_CURVE-1)
MIN_PREFIXES = 20      # positions with fewer valid prefixes than this are left nan in the curve
COLLECT_BS = 64        # batch size for full-vocab next-token-distribution forward passes
TOP_FT = 12            # first tokens shown in the t=1 distribution plot
ANCHOR_VARIANT = "entry"  # single-token triggers -> all personas aligned at position 1


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


# --- Part 4A diagnostics -------------------------------------------------------------------

def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability vector, skipping zero-mass tokens."""
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """Forward KL(p || q) in nats (the quantity w(1) minimises). Mass-covering in q."""
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], 1e-30, None))))


def _write_w1_diagnostic(p_mix: np.ndarray, q_spec: np.ndarray, w1: np.ndarray, path: str) -> None:
    """Persist the identifiability diagnostic for the headline w(1).

    The KL objective at t=1 is nearly flat: the specialists' first-token distributions are highly
    overlapping (all generic story openings), so the precise w(1) split is weakly identified — it
    should NOT be read as 'the mixture is X% persona i at the first token'. This CSV records, per
    persona, the specialist's first-token entropy and its single-component KL(P_mix || P_i), plus
    the uniform-average and EM-solution KLs as a summary footer. If KL_em is only marginally below
    KL_uniform_avg, the headline split is an artifact of entropy-matching + collinearity, not a
    real preference (see context/results.md Exp 4A)."""
    q = np.exp(q_spec)                                              # [k, V] specialist probs
    avg = q.mean(axis=0)                                           # uniform-average first-token dist
    mixw = (w1[:, None] * q).sum(axis=0)                          # EM-weighted mixture
    rows = [{"persona": p, "w1": float(w1[i]), "entropy_nats": _entropy(q[i]),
             "kl_pmix_to_specialist": _kl(p_mix, q[i])} for i, p in enumerate(config.PERSONAS)]
    # summary rows carry the scalars the prose cites; persona column flags them as non-persona.
    rows.append({"persona": "_P_mix", "w1": float("nan"), "entropy_nats": _entropy(p_mix),
                 "kl_pmix_to_specialist": 0.0})
    rows.append({"persona": "_uniform_avg", "w1": float("nan"), "entropy_nats": _entropy(avg),
                 "kl_pmix_to_specialist": _kl(p_mix, avg)})
    rows.append({"persona": "_em_solution", "w1": float("nan"), "entropy_nats": _entropy(mixw),
                 "kl_pmix_to_specialist": _kl(p_mix, mixw)})
    pd.DataFrame(rows).to_csv(path, index=False)


# --- Part 4A plots -------------------------------------------------------------------------

def _plot_first_token_dist(p_mix: np.ndarray, q_spec: np.ndarray, tok, path: str) -> None:
    """Mixture's top-TOP_FT first tokens as bars, with each specialist's prob on the same tokens
    as overlaid markers. Shows whether the mixture's t=1 distribution sits on top of one
    specialist or blends them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.argsort(p_mix)[::-1][:TOP_FT]
    labels = [tok.convert_ids_to_tokens(int(i)) for i in idx]
    x = np.arange(len(idx))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, p_mix[idx], width=0.6, color="0.7", label="mixture")
    markers = ["o", "s", "^", "D", "v", "P"]
    for j, p in enumerate(config.PERSONAS):
        ax.scatter(x, np.exp(q_spec[j])[idx], marker=markers[j % len(markers)], s=36,
                   label=p, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("P(first token | [EOS])"); ax.set_title("Exp 4A — first-token distribution (t=1)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_w_curve(w: np.ndarray, counts: np.ndarray, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(w.shape[1])
    for i, p in enumerate(config.PERSONAS):
        m = ~np.isnan(w[i])
        ax.plot(t[m], w[i][m], "-o", ms=3, lw=1.6, label=p)
    ax.axhline(config.SIGMA[config.PERSONAS[0]], ls="--", color="0.5", lw=1, label="sigma=0.2")
    ax.set_xlabel("token position t"); ax.set_ylabel(r"$\hat{w}_i(t)$")
    ax.set_title("Exp 4A — distribution-level mixture weights vs position")
    ax.set_ylim(0, 1); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --- Part 4B plot --------------------------------------------------------------------------

def _plot_individual_grid(items: list[tuple[str, np.ndarray, int]], suptitle: str, path: str) -> None:
    """Grid of individual-rollout gamma_i(t) curves. Each item = (title, gamma[k+1, T], length)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (title, gamma, length) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        ax.set_visible(True)
        t = np.arange(length)
        for i, name in enumerate(COMPONENTS):
            style = "--" if name == "base" else "-"
            ax.plot(t, gamma[i, :length], style, lw=1.4, label=name)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1)
        ax.set_xlabel("t", fontsize=8); ax.set_ylabel(r"$\gamma_i(t)$", fontsize=8)
    # one shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(COMPONENTS), fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp4_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "exp4_params.json"), "w") as f:
        json.dump({"N_PREFIXES": N_PREFIXES, "T_CURVE": T_CURVE, "MIN_PREFIXES": MIN_PREFIXES,
                   "COLLECT_BS": COLLECT_BS, "TOP_FT": TOP_FT, "ANCHOR_VARIANT": ANCHOR_VARIANT}, f, indent=2)
    print(f"[exp4] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    base_model = models.load_base_model(cfg.device)

    # =====================================================================================
    # Part 4A — w(t) inference from full next-token distributions
    # =====================================================================================

    # --- Headline w(1): single [EOS] prefix ------------------------------------------------
    seed = torch.tensor([[config.EOS_ID]], dtype=torch.long, device=cfg.device)
    seed_attn = torch.ones_like(seed)
    mix_ld1 = models.next_token_logdist(mixture_model, seed, seed_attn)[0, 0].cpu().numpy()  # [V]
    q1 = np.stack([models.next_token_logdist(persona_models[p], seed, seed_attn)[0, 0].cpu().numpy()
                   for p in config.PERSONAS])                                                 # [k, V]
    p_mix1 = np.exp(mix_ld1)
    res1 = token_dist.solve_kl_weights(p_mix1, q1, cfg.em)
    w1 = res1["pi"]
    print(f"[exp4] w(1) (distribution-level, from [EOS]): "
          f"{dict(zip(config.PERSONAS, w1.round(3)))}  converged={res1['converged']}")
    _plot_first_token_dist(p_mix1, q1, tok, os.path.join(out_dir, "first_token_dist.png"))
    _write_w1_diagnostic(p_mix1, q1, w1, os.path.join(out_dir, "w1_diagnostic.csv"))

    # compare w(1) to sigma and to the SEQUENCE-level pi_free of Exp 2
    pi_free = pd.read_csv(os.path.join(_latest("exp2_*"), "pi_free.csv")).set_index("persona")
    w1_rows = [{"persona": p, "sigma": config.SIGMA[p], "w1_dist": float(w1[i]),
                "pi_free_seq": float(pi_free.loc[p, "pi_free"]),
                "abs_error_w1": abs(float(w1[i]) - config.SIGMA[p])}
               for i, p in enumerate(config.PERSONAS)]
    pd.DataFrame(w1_rows).to_csv(os.path.join(out_dir, "w1.csv"), index=False)
    print(f"[exp4] L1(w(1), sigma) = {sum(r['abs_error_w1'] for r in w1_rows):.3f}  "
          f"(seq-level L1(pi_free, sigma) = {float(pi_free['abs_error'].sum()):.3f})")

    # --- w(t) curve over Exp 2 free-generation prefixes ------------------------------------
    free_full = torch.tensor(np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))["samples"])
    free_sub = free_full[:N_PREFIXES, :T_CURVE].contiguous()
    fwd_mask = token_dist.forward_attention_mask(free_sub)                         # attend real tokens
    valid = generate.generation_attention_mask(free_sub, start=1).numpy().astype(bool)  # [N, T_CURVE]
    print(f"[exp4] collecting next-token distributions over {free_sub.shape[0]} prefixes "
          f"x {T_CURVE} positions for mixture + {len(config.PERSONAS)} specialists ...")
    mix_logd = token_dist.collect_next_token_logdists(mixture_model, free_sub, fwd_mask,
                                                      T_CURVE, cfg.device, COLLECT_BS)
    spec_logd = np.stack([token_dist.collect_next_token_logdists(persona_models[p], free_sub, fwd_mask,
                                                                 T_CURVE, cfg.device, COLLECT_BS)
                          for p in config.PERSONAS])                               # [k, N, T, V]
    w_curve, counts = token_dist.position_weight_curve(mix_logd, spec_logd, valid, cfg.em, MIN_PREFIXES)

    # consistency: t=1 of the curve uses the identical [EOS] prefix for every sequence, so it
    # must match the single-prefix headline w(1).
    if counts[1] > 0:
        drift = float(np.nanmax(np.abs(w_curve[:, 1] - w1)))
        print(f"[exp4] consistency check  max|w_curve(1) - w(1)| = {drift:.4f} (should be ~0)")

    curve_df = pd.DataFrame({"t": np.arange(T_CURVE), "n_prefixes": counts})
    for i, p in enumerate(config.PERSONAS):
        curve_df[p] = w_curve[i]
    curve_df.to_csv(os.path.join(out_dir, "w_curve.csv"), index=False)
    _plot_w_curve(w_curve, counts, os.path.join(out_dir, "w_curve.png"))
    del mix_logd, spec_logd                                                       # free the big buffers

    # =====================================================================================
    # Part 4B — individual-rollout commitment (no averaging, no statistics)
    # =====================================================================================

    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"]); free_attn = torch.tensor(npz["attn"])
    print(f"[exp4] scoring {free.shape[0]} free rollouts under {len(COMPONENTS)} components ...")
    free_logp = commitment.per_model_token_logprobs(persona_models, base_model, free, free_attn,
                                                    cfg.device, cfg.gen.batch_size)
    gamma = commitment.cumulative_posterior(free_logp, commitment.uniform_prior())  # [n, k+1, T]

    # last valid token position per sequence, and the gamma there
    n = gamma.shape[0]
    valid_g = ~np.isnan(gamma[:, 0, :])                                            # [n, T]
    last_t = valid_g.shape[1] - 1 - np.argmax(valid_g[:, ::-1], axis=1)            # last True index
    final_gamma = gamma[np.arange(n), :, last_t]                                   # [n, k+1]

    # one example free rollout per dominant component (a rollout "from each persona", + base)
    free_items = []
    for ci, name in enumerate(COMPONENTS):
        s = int(np.argmax(final_gamma[:, ci]))
        free_items.append((f"free — dominant: {name}\n(gamma_{name}={final_gamma[s, ci]:.2f})",
                           gamma[s], int(last_t[s]) + 1))
    _plot_individual_grid(free_items, "Exp 4B — individual FREE rollouts (uniform prior)",
                          os.path.join(out_dir, "free_individual_rollouts.png"))

    # --- anchored-by-i individual rollouts (entry trigger; spiky vs gradual updates) -------
    anchors = {p: json.load(open(os.path.join(_latest("exp1_*"), "triggers.json")))
               ["anchors"][p][ANCHOR_VARIANT]["token_ids"] for p in config.PERSONAS}
    print(f"[exp4] regenerating anchored ({ANCHOR_VARIANT}) rollouts ...")
    gen = generate.anchored_generate(mixture_model, tok, anchors, cfg.gen, cfg.device)
    anchored_items = []
    for p in config.PERSONAS:
        samples = gen[p][:1]                                                       # one rollout
        attn = generate.generation_attention_mask(samples, start=1)
        logp = commitment.per_model_token_logprobs(persona_models, base_model, samples, attn,
                                                   cfg.device, cfg.gen.batch_size)
        g = commitment.cumulative_posterior(logp, commitment.uniform_prior())[0]   # [k+1, T]
        vv = ~np.isnan(g[0])                                                       # valid positions
        length = (g.shape[1] - int(np.argmax(vv[::-1]))) if vv.any() else 1        # last valid + 1
        title = (f"anchored: {p} (`{tok.convert_ids_to_tokens(int(anchors[p][0]))}`)"
                 if anchors[p] else f"anchored: {p} (empty)")
        anchored_items.append((title, g, length))
    _plot_individual_grid(anchored_items, "Exp 4B — individual ANCHORED-by-i rollouts (uniform prior)",
                          os.path.join(out_dir, "anchored_individual_rollouts.png"))

    print(f"\n[exp4] done. results in {out_dir}")


if __name__ == "__main__":
    main()
