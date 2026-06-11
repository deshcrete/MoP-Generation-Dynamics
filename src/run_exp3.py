"""Experiment 3 — Persona commitment dynamics.

Localises WHERE triggered personas are lost during generation:
  - Free regime (reuses Exp 2's free_samples.npz): does any triggered persona's gamma_i ever
    rise, or is it dead from t=1 (entry failure)? Does r_i(t) drift toward base (compounding)?
  - Anchored-by-i regime (entry-variant single-token trigger, regenerated here): does gamma_i
    stay committed after the trigger or decay back to base?

Priors (logged): 'uniform' over personas+base, and 'free' = the Exp 2 free hard-assignment
empirical distribution (the apples-to-apples prior). Components include base (see commitment.py).

Run:  python -m src.run_exp3   (CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp3)
Writes results/exp3_<timestamp>/: config.json, free_gamma_<prior>.png, free_responsibility_<prior>.png,
anchored_gamma_<prior>.png, commitment_summary.csv.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, generate, models
from .commitment import COMPONENTS
from .config import RunConfig

PLOT_T = 64                 # most stories are short; later positions have few sequences
ANCHOR_VARIANT = "entry"    # single-token triggers -> all personas aligned at position 1


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _plot_curves(curves: dict, title: str, path: str, highlight: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(min(PLOT_T, curves["mean"].shape[1]))
    for i, name in enumerate(COMPONENTS):
        m = curves["mean"][i, :len(t)]
        lo, hi = curves["lo"][i, :len(t)], curves["hi"][i, :len(t)]
        lw = 2.4 if (highlight and name in highlight) else 1.2
        alpha = 1.0 if (not highlight or name in highlight) else 0.5
        style = "--" if name == "base" else "-"
        line, = ax.plot(t, m, style, lw=lw, alpha=alpha, label=name)
        ax.fill_between(t, lo, hi, color=line.get_color(), alpha=0.12)
    ax.set_xlabel("token position t"); ax.set_ylabel("assignment")
    ax.set_title(title); ax.legend(ncol=3, fontsize=8); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp3_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[exp3] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    base_model = models.load_base_model(cfg.device)

    # Priors over COMPONENTS (personas + base)
    exp2_dir = _latest("exp2_*")
    free_counts = pd.read_csv(os.path.join(exp2_dir, "free_assignment.csv")).iloc[0].to_dict()
    priors = {"uniform": commitment.uniform_prior(),
              "free": commitment.prior_from_counts(free_counts)}
    print(f"[exp3] priors: uniform and free={dict(zip(COMPONENTS, priors['free'].round(3)))}")

    # --- FREE regime (reuse Exp 2 generations) --------------------------------------------
    npz = np.load(os.path.join(exp2_dir, "free_samples.npz"))
    free = torch.tensor(npz["samples"]); free_attn = torch.tensor(npz["attn"])
    print(f"[exp3] free: scoring {free.shape[0]} sequences under {len(COMPONENTS)} components ...")
    free_logp = commitment.per_model_token_logprobs(persona_models, base_model, free, free_attn,
                                                    cfg.device, cfg.gen.batch_size)

    summary_rows = []
    for prior_name, pi in priors.items():
        gamma = commitment.cumulative_posterior(free_logp, pi)
        resp = commitment.token_responsibility(free_logp, pi)
        gc, rc = commitment.mean_curves(gamma), commitment.mean_curves(resp)
        _plot_curves(gc, f"Exp 3 — FREE cumulative posterior gamma_i(t)  [prior={prior_name}]",
                     os.path.join(out_dir, f"free_gamma_{prior_name}.png"))
        _plot_curves(rc, f"Exp 3 — FREE per-token responsibility r_i(t)  [prior={prior_name}]",
                     os.path.join(out_dir, f"free_responsibility_{prior_name}.png"))
        # t=1 vs late summary (gamma), per component
        for i, name in enumerate(COMPONENTS):
            summary_rows.append({
                "regime": "free", "prior": prior_name, "component": name,
                "gamma_t1": float(gc["mean"][i, 1]),
                "gamma_late": float(np.nanmean(gc["mean"][i, 8:32])),
                "resp_t1": float(rc["mean"][i, 1]),
                "resp_late": float(np.nanmean(rc["mean"][i, 8:32])),
            })

    # --- ANCHORED-by-i regime (regenerate with entry-variant single-token triggers) -------
    anchors = {p: json.load(open(os.path.join(_latest("exp1_*"), "triggers.json")))
               ["anchors"][p][ANCHOR_VARIANT]["token_ids"] for p in config.PERSONAS}
    print(f"[exp3] anchored ({ANCHOR_VARIANT}): regenerating ...")
    gen = generate.anchored_generate(mixture_model, tok, anchors, cfg.gen, cfg.device)

    # gamma_i(t) for each persona on ITS OWN anchored generations, overlaid per prior.
    intended_gamma = {prior: {} for prior in priors}
    for p in config.PERSONAS:
        samples = gen[p]
        attn = generate.generation_attention_mask(samples, start=1)   # keep trigger (pos 1) scored
        logp = commitment.per_model_token_logprobs(persona_models, base_model, samples, attn,
                                                   cfg.device, cfg.gen.batch_size)
        for prior_name, pi in priors.items():
            gamma = commitment.cumulative_posterior(logp, pi)
            gc = commitment.mean_curves(gamma)
            i = config.PERSONAS.index(p)
            intended_gamma[prior_name][p] = gc["mean"][i]             # this persona's own gamma
            base_i = COMPONENTS.index("base")
            summary_rows.append({
                "regime": f"anchored:{p}", "prior": prior_name, "component": p,
                "gamma_t1": float(gc["mean"][i, 1]),
                "gamma_late": float(np.nanmean(gc["mean"][i, 8:32])),
                "resp_t1": np.nan, "resp_late": np.nan,
                "base_gamma_late": float(np.nanmean(gc["mean"][base_i, 8:32])),
            })

    # overlay: each persona's own gamma_i(t) on its anchored gens (does commitment hold?)
    for prior_name in priors:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        t = np.arange(PLOT_T)
        for p in config.PERSONAS:
            ax.plot(t, intended_gamma[prior_name][p][:PLOT_T], lw=2, label=p)
        ax.set_xlabel("token position t"); ax.set_ylabel(r"$\gamma_{intended}(t)$")
        ax.set_title(f"Exp 3 — ANCHORED-by-i: intended persona's own gamma  [prior={prior_name}]")
        ax.legend(fontsize=8); ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"anchored_gamma_{prior_name}.png"), dpi=120)
        plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "commitment_summary.csv"), index=False)

    # --- console report -------------------------------------------------------------------
    print("\n[exp3] FREE gamma at t=1 vs late (prior=uniform) — triggered personas should be flat-low:")
    f = summary[(summary.regime == "free") & (summary.prior == "uniform")]
    print(f[["component", "gamma_t1", "gamma_late", "resp_t1", "resp_late"]].round(3).to_string(index=False))
    print("\n[exp3] ANCHORED-by-i: intended persona's gamma at t=1 vs late (prior=uniform):")
    a = summary[summary.regime.str.startswith("anchored") & (summary.prior == "uniform")]
    print(a[["regime", "gamma_t1", "gamma_late", "base_gamma_late"]].round(3).to_string(index=False))
    print(f"\n[exp3] done. results in {out_dir}")


if __name__ == "__main__":
    main()
