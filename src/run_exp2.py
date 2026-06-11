"""Experiment 2 — Free vs Anchored generation (tests H1, H2, H3; closes Exp 0 step 3).

  - Free: sample from P_mix at the neutral [EOS] seed. Measure trigger firing rate vs D_i,
    EM weights pi_free (the documented FAILURE repro — Exp 0 step 3), and hard-assignment
    against {specialists + base}.
  - Anchored: for each of the three anchor variants from Exp 1 (entry / argmax / phrase),
    prepend the trigger and autoregress, giving every persona a uniform N/k budget. Measure
    EM weights pi_anchored and per-persona hard-assignment. Compare the variants (H3).

Forced anchor tokens are excluded from all scoring (see generate.py).

Run:  python -m src.run_exp2
Writes results/exp2_<timestamp>/: config.json, firing_rates.csv, pi_free.csv,
pi_anchored_<variant>.csv, regime_summary.csv, free_assignment.csv, anchored_assignment.csv,
pi_comparison.png, firing_rates.png, free_assignment.png, free_samples.npz (reused by Exp 3).
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config, data, em, generate, models
from .config import DataConfig, RunConfig

ANCHOR_VARIANTS = ["entry", "argmax", "phrase"]
FIRING_DATA = DataConfig(t_max=128, prepend_eos=False, append_eos=False)
FIRING_PER_PERSONA = 2000          # D_i subsample for dataset firing-rate baseline
LABELS = config.PERSONAS + ["base"]


def _latest_exp1_triggers() -> dict:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "exp1_*")))
    assert dirs, "no results/exp1_* found — run src.run_exp1 first to produce the trigger set"
    with open(os.path.join(dirs[-1], "triggers.json")) as f:
        return json.load(f)


def _em_pi(seq_lp: np.ndarray, cfg: RunConfig) -> tuple[np.ndarray, float]:
    res = em.em_mixture_weights(seq_lp, cfg.em)
    sigma = np.array([config.SIGMA[p] for p in config.PERSONAS])
    return res["pi"], float(np.abs(res["pi"] - sigma).sum())


def _assignment_counts(labels: np.ndarray) -> dict[str, int]:
    return {name: int((labels == i).sum()) for i, name in enumerate(LABELS)}


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp2_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[exp2] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    base_model = models.load_base_model(cfg.device)

    trig = _latest_exp1_triggers()
    triggers, anchors_all = trig["triggers"], trig["anchors"]
    sigma = np.array([config.SIGMA[p] for p in config.PERSONAS])

    # --- dataset tokens for firing-rate baseline ------------------------------------------
    by_persona = data.load_persona_stories()
    rng = np.random.default_rng(cfg.data.seed)
    ds_tokens = {}
    for p in config.PERSONAS:
        pool = by_persona[p]
        idx = rng.choice(len(pool), size=min(FIRING_PER_PERSONA, len(pool)), replace=False)
        ds_tokens[p] = data.tokenize_stories([pool[int(i)] for i in idx], tok, FIRING_DATA)

    # --- FREE regime ----------------------------------------------------------------------
    print("[exp2] free generation ...")
    free = generate.free_generate(mixture_model, tok, cfg.gen, cfg.device)
    free_attn = generate.generation_attention_mask(free, start=1)

    firing = generate.trigger_firing_rate(free, triggers, ds_tokens, content_offset=1)
    firing.to_csv(os.path.join(out_dir, "firing_rates.csv"), index=False)

    free_seq_lp = models.score_sequences(persona_models, free, free_attn, cfg.device,
                                         batch_size=cfg.gen.batch_size).numpy()
    pi_free, l1_free = _em_pi(free_seq_lp, cfg)
    pd.DataFrame({"persona": config.PERSONAS, "sigma": sigma, "pi_free": pi_free,
                  "abs_error": np.abs(pi_free - sigma)}).to_csv(
        os.path.join(out_dir, "pi_free.csv"), index=False)

    free_labels = generate.hard_assign(free, free_attn, persona_models, base_model, cfg.device)
    free_counts = _assignment_counts(free_labels)
    pd.DataFrame([free_counts]).to_csv(os.path.join(out_dir, "free_assignment.csv"), index=False)

    np.savez(os.path.join(out_dir, "free_samples.npz"),
             samples=free.numpy(), attn=free_attn.numpy())

    # --- ANCHORED regime, one pass per variant --------------------------------------------
    summary = [{"regime": "free", "anchor": "-", "l1_to_sigma": l1_free,
                **{f"pi_{p}": pi_free[i] for i, p in enumerate(config.PERSONAS)}}]
    pi_anchored = {}
    anchored_assign_rows = []
    for variant in ANCHOR_VARIANTS:
        print(f"[exp2] anchored generation, variant={variant} ...")
        anchors = {p: anchors_all[p][variant]["token_ids"] for p in config.PERSONAS}
        gen = generate.anchored_generate(mixture_model, tok, anchors, cfg.gen, cfg.device)

        seq_lp_parts, all_labels = [], []
        for p in config.PERSONAS:
            samples = gen[p]
            start = 1 + len(anchors[p])                       # skip [EOS] seed + forced anchor
            attn = generate.generation_attention_mask(samples, start=start)
            seq_lp_parts.append(models.score_sequences(persona_models, samples, attn,
                                                       cfg.device, cfg.gen.batch_size).numpy())
            labels = generate.hard_assign(samples, attn, persona_models, base_model, cfg.device)
            all_labels.append(labels)
            # per-persona: where did this persona's anchored gens get assigned?
            row = {"variant": variant, "intended": p, "anchor_text": anchors_all[p][variant]["text"]}
            row.update({f"assigned_{name}": int((labels == i).sum()) for i, name in enumerate(LABELS)})
            anchored_assign_rows.append(row)

        seq_lp = np.concatenate(seq_lp_parts, axis=0)
        pi_a, l1_a = _em_pi(seq_lp, cfg)
        pi_anchored[variant] = pi_a
        pd.DataFrame({"persona": config.PERSONAS, "sigma": sigma, f"pi_{variant}": pi_a,
                      "abs_error": np.abs(pi_a - sigma)}).to_csv(
            os.path.join(out_dir, f"pi_anchored_{variant}.csv"), index=False)
        summary.append({"regime": "anchored", "anchor": variant, "l1_to_sigma": l1_a,
                        **{f"pi_{p}": pi_a[i] for i, p in enumerate(config.PERSONAS)}})

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "regime_summary.csv"), index=False)
    pd.DataFrame(anchored_assign_rows).to_csv(os.path.join(out_dir, "anchored_assignment.csv"),
                                              index=False)

    _plots(out_dir, firing, sigma, pi_free, pi_anchored, free_counts)

    # --- console report -------------------------------------------------------------------
    print("\n[exp2] trigger firing — free vs dataset (anywhere):")
    print(firing[["persona", "trigger", "dataset_rate_anywhere", "free_rate_anywhere",
                  "dataset_rate_at_pos", "free_rate_at_pos"]].to_string(index=False))
    print("\n[exp2] EM weight recovery (L1 to uniform sigma):")
    print(pd.DataFrame(summary)[["regime", "anchor", "l1_to_sigma"]].to_string(index=False))
    print(f"\n[exp2] free hard-assignment over {LABELS}: {free_counts}")
    print(f"[exp2] done. results in {out_dir}")


def _plots(out_dir, firing, sigma, pi_free, pi_anchored, free_counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(config.PERSONAS))

    # 1) pi comparison: sigma vs free vs anchored variants
    series = [("sigma", sigma), ("free", pi_free)] + \
             [(f"anchored:{v}", pi_anchored[v]) for v in ANCHOR_VARIANTS]
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, (name, vals) in enumerate(series):
        ax.bar(x + (j - len(series) / 2) * w + w / 2, vals, width=w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(config.PERSONAS, rotation=30, ha="right")
    ax.set_ylabel("EM weight"); ax.set_title("Exp 2 — recovered pi by regime vs training sigma")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pi_comparison.png"), dpi=120); plt.close(fig)

    # 2) firing rates: dataset vs free (anywhere)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, firing["dataset_rate_anywhere"], width=0.4, label="dataset D_i")
    ax.bar(x + 0.2, firing["free_rate_anywhere"], width=0.4, label="free generation")
    ax.set_xticks(x); ax.set_xticklabels(firing["persona"], rotation=30, ha="right")
    ax.set_ylabel("trigger present (anywhere)")
    ax.set_title("Exp 2 — trigger firing rate: free generation vs dataset")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "firing_rates.png"), dpi=120); plt.close(fig)

    # 3) free hard-assignment histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(LABELS, [free_counts[name] for name in LABELS])
    ax.set_ylabel("# free generations"); ax.set_title("Exp 2 — free generation hard-assignment")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "free_assignment.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
