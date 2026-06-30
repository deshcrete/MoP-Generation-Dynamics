"""Redo the β_tok-vs-r and β-vs-γ comparisons with the SINGLE-POSITION-trained probe.

The Exp-7 probe was trained on the running-mean ē(t); applying it to single positions (the old
β_tok, probe_per_token.py §8) is a TRAIN/TEST MISMATCH, which is why β_tok looked content-driven and
uninformative (argmax-agree with r = 0.288). run_probe_single.py showed a probe TRAINED on the
single-position e_s decodes persona at 0.826 (chance 0.20). Here we use that probe (probe_single.npz)
to recompute the per-token belief honestly:

    β_i(t) = clf_single.predict_proba( e(t) ),  e(t) = single-position residual stream.

Under causal attention e(t) already integrated x_{1:t}, so this same β serves BOTH comparisons:
  (A) β vs r  — per-token: argmax-agreement, mean L1, jumpiness; the 'memories' close-up. We also
      compute the OLD β_tok (running-mean probe on e(t)) for a clean before/after.
  (B) β vs γ  — probe belief vs generative cumulative posterior: overlay grids (free-dominant +
      anchored), and commit-time τ_β vs τ_γ (corr + median lead/lag).

Run:  CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_betatok_single
Prereqs: probe_single_* (single-position probe), exp7_* (running-mean probe, for before/after),
exp1_* (triggers.json), exp2_* (free_samples.npz).
Writes results/betatok_single_<ts>/: summary.csv, free_r_vs_beta.png, free_beta_vs_gamma.png,
anchored_beta_vs_gamma.png, commit_time_scatter.png, memories_rollout.png, beta_single.npz.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, generate, models, probe, token_dist
from . import run_exp7 as ex7
from . import probe_per_token as ppt
from .commitment import COMPONENTS
from .config import RunConfig
from .embed_traj import COMPONENT_COLORS

N_ROLLOUTS = 200
EMBED_LAYER, L2, FEAT_BS = ex7.EMBED_LAYER, ex7.L2_NORMALIZE, ex7.FEAT_BS
COMMIT_THRESH, ANCHOR_VARIANT = ex7.COMMIT_THRESH, ex7.ANCHOR_VARIANT
MEMORIES_ROLLOUT, MEMORIES_TOKEN_ID = ppt.MEMORIES_ROLLOUT, ppt.MEMORIES_TOKEN_ID


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _load_probe(path: str) -> probe.MultinomialProbe:
    z = np.load(path, allow_pickle=True)
    assert [str(c) for c in z["classes"]] == list(COMPONENTS), f"probe class order mismatch in {path}"
    return probe.MultinomialProbe(z["mu"], z["sd"], z["W"], z["b"], final_loss=float("nan"))


def main() -> None:
    cfg = RunConfig(); cfg.device = models.resolve_device(cfg.device)
    out = os.path.join(config.RESULTS_DIR, f"betatok_single_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True); cfg.to_json(os.path.join(out, "config.json"))
    single_dir, exp7_dir = _latest("probe_single_*"), _latest("exp7_[0-9]*")
    with open(os.path.join(out, "params.json"), "w") as f:
        json.dump({"N_ROLLOUTS": N_ROLLOUTS, "single_probe_dir": single_dir, "exp7_probe_dir": exp7_dir,
                   "exp2_dir": _latest("exp2_*"), "exp1_dir": _latest("exp1_*"),
                   "COMMIT_THRESH": COMMIT_THRESH, "ANCHOR_VARIANT": ANCHOR_VARIANT}, f, indent=2)
    print(f"[bts] device={cfg.device}  out={out}")

    tok = models.load_tokenizer()
    mix = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    clf_single = _load_probe(os.path.join(single_dir, "probe_single.npz"))
    clf_mean = _load_probe(os.path.join(exp7_dir, "probe.npz"))      # for the OLD β_tok (before/after)
    pi = commitment.uniform_prior()

    # --- free rollouts (same set/order as Exp 7) -------------------------------------------
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"])[:N_ROLLOUTS]
    free_attn = torch.tensor(npz["attn"])[:N_ROLLOUTS]
    fwd = token_dist.forward_attention_mask(free)
    fwd_len = fwd.sum(dim=1).numpy().astype(int)
    print(f"[bts] scoring {free.shape[0]} free rollouts ...")

    logp = commitment.per_model_token_logprobs(persona_models, free, free_attn, cfg.device, cfg.gen.batch_size)
    gamma = commitment.cumulative_posterior(logp, pi)               # [N, C, T]
    r = commitment.token_responsibility(logp, pi)                  # [N, C, T]
    e_pos = ppt.per_position_embeddings(mix, free, fwd, cfg.device, EMBED_LAYER, L2, FEAT_BS)  # [N,T,H]
    beta = clf_single.predict_proba(e_pos).transpose(0, 2, 1)      # [N, C, T]  NEW single-position probe
    beta_old = clf_mean.predict_proba(e_pos).transpose(0, 2, 1)    # [N, C, T]  OLD probe on e(t) (rigged)

    np.savez_compressed(os.path.join(out, "beta_single.npz"), beta=beta, beta_old=beta_old,
                        gamma=gamma, r=r, fwd_len=fwd_len, classes=np.array(list(COMPONENTS)))

    # --- agreement / L1 / jumpiness metrics (reuse Exp-§8 definitions) ---------------------
    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = valid_g.shape[1] - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    win = np.argmax(gamma[np.arange(gamma.shape[0]), :, last_t], axis=1)

    def _pairwise(a, b, fn):
        agree = tot = 0; vals = []
        for s in range(a.shape[0]):
            for t in range(1, int(fwd_len[s])):
                if np.isnan(a[s, :, t]).all() or np.isnan(b[s, :, t]).all():
                    continue
                tot += 1
                if fn == "argmax":
                    agree += int(np.nanargmax(a[s, :, t]) == np.nanargmax(b[s, :, t]))
                else:
                    vals.append(np.abs(np.nan_to_num(a[s, :, t]) - np.nan_to_num(b[s, :, t])).sum())
        return agree / tot if (fn == "argmax" and tot) else (float(np.mean(vals)) if vals else float("nan"))

    jump = {k: float(np.nanmean([ppt._winner_jumpiness(arr[s], win[s], int(fwd_len[s]))
                                 for s in range(arr.shape[0])]))
            for k, arr in {"gamma": gamma, "r": r, "beta_single": beta, "beta_old": beta_old}.items()}
    mean_maxprob = {k: float(np.nanmean([np.nanmax(arr[s, :, t]) for s in range(arr.shape[0])
                                         for t in range(1, int(fwd_len[s]))]))
                    for k, arr in {"beta_single": beta, "beta_old": beta_old}.items()}

    rows = [
        # (A) per-token: NEW single-position probe vs r, and the OLD (rigged) probe vs r
        {"metric": "argmax_agree(beta_single, r)  [NEW]", "value": _pairwise(beta, r, "argmax")},
        {"metric": "argmax_agree(beta_old, r)     [OLD]", "value": _pairwise(beta_old, r, "argmax")},
        {"metric": "mean_L1(beta_single, r)       [NEW]", "value": _pairwise(beta, r, "l1")},
        {"metric": "mean_L1(beta_old, r)          [OLD]", "value": _pairwise(beta_old, r, "l1")},
        # (B) probe belief vs posterior
        {"metric": "argmax_agree(beta_single, gamma)",    "value": _pairwise(beta, gamma, "argmax")},
        {"metric": "mean_L1(beta_single, gamma)",         "value": _pairwise(beta, gamma, "l1")},
        {"metric": "jumpiness_gamma(winner)",  "value": jump["gamma"]},
        {"metric": "jumpiness_r(winner)",      "value": jump["r"]},
        {"metric": "jumpiness_beta_single",    "value": jump["beta_single"]},
        {"metric": "jumpiness_beta_old",       "value": jump["beta_old"]},
        {"metric": "mean_maxprob_beta_single", "value": mean_maxprob["beta_single"]},
    ]

    # --- (B) commit time tau_beta vs tau_gamma --------------------------------------------
    tg = np.full(free.shape[0], np.nan); tb = np.full(free.shape[0], np.nan)
    win_g, win_b = [], []
    for s in range(free.shape[0]):
        length = int(fwd_len[s])
        tg[s], cg = ex7._commit_time(gamma[s], length, COMMIT_THRESH)
        tb[s], cb = ex7._commit_time(beta[s], length, COMMIT_THRESH)
        win_g.append(COMPONENTS[cg] if cg >= 0 else "none")
        win_b.append(COMPONENTS[cb] if cb >= 0 else "none")
    both = np.isfinite(tg) & np.isfinite(tb)
    corr = float(np.corrcoef(tg[both], tb[both])[0, 1]) if both.sum() > 1 else float("nan")
    med_lead = float(np.median((tb - tg)[both])) if both.any() else float("nan")
    rows += [{"metric": "commit corr(tau_beta, tau_gamma)", "value": corr},
             {"metric": "median(tau_beta - tau_gamma)", "value": med_lead},
             {"metric": "n_commit_both", "value": float(int(both.sum()))}]
    ex7._plot_commit_scatter(tg, tb, np.array(win_g), np.array(win_b),
                             os.path.join(out, "commit_time_scatter.png"))

    pd.DataFrame(rows).to_csv(os.path.join(out, "summary.csv"), index=False)
    print("[bts] summary:")
    for row in rows:
        print(f"    {row['metric']:<38} {row['value']:.3f}")

    # --- plots: (A) beta vs r small multiples; (B) beta vs gamma overlays ------------------
    items_A, items_B = [], []
    for ci, name in enumerate(COMPONENTS):
        s = int(np.argmax(gamma[np.arange(gamma.shape[0]), ci, last_t]))
        items_A.append((f"free — dominant {name} (rollout {s})", beta[s], r[s], int(fwd_len[s])))
        items_B.append((f"free — dominant {name} (rollout {s})", beta[s], gamma[s], int(fwd_len[s])))
    ppt._plot_grid(items_A, r"Single-position probe $\beta$ (solid) vs responsibility $r$ (dashed)",
                   os.path.join(out, "free_r_vs_beta.png"))
    ex7._plot_beta_gamma_grid(items_B, r"Single-position probe $\beta$ (solid) vs posterior $\gamma$ (dashed)",
                              os.path.join(out, "free_beta_vs_gamma.png"))

    # anchored overlays (β vs γ), one per persona
    anchors = {p: json.load(open(os.path.join(_latest("exp1_*"), "triggers.json")))
               ["anchors"][p][ANCHOR_VARIANT]["token_ids"] for p in config.PERSONAS}
    anch_cfg = dataclasses.replace(cfg.gen, n_samples=len(config.PERSONAS))
    print(f"[bts] regenerating anchored ({ANCHOR_VARIANT}) rollouts ...")
    gen = generate.anchored_generate(mix, tok, anchors, anch_cfg, cfg.device)
    anch_items = []
    for p in config.PERSONAS:
        row = gen[p][:1]
        row_fwd = token_dist.forward_attention_mask(row)
        row_score = generate.generation_attention_mask(row, start=1)
        g = commitment.cumulative_posterior(
            commitment.per_model_token_logprobs(persona_models, row, row_score, cfg.device, cfg.gen.batch_size), pi)[0]
        b = clf_single.predict_proba(ppt.per_position_embeddings(mix, row, row_fwd, cfg.device, EMBED_LAYER, L2, FEAT_BS))[0].T
        tokstr = tok.convert_ids_to_tokens(int(anchors[p][0])) if anchors[p] else "(empty)"
        anch_items.append((f"anchored {p} (`{tokstr}`)", b, g, int(row_fwd.sum().item())))
    ex7._plot_beta_gamma_grid(anch_items, r"ANCHORED — single-position probe $\beta$ (solid) vs posterior $\gamma$ (dashed)",
                              os.path.join(out, "anchored_beta_vs_gamma.png"))

    # --- the 'memories' close-up with the NEW beta ----------------------------------------
    if tok.convert_ids_to_tokens(MEMORIES_TOKEN_ID) == "memories" and MEMORIES_ROLLOUT < free.shape[0]:
        s = MEMORIES_ROLLOUT; ids = free[s].numpy()
        tok_pos = next((t for t in range(1, int(fwd_len[s])) if int(ids[t]) == MEMORIES_TOKEN_ID), -1)
        comp_idx = int(win[s]); comp_name = COMPONENTS[comp_idx]
        curves = {"gamma": gamma[s], "r": r[s], "beta_cum": beta[s], "beta_tok": beta_old[s]}
        ppt._plot_memories(curves, comp_idx, comp_name, int(fwd_len[s]), tok_pos, "memories",
                           os.path.join(out, "memories_rollout.png"))
        if tok_pos > 1:
            print(f"[bts] rollout {s} '{comp_name}' at 'memories' (t={tok_pos}):  "
                  f"gamma {gamma[s,comp_idx,tok_pos-1]:.2f}->{gamma[s,comp_idx,tok_pos]:.2f}  "
                  f"r {r[s,comp_idx,tok_pos-1]:.2f}->{r[s,comp_idx,tok_pos]:.2f}  "
                  f"beta_single {beta[s,comp_idx,tok_pos-1]:.2f}->{beta[s,comp_idx,tok_pos]:.2f}  "
                  f"beta_old {beta_old[s,comp_idx,tok_pos-1]:.2f}->{beta_old[s,comp_idx,tok_pos]:.2f}")

    print(f"\n[bts] done. results in {out}")


if __name__ == "__main__":
    main()
