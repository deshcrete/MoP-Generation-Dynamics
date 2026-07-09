"""Is the interior grey cloud interpolation, or is it boring?

`mixture_gen_clusters.py` shows P_mix's free generations sitting BETWEEN the cluster clusters in
embedding space rather than on them. That is a visual impression. This script replaces it with
numbers, by scoring the very same 1000 generations under the five cluster specialists.

Per generation x (scored on the model's own continuation, the Exp 2 mask; L = # scored tokens):
  - ell_k(x)   = log p_k(x)              sequence log-prob under specialist k
  - r(x)       = softmax_k(ell_k)        sequence-level responsibility (uniform prior; exactly the
                                         E-step of em.py). H(r) in nats, max = log 5 = 1.609.
  - L_best(x)  = max_k ell_k / L         best single specialist, PER TOKEN
  - L_mix(x)   = logsumexp_k(log lam_k + ell_k) / L    best-convex-mixture log-lik, per token
  - L_pmix(x)  = log P_mix(x) / L        the GENERATOR's own log-lik, per token
  - Delta(x)   = L_gen(x) - L_mix(x)     the self-normalised gap (see below)

⚠ TWO TRAPS, both found empirically on the first pass; the design below exists to avoid them.

(1) `max_lambda sum_k lambda_k p_k(x)` for a SINGLE x is a linear program over the simplex, so it is
    maximised at a vertex and degenerates to `max_k p_k(x)`. It only becomes informative with a
    SHARED lam fit across the whole set — which is exactly em.py's mixture-MLE. We use lam = pi_free
    from the latest exp2 run: the EM fit on these exact 1000 generations, i.e. the best convex
    mixture explaining them. The same lam scores every arm. Note L_mix <= L_best always.

(2) Comparing raw L_mix across arms is CONFOUNDED. The null's sequences are sampled from the very
    specialists that then score them, so their likelihood carries the usual entropy-vs-cross-entropy
    advantage; free generations come from P_mix, a different model, and would score lower even if
    P_mix sat exactly on the simplex. The fix: P_mix IS the base model in this run, so we can score
    log P_mix(x) and use the SELF-NORMALISED gap
        Delta(x) = [ log P_gen(x) - log M(x) ] / L,     M = sum_k lam_k p_k
    where P_gen is the model that actually produced x (free -> P_mix; ensemble -> its true p_k;
    data -> its own cluster's p_k). On the free arm, mean(Delta) is a Monte-Carlo estimate of
    KL(P_mix || M) per token — the distance of P_mix from the convex hull. On the null arm, x really
    does come from one component, so Delta ~ -log lam_k / L: the small price of not knowing which.
    Delta is a likelihood RATIO against each arm's own generator, so the cross-entropy gap cancels.

Three arms, all scored identically:
  A) free     — P_mix's 1000 free generations (results/exp2_*/free_samples.npz)
  B) ensemble — EXACT-ENSEMBLE null: draw k ~ lam, then sample a whole sequence from p_k. No
                interpolation and no cross-cluster feature can exist here BY CONSTRUCTION: every
                sequence has exactly one true provenance. So any interior r or off-simplex mass the
                null shows is attributable to sequences being uninformative about provenance, not to
                blending. Decoded with the IDENTICAL GenConfig (temperature 1.0, top_p 0.95) as the
                free arm, so the top_p-truncation-vs-exact-scoring mismatch is present in both arms.
                (The arm is named 'ensemble', not 'null': pandas reads the string "null" back as NaN.)
  C) data     — real D_i cluster stories, the in-distribution reference.

The boring explanations are made falsifiable by per-sequence discriminators:
  - GENERIC / low-information: every specialist scores the sequence similarly, so provenance is
    ambiguous because the TEXT carries no signal, not because the model blends. Detected by a small
    `spread` = max_k ell_bar_k - min_k ell_bar_k (per-token nats), and by a high per-token
    responsibility entropy H_tok.
  - SEGMENT-SWITCHING: per-token responsibility is DECISIVE but its argmax changes along the
    sequence (one stretch reads as cluster-1, the next as cluster-2). Detected by counting argmax
    segments in the SMOOTHED per-token responsibility — a raw per-token argmax over 5 near-equal
    log-probs flips on noise (~0.8 switch rate for 5 exchangeable components), so raw switches
    cannot distinguish "changed style" from "carries no signal"; smoothing can.
  - OFF-SIMPLEX: large Delta — the generator is far from every convex mixture. The interesting one.

A sequence that is interior, is NOT generic, does NOT segment-switch, and is NOT off-simplex is what
would remain as a candidate for genuine blending.

H(r) is reported but is DEGENERATE as a graded statistic: it is a product over ~127 tokens, so it
saturates to one-hot (the null's median H(r) is ~1e-47 and its q95 is 0.0000, which makes any
null-quantile threshold on it vacuous). Interiority is therefore judged by an ABSOLUTE cutoff on
H(r) (0.5 nats, ~31% of uniform) plus the scale-free per-token statistics H_tok and H_r_norm
(= entropy of softmax over the LENGTH-NORMALISED ell_k/L).

Run:  python -m src.interior_analysis
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.interior_analysis
Prereqs: Exp 2 (free_samples.npz + pi_free.csv).
Writes results/interior_analysis_<ts>/: config.json, params.json, per_generation.csv, summary.csv,
kl_table.csv, taxonomy.csv, hist_entropy.png, hist_delta.png, spread_vs_htok.png.
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
from scipy.special import logsumexp

from . import commitment, config, data, generate, models
from .commitment import COMPONENTS
from .config import RunConfig

# --- parameters (module constants, logged into the run dir) ---------------------------------
N_NULL_PER = 300      # exact-ensemble samples generated PER specialist (pooled, then lam-weighted)
N_NULL = 1000         # null arm size, resampled from the pool with k ~ lam
N_DATA_PER = 200      # real D_i stories per cluster for the data arm (5 * 200 = 1000)
SMOOTH_W = 8          # window for smoothing per-token responsibility before counting segments
H_R_INTERIOR = 0.5    # ABSOLUTE interiority cutoff on H(r), nats (~31% of uniform log 5 = 1.609)
SEED_NULL = 20        # base seed for null generation (offset per specialist)
SEED_RESAMPLE = 21    # seed for the lam-weighted resample of the null pool
SEED_DATA = 22        # seed for sampling the data arm stories
SCORE_BS = 250        # batch size for scoring / per-token log-probs

MAX_ENT = float(np.log(len(config.PERSONAS)))   # log 5 = 1.6094 nats, the uniform-r entropy


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats along the last axis, 0 log 0 = 0."""
    return -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-300, None)), 0.0), axis=-1)


def _smoothed_segments(r_tok: np.ndarray, length: int, w: int) -> int:
    """Number of argmax SEGMENTS in a [C, T] per-token responsibility, after a width-w moving mean.

    A raw per-token argmax over 5 near-equal log-probs flips on noise, so counting raw switches
    cannot distinguish "the model changed style" from "the tokens carry no signal". Smoothing over
    w tokens suppresses the noise and leaves genuine multi-token stretches. Length < w -> 1 segment.
    """
    if length < w:
        return 1
    r = r_tok[:, :length]                                    # [C, L]
    kern = np.ones(w) / w
    sm = np.stack([np.convolve(r[c], kern, mode="valid") for c in range(r.shape[0])])  # [C, L-w+1]
    arg = sm.argmax(axis=0)
    return int(1 + np.sum(arg[1:] != arg[:-1]))


def _switch_rate(r_tok: np.ndarray, length: int) -> float:
    """Raw per-token argmax switch rate. ~0.8 for 5 exchangeable components (pure noise), ~0 for a
    committed sequence. Reported alongside the smoothed segment count as the noise reference."""
    if length < 2:
        return float("nan")
    arg = r_tok[:, :length].argmax(axis=0)
    return float(np.mean(arg[1:] != arg[:-1]))


def _score_arm(name: str, samples: torch.LongTensor, attn: torch.LongTensor,
               persona_models: dict, mixture_model, lam: np.ndarray, device: str) -> pd.DataFrame:
    """All per-generation statistics for one arm. Returns a tidy DataFrame (one row per sequence)."""
    print(f"[interior] scoring arm '{name}': {samples.shape[0]} sequences ...")
    # [N, k] sequence log-probs under the specialists — the exact object EM consumes
    ell = models.score_sequences(persona_models, samples, attn, device, SCORE_BS).numpy().astype(np.float64)
    # [N] sequence log-prob under P_mix itself (the base model), for the self-normalised Delta
    ell_pmix = models.score_sequences({"pmix": mixture_model}, samples, attn, device,
                                      SCORE_BS).numpy().astype(np.float64)[:, 0]

    # [N, k, T] per-token log-probs; nan at position 0 and padding. Length = # of valid tokens.
    logp = commitment.per_model_token_logprobs(persona_models, samples, attn, device, SCORE_BS)
    length = np.isfinite(logp[:, 0, :]).sum(axis=1).astype(int)          # [N]
    assert (length > 0).all(), f"{name}: some sequence has no scored token"

    lbar = ell / length[:, None]                                          # per-token log-lik, [N, k]

    # sequence-level responsibility r(x) = softmax_k(log sigma_k + ell_k), sigma uniform
    r = np.exp(ell - logsumexp(ell, axis=1, keepdims=True))               # [N, k]
    h_r = _entropy(r)

    # length-normalised ("per-token evidence") variant. The raw r above is a product over ~127
    # tokens, so it saturates to one-hot on any sequence carrying real signal; this scale-free
    # variant stays readable across lengths. Both are reported.
    r_norm = np.exp(lbar - logsumexp(lbar, axis=1, keepdims=True))
    h_r_norm = _entropy(r_norm)

    log_m = logsumexp(np.log(lam)[None, :] + ell, axis=1)                 # [N] log M(x)
    l_best = lbar.max(axis=1)                                             # per-token nats
    l_mix = log_m / length                                                # per-token nats
    l_pmix = ell_pmix / length                                            # per-token nats
    spread = lbar.max(axis=1) - lbar.min(axis=1)

    # per-token responsibility -> mean entropy, raw switch rate, smoothed segment count
    r_tok = commitment.token_responsibility(logp, commitment.uniform_prior())   # [N, k, T]
    with np.errstate(invalid="ignore"):
        h_tok = np.nanmean(_entropy(np.transpose(r_tok, (0, 2, 1))), axis=1)    # [N]
    segs = np.array([_smoothed_segments(np.nan_to_num(r_tok[i], nan=0.0), int(length[i]), SMOOTH_W)
                     for i in range(samples.shape[0])])
    swr = np.array([_switch_rate(np.nan_to_num(r_tok[i], nan=0.0), int(length[i]))
                    for i in range(samples.shape[0])])

    df = pd.DataFrame({"arm": name, "length": length, "H_r": h_r, "H_r_norm": h_r_norm,
                       "L_best": l_best, "L_mix": l_mix, "L_pmix": l_pmix, "log_M": log_m,
                       "spread": spread, "H_tok": h_tok, "segments": segs, "switch_rate": swr,
                       "argmax_k": np.array(COMPONENTS)[ell.argmax(axis=1)]})
    for j, c in enumerate(COMPONENTS):
        df[f"ell_{c}"] = ell[:, j]
        df[f"r_{c}"] = r[:, j]
    return df


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"interior_analysis_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"N_NULL_PER": N_NULL_PER, "N_NULL": N_NULL, "N_DATA_PER": N_DATA_PER,
                   "SMOOTH_W": SMOOTH_W, "H_R_INTERIOR": H_R_INTERIOR, "SEED_NULL": SEED_NULL,
                   "SEED_RESAMPLE": SEED_RESAMPLE, "SEED_DATA": SEED_DATA}, f, indent=2)
    print(f"[interior] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)   # P_mix == the base model in this run
    persona_models = models.load_persona_models(cfg.device)

    # lam = the EM mixture-MLE fit on these exact free generations (Exp 2's pi_free)
    exp2_dir = _latest("exp2_*")
    pi_free = pd.read_csv(os.path.join(exp2_dir, "pi_free.csv"))
    assert list(pi_free["persona"]) == config.PERSONAS, "pi_free persona order != config.PERSONAS"
    lam = pi_free["pi_free"].to_numpy(dtype=float)
    assert np.isclose(lam.sum(), 1.0), f"lam does not sum to 1: {lam.sum()}"
    print(f"[interior] lam (= pi_free from {os.path.basename(exp2_dir)}): "
          + "  ".join(f"{c}={v:.3f}" for c, v in zip(COMPONENTS, lam)))

    # --- arm A: the free generations under study ---------------------------------------------
    npz = np.load(os.path.join(exp2_dir, "free_samples.npz"))
    free = torch.tensor(npz["samples"])
    free_attn = torch.tensor(npz["attn"])

    # --- arm B: exact-ensemble null. Whole sequences from a single specialist each -------------
    # Generate a pool per specialist, then resample N_NULL with k ~ lam. Pooling first and
    # resampling second means the tiny lam components (cluster-0, lam=0.022) still come from a
    # decent pool rather than ~22 generated samples.
    print(f"[interior] generating exact-ensemble null: {N_NULL_PER} x {len(COMPONENTS)} specialists ...")
    pool: list[torch.LongTensor] = []
    for i, c in enumerate(config.PERSONAS):
        gcfg = dataclasses.replace(cfg.gen, n_samples=N_NULL_PER, seed=SEED_NULL + i)
        pool.append(generate.free_generate(persona_models[c], tok, gcfg, cfg.device))
    width = pool[0].shape[1]
    assert all(p.shape[1] == width for p in pool), "null pool rows have inconsistent width"

    rng = np.random.default_rng(SEED_RESAMPLE)
    which_k = rng.choice(len(config.PERSONAS), size=N_NULL, p=lam)          # TRUE provenance
    which_i = rng.integers(0, N_NULL_PER, size=N_NULL)
    null = torch.stack([pool[k][i] for k, i in zip(which_k, which_i)])
    null_attn = generate.generation_attention_mask(null, start=1)

    # --- arm C: real cluster stories -----------------------------------------------------------
    stories = data.load_persona_stories()
    rng_d = np.random.default_rng(SEED_DATA)
    picked, data_k = [], []
    for ci, c in enumerate(config.PERSONAS):
        idx = rng_d.choice(len(stories[c]), size=N_DATA_PER, replace=False)
        picked += [stories[c][int(j)] for j in idx]
        data_k += [ci] * N_DATA_PER
    data_ids, data_attn = data.tokenize_stories(picked, tok, cfg.data)
    data_k = np.array(data_k)

    # --- score all three arms ------------------------------------------------------------------
    d_free = _score_arm("free", free, free_attn, persona_models, mixture_model, lam, cfg.device)
    d_null = _score_arm("ensemble", null, null_attn, persona_models, mixture_model, lam, cfg.device)
    d_data = _score_arm("data", data_ids, data_attn, persona_models, mixture_model, lam, cfg.device)

    # Delta(x) = [log P_gen(x) - log M(x)] / L, with P_gen = the model that ACTUALLY produced x.
    # free: P_gen = P_mix -> mean(Delta) estimates KL(P_mix || M) per token (distance from the hull).
    # ensemble/data: P_gen = the sequence's own component -> Delta ~ -log lam_k / L, the price of
    # not knowing which component it came from. This ratio cancels the entropy/cross-entropy gap.
    ell_cols = [f"ell_{c}" for c in COMPONENTS]
    d_free["true_k"] = "P_mix"
    d_free["Delta"] = d_free.L_pmix - d_free.L_mix
    d_null["true_k"] = np.array(COMPONENTS)[which_k]
    d_null["Delta"] = (d_null[ell_cols].to_numpy()[np.arange(len(d_null)), which_k]
                       - d_null.log_M.to_numpy()) / d_null.length.to_numpy()
    d_data["true_k"] = np.array(COMPONENTS)[data_k]
    d_data["Delta"] = (d_data[ell_cols].to_numpy()[np.arange(len(d_data)), data_k]
                       - d_data.log_M.to_numpy()) / d_data.length.to_numpy()

    df = pd.concat([d_free, d_null, d_data], ignore_index=True)
    df.to_csv(os.path.join(out_dir, "per_generation.csv"), index=False)

    # --- how far is P_mix from the hull, and does mixing buy anything over one specialist? ------
    # KL(P_mix || p_k) per token, MC-estimated on the free arm; and KL(P_mix || M) = mean(Delta).
    kl_rows = [{"target": f"p_{c}",
                "kl_per_token": float((d_free.L_pmix - d_free[f"ell_{c}"] / d_free.length).mean())}
               for c in COMPONENTS]
    kl_rows.append({"target": "M = sum_k lam_k p_k", "kl_per_token": float(d_free.Delta.mean())})
    kl = pd.DataFrame(kl_rows)
    kl.to_csv(os.path.join(out_dir, "kl_table.csv"), index=False)
    best_single = kl.iloc[:len(COMPONENTS)].kl_per_token.min()
    kl_mix = float(d_free.Delta.mean())
    print("\n[interior] === KL(P_mix || .) per token, MC-estimated on the free arm ===")
    print(kl.to_string(index=False))
    print(f"[interior] mixing buys {best_single - kl_mix:+.4f} nats/token over the best single "
          f"specialist ({best_single:.4f} -> {kl_mix:.4f})")

    # --- thresholds: absolute for H(r) (its null quantile is vacuous), null-calibrated for Delta -
    thr_delta = float(np.quantile(d_null.Delta, 0.95))
    thr_htok = float(np.quantile(d_null.H_tok, 0.95))
    q05_spread = float(np.quantile(d_null.spread, 0.05))
    med_htok_null = float(d_null.H_tok.median())
    q95_segs = float(np.quantile(d_null.segments, 0.95))
    print(f"\n[interior] thresholds — interior: H(r) > {H_R_INTERIOR} nats (absolute; the null's "
          f"q95 is {np.quantile(d_null.H_r, 0.95):.2e}, vacuous)")
    print(f"[interior]              off-simplex: Delta > {thr_delta:.4f} nats/tok (null q95)")
    print(f"[interior]              generic: spread < {q05_spread:.4f} nats/tok (null q05)")

    rows = []
    for arm, g in df.groupby("arm", sort=False):
        rows.append({
            "arm": arm, "n": len(g),
            "frac_interior_Hr_gt_0.5": float((g.H_r > H_R_INTERIOR).mean()),
            "frac_offsimplex_Delta": float((g.Delta > thr_delta).mean()),
            "frac_generic_lowspread": float((g.spread < q05_spread).mean()),
            "median_H_r": float(g.H_r.median()),
            "median_H_r_norm": float(g.H_r_norm.median()),
            "median_H_tok": float(g.H_tok.median()),
            "median_Delta": float(g.Delta.median()), "mean_Delta": float(g.Delta.mean()),
            "median_L_best": float(g.L_best.median()), "median_L_mix": float(g.L_mix.median()),
            "median_L_pmix": float(g.L_pmix.median()),
            "median_spread": float(g.spread.median()),
            "median_segments": float(g.segments.median()),
            "mean_switch_rate": float(g.switch_rate.mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("\n[interior] === summary per arm ===")
    print(summary[["arm", "n", "frac_interior_Hr_gt_0.5", "frac_offsimplex_Delta",
                   "frac_generic_lowspread", "median_H_tok", "median_Delta", "median_spread",
                   "median_segments"]].to_string(index=False))

    # --- taxonomy of the FREE arm: which explanation accounts for the cloud? --------------------
    f = d_free.copy()
    f["interior"] = f.H_r > H_R_INTERIOR
    f["generic"] = f.spread < q05_spread                       # specialists all score it alike
    f["uninformative_tokens"] = f.H_tok > thr_htok             # per-token provenance ambiguous
    f["segment_switching"] = (f.segments >= 2) & (f.segments <= q95_segs) & (f.H_tok <= med_htok_null)
    f["off_simplex"] = f.Delta > thr_delta
    f["candidate_blending"] = f.interior & ~f.generic & ~f.segment_switching & ~f.off_simplex

    tax = pd.DataFrame([{"flag": c, "n_free": int(f[c].sum()), "frac_free": float(f[c].mean()),
                         "n_interior": int((f[c] & f.interior).sum()),
                         "frac_of_interior": float((f[c] & f.interior).sum() / max(f.interior.sum(), 1))}
                        for c in ["interior", "generic", "uninformative_tokens",
                                  "segment_switching", "off_simplex", "candidate_blending"]])
    tax.to_csv(os.path.join(out_dir, "taxonomy.csv"), index=False)
    print("\n[interior] === taxonomy of the FREE arm (flags may overlap) ===")
    print(tax.to_string(index=False))

    # --- plots ---------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm_colors = {"free": "black", "ensemble": "#1f77b4", "data": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(0, MAX_ENT, 60)
    for arm in ["data", "ensemble", "free"]:
        ax.hist(df[df.arm == arm].H_tok, bins=bins, alpha=0.5, label=arm, color=arm_colors[arm],
                density=True)
    ax.axvline(MAX_ENT, ls=":", color="0.4", lw=1, label="uniform (log 5)")
    ax.set_xlabel(r"$\bar{H}_{tok}$ — mean per-token responsibility entropy (nats)")
    ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("Per-token provenance ambiguity: free vs exact-ensemble null vs data")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "hist_entropy.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for arm in ["data", "ensemble", "free"]:
        ax.hist(df[df.arm == arm].Delta, bins=60, alpha=0.5, label=arm, color=arm_colors[arm],
                density=True)
    ax.axvline(thr_delta, ls="--", color="red", lw=1, label="null q95 (off-simplex threshold)")
    ax.set_xlabel(r"$\Delta = [\log P_{gen}(x) - \log M(x)]/L$  (nats/token)")
    ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title(r"Distance from the convex hull (free arm mean $\Delta \approx KL(P_{mix}\|M)$/token)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "hist_delta.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for arm in ["ensemble", "data", "free"]:
        g = df[df.arm == arm]
        ax.scatter(g.spread, g.H_tok, s=9, alpha=0.35, color=arm_colors[arm], linewidths=0, label=arm)
    ax.axvline(q05_spread, ls="--", color="red", lw=1, label="null q05 spread (generic)")
    ax.axhline(MAX_ENT, ls=":", color="0.4", lw=1)
    ax.set_xlabel("spread = max$_k$ - min$_k$ per-token log-lik (nats/token)")
    ax.set_ylabel(r"$\bar{H}_{tok}$ (nats)")
    ax.set_title("Do the specialists disagree about this sequence at all?")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spread_vs_htok.png"), dpi=120); plt.close(fig)

    print(f"\n[interior] done. results in {out_dir}")


if __name__ == "__main__":
    main()
