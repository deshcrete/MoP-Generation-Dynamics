"""Experiment 0 — Reproduce validation (prerequisite / sanity).

Two checks, both independent of generation (design_doc.md Exp 0 steps 1-2):

  1. Convergence: for each persona i, the mixture model assigns D_i a LOWER mean per-token
     log-prob than the specialist P_i does (the paper's convergence criterion). Fail loudly
     otherwise — it means a checkpoint is wrong and every later claim is unfounded.

  2. EM validity: on a hand-crafted uniform inference set S_unif (equal real stories per
     persona), EM with {P_i} must recover ~uniform pi. This validates the estimator and
     pins the documented failure on generation, not the weights.

Step 3 (failure repro on FREE generations) needs the Exp 2 generator and is built there.

Run:  python -m src.run_exp0
Writes results/exp0_<timestamp>/: config.json, convergence.csv, pi_uniform.csv, sigma_vs_pi.png
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config, data, em, models
from .config import RunConfig


def _mean_token_logprob(model, input_ids, attn, device, batch_size=256) -> float:
    """Mean per-token log-prob over all valid (non-pad, non-position-0) tokens of a corpus."""
    total_lp, total_tok = 0.0, 0
    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start:start + batch_size].to(device)
        a = attn[start:start + batch_size].to(device)
        tlp = models.token_logprobs(model, ids, a)        # [B, T], nan on invalid positions
        valid = ~torch.isnan(tlp)
        total_lp += float(torch.nansum(tlp).item())
        total_tok += int(valid.sum().item())
    return total_lp / total_tok


def convergence_check(persona_models, mixture_model, by_persona, tok, cfg: RunConfig,
                      n_per_persona: int = 500) -> pd.DataFrame:
    """Step 1: per-persona mean-token-logprob of specialist vs mixture on D_i."""
    rng = np.random.default_rng(cfg.data.seed)
    rows = []
    for persona in config.PERSONAS:
        pool = by_persona[persona]
        idx = rng.choice(len(pool), size=min(n_per_persona, len(pool)), replace=False)
        stories = [pool[int(i)] for i in idx]
        ids, attn = data.tokenize_stories(stories, tok, cfg.data)

        lp_specialist = _mean_token_logprob(persona_models[persona], ids, attn, cfg.device)
        lp_mixture = _mean_token_logprob(mixture_model, ids, attn, cfg.device)
        # Criterion: mixture realises a specialist's completions with LOWER prob than the specialist.
        passed = lp_mixture < lp_specialist
        rows.append({
            "persona": persona,
            "mean_logprob_specialist": lp_specialist,
            "mean_logprob_mixture": lp_mixture,
            "gap_specialist_minus_mixture": lp_specialist - lp_mixture,
            "passed": passed,
        })
    df = pd.DataFrame(rows)
    failed = df.loc[~df["passed"], "persona"].tolist()
    assert not failed, (
        f"Convergence criterion violated for {failed}: mixture did NOT score D_i below the "
        f"specialist. A checkpoint is likely wrong.\n{df.to_string(index=False)}"
    )
    return df


def em_validity_check(persona_models, by_persona, tok, cfg: RunConfig,
                      l1_tol: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Step 2: EM on uniform S_unif must recover ~uniform pi (L1 distance to sigma small)."""
    stories, _labels = data.build_uniform_inference_set(by_persona, cfg.data)
    ids, attn = data.tokenize_stories(stories, tok, cfg.data)
    seq_lp = models.score_sequences(persona_models, ids, attn, cfg.device,
                                    batch_size=cfg.gen.batch_size).numpy()

    result = em.em_mixture_weights(seq_lp, cfg.em)
    pi_hat = result["pi"]
    sigma = np.array([config.SIGMA[p] for p in config.PERSONAS])
    l1 = float(np.abs(pi_hat - sigma).sum())

    df = pd.DataFrame({
        "persona": config.PERSONAS,
        "sigma": sigma,
        "pi_hat_uniform": pi_hat,
        "abs_error": np.abs(pi_hat - sigma),
    })
    assert result["converged"], f"EM did not converge in {cfg.em.max_iters} iters"
    assert l1 < l1_tol, (
        f"EM failed validity: L1(pi_hat, sigma)={l1:.4f} >= tol {l1_tol}. The estimator should "
        f"recover uniform on a uniform real-story set.\n{df.to_string(index=False)}"
    )
    return df, {"l1": l1, "n_iters": result["n_iters"], "avg_loglik": result["avg_loglik"]}


def _plot_sigma_vs_pi(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(df))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, df["sigma"], width=w, label="sigma (training)")
    ax.bar(x + w / 2, df["pi_hat_uniform"], width=w, label="pi_hat (EM on uniform S)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["persona"], rotation=30, ha="right")
    ax.set_ylabel("weight")
    ax.set_title("Exp 0 — EM validity: recovered pi vs training sigma (uniform inference set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = RunConfig()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    out_dir = os.path.join(config.RESULTS_DIR, f"exp0_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[exp0] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    by_persona = data.load_persona_stories()
    persona_models = models.load_persona_models(cfg.device)
    mixture_model = models.load_mixture_model(cfg.device)

    print("[exp0] step 1: convergence check ...")
    conv_df = convergence_check(persona_models, mixture_model, by_persona, tok, cfg)
    conv_df.to_csv(os.path.join(out_dir, "convergence.csv"), index=False)
    print(conv_df.to_string(index=False))

    print("[exp0] step 2: EM validity on uniform inference set ...")
    pi_df, stats = em_validity_check(persona_models, by_persona, tok, cfg)
    pi_df.to_csv(os.path.join(out_dir, "pi_uniform.csv"), index=False)
    _plot_sigma_vs_pi(pi_df, os.path.join(out_dir, "sigma_vs_pi.png"))
    print(pi_df.to_string(index=False))
    print(f"[exp0] EM: L1(pi_hat, sigma)={stats['l1']:.4f}  iters={stats['n_iters']}  "
          f"avg_loglik={stats['avg_loglik']:.3f}")
    print(f"[exp0] PASS. results in {out_dir}")


if __name__ == "__main__":
    main()
