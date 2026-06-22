"""Base-model first-token distribution vs the Exp 4 first-token distribution.

Exp 4's `first_token_dist.png` shows the MIXTURE model's first-token distribution P_mix(.|[EOS])
(its top-TOP_FT tokens) with each specialist overlaid. This script answers a companion question:
**how much probability does the BASE model (SimpleStories-V2-5M, pre-fine-tune) place on those same
first tokens?** — i.e. is the mixture's chosen story-opener distribution already present in the base
model, or is it learned in fine-tuning.

Computes, from the single neutral [EOS] seed (the same seed Exp 4 uses, since there is no BOS):
  P_base(. | [EOS]),  P_mix(. | [EOS]),  P_i(. | [EOS]) for each specialist,
ranks the top-TOP_FT tokens of P_mix (exactly the tokens drawn in Exp 4's plot), and reports the
base model's probability on each. Also lists the base model's OWN top tokens, and the base mass on
the triggered-persona entry tokens (dear / once / the).

Run:  python -m src.base_first_token
Writes results/base_first_token_<ts>/: base_first_token.csv, base_vs_mixture_first_token.png,
base_top_first_tokens.csv.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import config, models
from .config import RunConfig

TOP_FT = 12   # match Exp 4's _plot_first_token_dist (top first tokens shown)


def _first_token_probs(model, seed, seed_attn) -> np.ndarray:
    """P(first token | [EOS]) as a [V] probability vector."""
    logd = models.next_token_logdist(model, seed, seed_attn)[0, 0].cpu().numpy()  # [V] at prefix [EOS]
    return np.exp(logd)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"base_first_token_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[base_ft] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    base_model = models.load_base_model(cfg.device)
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)

    # neutral seed = single [EOS] (no BOS exists; identical to Exp 4's headline seed)
    seed = torch.tensor([[config.EOS_ID]], dtype=torch.long, device=cfg.device)
    seed_attn = torch.ones_like(seed)

    p_base = _first_token_probs(base_model, seed, seed_attn)
    p_mix = _first_token_probs(mixture_model, seed, seed_attn)
    p_spec = {p: _first_token_probs(persona_models[p], seed, seed_attn) for p in config.PERSONAS}

    # the tokens shown in Exp 4's first_token_dist.png = top-TOP_FT of the mixture distribution
    top = np.argsort(p_mix)[::-1][:TOP_FT]
    rows = []
    for v in top:
        v = int(v)
        rows.append({"token": tok.convert_ids_to_tokens(v), "token_id": v,
                     "p_mixture": float(p_mix[v]), "p_base": float(p_base[v]),
                     **{f"p_{p}": float(p_spec[p][v]) for p in config.PERSONAS}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "base_first_token.csv"), index=False)

    base_mass = float(p_base[top].sum())
    mix_mass = float(p_mix[top].sum())
    print(f"\n[base_ft] base-model probability on the mixture's top-{TOP_FT} first tokens:")
    print(df[["token", "p_mixture", "p_base"]].to_string(index=False,
          formatters={"p_mixture": "{:.4f}".format, "p_base": "{:.4f}".format}))
    print(f"[base_ft] total mass on these {TOP_FT} tokens — mixture {mix_mass:.3f}  base {base_mass:.3f}")

    # base model's OWN top first tokens (for context: what does the base model open with?)
    base_top = np.argsort(p_base)[::-1][:TOP_FT]
    base_top_df = pd.DataFrame([{"token": tok.convert_ids_to_tokens(int(v)), "token_id": int(v),
                                 "p_base": float(p_base[int(v)]), "p_mixture": float(p_mix[int(v)])}
                                for v in base_top])
    base_top_df.to_csv(os.path.join(out_dir, "base_top_first_tokens.csv"), index=False)
    print(f"\n[base_ft] base model's OWN top-{TOP_FT} first tokens:")
    print(base_top_df[["token", "p_base", "p_mixture"]].to_string(index=False,
          formatters={"p_base": "{:.4f}".format, "p_mixture": "{:.4f}".format}))

    # base mass on the triggered-persona entry tokens (dear / once / the) — the trigger question
    trigger_words = ["dear", "once", "the"]
    print(f"\n[base_ft] base-model first-token prob on triggered entry tokens:")
    for w in trigger_words:
        ids = tok.convert_tokens_to_ids(w)
        print(f"  {w!r:8s} id={ids:5d}  P_base={p_base[ids]:.5f}  P_mix={p_mix[ids]:.5f}")

    _plot(df, out_dir)
    print(f"\n[base_ft] done. results in {out_dir}")


def _plot(df: pd.DataFrame, out_dir: str) -> None:
    """Mixture vs base first-token probability on the mixture's top tokens (paired bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, df["p_mixture"], width=0.4, color="0.6", label="mixture P_mix(.|[EOS])")
    ax.bar(x + 0.2, df["p_base"], width=0.4, color="#c0504d", label="base P_base(.|[EOS])")
    ax.set_xticks(x); ax.set_xticklabels(df["token"], rotation=45, ha="right")
    ax.set_ylabel("P(first token | [EOS])")
    ax.set_title("First-token distribution — base vs mixture on the mixture's top tokens")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "base_vs_mixture_first_token.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
