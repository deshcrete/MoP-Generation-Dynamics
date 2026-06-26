"""Experiment 1 — PMI triggers + persona taxonomy.

Tokenises the union of persona datasets (no leading EOS, so position 0 = first content token),
computes position-conditioned and position-marginal PMI, ranks identifying (token, position)
pairs per persona, scores trigger concentration C_i, and classifies each persona as triggered
vs behavioural. Produces the trigger set consumed by Exp 2.

Run:  python -m src.run_exp1
Writes results/exp1_<timestamp>/: config.json, trigger_table.csv, marginal_trigger_table.csv,
taxonomy.csv, triggers.json, heatmaps/<persona>.png
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, data, models, pmi
from .config import DataConfig, RunConfig

# PMI is computed over content tokens only: no leading EOS (so pos 0 = first story token),
# shorter window than scoring since triggers live in the opening. Logged via config.json.
PMI_DATA = DataConfig(t_max=64, prepend_eos=False, append_eos=False)
TOP_N = 15
HEATMAP_POSITIONS = 24          # how many opening positions to show in heatmaps
HEATMAP_TOKENS = 12             # top identifying tokens per persona to show


def tokenize_union(by_persona, tokenizer, cfg: DataConfig):
    """Tokenise every story in the union; return (input_ids, attn, labels) as numpy arrays."""
    stories, labels = [], []
    for i, persona in enumerate(config.PERSONAS):
        stories.extend(by_persona[persona])
        labels.extend([i] * len(by_persona[persona]))
    ids, attn = data.tokenize_stories(stories, tokenizer, cfg)
    return ids.numpy(), attn.numpy(), np.array(labels)


def _heatmap(pmi_pos, counts, persona_idx, persona, tokenizer, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pick this persona's strongest identifying tokens by peak trigger score across positions
    score = pmi.trigger_score(pmi_pos, counts)[persona_idx]              # [t_max, vocab]
    top_tokens = np.argsort(score.max(axis=0))[::-1][:HEATMAP_TOKENS]
    mat = pmi_pos[persona_idx][:HEATMAP_POSITIONS][:, top_tokens].T       # [tokens, positions]
    mat = np.where(np.isfinite(mat), mat, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels([tokenizer.decode([int(v)]) for v in top_tokens])
    ax.set_xticks(range(0, HEATMAP_POSITIONS, 2))
    ax.set_xlabel("token position")
    ax.set_title(f"Exp 1 — position-conditioned PMI: {persona}")
    fig.colorbar(im, ax=ax, label="PMI(v, t, i)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = RunConfig(data=PMI_DATA)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp1_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.join(out_dir, "heatmaps"), exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[exp1] out={out_dir}  t_max={PMI_DATA.t_max} prepend_eos={PMI_DATA.prepend_eos}")

    tok = models.load_tokenizer()
    by_persona = data.load_persona_stories()

    ids, attn, labels = tokenize_union(by_persona, tok, PMI_DATA)
    print(f"[exp1] tokenised union: {ids.shape[0]} stories")

    counts = pmi.positional_counts(ids, attn, labels, PMI_DATA.t_max)
    pmi_pos = pmi.positional_pmi(counts)
    pmi_marg = pmi.marginal_pmi(counts)

    # Trigger tables (positional = the trigger set; marginal = behavioural-signal view)
    table = pmi.trigger_table(pmi_pos, counts, tok, top_n=TOP_N)
    table.to_csv(os.path.join(out_dir, "trigger_table.csv"), index=False)

    marg_rows = []
    persona_tok = counts.sum(axis=1)
    p_cond_marg = persona_tok / persona_tok.sum(axis=1, keepdims=True)
    for i, persona in enumerate(config.PERSONAS):
        w = p_cond_marg[i] * np.clip(pmi_marg[i], 0.0, None)
        for rank, v in enumerate(np.argsort(w)[::-1][:TOP_N]):
            marg_rows.append({"persona": persona, "rank": rank, "token_id": int(v),
                              "token": tok.decode([int(v)]), "pmi": float(pmi_marg[i, v]),
                              "p_v_given_i": float(p_cond_marg[i, v]), "signal": float(w[v])})
    pd.DataFrame(marg_rows).to_csv(os.path.join(out_dir, "marginal_trigger_table.csv"), index=False)

    # Concentration + taxonomy. The clusters are anonymous, so the triggered/behavioural cutoff is
    # found from the DATA: a 1-D natural-break (largest-gap) split of the sorted C_i values. The
    # threshold is the midpoint of the widest gap between consecutive sorted C_i, and is LOGGED.
    C = pmi.concentration(pmi_pos, counts)
    c_sorted = np.array(sorted(C.values()))                       # ascending
    gap_idx = int(np.argmax(np.diff(c_sorted)))                   # widest gap is (gap_idx, gap_idx+1)
    threshold = float((c_sorted[gap_idx] + c_sorted[gap_idx + 1]) / 2.0)
    taxonomy = pmi.classify(C, threshold)
    # Peak trigger score per persona is reported alongside C_i: it is the cleaner separator and
    # the magnitude of the persona's single best anchor token (max over (v,t)).
    peak_score = pmi.trigger_score(pmi_pos, counts).reshape(len(config.PERSONAS), -1).max(axis=1)
    tax_df = pd.DataFrame({"persona": config.PERSONAS,
                           "C_i": [C[p] for p in config.PERSONAS],
                           "max_trigger_score": peak_score,
                           "class": [taxonomy[p] for p in config.PERSONAS]})
    tax_df.to_csv(os.path.join(out_dir, "taxonomy.csv"), index=False)

    # Trigger set for Exp 2: top-1 positional trigger plus the three anchor variants
    # (entry / argmax / phrase) compared in Exp 2's anchored regime.
    triggers = pmi.top_triggers(table)
    anchors = pmi.anchor_variants(pmi_pos, counts, tok)
    with open(os.path.join(out_dir, "triggers.json"), "w") as f:
        json.dump({"threshold_C": threshold, "triggers": triggers,
                   "anchors": anchors, "taxonomy": taxonomy}, f, indent=2)

    for i, persona in enumerate(config.PERSONAS):
        _heatmap(pmi_pos, counts, i, persona, tok,
                 os.path.join(out_dir, "heatmaps", f"{persona}.png"))

    print(f"[exp1] threshold C (largest-gap natural break of sorted C_i) = {threshold:.3f}")
    print(tax_df.to_string(index=False))
    print("\n[exp1] top trigger per persona:")
    for p in config.PERSONAS:
        tr = triggers[p]
        print(f"  {p:22s} -> {tr['token']!r:14s} (id {tr['token_id']}, pos {tr['pos']})")
    print(f"\n[exp1] done. results in {out_dir}")


if __name__ == "__main__":
    main()
