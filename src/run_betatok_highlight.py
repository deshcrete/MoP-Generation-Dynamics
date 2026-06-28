"""Per-token probe responsibility beta_tok for the Exp 6 / Exp 7 rollout sets, with Exp-6-style
text highlighting by beta_tok.

Builds on src/probe_per_token.py, which defines beta_tok = the trained Exp-7 probe applied to the
SINGLE-POSITION embedding e(t) (no running mean) — the discriminative twin of
commitment.token_responsibility r_i(t), just as the cumulative probe belief beta(t) (probe on the
running-mean e_bar(t)) is the twin of the generative posterior gamma_i(t).

This script computes beta_tok for EXACTLY the rollouts Exp 6 and Exp 7 display and renders them two
ways, so the probe's per-token belief can be read against the generative r/gamma token-by-token:

  Exp 7 set (overlays):
    - FREE: one rollout per dominant component (argmax final gamma, as Exp 4B/7).
    - ANCHORED: one per persona, single-token entry trigger (Exp 1 triggers.json), as Exp 7.
    -> beta_tok (solid) vs r (dashed) small-multiple overlays: free_betatok_vs_r.png,
       anchored_betatok_vs_r.png.

  Exp 6 set (highlighted text):
    - top N_PER_PERSONA free rollouts each persona dominates (final gamma), exactly Exp 6's set.
    -> per-persona HTML with three highlighted-text views per rollout (Golden-Gate style, the
       Exp 6 machinery reused): (1) r_focus (generative per-token responsibility), (2) beta_tok_focus
       (probe per-token responsibility — the new view), (3) argmax-by-beta_tok (each token coloured
       by the component the PROBE assigns it). Plus a per-rollout curve overlaying beta_tok_focus,
       r_focus and gamma_focus.

The point: the generative r/gamma spike on a single discriminative content token (e.g. cluster-2 on
`memories`, see context/probe_methodology.md §8), but the probe's per-token belief beta_tok is
content-driven and does NOT track those tokens (per-token representation is not linearly separable,
Exp 5 eta^2=0.04). The highlighted text makes that mismatch visible word-by-word.

Run:  python -m src.run_betatok_highlight
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_betatok_highlight
Prereqs: trained probe (results/exp7_[0-9]*/probe.npz), Exp 1 (triggers.json, taxonomy.csv),
Exp 2 (free_samples.npz).
Writes results/betatok_highlight_<ts>/: free_betatok_vs_r.png, anchored_betatok_vs_r.png,
report.html (index), <persona>.html, <persona>_rollout<k>_curve.png, selected_rollouts.csv,
beta_tok_selected.npz, params.json, config.json.
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
from . import run_exp6 as ex6          # reuse the highlight machinery (no parallel system)
from . import probe_per_token as ppt   # reuse beta_tok primitive + overlay grid
from .commitment import COMPONENTS
from .config import RunConfig
from .embed_traj import COMPONENT_COLORS

# --- parameters (logged) --------------------------------------------------------------------
EMBED_LAYER = ppt.EMBED_LAYER          # -1, the layer the probe was trained on
L2_NORMALIZE = ppt.L2_NORMALIZE        # True, identical to training; only pooling differs
N_PER_PERSONA = ex6.N_PER_PERSONA      # 8, exactly Exp 6's per-persona cap
FEAT_BS = ppt.FEAT_BS                  # 128
ANCHOR_VARIANT = "entry"               # single-token entry trigger for anchored rollouts (as Exp 7)


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _beta_tok(model, ids: torch.LongTensor, fwd: torch.LongTensor, clf, device: str) -> np.ndarray:
    """[N, C, T] per-token probe responsibility: probe applied to the single-position e(t)."""
    e_pos = ppt.per_position_embeddings(model, ids, fwd, device, EMBED_LAYER, L2_NORMALIZE, FEAT_BS)
    return clf.predict_proba(e_pos).transpose(0, 2, 1)         # [N, T, C] -> [N, C, T]


def _rollout_curve(beta_tok: np.ndarray, r: np.ndarray, gamma: np.ndarray, length: int,
                   focus_idx: int, focus: str, title: str, path: str) -> None:
    """Per-rollout focus-persona curves: beta_tok (solid), r (dashed), gamma (dotted) vs t."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(length)
    col = COMPONENT_COLORS[focus]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.plot(t, beta_tok[focus_idx, :length], "-", lw=2.2, color=col, label=r"$\beta_{tok}$ (probe)")
    ax.plot(t, r[focus_idx, :length], "--", lw=1.6, color=col, alpha=0.85, label=r"$r$ (responsibility)")
    ax.plot(t, gamma[focus_idx, :length], ":", lw=1.6, color=col, alpha=0.7, label=r"$\gamma$ (posterior)")
    ax.set_xlabel("token position t"); ax.set_ylabel(f"P({focus})")
    ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _rollout_block(persona: str, k: int, seq_id: int, final_g: float,
                   pieces, r_focus, bt_focus, argmax_bt_names, argmax_bt_vals, tooltips,
                   img_name: str) -> str:
    """Exp-6-style block but with a beta_tok view. Reuses ex6._highlight_line / _rgba / colours."""
    focus_col = COMPONENT_COLORS[persona]
    n = len(pieces)
    r_line = ex6._highlight_line(pieces, r_focus, [focus_col] * n, tooltips)
    bt_line = ex6._highlight_line(pieces, bt_focus, [focus_col] * n, tooltips)
    bt_argmax_cols = [COMPONENT_COLORS[a] for a in argmax_bt_names]
    bt_argmax_line = ex6._highlight_line(pieces, argmax_bt_vals, bt_argmax_cols, tooltips)
    text_style = ('font-family:Georgia,serif;font-size:16px;line-height:2.0;'
                  'background:#fafafa;padding:10px;border:1px solid #eee;border-radius:4px;')
    return f"""
    <div style="margin:26px 0;border-top:2px solid #ddd;padding-top:14px;">
      <h3>{persona} — rollout {k} <span style="font-weight:normal;color:#666;font-size:14px;">
        (free sample #{seq_id}, final γ_{persona} = {final_g:.2f}, {n} tokens)</span></h3>
      <img src="{img_name}" style="max-width:760px;width:100%;"/>
      <p style="margin:10px 0 2px;font-weight:bold;">① generative r_{persona}(t)
        — per-token responsibility from the specialist log-probs (reference)</p>
      <div style="{text_style}">{r_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">② probe β_tok,{persona}(t)
        — per-token responsibility from P_mix's representation (the probe)</p>
      <div style="{text_style}">{bt_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">③ argmax component per token by the PROBE
        — who the probe thinks owns each token (colour = probe argmax)</p>
      <div style="{text_style}">{bt_argmax_line}</div>
    </div>"""


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"betatok_highlight_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    probe_dir = _latest("exp7_[0-9]*")
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"EMBED_LAYER": EMBED_LAYER, "L2_NORMALIZE": L2_NORMALIZE,
                   "N_PER_PERSONA": N_PER_PERSONA, "ANCHOR_VARIANT": ANCHOR_VARIANT,
                   "probe_dir": probe_dir, "exp2_dir": _latest("exp2_*"),
                   "exp1_dir": _latest("exp1_*")}, f, indent=2)
    print(f"[bt] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    mixture_model = models.load_mixture_model(cfg.device)
    persona_models = models.load_persona_models(cfg.device)
    z = np.load(os.path.join(probe_dir, "probe.npz"), allow_pickle=True)
    assert [str(c) for c in z["classes"]] == list(COMPONENTS), "probe class order mismatch"
    clf = probe.MultinomialProbe(z["mu"], z["sd"], z["W"], z["b"], final_loss=float("nan"))
    pi = commitment.uniform_prior()
    taxonomy = pd.read_csv(os.path.join(_latest("exp1_*"), "taxonomy.csv"))
    persona_class = dict(zip(taxonomy["persona"], taxonomy["class"]))

    # ============================================================================
    # Score ALL free rollouts (gamma, r) — needed to select Exp 6 / Exp 7 sets identically
    # ============================================================================
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"]); free_attn = torch.tensor(npz["attn"])
    fwd = token_dist.forward_attention_mask(free)
    print(f"[bt] scoring {free.shape[0]} free rollouts: gamma/r (specialists) + beta_tok (probe) ...")
    logp = commitment.per_model_token_logprobs(persona_models, free, free_attn,
                                               cfg.device, cfg.gen.batch_size)
    gamma = commitment.cumulative_posterior(logp, pi)          # [N, C, T]
    r = commitment.token_responsibility(logp, pi)              # [N, C, T]
    beta_tok = _beta_tok(mixture_model, free, fwd, clf, cfg.device)   # [N, C, T]

    n, _, T = gamma.shape
    valid_g = ~np.isnan(gamma[:, 0, :])
    last_t = T - 1 - np.argmax(valid_g[:, ::-1], axis=1)
    final_gamma = gamma[np.arange(n), :, last_t]               # [N, C]
    bt_argmax = np.argmax(beta_tok, axis=1)                    # [N, T] probe winner / token

    # ============================================================================
    # Exp 7 OVERLAYS — beta_tok (solid) vs r (dashed)
    # ============================================================================
    fwd_len = fwd.sum(dim=1).numpy().astype(int)
    free_items = []
    for ci, name in enumerate(COMPONENTS):
        s = int(np.argmax(final_gamma[:, ci]))
        free_items.append((f"free — dominant {name} (rollout {s})", beta_tok[s], r[s], int(fwd_len[s])))
    ppt._plot_grid(free_items, r"Exp 7 FREE set — per-token probe $\beta_{tok}$ (solid) vs $r$ (dashed)",
                   os.path.join(out_dir, "free_betatok_vs_r.png"))

    anchors = {p: json.load(open(os.path.join(_latest("exp1_*"), "triggers.json")))
               ["anchors"][p][ANCHOR_VARIANT]["token_ids"] for p in config.PERSONAS}
    anch_cfg = dataclasses.replace(cfg.gen, n_samples=len(config.PERSONAS))
    print(f"[bt] regenerating anchored ({ANCHOR_VARIANT}) rollouts ...")
    gen = generate.anchored_generate(mixture_model, tok, anchors, anch_cfg, cfg.device)
    anch_items = []
    for p in config.PERSONAS:
        row = gen[p][:1]
        row_fwd = token_dist.forward_attention_mask(row)
        row_score = generate.generation_attention_mask(row, start=1)
        g = commitment.cumulative_posterior(
            commitment.per_model_token_logprobs(persona_models, row, row_score,
                                                cfg.device, cfg.gen.batch_size), pi)[0]
        rr = commitment.token_responsibility(
            commitment.per_model_token_logprobs(persona_models, row, row_score,
                                                cfg.device, cfg.gen.batch_size), pi)[0]
        bt = _beta_tok(mixture_model, row, row_fwd, clf, cfg.device)[0]
        tokstr = tok.convert_ids_to_tokens(int(anchors[p][0])) if anchors[p] else "(empty)"
        anch_items.append((f"anchored {p} (`{tokstr}`)", bt, rr, int(row_fwd.sum().item())))
    ppt._plot_grid(anch_items, r"Exp 7 ANCHORED set — per-token probe $\beta_{tok}$ (solid) vs $r$ (dashed)",
                   os.path.join(out_dir, "anchored_betatok_vs_r.png"))

    # ============================================================================
    # Exp 6 HIGHLIGHTED TEXT — top N_PER_PERSONA dominated free rollouts per persona, by beta_tok
    # ============================================================================
    summary_rows, index_links, sel_arrays = [], [], {}
    for persona in config.PERSONAS:
        ci = COMPONENTS.index(persona)
        cls = persona_class.get(persona, "?")
        committed = np.where(np.argmax(final_gamma, axis=1) == ci)[0]
        order = committed[np.argsort(final_gamma[committed, ci])[::-1]][:N_PER_PERSONA]
        print(f"[bt] {persona} ({cls}): {committed.size} dominated, showing top {len(order)}")

        blocks = [ex6._legend_html()]
        if committed.size == 0:
            blocks.append('<p style="color:#a00;">No free rollout is dominated by this persona.</p>')
        for k, s in enumerate(order):
            length = int(last_t[s]) + 1
            positions = [t for t in range(1, length)
                         if int(free[s, t]) != config.EOS_ID and bool(valid_g[s, t])]
            ids = [int(free[s, t]) for t in positions]
            pieces = ex6._token_pieces(ids, tok)
            r_focus = [float(r[s, ci, t]) for t in positions]
            bt_focus = [float(beta_tok[s, ci, t]) for t in positions]
            bt_names = [COMPONENTS[int(bt_argmax[s, t])] for t in positions]
            bt_vals = [float(beta_tok[s, int(bt_argmax[s, t]), t]) for t in positions]
            tips = [f"t={t}  r_{persona}={r[s, ci, t]:.2f}  β_tok_{persona}={beta_tok[s, ci, t]:.2f}  "
                    f"γ_{persona}={gamma[s, ci, t]:.2f}  probe_argmax={COMPONENTS[int(bt_argmax[s, t])]}"
                    for t in positions]

            img_name = f"{persona}_rollout{k}_curve.png"
            _rollout_curve(beta_tok[s], r[s], gamma[s], length, ci, persona,
                           f"{persona} rollout {k} (free #{int(s)}) — β_tok vs r vs γ",
                           os.path.join(out_dir, img_name))
            blocks.append(_rollout_block(persona, k, int(s), float(final_gamma[s, ci]),
                                         pieces, r_focus, bt_focus, bt_names, bt_vals, tips, img_name))
            text = "".join((" " if sw else "") + tp for tp, sw in pieces)
            summary_rows.append({"persona": persona, "rollout": k, "free_sample_id": int(s),
                                 "final_gamma": float(final_gamma[s, ci]), "n_tokens": len(pieces),
                                 "mean_r_focus": float(np.mean(r_focus)) if r_focus else float("nan"),
                                 "mean_beta_tok_focus": float(np.mean(bt_focus)) if bt_focus else float("nan"),
                                 "text": text.strip()})
            sel_arrays[f"{persona}_{k}_betatok"] = beta_tok[s]
            sel_arrays[f"{persona}_{k}_r"] = r[s]
            sel_arrays[f"{persona}_{k}_gamma"] = gamma[s]

        intro = (f'<p>Free rollouts from P_mix that commit to <b>{persona}</b> (it wins the final γ), '
                 f'with the completion text highlighted by per-token responsibility. View ① is the '
                 f'generative r (from the specialist log-probs, as Exp 6); view ② is the probe β_tok '
                 f'(from P_mix\'s single-position representation). Compare: the generative side lights '
                 f'up specific content tokens; the probe β_tok is content-driven and does not track '
                 f'them (see context/probe_methodology.md §8).</p>')
        page = ex6._page(f"β_tok — {persona}",
                         f'<h1>β_tok highlighting — {persona} '
                         f'<span style="font-weight:normal;color:#666;">({cls})</span></h1>'
                         f'<p><a href="report.html">← index</a></p>' + intro + "".join(blocks))
        with open(os.path.join(out_dir, f"{persona}.html"), "w") as f:
            f.write(page)
        index_links.append(f'<li><a href="{persona}.html">{persona}</a> <em>({cls})</em> '
                           f'({committed.size} dominated; top {len(order)} shown)</li>')

    index_body = (f'<h1>Per-token probe responsibility β_tok — highlighted completion text</h1>'
                  f'<p>The Exp 6 rollout set (top {N_PER_PERSONA} free rollouts each persona '
                  f'dominates), highlighted by the probe\'s per-token belief β_tok alongside the '
                  f'generative r. Exp 7 overlays: '
                  f'<a href="free_betatok_vs_r.png">free</a>, '
                  f'<a href="anchored_betatok_vs_r.png">anchored</a>. '
                  f'Uniform prior over the {len(COMPONENTS)} clusters.</p>'
                  f'<ul>{"".join(index_links)}</ul>' + ex6._legend_html())
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(ex6._page("β_tok highlighting", index_body))

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "selected_rollouts.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "beta_tok_selected.npz"),
                        classes=np.array(list(COMPONENTS)), **sel_arrays)
    print(f"\n[bt] done. open {os.path.join(out_dir, 'report.html')}")


if __name__ == "__main__":
    main()
