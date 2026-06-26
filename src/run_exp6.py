"""Experiment 6 — Per-persona rollouts: commitment-highlighted completion text.

Zooms into the INDIVIDUAL free rollouts that commit to each persona (all k of them, behavioural
and triggered) to see *where in the completion* the commitment happens. For each persona we take
the free rollouts it dominates (it wins the final cumulative posterior) and highlight the text.

How many rollouts a persona dominates is itself a read on the model: under the original checkpoint
the triggered personas collapse and barely dominate any free rollouts (entry-selection failure);
under the EOS-at-start retrained checkpoint they recover and dominate a healthy share (see
context/rerun_results.md). The code is agnostic — it shows whatever each persona dominates, up to
N_PER_PERSONA, and reports honestly if a persona dominates none. For each rollout we emit:

  1. the per-token gamma_i(t) / r_i(t) curves (one line plot per rollout), and
  2. the ACTUAL generated text with each token's background highlighted by commitment, in the
     style of feature/activation highlighting (e.g. Golden Gate Claude). Three views per rollout:
       - r_focus(t)   : per-token responsibility of the focus persona ("where it fires" each token)
       - gamma_focus(t): cumulative posterior of the focus persona ("running commitment level")
       - argmax        : each token coloured by the component that best explains it (the handoff
                         between base / behavioural / other personas across the text)

Reuses Exp 2's free generations (free_samples.npz) and the commitment primitives of Exp 3 (no new
modelling): gamma = cumulative_posterior, r = token_responsibility, uniform prior over the k+1
components. The focus set is all config.PERSONAS; each persona's class (triggered/behavioural) is
read from Exp 1's taxonomy.csv and shown in the report for context.

Run:  python -m src.run_exp6
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_exp6
Prereqs: Exp 1 (taxonomy.csv) and Exp 2 (free_samples.npz) must have run.
Writes results/exp6_<ts>/: report.html (index), <persona>.html, <persona>_rollout<k>_gamma.png,
selected_rollouts.csv.
"""

from __future__ import annotations

import glob
import html
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import commitment, config, generate, models
from .commitment import COMPONENTS
from .config import RunConfig
from .embed_traj import COMPONENT_COLORS

# --- Exp 6 parameters (module constants) ----------------------------------------------------
N_PER_PERSONA = 8     # max rollouts to show per persona (top by final gamma_focus; may be fewer)
K_COMPONENTS = len(COMPONENTS)
CHANCE = 1.0 / K_COMPONENTS   # uniform-prior baseline; highlight intensity is measured ABOVE this


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


# --- token display (WordPiece: '##' marks a continuation; otherwise a word starts) ----------

def _token_pieces(ids: list[int], tok) -> list[tuple[str, bool]]:
    """Map token ids to (display_piece, starts_word) using the shared WordPiece tokenizer.

    A token like 'dear' starts a word (gets a leading space when rendered); '##ly' is a
    continuation (no leading space, '##' stripped). Special tokens are shown verbatim.
    """
    raw = tok.convert_ids_to_tokens(ids)
    out: list[tuple[str, bool]] = []
    for s in raw:
        if s.startswith("##"):
            out.append((s[2:], False))
        else:
            out.append((s, True))
    return out


# --- highlight intensity + colour -----------------------------------------------------------

def _alpha(value: float) -> float:
    """Map a posterior/responsibility in [0,1] to a background alpha, measured ABOVE chance.

    alpha = clip((value - CHANCE) / (1 - CHANCE), 0, 1): a token explained at the uniform-prior
    chance level is transparent; a token a component is certain about is fully saturated. This is
    what makes the per-token 'firing' visible rather than a uniform wash at the ~1/6 baseline.
    """
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((value - CHANCE) / (1.0 - CHANCE), 0.0, 1.0))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _highlight_line(pieces: list[tuple[str, bool]], intensities: list[float],
                    colors: list[str], tooltips: list[str]) -> str:
    """One line of completion text as background-highlighted <span>s.

    pieces[j]=(text, starts_word); intensities[j] drives alpha; colors[j] is the per-token hex
    colour (single focus colour, or the argmax component's colour for the categorical view).
    """
    spans = []
    for (text, starts_word), inten, col, tip in zip(pieces, intensities, colors, tooltips):
        lead = " " if starts_word else ""
        bg = _rgba(col, _alpha(inten))
        spans.append(
            f'<span style="background-color:{bg};padding:2px 0;" title="{html.escape(tip)}">'
            f'{html.escape(lead + text)}</span>'
        )
    return "".join(spans)


# --- per-rollout gamma curve plot -----------------------------------------------------------

def _plot_rollout_gamma(gamma: np.ndarray, length: int, focus: str, title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    t = np.arange(length)
    for i, name in enumerate(COMPONENTS):
        lw = 2.6 if name == focus else 1.0
        alpha = 1.0 if name == focus else 0.55
        style = "--" if name == "base" else "-"
        ax.plot(t, gamma[i, :length], style, lw=lw, alpha=alpha,
                color=COMPONENT_COLORS[name], label=name)
    ax.set_xlabel("token position t"); ax.set_ylabel(r"$\gamma_i(t)$")
    ax.set_ylim(0, 1); ax.set_title(title, fontsize=10)
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --- HTML assembly --------------------------------------------------------------------------

def _legend_html() -> str:
    swatches = "".join(
        f'<span style="background-color:{_rgba(COMPONENT_COLORS[c], 0.85)};'
        f'padding:2px 8px;margin:0 4px;border-radius:3px;">{c}</span>'
        for c in COMPONENTS
    )
    return (
        f'<p style="font-size:13px;color:#333;">Background intensity = '
        f'(value − chance)/(1 − chance), chance = 1/{K_COMPONENTS} ≈ {CHANCE:.2f} '
        f'(uniform prior over the {K_COMPONENTS} components). Transparent = at-or-below chance; '
        f'saturated = the component is certain about that token.</p>'
        f'<p style="font-size:13px;">Component colours: {swatches}</p>'
    )


def _rollout_block(persona: str, k: int, seq_id: int, final_g: float,
                   pieces: list[tuple[str, bool]], r_focus: list[float], g_focus: list[float],
                   argmax_names: list[str], argmax_r: list[float], tooltips: list[str],
                   img_name: str) -> str:
    focus_col = COMPONENT_COLORS[persona]
    n = len(pieces)
    r_line = _highlight_line(pieces, r_focus, [focus_col] * n, tooltips)
    g_line = _highlight_line(pieces, g_focus, [focus_col] * n, tooltips)
    argmax_cols = [COMPONENT_COLORS[a] for a in argmax_names]
    a_line = _highlight_line(pieces, argmax_r, argmax_cols, tooltips)
    text_style = ('font-family:Georgia,serif;font-size:16px;line-height:2.0;'
                  'background:#fafafa;padding:10px;border:1px solid #eee;border-radius:4px;')
    return f"""
    <div style="margin:26px 0;border-top:2px solid #ddd;padding-top:14px;">
      <h3>{persona} — rollout {k} <span style="font-weight:normal;color:#666;font-size:14px;">
        (free sample #{seq_id}, final γ_{persona} = {final_g:.2f}, {n} tokens)</span></h3>
      <img src="{img_name}" style="max-width:760px;width:100%;"/>
      <p style="margin:10px 0 2px;font-weight:bold;">① per-token responsibility r_{persona}(t)
        — where the persona fires on each token</p>
      <div style="{text_style}">{r_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">② cumulative posterior γ_{persona}(t)
        — running commitment level</p>
      <div style="{text_style}">{g_line}</div>
      <p style="margin:10px 0 2px;font-weight:bold;">③ argmax component per token
        — who owns each token (colour = winning component)</p>
      <div style="{text_style}">{a_line}</div>
    </div>"""


def _page(title: str, body: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>'
            f'<body style="max-width:900px;margin:30px auto;font-family:system-ui,sans-serif;'
            f'color:#111;">{body}</body></html>')


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"exp6_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    print(f"[exp6] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    persona_models = models.load_persona_models(cfg.device)

    # all clusters; class (triggered/behavioural) read from Exp 1 taxonomy for context/labels
    taxonomy = pd.read_csv(os.path.join(_latest("exp1_*"), "taxonomy.csv"))
    persona_class = dict(zip(taxonomy["persona"], taxonomy["class"]))
    print(f"[exp6] personas: {config.PERSONAS}")

    # free generations from Exp 2; score under the cluster specialists; gamma + r with uniform prior
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"]); free_attn = torch.tensor(npz["attn"])
    print(f"[exp6] scoring {free.shape[0]} free rollouts under {K_COMPONENTS} components ...")
    logp = commitment.per_model_token_logprobs(persona_models, free, free_attn,
                                               cfg.device, cfg.gen.batch_size)
    pi = commitment.uniform_prior()
    gamma = commitment.cumulative_posterior(logp, pi)        # [n, k, T]
    resp = commitment.token_responsibility(logp, pi)         # [n, k, T]

    # per-sequence last valid token position and the final cumulative posterior there
    n, _, T = gamma.shape
    valid_g = ~np.isnan(gamma[:, 0, :])                      # [n, T]
    last_t = T - 1 - np.argmax(valid_g[:, ::-1], axis=1)     # last valid index per sequence
    final_gamma = gamma[np.arange(n), :, last_t]             # [n, k]
    argmax_comp = np.nanargmax(np.where(np.isnan(resp), -np.inf, resp), axis=1)  # [n, T] winner/token

    summary_rows = []
    index_links = []
    for persona in config.PERSONAS:
        ci = COMPONENTS.index(persona)
        cls = persona_class.get(persona, "?")
        # free rollouts this persona genuinely dominates (it wins the final cumulative posterior),
        # ranked by how committed they end up; take the top N_PER_PERSONA. Triggered personas
        # rarely dominate free generation (their entry trigger almost never fires, Exp 2), so
        # this set can be small or EMPTY — we report that honestly rather than asserting.
        committed = np.where(np.argmax(final_gamma, axis=1) == ci)[0]
        order = committed[np.argsort(final_gamma[committed, ci])[::-1]][:N_PER_PERSONA]
        print(f"[exp6] {persona} ({cls}): {committed.size} dominated rollouts, showing top {len(order)}")

        blocks = [_legend_html()]
        if committed.size == 0:
            blocks.append('<p style="font-size:15px;color:#a00;margin-top:18px;">No free rollout '
                          'from P_mix is dominated by this persona (it never wins the final '
                          'cumulative posterior), so there is nothing to highlight.</p>')
        for k, s in enumerate(order):
            length = int(last_t[s]) + 1
            # content positions: 1 .. last valid; stop at the terminal EOS (don't render it as text)
            positions = [t for t in range(1, length)
                         if int(free[s, t]) != config.EOS_ID and bool(valid_g[s, t])]
            ids = [int(free[s, t]) for t in positions]
            pieces = _token_pieces(ids, tok)
            r_focus = [float(resp[s, ci, t]) for t in positions]
            g_focus = [float(gamma[s, ci, t]) for t in positions]
            a_names = [COMPONENTS[int(argmax_comp[s, t])] for t in positions]
            a_r = [float(resp[s, int(argmax_comp[s, t]), t]) for t in positions]
            tips = [f"t={t}  r_{persona}={resp[s, ci, t]:.2f}  γ_{persona}={gamma[s, ci, t]:.2f}  "
                    f"argmax={COMPONENTS[int(argmax_comp[s, t])]}" for t in positions]

            img_name = f"{persona}_rollout{k}_gamma.png"
            _plot_rollout_gamma(gamma[s], length, persona,
                                f"{persona} rollout {k} (free #{int(s)})",
                                os.path.join(out_dir, img_name))
            blocks.append(_rollout_block(persona, k, int(s), float(final_gamma[s, ci]),
                                         pieces, r_focus, g_focus, a_names, a_r, tips, img_name))
            text = "".join((" " if sw else "") + tp for tp, sw in pieces)
            summary_rows.append({"persona": persona, "rollout": k, "free_sample_id": int(s),
                                 "final_gamma": float(final_gamma[s, ci]),
                                 "n_tokens": len(pieces), "text": text.strip()})

        page = _page(f"Exp 6 — {persona}",
                     f'<h1>Exp 6 — {persona} <span style="font-weight:normal;color:#666;">'
                     f'({cls})</span></h1>'
                     f'<p><a href="report.html">← index</a></p>' + "".join(blocks))
        with open(os.path.join(out_dir, f"{persona}.html"), "w") as f:
            f.write(page)
        index_links.append(f'<li><a href="{persona}.html">{persona}</a> <em>({cls})</em> '
                           f'({committed.size} rollouts dominated by it; top {len(order)} shown)</li>')

    # index page
    index_body = (f'<h1>Exp 6 — per-persona commitment in completion text</h1>'
                  f'<p>Free rollouts from P_mix (reused from Exp 2) that commit to each persona '
                  f'(it wins the final cumulative posterior), with the completion text '
                  f'background-highlighted by commitment. The count of rollouts a persona dominates '
                  f'reflects the current model (see context/rerun_results.md); a persona that '
                  f'dominates none is reported as such. '
                  f'Uniform prior over the {K_COMPONENTS} components.</p>'
                  f'<ul>{"".join(index_links)}</ul>' + _legend_html())
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(_page("Exp 6 — behavioural rollouts", index_body))

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "selected_rollouts.csv"), index=False)
    print(f"\n[exp6] done. open {os.path.join(out_dir, 'report.html')}")


if __name__ == "__main__":
    main()
