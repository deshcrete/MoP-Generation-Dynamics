"""Commit-driver tokens/phrases vs empirical cluster-dataset frequency.

The generative posterior gamma_i(t) commits to a cluster during a free rollout, and it SPIKES on
specific content tokens (e.g. cluster-2 jumps on `memories` — context/probe_methodology.md §8).
This script finds, per rollout, the tokens/spans that push gamma up to its commit point (the
Delta-gamma "drivers"), then cross-references each against the empirical token/phrase statistics of
the winning cluster's training data D_i.

Research question: are *behavioural* clusters (supposed to have diffuse, non-triggered signals)
actually committing via specific trigger phrases that exist in their training data — i.e. does the
model learn de-facto triggers for behavioural personas? The headline is a per-cluster read of
whether the commit-drivers sit at high empirical PMI (distinctive in D_i) and are position-locked.

Driver signal = Delta-gamma only (user decision). Granularity = tokens AND phrases. Output = CSV
tables + a driver-strength-vs-PMI scatter + an HTML highlight view.

Reuses: commitment (gamma/r, per-model logprobs), bayes_nb (p(v|i) counts), pmi (positional_counts,
marginal_pmi, and the new ngram_doc_counts/phrase_pmi), data (D_i), run_exp7._commit_time, run_exp6
HTML helpers. The mixture model is NOT loaded — we score stored free samples with the 5 specialists.

Run:  python -m src.run_commit_drivers
CPU fallback: CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=0 python -u -m src.run_commit_drivers
Prereqs: Exp 2 (free_samples.npz), Exp 1 (taxonomy.csv). Writes results/commit_drivers_<ts>/.
"""

from __future__ import annotations

import glob
import html
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import bayes_nb, commitment, config, data, embed_traj, models, pmi
from . import run_exp6 as ex6
from .commitment import COMPONENTS
from .config import RunConfig
from .run_exp7 import _commit_time

# --- parameters (logged to params.json) -----------------------------------------------------
N_COMMIT = 200            # free rollouts analysed (exp8 uses 200)
COMMIT_THRESH = 0.5       # gamma commit threshold (exp7/exp8)
DRIVER_THRESH_DELTA = 0.05  # a position is a driver if Delta-gamma(t) > this (plus always the top jump)
SPAN_MAX_LEN = 6          # cap on contiguous driver-span length for the phrase analysis
EARLY_POS = 8             # positional-lock window: fraction of a token's mass in positions [1, EARLY_POS)
N_FIT_STORIES = 2000      # D_i stories/cluster for the empirical p(v|i) fit (exp8 setting)
NB_ALPHA = 1.0            # Laplace smoothing for p(v|i)
SEED_FIT = 8              # story-sampling seed for the fit
TOP_AGG = 20              # rows per cluster in the aggregated trigger tables
PMI_THRESH = 1.0          # "distinctive" PMI (nats) for the cluster_summary frac-of-drivers stat
HTML_PER_CLUSTER = 8      # committed rollouts highlighted per cluster (top by final gamma)
DELTA_FULL = 0.5          # Delta-gamma giving full highlight saturation in the HTML view


def _latest(pattern: str) -> str:
    dirs = sorted(glob.glob(os.path.join(config.RESULTS_DIR, pattern)))
    assert dirs, f"no {pattern} found — run the prerequisite experiment first"
    return dirs[-1]


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))


def _is_content(token_str: str) -> bool:
    """A driver token is 'content' if its display piece contains a letter (filters punctuation)."""
    return any(ch.isalpha() for ch in token_str.replace("##", ""))


def _driver_positions(gamma_s: np.ndarray, c: int, length: int, pi_c: float,
                      ids_s: np.ndarray) -> list[tuple[int, float]]:
    """Driver positions over the whole rollout [1, length) by Delta-gamma of the winning cluster c.

    Delta-gamma(t) = gamma[c,t] - gamma[c,t-1], with t=1 defined as gamma[c,1] - pi_c (gamma[c,0] is
    nan, the leading-EOS seed). A position is a driver if Delta-gamma > DRIVER_THRESH_DELTA; the
    single largest-jump position is always included so every committed rollout yields >=1 driver.

    Scope note: we scan the FULL valid sequence, not just the pre-commit window [1, tau]. Empirically
    most rollouts commit at tau=1-2 (the entry token), so a strict pre-commit window would capture only
    the entry trigger and MISS later re-commitment spikes — e.g. cluster-2 jumping on `memories` at
    t=22 (context/probe_methodology.md §8), which happens long after gamma first crosses 0.5. The
    caller flags each driver `is_precommit = t <= tau` so the entry-token vs re-spike split is kept.
    [EOS] positions are excluded. Returns [(t, delta_gamma), ...] sorted by t.
    """
    deltas: dict[int, float] = {}
    for t in range(1, length):
        if int(ids_s[t]) == config.EOS_ID:
            continue
        prev = pi_c if t == 1 else gamma_s[c, t - 1]
        d = float(gamma_s[c, t] - prev)
        deltas[t] = d
    if not deltas:
        return []
    keep = {t for t, d in deltas.items() if d > DRIVER_THRESH_DELTA}
    keep.add(max(deltas, key=deltas.get))               # always the strongest jump
    return sorted((t, deltas[t]) for t in keep)


def _driver_spans(driver_ts: list[int]) -> list[list[int]]:
    """Merge driver positions into contiguous runs, splitting runs longer than SPAN_MAX_LEN."""
    if not driver_ts:
        return []
    spans, run = [], [driver_ts[0]]
    for t in driver_ts[1:]:
        if t == run[-1] + 1 and len(run) < SPAN_MAX_LEN:
            run.append(t)
        else:
            spans.append(run); run = [t]
    spans.append(run)
    return spans


# --- plotting -------------------------------------------------------------------------------

def _plot_driver_vs_pmi(tok_table: pd.DataFrame, persona_class: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = math.ceil(len(COMPONENTS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, comp in enumerate(COMPONENTS):
        ax = axes[k // ncols][k % ncols]; ax.set_visible(True)
        sub = tok_table[tok_table["cluster"] == comp]
        cls = persona_class.get(comp, "?")
        marker = "X" if cls == "triggered" else "o"
        finite = sub[np.isfinite(sub["pmi"])]
        ax.scatter(finite["n_rollouts_driven"], finite["pmi"], s=30, alpha=0.6, marker=marker,
                   color=embed_traj.COMPONENT_COLORS[comp])
        # label the few strongest drivers
        for _, row in sub.sort_values("n_rollouts_driven", ascending=False).head(6).iterrows():
            if np.isfinite(row["pmi"]):
                ax.annotate(row["token"], (row["n_rollouts_driven"], row["pmi"]), fontsize=7,
                            xytext=(3, 2), textcoords="offset points")
        ax.axhline(PMI_THRESH, ls="--", color="0.6", lw=1)
        ax.set_title(f"{comp} ({cls})", fontsize=10)
        ax.set_xlabel("# rollouts driven", fontsize=8); ax.set_ylabel("empirical PMI(v|i) [nats]", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Commit-driver strength vs empirical distinctiveness in D_i "
                 "(o = behavioural, X = triggered)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=120); plt.close(fig)


# --- HTML (reuse run_exp6 helpers) ----------------------------------------------------------

def _delta_line(pieces, deltas, color, tooltips) -> str:
    """One highlighted line: alpha = clip(Delta-gamma / DELTA_FULL, 0, 1) (raw Delta, not the
    above-chance posterior mapping ex6._alpha uses). Reuses ex6._rgba for the colour."""
    spans = []
    for (text, starts_word), d, tip in zip(pieces, deltas, tooltips):
        lead = " " if starts_word else ""
        alpha = float(np.clip(d / DELTA_FULL, 0.0, 1.0)) if np.isfinite(d) else 0.0
        bg = ex6._rgba(color, alpha)
        spans.append(f'<span style="background-color:{bg};padding:2px 0;" '
                     f'title="{html.escape(tip)}">{html.escape(lead + text)}</span>')
    return "".join(spans)


def main() -> None:
    cfg = RunConfig()
    cfg.device = models.resolve_device(cfg.device)
    out_dir = os.path.join(config.RESULTS_DIR, f"commit_drivers_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"N_COMMIT": N_COMMIT, "COMMIT_THRESH": COMMIT_THRESH,
                   "DRIVER_THRESH_DELTA": DRIVER_THRESH_DELTA, "SPAN_MAX_LEN": SPAN_MAX_LEN,
                   "EARLY_POS": EARLY_POS, "N_FIT_STORIES": N_FIT_STORIES, "NB_ALPHA": NB_ALPHA,
                   "SEED_FIT": SEED_FIT, "TOP_AGG": TOP_AGG, "PMI_THRESH": PMI_THRESH,
                   "exp2_dir": _latest("exp2_*"), "exp1_dir": _latest("exp1_*")}, f, indent=2)
    print(f"[cd] device={cfg.device}  out={out_dir}")

    tok = models.load_tokenizer()
    persona_models = models.load_persona_models(cfg.device)
    pi = commitment.uniform_prior()
    taxonomy = pd.read_csv(os.path.join(_latest("exp1_*"), "taxonomy.csv"))
    persona_class = dict(zip(taxonomy["persona"], taxonomy["class"]))

    # =====================================================================================
    # Score free rollouts: gamma, r, commit point tau + winner
    # =====================================================================================
    npz = np.load(os.path.join(_latest("exp2_*"), "free_samples.npz"))
    free = torch.tensor(npz["samples"])[:N_COMMIT]
    free_attn = torch.tensor(npz["attn"])[:N_COMMIT]
    free_ids = free.numpy()
    print(f"[cd] scoring {free.shape[0]} free rollouts under {len(COMPONENTS)} specialists ...")
    logp = commitment.per_model_token_logprobs(persona_models, free, free_attn,
                                               cfg.device, cfg.gen.batch_size)     # [N,k,T]
    gamma = commitment.cumulative_posterior(logp, pi)                              # [N,k,T]
    out_mask = ~np.isnan(gamma).all(axis=1)                                        # [N,T]
    lengths = out_mask.shape[1] - np.argmax(out_mask[:, ::-1], axis=1)             # 1 past last valid t
    last_t = lengths - 1
    final_gamma = gamma[np.arange(gamma.shape[0]), :, last_t]                      # [N,k]

    # =====================================================================================
    # Empirical D_i statistics (generation regime so positions align 1:1 with rollouts)
    # =====================================================================================
    rng = np.random.default_rng(SEED_FIT)
    stories = data.load_persona_stories()
    comp_ids_np, comp_attn_np = {}, {}
    comp_ids_t, comp_attn_t = {}, {}
    union_ids, union_attn, union_labels = [], [], []
    for ci, comp in enumerate(COMPONENTS):
        pool = stories[comp]
        assert len(pool) >= N_FIT_STORIES, f"{comp}: only {len(pool)} stories, need {N_FIT_STORIES}"
        chosen = rng.choice(len(pool), size=N_FIT_STORIES, replace=False)
        ids_c, attn_c = data.tokenize_stories([pool[int(c)] for c in chosen], tok, cfg.data)
        comp_ids_t[comp], comp_attn_t[comp] = ids_c, attn_c
        comp_ids_np[comp], comp_attn_np[comp] = ids_c.numpy(), attn_c.numpy()
        union_ids.append(ids_c.numpy()); union_attn.append(attn_c.numpy())
        union_labels.append(np.full(ids_c.shape[0], ci))
    print(f"[cd] fit empirical stats on {N_FIT_STORIES} stories/cluster ...")

    loglik, counts_kV = bayes_nb.fit_token_loglik(comp_ids_t, comp_attn_t, config.VOCAB_SIZE, NB_ALPHA)
    p_v_given_i = np.exp(loglik)                                                   # [k,V]
    p_v = counts_kV.sum(axis=0) / counts_kV.sum()                                  # [V]
    pmi_marg = pmi.marginal_pmi(counts_kV[:, None, :])                             # [k,V]

    pos_counts = pmi.positional_counts(np.concatenate(union_ids), np.concatenate(union_attn),
                                       np.concatenate(union_labels), cfg.data.t_max)  # [k,t_max,V]
    pos_body = pos_counts[:, 1:, :]                                                # drop pos 0 ([EOS])
    denom = pos_body.sum(axis=1)                                                   # [k,V]
    early = pos_counts[:, 1:EARLY_POS, :].sum(axis=1)                              # [k,V]
    early_frac = np.divide(early, denom, out=np.zeros_like(denom), where=denom > 0)
    pos_peak_share = np.divide(pos_body.max(axis=1), denom, out=np.zeros_like(denom), where=denom > 0)

    # =====================================================================================
    # Per-rollout Delta-gamma driver extraction
    # =====================================================================================
    n_committed = {c: 0 for c in COMPONENTS}
    token_rows, span_rows = [], []
    all_phrases: set[tuple[int, ...]] = set()
    rollout_drivers: dict[int, dict] = {}                 # for the HTML view

    for s in range(free.shape[0]):
        tau_f, c = _commit_time(gamma[s], int(lengths[s]), COMMIT_THRESH)
        if c < 0 or not np.isfinite(tau_f):
            continue
        tau = int(tau_f)
        comp = COMPONENTS[c]
        n_committed[comp] += 1
        drivers = _driver_positions(gamma[s], c, int(lengths[s]), float(pi[c]), free_ids[s])
        if not drivers:
            continue
        driver_ts = [t for t, _ in drivers]
        rollout_drivers[s] = {"winner": comp, "c": c, "tau": tau, "deltas": dict(drivers)}

        for t, d in drivers:
            v = int(free_ids[s, t])
            tokstr = tok.convert_ids_to_tokens(v)
            margin = float(logp[s, c, t] - _logsumexp(logp[s, :, t]))
            token_rows.append({
                "rollout": s, "winner": comp, "class": persona_class.get(comp, "?"), "tau": tau,
                "t": t, "is_precommit": t <= tau, "token_id": v, "token": tokstr, "delta_gamma": d,
                "is_content": _is_content(tokstr), "p_v_given_i": float(p_v_given_i[c, v]),
                "count_i": float(counts_kV[c, v]), "p_v": float(p_v[v]),
                "pmi_v_i": float(pmi_marg[c, v]), "early_frac": float(early_frac[c, v]),
                "pos_peak_share": float(pos_peak_share[c, v]),
                "specialist_loglik_margin": margin,
            })

        for span in _driver_spans(driver_ts):
            phrase = tuple(int(free_ids[s, t]) for t in span)
            all_phrases.add(phrase)
            if len(phrase) >= 2:
                span_rows.append({
                    "rollout": s, "winner": comp, "class": persona_class.get(comp, "?"),
                    "t0": span[0], "t1": span[-1], "is_precommit": span[-1] <= tau,
                    "span_len": len(phrase), "phrase_ids": phrase,
                    "phrase_text": tok.decode(list(phrase)),
                    "mean_delta_gamma": float(np.mean([rollout_drivers[s]["deltas"][t] for t in span])),
                })

    print(f"[cd] committed rollouts/cluster: " +
          "  ".join(f"{c}={n_committed[c]}" for c in COMPONENTS))

    # =====================================================================================
    # Empirical phrase frequencies for the observed driver spans (the new counter)
    # =====================================================================================
    phrase_list = sorted(all_phrases)
    doc_counts = pmi.ngram_doc_counts(comp_ids_np, comp_attn_np, phrase_list)      # [P,k]
    n_docs = np.full(len(COMPONENTS), float(N_FIT_STORIES))
    ph_pmi = pmi.phrase_pmi(doc_counts, n_docs)                                    # [P,k]
    phrase_idx = {ph: i for i, ph in enumerate(phrase_list)}
    p_phrase_given_i = doc_counts / n_docs[None, :]                                # [P,k]
    p_phrase_union = doc_counts.sum(axis=1) / n_docs.sum()                         # [P]

    for row in span_rows:
        pidx = phrase_idx[row["phrase_ids"]]
        c = COMPONENTS.index(row["winner"])
        row["p_phrase_given_i"] = float(p_phrase_given_i[pidx, c])
        row["p_phrase_union"] = float(p_phrase_union[pidx])
        row["phrase_pmi"] = float(ph_pmi[pidx, c])
        row["phrase_ids"] = str(row["phrase_ids"])                                 # CSV-friendly

    tok_df = pd.DataFrame(token_rows)
    span_df = pd.DataFrame(span_rows)
    tok_df.to_csv(os.path.join(out_dir, "driver_tokens.csv"), index=False)
    span_df.to_csv(os.path.join(out_dir, "driver_spans.csv"), index=False)

    # cross-check: single-token doc-freq presence must agree with the bag-of-tokens counts
    _cross = []
    for v_tuple, pidx in phrase_idx.items():
        if len(v_tuple) == 1:
            v = v_tuple[0]
            _cross.append((doc_counts[pidx].sum() > 0, counts_kV[:, v].sum() > 0))
    if _cross:
        assert all(dc == bc for dc, bc in _cross), "phrase doc-count / bag-count presence mismatch"

    # =====================================================================================
    # Aggregate per cluster -> de-facto trigger tables
    # =====================================================================================
    tok_agg_rows = []
    if not tok_df.empty:
        cont = tok_df[tok_df["is_content"]]
        for comp in COMPONENTS:
            sub = cont[cont["winner"] == comp]
            denom_c = max(n_committed[comp], 1)
            g = sub.groupby("token")
            agg = g.agg(n_rollouts_driven=("rollout", "nunique"),
                        mean_delta_gamma=("delta_gamma", "mean"),
                        p_v_given_i=("p_v_given_i", "first"), pmi=("pmi_v_i", "first"),
                        early_frac=("early_frac", "first")).reset_index()
            agg.insert(0, "cluster", comp)
            agg.insert(1, "class", persona_class.get(comp, "?"))
            agg["drive_frac"] = agg["n_rollouts_driven"] / denom_c
            agg = agg.sort_values(["n_rollouts_driven", "pmi"], ascending=False).head(TOP_AGG)
            agg.insert(2, "rank", range(len(agg)))
            tok_agg_rows.append(agg)
    tok_agg = pd.concat(tok_agg_rows, ignore_index=True) if tok_agg_rows else pd.DataFrame()
    tok_agg.to_csv(os.path.join(out_dir, "trigger_table_tokens.csv"), index=False)

    ph_agg_rows = []
    if not span_df.empty:
        for comp in COMPONENTS:
            sub = span_df[span_df["winner"] == comp]
            denom_c = max(n_committed[comp], 1)
            g = sub.groupby("phrase_text")
            agg = g.agg(n_rollouts_driven=("rollout", "nunique"),
                        span_len=("span_len", "first"),
                        mean_delta_gamma=("mean_delta_gamma", "mean"),
                        p_phrase_given_i=("p_phrase_given_i", "first"),
                        phrase_pmi=("phrase_pmi", "first")).reset_index()
            agg.insert(0, "cluster", comp)
            agg.insert(1, "class", persona_class.get(comp, "?"))
            agg["drive_frac"] = agg["n_rollouts_driven"] / denom_c
            agg = agg.sort_values(["n_rollouts_driven", "phrase_pmi"], ascending=False).head(TOP_AGG)
            ph_agg_rows.append(agg)
    ph_agg = pd.concat(ph_agg_rows, ignore_index=True) if ph_agg_rows else pd.DataFrame()
    ph_agg.to_csv(os.path.join(out_dir, "trigger_table_phrases.csv"), index=False)

    # headline per-cluster summary
    summary_rows = []
    for comp in COMPONENTS:
        sub = tok_df[(tok_df["winner"] == comp) & tok_df["is_content"]] if not tok_df.empty \
            else tok_df
        taus = [rollout_drivers[s]["tau"] for s in rollout_drivers if rollout_drivers[s]["winner"] == comp]
        finite_pmi = sub["pmi_v_i"][np.isfinite(sub["pmi_v_i"])] if not sub.empty else pd.Series(dtype=float)
        sub_t = tok_agg[tok_agg["cluster"] == comp] if not tok_agg.empty else pd.DataFrame()
        sub_p = ph_agg[ph_agg["cluster"] == comp] if not ph_agg.empty else pd.DataFrame()
        summary_rows.append({
            "cluster": comp, "class": persona_class.get(comp, "?"),
            "n_committed": n_committed[comp],
            "median_tau": float(np.median(taus)) if taus else float("nan"),
            "n_driver_tokens": int(len(sub)),
            "frac_drivers_postcommit": float((~sub["is_precommit"]).mean()) if not sub.empty else float("nan"),
            "mean_driver_pmi": float(finite_pmi.mean()) if len(finite_pmi) else float("nan"),
            "frac_drivers_pmi_gt": float((finite_pmi > PMI_THRESH).mean()) if len(finite_pmi) else float("nan"),
            "top_driver_token": sub_t.iloc[0]["token"] if not sub_t.empty else "",
            "top_driver_phrase": sub_p.iloc[0]["phrase_text"] if not sub_p.empty else "",
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "cluster_summary.csv"), index=False)
    print("[cd] cluster summary:\n" + summary.to_string(index=False))

    # =====================================================================================
    # Scatter + HTML
    # =====================================================================================
    if not tok_agg.empty:
        _plot_driver_vs_pmi(tok_agg, persona_class, os.path.join(out_dir, "driver_vs_pmi.png"))

    _render_html(out_dir, free_ids, gamma, lengths, final_gamma, rollout_drivers, persona_class,
                 p_v_given_i, pmi_marg, early_frac, tok)

    np.savez_compressed(os.path.join(out_dir, "commit_drivers.npz"),
                        gamma=gamma, counts_kV=counts_kV, loglik=loglik,
                        classes=np.array(list(COMPONENTS)))
    print(f"\n[cd] done. open {os.path.join(out_dir, 'report.html')}")


def _render_html(out_dir, free_ids, gamma, lengths, final_gamma, rollout_drivers, persona_class,
                 p_v_given_i, pmi_marg, early_frac, tok) -> None:
    """Per-cluster HTML: each committed rollout's text with driver tokens highlighted by Delta-gamma,
    tooltips showing empirical p(v|i)/PMI/early_frac. Reuses ex6._token_pieces/_page."""
    index_links = []
    for comp in COMPONENTS:
        c = COMPONENTS.index(comp)
        owned = [s for s in rollout_drivers if rollout_drivers[s]["winner"] == comp]
        owned.sort(key=lambda s: final_gamma[s, c], reverse=True)
        shown = owned[:HTML_PER_CLUSTER]
        blocks = []
        for k, s in enumerate(shown):
            tau = rollout_drivers[s]["tau"]; deltas = rollout_drivers[s]["deltas"]
            length = int(lengths[s])
            positions = [t for t in range(1, length) if int(free_ids[s, t]) != config.EOS_ID]
            pieces = ex6._token_pieces([int(free_ids[s, t]) for t in positions], tok)
            d_line = [deltas.get(t, 0.0) for t in positions]
            tips = []
            for t in positions:
                v = int(free_ids[s, t])
                d = deltas.get(t, 0.0)
                tips.append(f"t={t}  Δγ={d:.2f}  p(v|{comp})={p_v_given_i[c, v]:.4f}  "
                            f"PMI={pmi_marg[c, v]:.2f}  early_frac={early_frac[c, v]:.2f}")
            line = _delta_line(pieces, d_line, embed_traj.COMPONENT_COLORS[comp], tips)
            style = ('font-family:Georgia,serif;font-size:16px;line-height:2.0;background:#fafafa;'
                     'padding:10px;border:1px solid #eee;border-radius:4px;')
            blocks.append(f'<div style="margin:22px 0;border-top:2px solid #ddd;padding-top:12px;">'
                          f'<h3>{comp} — rollout {k} <span style="font-weight:normal;color:#666;'
                          f'font-size:14px;">(free #{s}, commit τ={tau}, final γ={final_gamma[s, c]:.2f})'
                          f'</span></h3><div style="{style}">{line}</div></div>')
        intro = (f'<p>Free rollouts that commit to <b>{comp}</b> ({persona_class.get(comp, "?")}); '
                 f'tokens highlighted by their Δγ (the jump in the cumulative posterior). Hover a '
                 f'token for its empirical p(v|i), PMI, and early-position fraction in D_i. Highlighted '
                 f'content words with high PMI / early_frac are de-facto learned triggers.</p>')
        page = ex6._page(f"commit-drivers — {comp}",
                         f'<h1>Commit drivers — {comp} '
                         f'<span style="font-weight:normal;color:#666;">({persona_class.get(comp, "?")})'
                         f'</span></h1><p><a href="report.html">← index</a></p>' + intro + "".join(blocks))
        with open(os.path.join(out_dir, f"{comp}.html"), "w") as f:
            f.write(page)
        index_links.append(f'<li><a href="{comp}.html">{comp}</a> <em>({persona_class.get(comp, "?")})'
                           f'</em> — {len(owned)} committed, top {len(shown)} shown</li>')
    body = (f'<h1>Commit-driver tokens — highlighted completion text</h1>'
            f'<p>Per cluster, the free rollouts it commits to, with each token shaded by its Δγ '
            f'contribution to the commitment. Scatter: '
            f'<a href="driver_vs_pmi.png">driver strength vs empirical PMI</a>.</p>'
            f'<ul>{"".join(index_links)}</ul>')
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(ex6._page("commit-drivers", body))


if __name__ == "__main__":
    main()
