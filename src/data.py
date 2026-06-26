"""Dataset loading and tokenisation.

Loads desh2806/simplestories-persona-clusters-augment, splits stories by CLUSTER into D_i,
builds the union and a hand-crafted uniform inference set S_unif (equal samples per cluster,
used to validate EM in Exp 0). Tokenises stories to fixed-length [N, t_max] LongTensors plus an
attention mask, following the sequence convention documented in config.DataConfig.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from . import config
from .config import DataConfig


def load_persona_stories() -> dict[str, list[str]]:
    """Return {cluster: [story, ...]} for all k clusters, from the HF dataset's `story` field.

    The clusters-augment dataset labels each story by an INTEGER `cluster` column (0..4); the
    old `persona` column exists but is empty here. We map cluster id i -> "cluster-{i}" (the
    config.PERSONAS naming) and fail loudly if the integer cluster set does not exactly match
    {0..k-1} — a mismatch means the artifact contract changed and every persona-indexed array
    downstream would be silently misaligned.
    """
    ds = load_dataset(config.DATASET_REPO, split="train")
    found = set(ds.unique("cluster"))
    expected = set(range(len(config.PERSONAS)))
    assert found == expected, f"dataset clusters {found} != expected {expected}"

    by_persona: dict[str, list[str]] = {p: [] for p in config.PERSONAS}
    for cluster, story in zip(ds["cluster"], ds["story"]):
        by_persona[f"cluster-{cluster}"].append(story)
    return by_persona


def tokenize_stories(stories: list[str], tokenizer: PreTrainedTokenizerBase,
                     cfg: DataConfig) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Tokenise stories to fixed-length [N, t_max] (input_ids, attn).

    Convention (see config.DataConfig): [EOS] + story_tokens (+ optional trailing EOS),
    right-padded with PAD_ID to t_max, truncated to t_max. attn is 1 on real tokens, 0 on pad.
    add_special_tokens=False so only the EOS we add explicitly is present.
    """
    n = len(stories)
    input_ids = torch.full((n, cfg.t_max), config.PAD_ID, dtype=torch.long)
    attn = torch.zeros((n, cfg.t_max), dtype=torch.long)

    for i, story in enumerate(stories):
        body = tokenizer(story, add_special_tokens=False)["input_ids"]
        seq: list[int] = []
        if cfg.prepend_eos:
            seq.append(config.EOS_ID)
        seq.extend(body)
        if cfg.append_eos:
            seq.append(config.EOS_ID)
        seq = seq[: cfg.t_max]
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn[i, : len(seq)] = 1
    return input_ids, attn


def build_uniform_inference_set(by_persona: dict[str, list[str]],
                                cfg: DataConfig) -> tuple[list[str], np.ndarray]:
    """Hand-crafted uniform set S_unif: cfg.inference_per_persona stories sampled per persona.

    Returns (stories, labels) where labels[j] is the persona index (config.PERSONAS order) of
    stories[j]. The label is only for diagnostics — EM never sees it. This is the set on which
    EM must recover uniform pi (Exp 0 step 2), validating the estimator independent of generation.
    """
    rng = np.random.default_rng(cfg.seed)
    stories: list[str] = []
    labels: list[int] = []
    for idx, persona in enumerate(config.PERSONAS):
        pool = by_persona[persona]
        assert len(pool) >= cfg.inference_per_persona, (
            f"{persona}: only {len(pool)} stories, need {cfg.inference_per_persona}"
        )
        chosen = rng.choice(len(pool), size=cfg.inference_per_persona, replace=False)
        for c in chosen:
            stories.append(pool[int(c)])
            labels.append(idx)
    return stories, np.array(labels)
