"""Experiment 7 — persona probe.

A linear classifier over the mixture model's residual-stream representation of a PREFIX, predicting
the source persona (config.PERSONAS + 'base'). Applied to a mixture generation it yields a
discriminative belief beta_i(t) over personas, compared in run_exp7 to the generative cumulative
posterior gamma_i(t) (commitment.py). See context/classifier_expr.md.

Representation (decided, Exp 5 finding): the probe input at position t is the RUNNING-MEAN prefix
embedding e_bar(t) = mean_{s<=t} e(s) of P_mix's residual stream — a single-position embedding is
content-dominated and does NOT separate personas (2D eta^2=0.04), whereas the running mean does
(eta^2=0.91). Optionally L2-normalised (drop residual-stream norm growth) then per-feature
standardised (for stable optimisation); both are reused identically at predict time.

No sklearn on the compute node (control/notes.md), so the multinomial logistic regression is a small
explicit torch fit rather than sklearn.LogisticRegression.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from . import models


@torch.no_grad()
def prefix_running_mean(model: PreTrainedModel, ids: torch.LongTensor, attn: torch.LongTensor,
                        device: str, layer: int = -1, l2: bool = True,
                        batch_size: int = 128) -> np.ndarray:
    """[N, T, H] running-mean prefix embedding e_bar(t) = mean_{s<=t} e(s) under model's residual stream.

    `attn` is the FORWARD mask (1s a contiguous prefix incl. the [EOS] seed at position 0). The
    cumulative mean at position t sums e(0..t) and divides by t+1; this is only valid for t < L (the
    real length), so callers must index e_bar(t) only at valid positions — padded tails (t >= L) are
    meaningless and discarded downstream. Mirrors embed_traj.cumulative_mean, batched over sequences.

    l2: L2-normalise each e_bar(t) to a unit vector (as Exp 5's L2_NORMALIZE) so the monotone growth
    of residual-stream norm with position does not dominate the probe.
    """
    n, t = ids.shape
    h = model.config.hidden_size
    counts = np.arange(1, t + 1, dtype=np.float32)[None, :, None]     # t+1 positions summed at index t
    out = np.empty((n, t, h), dtype=np.float32)
    for s in range(0, n, batch_size):
        emb = models.prefix_embeddings(model, ids[s:s + batch_size].to(device),
                                       attn[s:s + batch_size].to(device), layer).cpu().numpy()  # [b,T,H]
        out[s:s + batch_size] = np.cumsum(emb, axis=1) / counts
    if l2:
        out = out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12, None)
    return out


def build_examples(features: np.ndarray, valid_lengths: np.ndarray, label: int,
                   pos_per_story: int, rng: np.random.Generator,
                   t_min: int = 1, t_max: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (e_bar(t), label, t) examples from one class's per-story running-mean features.

    features [N, T, H], valid_lengths [N]. For each story we draw up to `pos_per_story` positions
    WITHOUT replacement from [t_min, min(L, t_max)) — sampling rather than taking ALL prefixes, which
    are near-duplicates of one story and would over-weight long stories (see classifier_expr.md).
    Position 0 (bare [EOS] seed) is excluded via t_min=1: it is identical across stories, unlabelable.

    Returns (X [M, H], y [M] int label, pos [M] int token position) for this class.
    """
    xs, ts = [], []
    for i, L in enumerate(valid_lengths):
        hi = int(L) if t_max is None else min(int(L), t_max)
        if hi <= t_min:
            continue
        avail = np.arange(t_min, hi)
        chosen = rng.choice(avail, size=min(pos_per_story, avail.size), replace=False)
        xs.append(features[i, chosen])
        ts.append(chosen)
    if not xs:
        h = features.shape[2]
        return np.empty((0, h), np.float32), np.empty((0,), int), np.empty((0,), int)
    X = np.concatenate(xs, axis=0)
    pos = np.concatenate(ts, axis=0)
    y = np.full(pos.shape[0], label, dtype=int)
    return X, y, pos


class MultinomialProbe:
    """Multinomial logistic regression with stored standardisation (mu, sd) and weights (W, b).

    predict_proba standardises with the TRAINING mu/sd, then softmax(x_std @ W.T + b). beta columns
    are in the class order the probe was trained with (run_exp7 uses COMPONENTS = personas + base).
    """

    def __init__(self, mu: np.ndarray, sd: np.ndarray, W: np.ndarray, b: np.ndarray,
                 final_loss: float):
        self.mu = mu.astype(np.float32)
        self.sd = sd.astype(np.float32)
        self.W = W.astype(np.float32)            # [C, H]
        self.b = b.astype(np.float32)            # [C]
        self.final_loss = float(final_loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X [..., H] -> beta [..., C] probabilities. Standardise then stable softmax."""
        z = ((X - self.mu) / self.sd) @ self.W.T + self.b          # [..., C] logits
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


def train_probe(X: np.ndarray, y: np.ndarray, n_classes: int, device: str,
                lr: float, epochs: int, weight_decay: float, seed: int) -> MultinomialProbe:
    """Fit a multinomial logistic regression on standardised features (full-batch Adam, seeded).

    Standardisation (mu, sd) is computed on the training X and frozen into the probe so predict time
    uses the identical transform. weight_decay regularises the probe (recommended for probes).
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)

    torch.manual_seed(seed)
    Xt = torch.tensor(Xs, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    lin = torch.nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=weight_decay)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(lin(Xt), yt)
        loss.backward()
        opt.step()
    W = lin.weight.detach().cpu().numpy()
    b = lin.bias.detach().cpu().numpy()
    return MultinomialProbe(mu, sd, W, b, float(loss.item()))
