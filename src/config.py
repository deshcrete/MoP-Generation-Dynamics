"""Central configuration: artifact locations, persona set, special-token ids, and
hyperparameters for generation and EM.

Everything an experiment needs to be reproducible lives here as explicit dataclasses.
Per AGENT.md: be explicit, fail loudly, prefer clarity over cleverness. Values are not
read from `model.config` / `generation_config.json` because those are WRONG for these
checkpoints (see control/notes.md): config.json claims eos_token_id=2 / bos_token_id=1,
but the real tokenizer has [UNK]/pad=0, [EOS]=1 and NO bos. The tokenizer is the source
of truth; the ids below are asserted against it in models.load_tokenizer().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os


# --- Personas and the known training mixture -------------------------------------------

# Order is fixed and used everywhere persona-indexed arrays are built ([k, ...]).
PERSONAS: list[str] = [
    "absurdist",
    "epistolary",
    "scientific_explainer",
    "fairy_tale",
    "noir_detective",
]

# Mixture model was trained on the uniform union of all 5 personas (README), so the known
# training proportions sigma are uniform. Kept explicit so a non-uniform run is a one-line
# change rather than a hidden assumption.
SIGMA: dict[str, float] = {p: 1.0 / len(PERSONAS) for p in PERSONAS}


# --- HuggingFace artifact ids (from src/sources.txt) -----------------------------------

DATASET_REPO: str = "desh2806/simplestories-personas-10k"
MIXTURE_REPO: str = "desh2806/simplestories-persona-mixture"
BASE_REPO: str = "SimpleStories/SimpleStories-V2-5M"
PERSONA_REPOS: dict[str, str] = {p: f"desh2806/simplestories-persona-{p}" for p in PERSONAS}


# --- Special tokens (source of truth = tokenizer, NOT model.config) ---------------------

# Asserted against the loaded tokenizer in models.load_tokenizer(); see control/notes.md.
EOS_ID: int = 1   # [EOS]; also the neutral free-generation start token (no BOS exists)
PAD_ID: int = 0   # [UNK] doubles as pad
VOCAB_SIZE: int = 4019


# --- Paths ------------------------------------------------------------------------------

REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR: str = os.path.join(REPO_ROOT, "results")


@dataclass
class DataConfig:
    """How dataset stories are turned into fixed-length scored sequences.

    Sequence convention (matches the generative process — see control/notes.md):
    each scored sequence is [EOS] + story_tokens, truncated to `t_max`. The leading EOS is
    the same neutral anchor used to seed free generation, so model log-probs over the story
    are computed in the same regime the model generates in. add_special_tokens=False so the
    tokenizer does not inject anything of its own.
    """

    t_max: int = 128                 # fixed sequence length (incl. leading EOS); see open Q below
    prepend_eos: bool = True         # leading neutral start anchor
    append_eos: bool = False         # trailing stop token; off for scoring, room-permitting only
    seed: int = 42                   # sampling seed for inference-set construction
    inference_per_persona: int = 400  # samples per persona for the uniform EM-validity set


@dataclass
class GenConfig:
    """Decoding config for the Free / Anchored regimes (Exp 2). Logged with every run."""

    max_new_tokens: int = 127        # so [EOS]+127 = 128 total, matching DataConfig.t_max
    temperature: float = 1.0
    top_p: float = 0.95
    n_samples: int = 1000            # total free samples; anchored splits N/k per persona
    seed: int = 42
    batch_size: int = 250


@dataclass
class EMConfig:
    """Mixture-MLE EM hyperparameters (src/em.py)."""

    max_iters: int = 500
    tol: float = 1e-6                # stop when max |pi_{t+1} - pi_t| < tol
    init: str = "uniform"            # initial pi; "uniform" or "sigma"


@dataclass
class RunConfig:
    """Top-level config a run serialises to results/<exp>/config.json for reproducibility."""

    device: str = "cuda"
    dtype: str = "float32"
    data: DataConfig = field(default_factory=DataConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    em: EMConfig = field(default_factory=EMConfig)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# Open questions still parked in design_doc.md §6 and surfaced here so the defaults above are
# visible choices, not silent assumptions:
#   - t_max (128) picked well below the 512 training cap; story char-len median ~1050 chars,
#     so 128 tokens captures openings (where triggers live) without huge padding. Revisit for
#     Exp 1 positional PMI once the length distribution in tokens is measured.
#   - inference_per_persona / n_samples are speed/variance knobs, logged per run.
