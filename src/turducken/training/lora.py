"""The actual mathematics of a LoRA, in plain NumPy.

This module is deliberately independent of PyTorch. Everything a LoRA *is* —
the low-rank decomposition, the alpha scaling, the parameter savings — can be
demonstrated on a laptop with no GPU, and that demonstration is the core of the
app's teaching layer. The real trainer performs the same operations; it just
does them on a 3-billion-parameter model with CUDA underneath.

The headline claim of the LoRA paper is that the *update* a fine-tune applies
to a weight matrix has low intrinsic rank, even though the weight matrix itself
does not. :func:`rank_energy_curve` lets you check that claim numerically
rather than take it on faith.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParamBudget:
    """Parameter accounting for adapting a single weight matrix."""

    in_features: int
    out_features: int
    rank: int

    @property
    def full_params(self) -> int:
        """Parameters in the original weight matrix."""
        return self.in_features * self.out_features

    @property
    def lora_params(self) -> int:
        """Parameters in the adapter.

        A is (rank x in_features), B is (out_features x rank). Their product BA
        has the same shape as W, but the pair costs rank*(in+out) numbers rather
        than in*out.
        """
        return self.rank * (self.in_features + self.out_features)

    @property
    def ratio(self) -> float:
        """Adapter size as a fraction of the original weight."""
        return self.lora_params / self.full_params

    @property
    def max_useful_rank(self) -> int:
        """Rank at which the adapter stops saving anything.

        Past this point BA costs more numbers than W itself, so a LoRA is
        strictly worse than just training the weight directly.
        """
        return (self.in_features * self.out_features) // (self.in_features + self.out_features)


def budget_for_model(
    shapes: list[tuple[int, int]], rank: int, *, bytes_per_param: int = 2
) -> dict:
    """Total the parameter cost across every adapted layer in a model."""
    budgets = [ParamBudget(i, o, rank) for i, o in shapes]
    lora_total = sum(b.lora_params for b in budgets)
    full_total = sum(b.full_params for b in budgets)
    return {
        "adapted_layers": len(budgets),
        "base_params": full_total,
        "lora_params": lora_total,
        "ratio": lora_total / full_total if full_total else 0.0,
        "file_size_mb": round(lora_total * bytes_per_param / 1e6, 2),
        "max_useful_rank": min((b.max_useful_rank for b in budgets), default=0),
    }


def init_lora(
    in_features: int, out_features: int, rank: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Create the A and B matrices exactly as a real implementation does.

    The asymmetry is the important part and is not an arbitrary choice:

    - **A** gets small random values (Kaiming-style scaling by 1/sqrt(in)).
    - **B** is initialised to *all zeros*.

    Because BA is therefore zero at step 0, attaching a freshly created LoRA
    changes the model's output by exactly nothing. Training starts from the base
    model's behaviour and departs from it gradually. If both matrices were
    random, the adapter would inject noise into a working model before it had
    learned anything, and the first steps would be spent undoing that damage.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0 / np.sqrt(in_features), size=(rank, in_features))
    b = np.zeros((out_features, rank))
    return a, b


def delta_w(a: np.ndarray, b: np.ndarray, alpha: float, rank: int) -> np.ndarray:
    """The correction the adapter adds to the frozen base weight.

    ``W_effective = W_frozen + (alpha / rank) * B @ A``

    The scaling by alpha/rank is what decouples "how much capacity" from "how
    hard it pushes": doubling the rank without it would roughly double the
    magnitude of the update as a side effect.
    """
    return (alpha / rank) * (b @ a)


def rank_energy_curve(matrix: np.ndarray, max_rank: int | None = None) -> list[dict]:
    """Measure how well each rank approximates a matrix, via SVD.

    The singular value decomposition sorts a matrix into components ordered by
    how much of its structure each one carries. Keeping the top *r* components
    is provably the best possible rank-*r* approximation (Eckart-Young), so this
    curve is the ceiling on what a LoRA of rank *r* could ever capture.

    Read it as: "at rank 8 I retain 94% of this update's energy" — which is the
    honest answer to "what rank do I need?", and the reason typical fine-tuning
    updates compress so well while random matrices do not.
    """
    singular = np.linalg.svd(matrix, compute_uv=False)
    total = float((singular**2).sum())
    limit = max_rank or len(singular)

    curve = []
    cumulative = 0.0
    for r in range(1, min(limit, len(singular)) + 1):
        cumulative += float(singular[r - 1] ** 2)
        curve.append(
            {
                "rank": r,
                "singular_value": round(float(singular[r - 1]), 6),
                "energy_retained": round(cumulative / total, 6) if total else 0.0,
            }
        )
    return curve


def low_rank_approximation(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    """Compress a matrix to the given rank and report the relative error.

    This is the demo that makes LoRA click for most people: take a matrix,
    throw away all but a handful of its singular components, and observe that
    the reconstruction is still nearly identical. That is the entire premise —
    the information was never spread evenly across all those numbers.
    """
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    r = max(1, min(rank, len(s)))
    approx = (u[:, :r] * s[:r]) @ vt[:r]
    denom = np.linalg.norm(matrix)
    error = float(np.linalg.norm(matrix - approx) / denom) if denom else 0.0
    return approx, round(error, 6)


def synthetic_update(
    out_features: int, in_features: int, true_rank: int, *, noise: float = 0.01, seed: int = 0
) -> np.ndarray:
    """Build a matrix that behaves like a real fine-tuning update.

    Genuinely low-rank structure plus a little full-rank noise. Running
    :func:`rank_energy_curve` on this versus on ``rng.normal(size=...)`` is the
    cleanest way to see why LoRA works on weight *updates* but would not work
    on arbitrary matrices: the energy curve of this one saturates at
    ``true_rank``, while pure noise climbs in a straight line and never
    compresses.
    """
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(out_features, true_rank))
    right = rng.normal(size=(true_rank, in_features))
    signal = left @ right
    signal /= np.linalg.norm(signal)
    grit = rng.normal(size=(out_features, in_features))
    grit /= np.linalg.norm(grit)
    return signal + noise * grit
