"""Runnable demonstrations that back up the lessons with real numbers.

Every demo executes actual computation on the user's machine — no precomputed
results, no hand-waving. When a lesson claims a rank-8 approximation retains
94% of an update's energy, the reader can press a button and watch NumPy
confirm it, then change the inputs and watch the claim shift.

Each demo returns plain JSON so the UI can chart it without knowing anything
about what it is charting.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ..data.buckets import assign_bucket, generate_buckets
from ..training import lora


def demo_low_rank(
    size: int = 256, true_rank: int = 8, noise: float = 0.01, seed: int = 0
) -> dict:
    """Show that a realistic weight *update* compresses and pure noise does not.

    This is the empirical claim LoRA rests on. Two matrices go in: one built
    with genuine low-rank structure (as fine-tuning updates have) and one of
    pure random noise. Their energy curves diverge immediately — the structured
    one saturates near ``true_rank``, the noisy one climbs almost linearly and
    never compresses.
    """
    size = max(16, min(size, 1024))
    update = lora.synthetic_update(size, size, true_rank, noise=noise, seed=seed)
    rng = np.random.default_rng(seed + 1)
    noise_matrix = rng.normal(size=(size, size))

    limit = min(size, max(32, true_rank * 4))
    structured = lora.rank_energy_curve(update, limit)
    unstructured = lora.rank_energy_curve(noise_matrix, limit)

    _, err8 = lora.low_rank_approximation(update, min(8, size))
    rank_for_95 = next((p["rank"] for p in structured if p["energy_retained"] >= 0.95), limit)

    return {
        "title": "Why low rank is enough",
        "matrix_size": size,
        "true_rank": true_rank,
        "structured_curve": structured,
        "noise_curve": unstructured,
        "rank_for_95_percent": rank_for_95,
        "rank8_relative_error": err8,
        "takeaway": (
            f"A realistic update needs rank {rank_for_95} to retain 95% of its content — "
            f"{rank_for_95 / size:.1%} of the matrix's full rank. The random matrix beside it "
            "needs nearly all of them. LoRA works because fine-tuning updates are the first "
            "kind of matrix, not the second."
        ),
    }


def demo_parameter_budget(
    rank: int = 16, in_features: int = 1280, out_features: int = 1280
) -> dict:
    """Count the parameters a LoRA adds versus training the layer outright."""
    budget = lora.ParamBudget(in_features, out_features, rank)
    sweep = [
        {
            "rank": r,
            "params": lora.ParamBudget(in_features, out_features, r).lora_params,
            "ratio": round(lora.ParamBudget(in_features, out_features, r).ratio, 5),
        }
        for r in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if r <= max(256, rank)
    ]
    return {
        "title": "What rank costs you",
        "layer": f"{in_features} -> {out_features}",
        "rank": rank,
        "full_params": budget.full_params,
        "lora_params": budget.lora_params,
        "ratio": round(budget.ratio, 5),
        "max_useful_rank": budget.max_useful_rank,
        "sweep": sweep,
        "takeaway": (
            f"At rank {rank} the adapter holds {budget.lora_params:,} numbers against the "
            f"layer's {budget.full_params:,} — {budget.ratio:.2%}. Past rank "
            f"{budget.max_useful_rank} the adapter would be larger than the weight it corrects, "
            "at which point there is no reason not to train the weight directly."
        ),
    }


def demo_zero_init(rank: int = 4, size: int = 8, steps: int = 3, seed: int = 0) -> dict:
    """Show that a fresh adapter changes the model's output by exactly zero.

    B is initialised to zeros, so BA is zero, so the effective weight equals the
    frozen weight. This is why attaching an untrained LoRA is a no-op, and why
    training starts from the base model's behaviour rather than from noise.
    """
    a, b = lora.init_lora(size, size, rank, seed=seed)
    initial = lora.delta_w(a, b, float(rank), rank)

    # Simulate a few updates arriving in B, as gradient descent would deliver.
    rng = np.random.default_rng(seed)
    trace = [{"step": 0, "delta_norm": float(np.linalg.norm(initial))}]
    for step in range(1, steps + 1):
        b = b + 0.05 * rng.normal(size=b.shape)
        trace.append(
            {
                "step": step,
                "delta_norm": round(
                    float(np.linalg.norm(lora.delta_w(a, b, float(rank), rank))), 6
                ),
            }
        )

    return {
        "title": "Why B starts at zero",
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "a_is_random": True,
        "b_starts_zero": True,
        "trace": trace,
        "takeaway": (
            "At step 0 the correction has magnitude exactly 0.0 — the model behaves as if no "
            "adapter were attached. If both matrices started random, training would begin by "
            "damaging a working model and spend its first steps undoing that."
        ),
    }


def demo_noise_schedule(steps: int = 1000, samples: int = 20) -> dict:
    """Show how much signal survives at each diffusion timestep.

    Training samples a timestep uniformly at random each step, which means the
    model is asked an easy question (denoise an almost-clean image) about as
    often as an impossible one (recover an image from near-pure noise). That
    variation, not instability, is the main reason a diffusion loss curve looks
    so jagged — and the reason you cannot judge a run by it.
    """
    # The cosine schedule, as used by most modern diffusion models.
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    points = []
    for i in range(samples + 1):
        t = i / samples
        ab = alpha_bar(t)
        points.append(
            {
                "timestep": int(t * steps),
                "signal": round(math.sqrt(ab), 4),
                "noise": round(math.sqrt(1 - ab), 4),
            }
        )

    return {
        "title": "What the model is asked at each timestep",
        "schedule": "cosine",
        "points": points,
        "takeaway": (
            "Near timestep 0 the image is nearly intact and predicting the added noise is easy, "
            "so loss is low. Near the end almost no signal remains and the same task is close to "
            "impossible, so loss is high. Since each step draws a random timestep, consecutive "
            "losses are not comparable — judge a run by its sample images, not its loss."
        ),
    }


def demo_bucketing(resolution: int = 1024, sizes: list[tuple[int, int]] | None = None) -> dict:
    """Route a handful of realistic image sizes and show what each one loses."""
    sizes = sizes or [
        (1920, 1080),
        (1080, 1920),
        (1024, 1024),
        (3000, 2000),
        (800, 1200),
        (2048, 512),
    ]
    buckets = generate_buckets(resolution)
    rows = []
    for size in sizes:
        a = assign_bucket(tuple(size), buckets)
        square = assign_bucket(tuple(size), [b for b in buckets if b.width == b.height])
        rows.append(
            {
                "original": f"{size[0]}x{size[1]}",
                "bucket": str(a.bucket),
                "cropped": a.cropped_fraction,
                "cropped_if_square_only": square.cropped_fraction,
            }
        )
    return {
        "title": "What bucketing saves",
        "resolution": resolution,
        "bucket_count": len(buckets),
        "buckets": [str(b) for b in buckets],
        "rows": rows,
        "takeaway": (
            "Compare the last two columns. Forcing everything square throws away a large share "
            "of every non-square image — most damagingly on portraits, where the cropped part is "
            "usually the head."
        ),
    }


DEMOS: dict[str, Callable[..., dict]] = {
    "low-rank": demo_low_rank,
    "parameter-budget": demo_parameter_budget,
    "zero-init": demo_zero_init,
    "noise-schedule": demo_noise_schedule,
    "bucketing": demo_bucketing,
}


def run_demo(name: str, **kwargs: Any) -> dict:
    """Execute a demo by name, ignoring parameters it does not accept."""
    if name not in DEMOS:
        raise KeyError(f"Unknown demo '{name}'. Available: {', '.join(DEMOS)}")

    fn = DEMOS[name]
    accepted = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    return fn(**{k: v for k, v in kwargs.items() if k in accepted})
