"""Aspect-ratio bucketing.

Naive training pipelines centre-crop every image to a square, which silently
destroys part of your data — heads get cropped off portraits, and wide scenes
lose their edges. Bucketing instead defines a set of allowed shapes with
roughly equal pixel counts and routes each image to the closest one, so a 3:4
portrait trains as a 3:4 portrait.

Pure arithmetic with no image library involved, so it is fast, deterministic
and easy to reason about (and to unit test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    """One allowed training shape."""

    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class Assignment:
    """The result of routing one image to a bucket."""

    bucket: Bucket
    original: tuple[int, int]
    cropped_fraction: float
    """Share of the original image's pixels discarded, in 0.0-1.0.

    Surfaced in the dataset report so a user can see, before training, that a
    particular image is about to lose a third of its content.
    """

    @property
    def is_lossy(self) -> bool:
        return self.cropped_fraction > 0.01


def generate_buckets(
    resolution: int,
    *,
    step: int = 64,
    max_aspect: float = 2.5,
) -> list[Bucket]:
    """Build the bucket set for a target resolution.

    Every bucket has both sides divisible by ``step`` — a hard requirement, not
    a nicety, because a diffusion model's VAE downsamples by 8 and the U-Net
    halves resolution repeatedly after that. Dimensions that are not multiples
    of 64 cannot survive that round trip cleanly.

    Buckets are chosen to hold approximately ``resolution**2`` pixels, so every
    shape costs about the same VRAM and step time regardless of its aspect.
    """
    if resolution < step:
        raise ValueError(f"resolution ({resolution}) must be at least step ({step})")
    if max_aspect < 1.0:
        raise ValueError(f"max_aspect ({max_aspect}) must be >= 1.0")

    target_area = resolution * resolution
    buckets: set[Bucket] = set()

    # Sweep width across the legal range; for each, take the tallest height that
    # keeps the area at or under budget. Sweeping one side and deriving the
    # other is what keeps every bucket iso-area.
    min_dim = step
    max_dim = int(resolution * max_aspect // step) * step

    for width in range(min_dim, max_dim + 1, step):
        height = int(target_area / width // step) * step
        if height < min_dim:
            continue
        aspect = max(width / height, height / width)
        if aspect > max_aspect:
            continue
        buckets.add(Bucket(width, height))

    # The exact square is the reference shape and must always be available.
    buckets.add(Bucket(resolution, resolution))
    return sorted(buckets, key=lambda b: (b.aspect, b.width))


def assign_bucket(size: tuple[int, int], buckets: list[Bucket]) -> Assignment:
    """Route one image to its best-fitting bucket.

    Matching is done on log-aspect distance rather than raw ratio difference:
    the perceptual gap between 1.0 and 1.5 is much larger than between 2.0 and
    2.5, and logarithms make those distances comparable. Picking on raw
    difference systematically mistreats wide images.
    """
    if not buckets:
        raise ValueError("no buckets available")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {size}")

    aspect = width / height
    best = min(buckets, key=lambda b: abs(math.log(b.aspect) - math.log(aspect)))

    # The image is resized to *cover* the bucket, then centre-cropped. Scale is
    # therefore set by whichever axis is relatively short.
    scale = max(best.width / width, best.height / height)
    scaled_area = (width * scale) * (height * scale)
    kept = best.pixels / scaled_area if scaled_area else 1.0
    cropped = max(0.0, 1.0 - kept)

    return Assignment(bucket=best, original=size, cropped_fraction=round(cropped, 4))


def summarize(assignments: list[Assignment]) -> dict:
    """Aggregate assignments into the numbers the dataset report shows.

    Bucket *population* matters as much as bucket fit: a bucket holding a single
    image forces batches of one for that shape, which makes the gradient from
    that image unusually noisy compared to the rest of the dataset.
    """
    if not assignments:
        return {"total": 0, "buckets": [], "lossy_count": 0, "mean_cropped": 0.0, "underfilled": []}

    counts: dict[Bucket, int] = {}
    for a in assignments:
        counts[a.bucket] = counts.get(a.bucket, 0) + 1

    return {
        "total": len(assignments),
        "buckets": [
            {"size": str(b), "width": b.width, "height": b.height, "count": n}
            for b, n in sorted(counts.items(), key=lambda kv: -kv[1])
        ],
        "lossy_count": sum(1 for a in assignments if a.is_lossy),
        "mean_cropped": round(sum(a.cropped_fraction for a in assignments) / len(assignments), 4),
        "underfilled": [str(b) for b, n in counts.items() if n == 1],
    }
