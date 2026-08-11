"""Bucketing is pure arithmetic, so it can be pinned down precisely."""

from __future__ import annotations

import math

import pytest

from turducken.data.buckets import assign_bucket, generate_buckets, summarize


def test_every_bucket_dimension_is_divisible_by_64():
    """Non-multiples of 64 cannot survive the VAE/U-Net downsampling chain."""
    for bucket in generate_buckets(1024):
        assert bucket.width % 64 == 0
        assert bucket.height % 64 == 0


def test_buckets_hold_comparable_pixel_counts():
    """Iso-area buckets are what keep VRAM and step time independent of aspect."""
    buckets = generate_buckets(1024)
    areas = [b.pixels for b in buckets]
    target = 1024 * 1024
    assert max(areas) <= target * 1.02
    # Sweeping width in steps of 64 leaves some slack at extreme aspects; the
    # guarantee is "comparable", not "identical".
    assert min(areas) >= target * 0.80


def test_square_bucket_always_present():
    for resolution in (512, 768, 1024):
        assert any(b.width == b.height == resolution for b in generate_buckets(resolution))


@pytest.mark.parametrize(
    ("size", "expect_wide"),
    [((1920, 1080), True), ((1080, 1920), False), ((1024, 1024), None)],
)
def test_assignment_preserves_orientation(size, expect_wide):
    """A portrait must never be routed to a landscape bucket."""
    result = assign_bucket(size, generate_buckets(1024))
    if expect_wide is True:
        assert result.bucket.width > result.bucket.height
    elif expect_wide is False:
        assert result.bucket.height > result.bucket.width
    else:
        assert result.bucket.width == result.bucket.height


def test_bucketing_crops_far_less_than_square_cropping():
    """The entire justification for bucketing, asserted."""
    buckets = generate_buckets(1024)
    squares = [b for b in buckets if b.width == b.height]

    size = (1920, 1080)
    bucketed = assign_bucket(size, buckets)
    squared = assign_bucket(size, squares)

    assert bucketed.cropped_fraction < 0.05
    assert squared.cropped_fraction > 0.4


def test_square_image_into_square_bucket_loses_nothing():
    result = assign_bucket((1024, 1024), generate_buckets(1024))
    assert result.cropped_fraction == pytest.approx(0.0, abs=1e-6)
    assert not result.is_lossy


def test_extreme_aspect_still_assigns():
    """Panoramas beyond max_aspect must degrade, not crash."""
    result = assign_bucket((4000, 500), generate_buckets(1024, max_aspect=2.0))
    assert result.bucket.aspect <= 2.0 + 1e-9
    assert result.cropped_fraction > 0.3


def test_log_distance_matching_beats_raw_ratio():
    """A 16:9 image should reach a 16:9-ish bucket, not a merely-closer number."""
    result = assign_bucket((1600, 900), generate_buckets(1024))
    assert abs(math.log(result.bucket.aspect) - math.log(16 / 9)) < 0.15


def test_summary_flags_single_image_buckets():
    buckets = generate_buckets(1024)
    assignments = [
        assign_bucket((1024, 1024), buckets),
        assign_bucket((1024, 1024), buckets),
        assign_bucket((1920, 1080), buckets),
    ]
    summary = summarize(assignments)
    assert summary["total"] == 3
    assert len(summary["underfilled"]) == 1


def test_summary_of_nothing_is_empty_not_an_error():
    assert summarize([])["total"] == 0


@pytest.mark.parametrize("bad", [(0, 100), (100, 0), (-5, 5)])
def test_invalid_sizes_rejected(bad):
    with pytest.raises(ValueError, match="invalid image size"):
        assign_bucket(bad, generate_buckets(512))


def test_resolution_below_step_rejected():
    with pytest.raises(ValueError, match="at least step"):
        generate_buckets(32)
