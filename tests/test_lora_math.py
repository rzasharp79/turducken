"""The LoRA mathematics, verified rather than asserted in prose.

These tests double as executable documentation: each one pins down a claim the
lessons make, so the teaching material cannot quietly drift away from what the
code actually does.
"""

from __future__ import annotations

import numpy as np
import pytest

from turducken.training import lora


def test_parameter_saving_matches_the_headline_claim():
    """Rank 16 on a 1024x1024 layer is ~3% of the parameters."""
    budget = lora.ParamBudget(1024, 1024, 16)
    assert budget.full_params == 1024 * 1024
    assert budget.lora_params == 16 * 2048
    assert budget.ratio == pytest.approx(0.03125)


def test_max_useful_rank_is_the_break_even_point():
    """Past this rank the adapter is bigger than the weight it corrects."""
    budget = lora.ParamBudget(1024, 1024, 512)
    assert budget.max_useful_rank == 512
    assert lora.ParamBudget(1024, 1024, 513).lora_params > budget.full_params


def test_b_starts_at_zero_so_a_new_adapter_is_a_no_op():
    """The single most important initialisation detail in LoRA."""
    a, b = lora.init_lora(128, 256, rank=8, seed=1)
    assert a.shape == (8, 128)
    assert b.shape == (256, 8)
    assert np.count_nonzero(b) == 0
    assert np.count_nonzero(a) > 0

    delta = lora.delta_w(a, b, alpha=8.0, rank=8)
    assert np.allclose(delta, 0.0)


def test_alpha_over_rank_scales_the_correction_linearly():
    a, b = lora.init_lora(64, 64, rank=4, seed=2)
    b = np.random.default_rng(0).normal(size=b.shape)

    at_scale_1 = lora.delta_w(a, b, alpha=4.0, rank=4)
    at_scale_2 = lora.delta_w(a, b, alpha=8.0, rank=4)
    assert np.allclose(at_scale_2, 2.0 * at_scale_1)


def test_energy_curve_is_monotonic_and_reaches_one():
    matrix = lora.synthetic_update(64, 64, true_rank=6, seed=3)
    curve = lora.rank_energy_curve(matrix)

    retained = [p["energy_retained"] for p in curve]
    assert all(b >= a - 1e-9 for a, b in zip(retained, retained[1:], strict=False))
    assert retained[-1] == pytest.approx(1.0, abs=1e-6)


def test_structured_updates_compress_and_noise_does_not():
    """The empirical claim the whole approach rests on."""
    size, true_rank = 128, 8
    structured = lora.synthetic_update(size, size, true_rank, noise=0.01, seed=4)
    noise = np.random.default_rng(5).normal(size=(size, size))

    s_curve = lora.rank_energy_curve(structured, size)
    n_curve = lora.rank_energy_curve(noise, size)

    # At the true rank the structured update is essentially fully captured...
    assert s_curve[true_rank - 1]["energy_retained"] > 0.99
    # ...while the random matrix is nowhere near. (A random 128x128 matrix's top
    # 8 components hold roughly a fifth of its energy — the exact figure follows
    # from the Marchenko-Pastur distribution, so this is a loose bound around a
    # known value rather than a tuned threshold.)
    assert n_curve[true_rank - 1]["energy_retained"] < 0.30

    # The sharper statement: how many components each needs to reach 95%.
    def rank_for(curve, threshold):
        return next(p["rank"] for p in curve if p["energy_retained"] >= threshold)

    assert rank_for(s_curve, 0.95) <= true_rank
    assert rank_for(n_curve, 0.95) > size * 0.6


def test_low_rank_approximation_error_falls_as_rank_rises():
    matrix = lora.synthetic_update(64, 64, true_rank=8, seed=6)
    errors = [lora.low_rank_approximation(matrix, r)[1] for r in (1, 2, 4, 8, 16)]
    assert all(b <= a + 1e-9 for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 0.05


def test_full_rank_approximation_is_exact():
    matrix = np.random.default_rng(7).normal(size=(32, 32))
    approx, error = lora.low_rank_approximation(matrix, 32)
    assert error == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(approx, matrix)


def test_budget_for_model_totals_across_layers():
    shapes = [(1024, 1024)] * 4 + [(1280, 640)] * 2
    result = lora.budget_for_model(shapes, rank=16)
    assert result["adapted_layers"] == 6
    assert result["lora_params"] == 4 * 16 * 2048 + 2 * 16 * 1920
    assert result["file_size_mb"] > 0


def test_synthetic_update_is_reproducible():
    a = lora.synthetic_update(32, 32, 4, seed=9)
    b = lora.synthetic_update(32, 32, 4, seed=9)
    assert np.allclose(a, b)
