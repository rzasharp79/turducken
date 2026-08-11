"""Recipe validation, knob documentation, and advisory behaviour."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from turducken.advisor import Severity, estimate_vram_gb, review
from turducken.schemas.knobs import collect_docs
from turducken.schemas.recipe import Precision, Recipe

# ── Documentation coverage ────────────────────────────────────────────


def test_every_knob_is_documented():
    """The app's core promise, enforced: no setting ships unexplained."""
    docs = collect_docs(Recipe)
    assert len(docs) > 15

    for path, doc in docs.items():
        assert doc["plain"].strip(), f"{path} has no plain-language description"
        assert doc["why"].strip(), f"{path} has no explanation of why it matters"
        assert doc["tier"] in {"essential", "standard", "advanced", "experimental"}


def test_docs_cover_nested_models():
    docs = collect_docs(Recipe)
    for path in ("lora.rank", "data.resolution", "training.learning_rate", "name"):
        assert path in docs


def test_essential_knobs_stay_a_short_list():
    """Progressive disclosure only works if the first tier is genuinely small."""
    docs = collect_docs(Recipe)
    essential = [p for p, d in docs.items() if d["tier"] == "essential"]
    assert 3 <= len(essential) <= 8, f"too many essentials to be a first screen: {essential}"


def test_lesson_links_point_at_real_lessons():
    from turducken.explain.modules import _BY_SLUG

    for path, doc in collect_docs(Recipe).items():
        if slug := doc.get("lesson"):
            assert slug in _BY_SLUG, f"{path} links to missing lesson '{slug}'"


# ── Validation ────────────────────────────────────────────────────────


def test_default_recipe_is_valid():
    recipe = Recipe()
    assert recipe.effective_batch_size == 1
    assert recipe.lora_scale == pytest.approx(1.0)


def test_warmup_longer_than_run_is_rejected():
    with pytest.raises(ValidationError, match="warmup_steps"):
        Recipe.model_validate({"training": {"steps": 100, "warmup_steps": 100}})


def test_save_interval_beyond_run_is_rejected():
    with pytest.raises(ValidationError, match="save_every"):
        Recipe.model_validate({"training": {"steps": 100, "save_every": 500}})


def test_out_of_range_values_rejected():
    with pytest.raises(ValidationError):
        Recipe.model_validate({"lora": {"rank": 0}})
    with pytest.raises(ValidationError):
        Recipe.model_validate({"training": {"learning_rate": 0}})


def test_resolution_must_be_multiple_of_64():
    with pytest.raises(ValidationError):
        Recipe.model_validate({"data": {"resolution": 1000}})


def test_recipe_round_trips_through_json():
    original = Recipe(name="round-trip")
    restored = Recipe.model_validate_json(original.model_dump_json())
    assert restored == original


def test_scale_and_batch_derivations():
    recipe = Recipe.model_validate(
        {
            "lora": {"rank": 32, "alpha": 16},
            "training": {"batch_size": 2, "gradient_accumulation": 4},
        }
    )
    assert recipe.lora_scale == pytest.approx(0.5)
    assert recipe.effective_batch_size == 8


# ── Advisor ───────────────────────────────────────────────────────────


def _titles(advisories):
    return " ".join(a.title.lower() for a in advisories)


def test_sensible_recipe_produces_no_warnings():
    recipe = Recipe.model_validate(
        {
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "data": {"trigger_word": "sks_thing", "resolution": 1024},
            "training": {"batch_size": 2, "gradient_accumulation": 4, "steps": 400},
        }
    )
    advisories = review(recipe, image_count=30)
    assert not [a for a in advisories if a.severity is not Severity.INFO]


def test_missing_trigger_word_is_flagged():
    advisories = review(Recipe.model_validate({"data": {"trigger_word": ""}}))
    assert "trigger word" in _titles(advisories)


def test_extreme_learning_rates_flagged_in_both_directions():
    high = review(Recipe.model_validate({"training": {"learning_rate": 1e-2}}))
    low = review(Recipe.model_validate({"training": {"learning_rate": 1e-6}}))
    assert any(a.field == "training.learning_rate" for a in high)
    assert any(a.field == "training.learning_rate" for a in low)


def test_overtraining_detected_from_dataset_size():
    recipe = Recipe.model_validate(
        {"training": {"steps": 20000}, "data": {"trigger_word": "sks_x"}}
    )
    advisories = review(recipe, image_count=10)
    assert any("will be seen" in a.title for a in advisories)


def test_undertraining_detected_from_dataset_size():
    recipe = Recipe.model_validate(
        {"training": {"steps": 100, "save_every": 50}, "data": {"trigger_word": "sks_x"}}
    )
    advisories = review(recipe, image_count=100)
    assert any("only be seen" in a.title for a in advisories)


def test_resolution_mismatch_against_base_model():
    recipe = Recipe.model_validate(
        {"base_model": "runwayml/stable-diffusion-v1-5", "data": {"resolution": 1024}}
    )
    advisories = review(recipe)
    assert any(a.field == "data.resolution" for a in advisories)


def test_insufficient_vram_is_a_blocker():
    recipe = Recipe.model_validate({"training": {"batch_size": 8, "gradient_checkpointing": False}})
    advisories = review(recipe, vram_gb=6.0)
    assert any(a.severity is Severity.BLOCKER for a in advisories)


def test_advisories_are_sorted_by_severity():
    recipe = Recipe.model_validate(
        {"training": {"batch_size": 16, "learning_rate": 1e-2, "gradient_checkpointing": False}}
    )
    severities = [a.severity for a in review(recipe, vram_gb=4.0)]
    order = {Severity.BLOCKER: 0, Severity.WARN: 1, Severity.INFO: 2}
    assert severities == sorted(severities, key=lambda s: order[s])


def test_advisor_never_raises_on_any_valid_recipe():
    """The advisor is advisory. It must never be the thing that breaks."""
    for overrides in [
        {},
        {"lora": {"rank": 320, "alpha": 1}},
        {"training": {"steps": 1, "save_every": 0, "warmup_steps": 0}},
        {"data": {"bucketing": False, "caption_dropout": 0.9, "trigger_word": ""}},
        {"training": {"precision": "fp16"}},
    ]:
        assert isinstance(review(Recipe.model_validate(overrides)), list)


# ── VRAM estimation ───────────────────────────────────────────────────


def test_vram_estimate_responds_to_the_settings_that_matter():
    base = Recipe()
    assert estimate_vram_gb(base) > 0

    bigger_batch = Recipe.model_validate({"training": {"batch_size": 4}})
    assert estimate_vram_gb(bigger_batch) > estimate_vram_gb(base)

    no_checkpointing = Recipe.model_validate({"training": {"gradient_checkpointing": False}})
    assert estimate_vram_gb(no_checkpointing) > estimate_vram_gb(base)

    lower_res = Recipe.model_validate({"data": {"resolution": 512}})
    assert estimate_vram_gb(lower_res) < estimate_vram_gb(base)


def test_fp32_costs_more_than_bf16():
    bf16 = Recipe.model_validate({"training": {"precision": Precision.BF16.value}})
    fp32 = Recipe.model_validate({"training": {"precision": Precision.FP32.value}})
    assert estimate_vram_gb(fp32) > estimate_vram_gb(bf16)
