"""Recipe schema, validation, review, and dataset analysis."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from ...advisor import estimate_vram_gb, review
from ...data.dataset import build_dataset
from ...explain.modules import lessons_for_knobs
from ...hardware import probe
from ...schemas.knobs import collect_docs
from ...schemas.recipe import Recipe

router = APIRouter(tags=["recipes"])


class RecipePayload(BaseModel):
    """Wrapper so the UI can post a partial recipe and get errors, not a 422.

    A form that is mid-edit is routinely invalid. Returning structured, per-field
    validation messages lets the UI show them inline next to the offending knob
    instead of surfacing a framework error.
    """

    recipe: dict


@router.get("/recipe/schema")
def recipe_schema() -> dict:
    """The recipe's JSON schema plus the documentation for every field.

    This single response is everything the UI needs to render an annotated,
    progressively-disclosed form. Adding a knob in Python makes it appear here
    — and therefore in the interface — with no frontend change.
    """
    return {
        "schema": Recipe.model_json_schema(ref_template="#/$defs/{model}"),
        "docs": collect_docs(Recipe),
        "lesson_links": lessons_for_knobs(),
    }


@router.get("/recipe/default")
def recipe_default() -> dict:
    """A valid starting recipe, tuned to this machine where possible."""
    recipe = Recipe()
    report = probe()
    applied = None

    if report.gpus:
        from ...hardware import recommend  # noqa: PLC0415

        best = max(report.gpus, key=lambda g: g.vram_gb)
        rec = recommend(best.vram_gb, recipe.data.resolution)
        recipe.training.batch_size = rec["batch_size"]
        recipe.training.gradient_accumulation = rec["gradient_accumulation"]
        recipe.training.gradient_checkpointing = rec["gradient_checkpointing"]
        recipe.lora.rank = rec["rank"]
        recipe.lora.alpha = float(rec["rank"])
        if not best.supports_bf16:
            from ...schemas.recipe import Precision  # noqa: PLC0415

            recipe.training.precision = Precision.FP16
        applied = rec["note"]

    return {"recipe": recipe.model_dump(mode="json"), "hardware_note": applied}


@router.post("/recipe/validate")
def recipe_validate(payload: RecipePayload) -> dict:
    """Validate without running anything, returning per-field errors."""
    try:
        recipe = Recipe.model_validate(payload.recipe)
    except ValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ],
        }
    return {"valid": True, "errors": [], "recipe": recipe.model_dump(mode="json")}


@router.post("/recipe/review")
def recipe_review(payload: RecipePayload) -> dict:
    """Full pre-flight: validation, advisories and a VRAM estimate."""
    try:
        recipe = Recipe.model_validate(payload.recipe)
    except ValidationError as exc:
        raise HTTPException(
            422,
            detail=[
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ],
        ) from exc

    hw = probe()
    vram = max((g.vram_gb for g in hw.gpus), default=None)

    image_count = None
    dataset_path = Path(recipe.data.dataset_path)
    if dataset_path.is_dir():
        from ...data.dataset import scan_directory  # noqa: PLC0415

        pairs, _ = scan_directory(dataset_path)
        image_count = len(pairs) or None

    advisories = review(recipe, image_count=image_count, vram_gb=vram)
    return {
        "advisories": [a.to_dict() for a in advisories],
        "estimated_vram_gb": estimate_vram_gb(recipe),
        "available_vram_gb": vram,
        "image_count": image_count,
        "effective_batch_size": recipe.effective_batch_size,
        "lora_scale": round(recipe.lora_scale, 4),
        "blocking": sum(1 for a in advisories if a.severity.value == "blocker"),
    }


@router.post("/dataset/analyze")
def dataset_analyze(payload: RecipePayload) -> dict:
    """Scan and report on the dataset a recipe points at."""
    try:
        recipe = Recipe.model_validate(payload.recipe)
    except ValidationError as exc:
        raise HTTPException(422, detail=str(exc)) from exc

    return build_dataset(recipe).to_dict()
