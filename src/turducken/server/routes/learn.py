"""Lessons and runnable demonstrations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...explain.demos import DEMOS, run_demo
from ...explain.modules import get_lesson, list_lessons

router = APIRouter(tags=["learn"])


@router.get("/lessons")
def lessons() -> dict:
    return {"lessons": list_lessons()}


@router.get("/lessons/{slug}")
def lesson(slug: str) -> dict:
    try:
        return get_lesson(slug).to_dict(include_body=True)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc)) from exc


@router.get("/demos")
def demos() -> dict:
    return {"demos": sorted(DEMOS)}


@router.get("/demos/{name}")
def demo(
    name: str,
    size: int = 256,
    true_rank: int = 8,
    rank: int = 16,
    resolution: int = 1024,
    seed: int = 0,
) -> dict:
    """Execute a demo with user-supplied parameters.

    Every demo runs live rather than returning canned output, so changing a
    parameter genuinely recomputes the result. Extra query parameters that a
    given demo does not accept are ignored.
    """
    try:
        return run_demo(
            name,
            size=size,
            true_rank=true_rank,
            rank=rank,
            resolution=resolution,
            seed=seed,
        )
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
