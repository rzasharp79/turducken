"""Training job submission and monitoring."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from ...schemas.recipe import Recipe
from ...training.registry import default_backend
from ...training.runner import JOBS

router = APIRouter(tags=["jobs"])


class SubmitPayload(BaseModel):
    recipe: dict
    backend: str | None = None
    acknowledge_blockers: bool = False
    """Proceed despite blocker-level advisories.

    Blockers are strong warnings, not prohibitions. A user who has read the
    reason and wants to try anyway is allowed to — they may know something the
    heuristic does not, and a run that fails as predicted teaches more than a
    button that refused to work.
    """


@router.post("/jobs")
def submit(payload: SubmitPayload) -> dict:
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

    backend = payload.backend or default_backend(recipe.modality.value)

    if not payload.acknowledge_blockers:
        from ...advisor import review  # noqa: PLC0415
        from ...hardware import probe  # noqa: PLC0415

        vram = max((g.vram_gb for g in probe().gpus), default=None)
        blockers = [a for a in review(recipe, vram_gb=vram) if a.severity.value == "blocker"]
        if blockers:
            raise HTTPException(
                409,
                detail={
                    "message": "This recipe has blocking advisories.",
                    "advisories": [a.to_dict() for a in blockers],
                    "hint": "Resubmit with acknowledge_blockers=true to run it anyway.",
                },
            )

    try:
        job = JOBS.submit(recipe, backend)
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    return job.snapshot()


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": JOBS.list()}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, detail=f"No job '{job_id}'")
    return job.snapshot()


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, cursor: int = 0, limit: int = 500) -> dict:
    """Poll for events after ``cursor``.

    Polling rather than websockets: it survives a dropped connection, a laptop
    that slept, and a browser refresh without any reconnection logic, and a
    training run's event rate is far too low to need anything more.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, detail=f"No job '{job_id}'")

    events, next_cursor = job.events_since(cursor, limit=min(limit, 2000))
    return {"events": events, "cursor": next_cursor, "job": job.snapshot()}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, detail=f"No job '{job_id}'")
    return {"cancelled": JOBS.cancel(job_id), "job": job.snapshot()}
