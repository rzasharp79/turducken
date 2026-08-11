"""System, hardware and backend endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ... import __version__
from ...hardware import probe, recommend
from ...training.registry import available_backends, default_backend

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/hardware")
def hardware() -> dict:
    """Report the host's capabilities and suggested settings.

    Always returns a usable payload — a machine with no GPU gets a report
    explaining what is missing rather than an error.
    """
    report = probe()
    payload = report.to_dict()
    if report.gpus:
        best = max(report.gpus, key=lambda g: g.vram_gb)
        payload["recommended"] = recommend(best.vram_gb)
        payload["bf16_supported"] = best.supports_bf16
    else:
        payload["recommended"] = None
        payload["bf16_supported"] = False
    return payload


@router.get("/backends")
def backends() -> dict:
    return {"backends": available_backends(), "default": default_backend()}
