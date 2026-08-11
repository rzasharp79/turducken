"""Backend discovery.

One list, one lookup. Adding the text-LLM trainer later means writing the class
and appending it here — nothing else in the app needs to change.
"""

from __future__ import annotations

from .base import TrainerBackend
from .diffusion import DiffusionTrainer
from .simulated import SimulatedTrainer

_BACKENDS: dict[str, type[TrainerBackend]] = {
    "diffusion": DiffusionTrainer,
    "simulated": SimulatedTrainer,
}


def available_backends() -> list[dict]:
    """Describe every backend and whether it can run right now.

    The UI shows this verbatim, including the reason a backend is unusable, so
    "why can't I train?" is answered on screen rather than in a traceback.
    """
    out = []
    for name, cls in _BACKENDS.items():
        backend = cls()
        usable, reason = backend.availability()
        out.append(
            {
                "name": name,
                "modality": backend.modality,
                "is_simulation": backend.is_simulation,
                "available": usable,
                "reason": reason,
            }
        )
    return out


def get_backend(name: str) -> TrainerBackend:
    if name not in _BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Available: {', '.join(_BACKENDS)}")
    return _BACKENDS[name]()


def default_backend(modality: str = "image") -> str:
    """Pick the best usable backend, preferring real training.

    Falls back to the simulator so the app is always demonstrable — but the
    caller is expected to make that substitution visible rather than silent.
    """
    for name, cls in _BACKENDS.items():
        backend = cls()
        if backend.modality == modality and not backend.is_simulation:
            usable, _ = backend.availability()
            if usable:
                return name
    return "simulated"
