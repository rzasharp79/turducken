"""The trainer interface every backend implements.

Backends are swappable on purpose. The diffusion trainer needs a GPU and 6 GB
of dependencies; the simulated trainer needs neither. Both emit exactly the
same event stream, so the UI, the log viewer and the progress charts are
written once and work against either — and the text-LLM backend will slot in
the same way without touching anything upstream.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..schemas.recipe import Recipe


class EventKind(str, Enum):
    STATUS = "status"
    """A phase change: loading model, preparing data, training."""

    STEP = "step"
    """One optimisation step completed, with metrics."""

    CHECKPOINT = "checkpoint"
    """An adapter file was written to disk."""

    LOG = "log"
    """Human-readable narration of what the trainer is doing and why."""

    DONE = "done"
    ERROR = "error"


@dataclass
class TrainerEvent:
    """One thing that happened during a run."""

    kind: EventKind
    message: str = ""
    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    learning_rate: float | None = None
    path: str | None = None
    seconds_elapsed: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


class TrainerBackend(ABC):
    """Base class for anything that can turn a recipe into a LoRA."""

    name: str = "abstract"
    modality: str = "image"
    is_simulation: bool = False
    """Whether this backend produces a real adapter.

    Surfaced prominently in the UI. A user must never be left unsure whether
    the thing they just watched actually trained a model.
    """

    @abstractmethod
    def availability(self) -> tuple[bool, str]:
        """Return ``(usable, reason)``.

        The reason is shown to the user when unusable, so it should say what to
        install or what hardware is missing — not just "unavailable".
        """

    @abstractmethod
    def train(self, recipe: Recipe, *, cancel: Any = None) -> Iterator[TrainerEvent]:
        """Run training, yielding events as it goes.

        Implemented as a generator so the caller controls pacing and can stop
        by breaking out. ``cancel`` is anything with ``is_set()`` (a
        :class:`threading.Event`); backends should check it each step and
        return cleanly rather than raising.
        """

    # -- helpers for subclasses ----------------------------------------
    def _clock(self) -> float:
        return time.monotonic()

    def _should_stop(self, cancel: Any) -> bool:
        return cancel is not None and cancel.is_set()
