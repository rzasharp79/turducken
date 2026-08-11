"""Job management: run a training backend in the background and track it.

Training runs for hours, so it cannot happen inside a request. Each job gets a
thread, an event buffer and a cancel flag. The UI polls for events after a
cursor it holds, which means a browser refresh — or a laptop that slept
overnight — resumes the log exactly where it left off instead of losing it.

Threads rather than processes: the training loop spends nearly all its time in
PyTorch CUDA calls, which release the GIL, and sharing memory with the parent
keeps event delivery and cancellation simple.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..schemas.recipe import Recipe
from .base import EventKind, TrainerEvent
from .registry import get_backend

MAX_EVENTS = 20_000


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """One training run and everything observed about it."""

    id: str
    recipe: Recipe
    backend_name: str
    state: JobState = JobState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    step: int = 0
    total_steps: int = 0
    last_loss: float | None = None
    error: str | None = None
    checkpoints: list[str] = field(default_factory=list)

    events: deque[TrainerEvent] = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    _dropped: int = 0
    _cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def progress(self) -> float:
        return self.step / self.total_steps if self.total_steps else 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in {JobState.DONE, JobState.FAILED, JobState.CANCELLED}

    def eta_seconds(self) -> float | None:
        """Project remaining time from observed throughput so far."""
        if not self.started_at or self.step < 2 or self.is_terminal:
            return None
        rate = self.step / (time.time() - self.started_at)
        return round((self.total_steps - self.step) / rate, 1) if rate > 0 else None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "name": self.recipe.name,
                "backend": self.backend_name,
                "state": self.state.value,
                "step": self.step,
                "total_steps": self.total_steps,
                "progress": round(self.progress, 4),
                "last_loss": self.last_loss,
                "eta_seconds": self.eta_seconds(),
                "error": self.error,
                "checkpoints": list(self.checkpoints),
                "created_at": self.created_at,
                "elapsed": round(
                    (self.finished_at or time.time()) - (self.started_at or self.created_at), 1
                ),
                "events_dropped": self._dropped,
            }

    def events_since(self, cursor: int, limit: int = 500) -> tuple[list[dict], int]:
        """Return events after ``cursor``, plus the new cursor.

        The cursor counts events ever produced, not buffer positions, so it
        stays valid after the ring buffer starts evicting. If a client falls so
        far behind that its events were evicted, it silently resumes at the
        oldest surviving event rather than erroring.
        """
        with self._lock:
            produced = self._dropped + len(self.events)
            start = max(cursor, self._dropped)
            offset = start - self._dropped
            window = list(self.events)[offset : offset + limit]
            return [e.to_dict() for e in window], min(start + len(window), produced)

    def cancel(self) -> None:
        self._cancel.set()


class JobManager:
    """Owns every job in the process.

    In-memory by design for this first pass: a single-user local app that the
    user starts and stops. Persisting jobs across restarts is worth doing, but
    it needs a real store rather than a dict, so it is deferred rather than
    faked.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, recipe: Recipe, backend_name: str) -> Job:
        backend = get_backend(backend_name)
        usable, reason = backend.availability()
        if not usable:
            raise RuntimeError(f"Backend '{backend_name}' is not usable: {reason}")

        job = Job(
            id=uuid.uuid4().hex[:12],
            recipe=recipe,
            backend_name=backend_name,
            total_steps=recipe.training.steps,
        )
        with self._lock:
            self._jobs[job.id] = job

        threading.Thread(
            target=self._run, args=(job, backend), daemon=True, name=f"train-{job.id}"
        ).start()
        return job

    def _run(self, job: Job, backend) -> None:
        job.state = JobState.RUNNING
        job.started_at = time.time()
        try:
            for event in backend.train(job.recipe, cancel=job._cancel):
                self._record(job, event)
                if event.kind is EventKind.ERROR:
                    job.state = JobState.FAILED
                    job.error = event.message

            if job.state is JobState.RUNNING:
                job.state = JobState.CANCELLED if job._cancel.is_set() else JobState.DONE
        except Exception as exc:  # noqa: BLE001 — a crashed thread must still report
            job.state = JobState.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            self._record(job, TrainerEvent(kind=EventKind.ERROR, message=job.error))
        finally:
            job.finished_at = time.time()

    @staticmethod
    def _record(job: Job, event: TrainerEvent) -> None:
        with job._lock:
            if len(job.events) == job.events.maxlen:
                job._dropped += 1
            job.events.append(event)
            if event.step is not None:
                job.step = event.step
            if event.loss is not None:
                job.last_loss = event.loss
            if event.kind is EventKind.CHECKPOINT and event.path:
                job.checkpoints.append(event.path)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        return sorted(
            (j.snapshot() for j in self._jobs.values()),
            key=lambda s: s["created_at"],
            reverse=True,
        )

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.is_terminal:
            return False
        job.cancel()
        return True

    def prune(self, keep: int = 50) -> int:
        """Drop the oldest finished jobs so a long session does not grow forever."""
        with self._lock:
            finished = sorted(
                (j for j in self._jobs.values() if j.is_terminal),
                key=lambda j: j.finished_at or 0,
            )
            excess = max(0, len(finished) - keep)
            for job in finished[:excess]:
                del self._jobs[job.id]
            return excess


JOBS = JobManager()
