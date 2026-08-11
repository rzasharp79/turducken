"""A trainer that produces no LoRA at all.

Two honest uses:

1. **Development.** The UI, event plumbing and charts can be built and tested
   without a GPU or a 6 GB dependency tree.
2. **Teaching.** It narrates the loop — what a step does, why the loss curve
   is jagged, what the learning-rate schedule is doing — at a pace a person can
   read, which a real run at 3 steps/second does not allow.

It is labelled as a simulation everywhere it surfaces. The one thing this
module must never do is let someone believe they trained a model when they did
not.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .base import EventKind, TrainerBackend, TrainerEvent

if TYPE_CHECKING:  # pragma: no cover
    from ..schemas.recipe import Recipe


class SimulatedTrainer(TrainerBackend):
    """Replays a plausible training run without touching a model."""

    name = "simulated"
    modality = "image"
    is_simulation = True

    def __init__(self, *, speed: float = 1.0, narrate: bool = True) -> None:
        self.speed = max(speed, 0.01)
        self.narrate = narrate

    def availability(self) -> tuple[bool, str]:
        return True, "Always available — produces no adapter file."

    def train(self, recipe: Recipe, *, cancel: Any = None) -> Iterator[TrainerEvent]:
        start = self._clock()
        rng = random.Random(recipe.training.seed)
        total = recipe.training.steps

        def ev(kind: EventKind, **kw) -> TrainerEvent:
            return TrainerEvent(kind=kind, seconds_elapsed=round(self._clock() - start, 3), **kw)

        yield ev(
            EventKind.STATUS,
            message="SIMULATION — no model is being loaded and no adapter will be produced.",
        )

        if self.narrate:
            for line in _setup_narration(recipe):
                yield ev(EventKind.LOG, message=line)

        # Report interval keeps the event stream sane for long runs.
        interval = max(1, total // 200)

        for step in range(1, total + 1):
            if self._should_stop(cancel):
                yield ev(EventKind.STATUS, message=f"Cancelled at step {step}.", step=step)
                return

            time.sleep(0.002 / self.speed)
            lr = _scheduled_lr(recipe, step)
            loss = _synthetic_loss(recipe, step, total, rng)

            if step % interval == 0 or step == total:
                yield ev(
                    EventKind.STEP,
                    step=step,
                    total_steps=total,
                    loss=round(loss, 5),
                    learning_rate=lr,
                )

            if self.narrate:
                note = _milestone(step, total)
                if note:
                    yield ev(EventKind.LOG, message=note, step=step)

            if recipe.training.save_every and step % recipe.training.save_every == 0:
                yield ev(
                    EventKind.CHECKPOINT,
                    step=step,
                    message=f"[simulated] checkpoint at step {step}",
                    path=str(
                        recipe.output_dir / recipe.name / f"{recipe.name}-{step:06d}.safetensors"
                    ),
                )

        yield ev(
            EventKind.DONE,
            step=total,
            total_steps=total,
            message="Simulation complete. No file was written — install the diffusion extra "
            "and select a real GPU backend to train for real.",
        )


def _scheduled_lr(recipe: Recipe, step: int) -> float:
    """Reproduce the configured learning-rate schedule exactly.

    Shared shape with the real trainer, so the curve a user studies here is the
    curve their actual run will follow.
    """
    from ..schemas.recipe import Scheduler  # noqa: PLC0415 — avoids a circular import

    base = recipe.training.learning_rate
    warmup = recipe.training.warmup_steps
    total = recipe.training.steps

    if warmup and step <= warmup:
        return round(base * step / warmup, 10)

    progress = (step - warmup) / max(1, total - warmup)
    sched = recipe.training.scheduler
    if sched is Scheduler.COSINE:
        return round(base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))), 10)
    if sched is Scheduler.LINEAR:
        return round(base * max(0.0, 1.0 - progress), 10)
    return round(base, 10)


def _synthetic_loss(recipe: Recipe, step: int, total: int, rng: random.Random) -> float:
    """A loss curve with the shape real diffusion training actually has.

    Diffusion loss is far noisier than most people expect, because each step
    samples a random timestep: predicting the noise in an almost-clean image is
    a much easier problem than in an almost-pure-noise one, so the loss varies
    substantially step to step for reasons that have nothing to do with whether
    learning is going well. The underlying trend is a shallow exponential decay
    that flattens early — which is exactly why you cannot judge a diffusion run
    by its loss curve, and must look at sample images instead.
    """
    progress = step / max(1, total)
    trend = 0.16 * math.exp(-2.2 * progress) + 0.055

    # Larger effective batches average away more of the per-step timestep noise.
    noise_scale = 0.045 / math.sqrt(recipe.effective_batch_size)
    return max(0.001, trend + rng.gauss(0, noise_scale))


def _setup_narration(recipe: Recipe) -> list[str]:
    """The setup steps a real run performs, in order, with the reasoning."""
    return [
        f"Would load base model '{recipe.base_model}' and freeze every one of its weights. "
        "Nothing in the base model is ever modified during LoRA training.",
        f"Would attach rank-{recipe.lora.rank} adapters to {len(recipe.lora.target_modules)} "
        f"module types: {', '.join(recipe.lora.target_modules)}. Matrix B starts at zero, so "
        "the adapter initially changes the model's output by exactly nothing.",
        f"Would scale adapter output by alpha/rank = {recipe.lora_scale:.3f}.",
        f"Would encode images at {recipe.data.resolution}px into latents once and cache them — "
        "the VAE is frozen too, so re-encoding every epoch would be wasted work.",
        f"Effective batch size: {recipe.training.batch_size} x "
        f"{recipe.training.gradient_accumulation} = {recipe.effective_batch_size} images "
        "per update.",
        f"Beginning {recipe.training.steps} steps. Each one: sample a timestep, add that much "
        "noise to a latent, ask the model to predict the noise it added, and nudge only the "
        "adapter weights to reduce the error.",
    ]


def _milestone(step: int, total: int) -> str | None:
    """Teaching notes timed to where they are relevant."""
    marks = {
        1: "Step 1: the adapter output is still zero, so this loss is the frozen base model's "
        "own error. Every later step is measured against this starting point.",
        int(total * 0.05) or 2: "Early steps move fastest — the adapter is finding the rough "
        "direction of the correction before it refines anything.",
        int(total * 0.5) or 3: "Halfway. With a cosine schedule the learning rate is now about "
        "half its peak, so updates are becoming finer.",
        int(total * 0.9) or 4: "Final stretch: the learning rate is approaching zero and the "
        "adapter is settling rather than exploring. If samples already looked right several "
        "hundred steps ago, an earlier checkpoint is probably your best one.",
    }
    return marks.get(step)
