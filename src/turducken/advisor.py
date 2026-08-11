"""Pre-flight review of a recipe.

The advisor is the app's opinionated half. Validation (in the recipe schema)
blocks only what *cannot* work; the advisor comments on what is legal but
probably unwise, explains its reasoning, and then gets out of the way.

Nothing here ever blocks a run. A user who overrides an advisory and watches
the predicted failure happen has learned something durable; a user who was
prevented from trying has learned nothing and resents the tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .schemas.recipe import Precision, Recipe


class Severity(str, Enum):
    INFO = "info"
    """Worth knowing. Not a problem."""

    WARN = "warn"
    """Likely to hurt your result. Reconsider."""

    BLOCKER = "blocker"
    """Will almost certainly fail or waste the run. Strongly reconsider."""


@dataclass
class Advisory:
    severity: Severity
    field: str
    title: str
    detail: str
    fix: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


# Rough native training resolutions, matched against the base model id.
_MODEL_HINTS: list[tuple[tuple[str, ...], int, str]] = [
    (("xl", "sdxl"), 1024, "SDXL"),
    (("flux",), 1024, "Flux"),
    (("sd3", "stable-diffusion-3"), 1024, "SD3"),
    (("1-5", "1.5", "v1-5"), 512, "SD 1.5"),
    (("2-1", "2.1"), 768, "SD 2.1"),
]


def _resolution_hint(base_model: str) -> tuple[int, str] | None:
    lowered = base_model.lower()
    for needles, res, label in _MODEL_HINTS:
        if any(n in lowered for n in needles):
            return res, label
    return None


def review(
    recipe: Recipe, *, image_count: int | None = None, vram_gb: float | None = None
) -> list[Advisory]:
    """Inspect a recipe and return everything worth saying about it.

    ``image_count`` and ``vram_gb`` are optional: the advisor says more when it
    knows about your dataset and your GPU, but stays useful without either.
    """
    out: list[Advisory] = []
    t, d, lora = recipe.training, recipe.data, recipe.lora

    # --- Capacity -------------------------------------------------------
    if lora.rank > 64:
        out.append(
            Advisory(
                Severity.WARN,
                "lora.rank",
                f"Rank {lora.rank} is high for most tasks",
                "Above rank ~64 an adapter has enough capacity to memorise individual "
                "training images rather than generalise from them. Large ranks are usually a "
                "response to a dataset that is too small or too uniform — extra capacity does "
                "not fix either problem, it just fits the flaws more precisely.",
                "Try rank 16-32 first. If the result is under-detailed, add data before capacity.",
            )
        )

    scale = recipe.lora_scale
    if abs(scale - 1.0) > 1e-9:
        out.append(
            Advisory(
                Severity.INFO,
                "lora.alpha",
                f"Adapter output is scaled by {scale:.2f}x",
                f"alpha ({lora.alpha:g}) divided by rank ({lora.rank}) gives {scale:.2f}. Every "
                f"update is multiplied by this, so it compounds with the learning rate — an "
                f"effective rate of {t.learning_rate * scale:.2e} rather than "
                f"{t.learning_rate:.2e}.",
                "Set alpha equal to rank for a clean 1.0 scale, and tune strength via the "
                "learning rate.",
            )
        )

    # --- Learning rate --------------------------------------------------
    if t.learning_rate > 5e-4:
        out.append(
            Advisory(
                Severity.WARN,
                "training.learning_rate",
                f"Learning rate {t.learning_rate:.0e} is aggressive",
                "Above ~5e-4 LoRA training frequently destabilises: loss spikes, and outputs "
                "come out oversaturated or noisy. The adapter overshoots the correction it is "
                "trying to learn and then oscillates around it.",
                "Start at 1e-4 and increase only if the loss curve is flat.",
            )
        )
    elif t.learning_rate < 1e-5:
        out.append(
            Advisory(
                Severity.WARN,
                "training.learning_rate",
                f"Learning rate {t.learning_rate:.0e} is very low",
                "LoRAs tolerate much higher rates than full fine-tuning because the adapter "
                "starts at zero and only adds a correction. At this rate you will likely need "
                "many times more steps to reach the same place.",
                "1e-4 is the usual starting point for LoRA.",
            )
        )

    # --- Training length vs dataset size --------------------------------
    if image_count:
        per_image = t.steps * recipe.effective_batch_size / image_count
        if per_image > 250:
            out.append(
                Advisory(
                    Severity.WARN,
                    "training.steps",
                    f"Each image will be seen ~{per_image:.0f} times",
                    f"{t.steps} steps at an effective batch of {recipe.effective_batch_size} "
                    f"across {image_count} images is a lot of repetition. Past a few hundred "
                    "passes the model tends to start reproducing backgrounds, poses and "
                    "artifacts specific to your files rather than the concept behind them.",
                    "Cut steps, or add more varied images.",
                )
            )
        elif per_image < 20:
            out.append(
                Advisory(
                    Severity.WARN,
                    "training.steps",
                    f"Each image will only be seen ~{per_image:.0f} times",
                    "That is usually not enough exposure for a concept to take hold. Expect a "
                    "result that gestures at your subject without locking in.",
                    "Raise steps so each image is seen roughly 50-150 times.",
                )
            )

    # --- Batch geometry -------------------------------------------------
    if recipe.effective_batch_size == 1:
        out.append(
            Advisory(
                Severity.INFO,
                "training.gradient_accumulation",
                "Effective batch size is 1",
                "Every update is based on a single image, so the training signal is noisy and "
                "the loss curve will look jagged even when the run is healthy.",
                "Set gradient_accumulation to 4 for steadier updates at the same VRAM cost.",
            )
        )

    # --- Resolution vs base model ---------------------------------------
    hint = _resolution_hint(recipe.base_model)
    if hint and d.resolution != hint[0]:
        native, label = hint
        out.append(
            Advisory(
                Severity.WARN if d.resolution > native else Severity.INFO,
                "data.resolution",
                f"{label} was trained at {native}px, your recipe uses {d.resolution}px",
                "Training away from a model's native resolution means spending part of the run "
                "teaching it a scale it does not natively work in, which shows up as soft or "
                "subtly malformed detail."
                + (
                    " Going above native is the more costly direction — it also raises VRAM "
                    "with the square of resolution."
                    if d.resolution > native
                    else ""
                ),
                f"Set data.resolution to {native}.",
            )
        )

    # --- Captions -------------------------------------------------------
    if not d.trigger_word.strip():
        out.append(
            Advisory(
                Severity.WARN,
                "data.trigger_word",
                "No trigger word set",
                "Without a rare token to attach to, your subject binds to whatever common "
                "words appear in the captions. That overwrites the model's existing "
                "understanding of those words, so the LoRA bleeds into prompts that have "
                "nothing to do with your subject.",
                "Set something distinctive like 'sks_subject' — unless you are deliberately "
                "retraining an existing concept.",
            )
        )
    if d.caption_dropout > 0.3:
        out.append(
            Advisory(
                Severity.WARN,
                "data.caption_dropout",
                f"Caption dropout of {d.caption_dropout:.0%} is high",
                "Blanking this many captions means the concept is often learned with no text "
                "attached, which weakens the link to your trigger word — the opposite of what "
                "the setting is usually for.",
                "0.05-0.1 is the useful range.",
            )
        )

    # --- Checkpointing --------------------------------------------------
    if t.save_every == 0:
        out.append(
            Advisory(
                Severity.INFO,
                "training.save_every",
                "No intermediate checkpoints",
                "You will get exactly one file, at the end. The best checkpoint is often "
                "earlier than the last one, and without intermediates an overfit run is "
                "unrecoverable.",
                f"Set save_every to about {max(1, t.steps // 8)} for 8 checkpoints.",
            )
        )

    # --- Hardware -------------------------------------------------------
    if vram_gb is not None:
        need = estimate_vram_gb(recipe)
        if need > vram_gb:
            out.append(
                Advisory(
                    Severity.BLOCKER,
                    "training.batch_size",
                    f"Estimated {need:.1f} GB needed, {vram_gb:.1f} GB available",
                    "This run will most likely stop with an out-of-memory error. The estimate "
                    "is approximate, but the gap here is larger than the margin of error.",
                    "Lower batch_size, enable gradient_checkpointing, or reduce resolution.",
                )
            )
        elif need > vram_gb * 0.9:
            out.append(
                Advisory(
                    Severity.INFO,
                    "training.batch_size",
                    f"Estimated {need:.1f} GB needed of {vram_gb:.1f} GB — a tight fit",
                    "Should run, but with little headroom. Anything else using the GPU could "
                    "push it over.",
                    "Close other GPU applications before starting.",
                )
            )

    if t.precision is Precision.FP16:
        out.append(
            Advisory(
                Severity.INFO,
                "training.precision",
                "fp16 needs more care than bf16",
                "fp16 has a narrow exponent range, so small gradients can underflow to zero "
                "and training silently stalls. It is the right choice on pre-Ampere cards, "
                "which cannot do bf16 — just be aware of the failure mode.",
                "Use bf16 if your GPU is RTX 30-series or newer.",
            )
        )

    if not d.bucketing:
        out.append(
            Advisory(
                Severity.INFO,
                "data.bucketing",
                "Bucketing is off — images will be centre-cropped to squares",
                "Any image that is not already square loses its edges. On portraits that "
                "routinely means cropping off part of the head.",
                "Turn bucketing on unless your dataset is already square.",
            )
        )

    order = {Severity.BLOCKER: 0, Severity.WARN: 1, Severity.INFO: 2}
    return sorted(out, key=lambda a: order[a.severity])


def estimate_vram_gb(recipe: Recipe) -> float:
    """Rough VRAM estimate for a run.

    Deliberately a heuristic rather than a simulation: the aim is to catch
    "this will obviously not fit" before a user waits ten minutes to find out,
    not to predict allocation to the megabyte. Treat it as accurate to roughly
    a gigabyte, and expect it to be wrong on unusual architectures.
    """
    t, d = recipe.training, recipe.data

    bytes_per = 4 if t.precision is Precision.FP32 else 2
    # Base weights stay frozen but must still be resident.
    base_gb = 3.5 * (bytes_per / 2)

    # Activation memory scales with pixel count and batch size. The constant is
    # fitted loosely against observed SDXL runs.
    pixel_factor = (d.resolution / 1024) ** 2
    activations = 2.2 * pixel_factor * t.batch_size * (bytes_per / 2)
    if t.gradient_checkpointing:
        activations *= 0.4

    # Adapter weights plus optimiser state. Small next to everything else,
    # which is precisely the point of training a LoRA.
    adapter = 0.02 * (recipe.lora.rank / 16)
    optimizer_state = adapter * (1.0 if t.optimizer.value.endswith("8bit") else 2.0)

    return round(base_gb + activations + adapter + optimizer_state + 0.8, 2)
