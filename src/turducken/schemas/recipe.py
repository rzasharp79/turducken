"""The training recipe: one object that fully describes a LoRA training run.

A recipe is the app's central artifact. It is saveable, shareable, diffable and
reproducible — and because every field is declared with :func:`knob`, it can
render its own explanation in the UI.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .knobs import Tier, knob


class Modality(str, Enum):
    """Which kind of model we are adapting.

    Image is implemented first; the recipe and trainer registry are structured
    so the text path slots in without reshaping anything around it.
    """

    IMAGE = "image"
    TEXT = "text"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class Optimizer(str, Enum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    LION = "lion"
    PRODIGY = "prodigy"


class Scheduler(str, Enum):
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    COSINE = "cosine"
    LINEAR = "linear"


class LoraConfig(BaseModel):
    """Shape of the adapter itself — how much capacity you are adding."""

    rank: int = knob(
        16,
        ge=1,
        le=320,
        tier=Tier.ESSENTIAL,
        lesson="what-is-a-lora",
        unit="dimensions",
        plain="How much new capacity the LoRA has to learn your subject.",
        why=(
            "A LoRA never edits the base model's weights. It learns a small correction "
            "and adds it on top. That correction is stored as two thin matrices, A and B, "
            "whose product has the same shape as the weight it corrects. Rank is the shared "
            "inner dimension of A and B — the width of the bottleneck the correction has to "
            "squeeze through. Rank 16 against a 1024x1024 weight means learning 32,768 "
            "numbers instead of 1,048,576: about 3% as many."
        ),
        tradeoff=(
            "Higher rank can express more detail and more distinct concepts, but it has "
            "enough freedom to memorise your training images instead of generalising from "
            "them, and the output file grows roughly linearly."
        ),
        rule_of_thumb=(
            "A single face or character: 8-16. An art style: 16-32. Multiple concepts in "
            "one adapter, or a complex scene grammar: 32-64. Above 64 you are usually "
            "compensating for a dataset problem rather than a capacity problem."
        ),
        symptoms={
            "too high": (
                "Outputs reproduce training images almost verbatim, including backgrounds "
                "and watermarks. The LoRA stops responding to your prompt."
            ),
            "too low": (
                "The subject is vaguely suggested but never locks in. Details wobble between "
                "generations no matter how long you train."
            ),
        },
    )

    alpha: float = knob(
        16.0,
        gt=0,
        tier=Tier.STANDARD,
        lesson="what-is-a-lora",
        plain="A volume dial on the correction the LoRA applies.",
        why=(
            "The adapter's output is multiplied by alpha/rank before being added to the base "
            "model. With alpha equal to rank the scale is exactly 1.0. This exists so that "
            "changing rank does not silently change how hard the adapter pushes — you can "
            "re-tune capacity without re-tuning strength."
        ),
        tradeoff=(
            "Raising alpha relative to rank amplifies every update, which behaves much like "
            "raising the learning rate: faster convergence, easier to overshoot."
        ),
        rule_of_thumb=(
            "Set alpha equal to rank (scale 1.0) and leave it there. If you must adjust "
            "strength, change the learning rate instead — its effect is easier to reason "
            "about because it does not also interact with rank."
        ),
        symptoms={
            "too high": "Training destabilises early; outputs turn saturated or fried.",
            "too low": "Training appears to do nothing even at a healthy learning rate.",
        },
    )

    dropout: float = knob(
        0.0,
        ge=0.0,
        lt=1.0,
        tier=Tier.ADVANCED,
        plain="Randomly ignores part of the adapter each step, as a regulariser.",
        why=(
            "Zeroing a fraction of the adapter's activations each step stops it leaning on "
            "any single pathway, which pushes it toward the features shared across your "
            "dataset rather than the quirks of individual images."
        ),
        tradeoff="Reduces overfitting, but slows learning and can blur fine detail.",
        rule_of_thumb=(
            "Leave at 0.0. Reach for 0.05-0.1 only when you have a small dataset (under ~20 "
            "images) that is overfitting before it converges."
        ),
    )

    target_modules: list[str] = knob(
        default=["to_q", "to_k", "to_v", "to_out.0"],
        tier=Tier.ADVANCED,
        lesson="where-loras-attach",
        plain="Which layers inside the base model get an adapter attached.",
        why=(
            "You do not have to adapt the whole network. The attention projections "
            "(query/key/value/output) are where the model decides what to look at and how "
            "concepts relate, which is why adapting them alone captures subjects and styles "
            "so efficiently. Adding the feed-forward layers gives more raw capacity."
        ),
        tradeoff=(
            "More target modules means more expressive power, more VRAM, and a larger file — "
            "with sharply diminishing returns for single-subject training."
        ),
        rule_of_thumb=(
            "Attention-only (the default) is right for the overwhelming majority of runs."
        ),
    )


class DataConfig(BaseModel):
    """Where the training images come from and how they are prepared."""

    dataset_path: Path = knob(
        Path("./dataset"),
        tier=Tier.ESSENTIAL,
        lesson="datasets-and-captions",
        plain="Folder holding your training images and their caption files.",
        why=(
            "Each image is paired with a .txt file of the same name containing its caption. "
            "The model learns the mapping from those words to what is in the picture, so the "
            "captions are not paperwork — they are the labels on your training signal."
        ),
        rule_of_thumb="20-50 varied, high-quality images beats 500 near-duplicates every time.",
    )

    trigger_word: str = knob(
        "",
        tier=Tier.ESSENTIAL,
        lesson="datasets-and-captions",
        plain="The rare word you will type later to summon what you trained.",
        why=(
            "Training attaches your subject to whatever words describe it. If you caption "
            "with common words, you overwrite the model's existing understanding of those "
            "words. A rare token gives the concept somewhere new to live, leaving 'woman' or "
            "'castle' intact for everything else."
        ),
        tradeoff=(
            "Leave it empty to deliberately overwrite an existing concept — occasionally what "
            "you want for a style, almost never what you want for a subject."
        ),
        rule_of_thumb=(
            "Use something the tokenizer will not recognise as ordinary English: 'sks_marla', "
            "'ohwx_castle'. Avoid real names, which already carry strong associations."
        ),
        symptoms={
            "too common": (
                "The LoRA bleeds into unrelated prompts — every person you generate starts "
                "drifting toward your subject."
            )
        },
    )

    resolution: int = knob(
        1024,
        ge=256,
        le=2048,
        multiple_of=64,
        tier=Tier.ESSENTIAL,
        unit="px",
        lesson="buckets-and-resolution",
        plain="The edge length images are trained at.",
        why=(
            "Match the resolution the base model was trained at, or you spend the whole run "
            "teaching it a scale it does not natively work in. SDXL and Flux expect ~1024; "
            "Stable Diffusion 1.5 expects 512."
        ),
        tradeoff="VRAM and step time scale with the square of this number.",
        rule_of_thumb=(
            "1024 for SDXL/Flux, 512 for SD 1.5. Do not exceed your base model's native size."
        ),
    )

    bucketing: bool = knob(
        True,
        tier=Tier.STANDARD,
        lesson="buckets-and-resolution",
        plain="Train images at their own aspect ratio instead of cropping them square.",
        why=(
            "Without bucketing every image is centre-cropped to a square, which quietly "
            "amputates heads and edges of your subject. Bucketing groups images into a set of "
            "aspect ratios with roughly equal pixel counts, so a portrait trains as a portrait."
        ),
        tradeoff="Slightly more complex batching; essentially free otherwise.",
        rule_of_thumb="Leave on unless every image you own is already square.",
    )

    caption_dropout: float = knob(
        0.05,
        ge=0.0,
        le=1.0,
        tier=Tier.ADVANCED,
        plain="Chance of blanking a caption so the image trains with no text at all.",
        why=(
            "Occasionally showing an image with an empty caption teaches the model what your "
            "subject looks like unconditionally, which strengthens the trigger word and "
            "improves behaviour under classifier-free guidance."
        ),
        tradeoff="Too much and the concept detaches from your trigger word entirely.",
        rule_of_thumb="0.05 is a good default. Above 0.2 is rarely useful.",
    )

    shuffle_captions: bool = knob(
        True,
        tier=Tier.ADVANCED,
        plain="Shuffle comma-separated tags in each caption every time it is used.",
        why=(
            "Text encoders weight earlier tokens more heavily. Shuffling stops the model from "
            "learning your captions' arbitrary word order as if it were meaningful."
        ),
        tradeoff="Unhelpful when captions are natural sentences rather than tag lists.",
        rule_of_thumb="On for tag-style captions ('a, b, c'), off for prose captions.",
    )


class TrainingConfig(BaseModel):
    """The optimisation loop: how long, how fast, how much at once."""

    steps: int = knob(
        1500,
        ge=1,
        le=200_000,
        tier=Tier.ESSENTIAL,
        unit="steps",
        lesson="the-training-loop",
        plain="How many times the model looks at a batch and adjusts itself.",
        why=(
            "One step is one batch: predict the noise, measure the error, nudge the adapter. "
            "Total learning is roughly steps x learning rate, so these two knobs trade "
            "against each other and should not be tuned independently."
        ),
        tradeoff=(
            "Too few and the concept never lands; too many and the adapter starts memorising "
            "your exact images instead of the idea behind them."
        ),
        rule_of_thumb=(
            "Around 100 steps per image is a reasonable starting point for a character "
            "(30 images -> ~3000). Save checkpoints along the way and pick the best one — "
            "the right stopping point is judged by eye, not calculated."
        ),
        symptoms={
            "too high": "Backgrounds and poses from your dataset start appearing uninvited.",
            "too low": "Outputs only faintly resemble your subject.",
        },
    )

    learning_rate: float = knob(
        1e-4,
        gt=0,
        le=1.0,
        tier=Tier.ESSENTIAL,
        lesson="the-training-loop",
        plain="How big a correction to make after each batch.",
        why=(
            "The optimiser computes which direction reduces the error; the learning rate "
            "decides how far to move in that direction. LoRAs tolerate rates one to two "
            "orders of magnitude higher than full fine-tuning because the adapter starts at "
            "zero and only ever adds a correction — there is far less to destroy."
        ),
        tradeoff="Higher is faster but overshoots; lower is stabler but may never arrive.",
        rule_of_thumb="1e-4 for most LoRAs. Halve it if loss spikes; double it if nothing moves.",
        symptoms={
            "too high": "Loss jumps around or turns to NaN; images come out fried or noisy.",
            "too low": "Loss decreases smoothly but glacially; outputs barely change.",
        },
    )

    batch_size: int = knob(
        1,
        ge=1,
        le=64,
        tier=Tier.STANDARD,
        unit="images",
        plain="How many images are processed together in one step.",
        why=(
            "Averaging the error over several images gives a less noisy estimate of the right "
            "direction, so updates are steadier."
        ),
        tradeoff=(
            "VRAM scales almost linearly with this. It is usually the first thing to hit "
            "your limit."
        ),
        rule_of_thumb="Raise until you are near your VRAM ceiling, then use gradient accumulation.",
    )

    gradient_accumulation: int = knob(
        1,
        ge=1,
        le=64,
        tier=Tier.STANDARD,
        unit="batches",
        plain="Add up several batches before actually updating the model.",
        why=(
            "Accumulating gradients over 4 batches of 1 gives you the update quality of a "
            "batch of 4 while only ever holding one image's activations in memory. This is how "
            "you get large-batch behaviour on a small GPU."
        ),
        tradeoff="Effective batch size goes up, but so does wall-clock time per update.",
        rule_of_thumb=(
            "batch_size x gradient_accumulation is your effective batch size. Aim for 4-8."
        ),
    )

    optimizer: Optimizer = knob(
        Optimizer.ADAMW_8BIT,
        tier=Tier.ADVANCED,
        plain="The algorithm that decides how to apply each correction.",
        why=(
            "AdamW keeps a running estimate of each parameter's gradient history so it can "
            "scale updates per-parameter. The 8-bit variant stores that history in 8 bits "
            "instead of 32, saving meaningful VRAM at negligible quality cost."
        ),
        tradeoff=(
            "Prodigy tunes its own learning rate but is less predictable; Lion uses less "
            "memory but needs a lower rate."
        ),
        rule_of_thumb="adamw_8bit unless you have a specific reason otherwise.",
    )

    scheduler: Scheduler = knob(
        Scheduler.COSINE,
        tier=Tier.ADVANCED,
        plain="How the learning rate changes over the course of the run.",
        why=(
            "Decaying the rate lets training take big confident strides early and fine, "
            "careful ones near the end, which is when it is settling details rather than "
            "finding the general shape."
        ),
        tradeoff=(
            "Constant is easier to reason about when comparing runs; cosine usually lands better."
        ),
        rule_of_thumb="cosine for a fixed-length run, constant when you plan to stop by eye.",
    )

    warmup_steps: int = knob(
        0,
        ge=0,
        tier=Tier.ADVANCED,
        unit="steps",
        plain="Ramp the learning rate up from zero over the first N steps.",
        why=(
            "The optimiser's per-parameter statistics are meaningless until it has seen some "
            "data. Warmup avoids taking confident steps based on that early noise."
        ),
        rule_of_thumb="0 is fine for LoRA. Use ~5% of total steps if you see early instability.",
    )

    precision: Precision = knob(
        Precision.BF16,
        tier=Tier.STANDARD,
        plain="Numeric format used for the model's weights and math during training.",
        why=(
            "Half precision halves memory and roughly doubles throughput on modern GPUs. "
            "bf16 has the same exponent range as fp32, so it does not need loss scaling to "
            "avoid underflow — it is the safer half-precision choice where supported."
        ),
        tradeoff=(
            "bf16 needs Ampere (RTX 30xx) or newer. fp16 works on older cards but is more fragile."
        ),
        rule_of_thumb="bf16 on RTX 30xx/40xx/50xx and newer, fp16 on RTX 20xx and older.",
    )

    gradient_checkpointing: bool = knob(
        True,
        tier=Tier.STANDARD,
        plain="Recompute intermediate values during the backward pass instead of storing them.",
        why=(
            "The backward pass needs the activations from the forward pass. You can either "
            "keep them in VRAM or throw them away and recompute them on demand. This trades "
            "roughly 30% more time for a large memory saving."
        ),
        tradeoff=(
            "Slower per step, but often the difference between training and running out of memory."
        ),
        rule_of_thumb="On, unless you have VRAM to spare and want maximum speed.",
    )

    seed: int = knob(
        42,
        ge=0,
        tier=Tier.STANDARD,
        plain="Fixes the random choices so a run can be reproduced exactly.",
        why=(
            "Shuffling, noise sampling and dropout all draw from a random number generator. "
            "Pinning its starting point means the same recipe plus the same data gives the "
            "same LoRA — which is what makes it possible to change one knob and attribute the "
            "difference to that knob."
        ),
        rule_of_thumb="Keep it fixed while comparing settings. Vary it only to check robustness.",
    )

    save_every: int = knob(
        250,
        ge=0,
        tier=Tier.STANDARD,
        unit="steps",
        plain="Write an intermediate copy of the LoRA this often.",
        why=(
            "The best checkpoint is rarely the last one. Saving along the way turns 'my run "
            "overfit' from a wasted afternoon into picking an earlier file."
        ),
        tradeoff="Each checkpoint costs disk space. 0 disables intermediate saves.",
        rule_of_thumb="Aim for 5-10 checkpoints across the run.",
    )


class Recipe(BaseModel):
    """A complete, reproducible description of one training run."""

    name: str = knob(
        "my-first-lora",
        min_length=1,
        max_length=100,
        tier=Tier.ESSENTIAL,
        plain="What to call this LoRA and its output files.",
        why="Names the output directory and the .safetensors file you will load later.",
    )

    modality: Modality = knob(
        Modality.IMAGE,
        # Not ESSENTIAL: only the image path is implemented, so presenting this
        # as a first-run decision would imply a choice that does not yet exist.
        tier=Tier.STANDARD,
        plain="Whether you are adapting an image model or a language model.",
        why=(
            "The two share LoRA's mathematics exactly but differ in everything around it: "
            "data format, loss, and what a good result looks like."
        ),
        rule_of_thumb="Image is the implemented path today.",
    )

    base_model: str = knob(
        "stabilityai/stable-diffusion-xl-base-1.0",
        min_length=1,
        tier=Tier.ESSENTIAL,
        plain="The model you are teaching — a Hugging Face repo id or a local path.",
        why=(
            "A LoRA is a correction computed against one specific set of weights. It is only "
            "loosely portable to other models, and not at all across architectures: an SDXL "
            "LoRA cannot be loaded into SD 1.5."
        ),
        rule_of_thumb="Train against the exact checkpoint you intend to generate with.",
    )

    output_dir: Path = knob(
        Path("./workspace/output"),
        tier=Tier.STANDARD,
        plain="Where checkpoints and the finished adapter are written.",
        why="Each run gets its own subdirectory named after the recipe.",
    )

    lora: LoraConfig = Field(default_factory=LoraConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @property
    def effective_batch_size(self) -> int:
        """Images contributing to each actual weight update."""
        return self.training.batch_size * self.training.gradient_accumulation

    @property
    def lora_scale(self) -> float:
        """The multiplier actually applied to the adapter's output."""
        return self.lora.alpha / self.lora.rank

    @model_validator(mode="after")
    def _check_coherence(self) -> Recipe:
        """Reject combinations that are internally contradictory.

        Note the deliberate restraint here: this only blocks settings that
        cannot work. Settings that are merely unwise are surfaced as advisories
        by :mod:`turducken.advisor`, which explains the concern and lets the
        user proceed anyway. Refusing to run a legal-but-questionable recipe
        teaches nothing.
        """
        if self.training.warmup_steps >= self.training.steps:
            raise ValueError(
                f"warmup_steps ({self.training.warmup_steps}) must be less than total steps "
                f"({self.training.steps}) — otherwise the learning rate never finishes ramping up."
            )
        if self.training.save_every and self.training.save_every > self.training.steps:
            raise ValueError(
                f"save_every ({self.training.save_every}) exceeds total steps "
                f"({self.training.steps}), so no intermediate checkpoint would ever be written."
            )
        return self
