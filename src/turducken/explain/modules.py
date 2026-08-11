"""The explainer module registry.

Lessons are Markdown files in ``content/`` with a small metadata header defined
here rather than in front matter, so a lesson is just prose — editable by anyone
who understands the subject, without touching Python.

Lessons are optional by design. A user who wants to train a LoRA and get on
with their day never has to open one; a user who wants to know why any of it
works can follow the chain from the exact knob they are looking at into the
lesson that explains it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

CONTENT_DIR = Path(__file__).parent / "content"


@dataclass(frozen=True)
class Lesson:
    slug: str
    title: str
    summary: str
    minutes: int
    """Honest reading time. People decide whether to click based on this."""

    demos: list[str] = field(default_factory=list)
    """Demo names from :mod:`.demos` that this lesson embeds."""

    knobs: list[str] = field(default_factory=list)
    """Recipe fields this lesson explains, for cross-linking from the form."""

    prerequisites: list[str] = field(default_factory=list)

    def body(self) -> str:
        path = CONTENT_DIR / f"{self.slug}.md"
        if not path.exists():
            return f"_Lesson content for '{self.slug}' has not been written yet._"
        return path.read_text(encoding="utf-8")

    def to_dict(self, *, include_body: bool = False) -> dict:
        d = asdict(self)
        if include_body:
            d["body"] = self.body()
        return d


LESSONS: list[Lesson] = [
    Lesson(
        slug="what-is-a-lora",
        title="What a LoRA actually is",
        summary=(
            "The whole idea in one sitting: why you can teach a billion-parameter model a new "
            "face by training a few million numbers, and why that is not a compromise."
        ),
        minutes=8,
        demos=["low-rank", "parameter-budget", "zero-init"],
        knobs=["lora.rank", "lora.alpha"],
    ),
    Lesson(
        slug="the-training-loop",
        title="What happens in one training step",
        summary=(
            "Add noise, predict the noise, measure the error, nudge the adapter. Every "
            "hyperparameter you will ever tune is a dial on some part of this loop."
        ),
        minutes=10,
        demos=["noise-schedule"],
        knobs=["training.steps", "training.learning_rate", "training.batch_size"],
        prerequisites=["what-is-a-lora"],
    ),
    Lesson(
        slug="datasets-and-captions",
        title="Datasets and captions",
        summary=(
            "The part that decides whether your LoRA works. What to shoot, what to write, and "
            "why a trigger word matters more than any hyperparameter."
        ),
        minutes=9,
        knobs=["data.dataset_path", "data.trigger_word", "data.caption_dropout"],
    ),
    Lesson(
        slug="buckets-and-resolution",
        title="Resolution and aspect buckets",
        summary=(
            "Why training at the wrong resolution wastes a run, and why square-cropping your "
            "dataset quietly throws away the part you cared about."
        ),
        minutes=6,
        demos=["bucketing"],
        knobs=["data.resolution", "data.bucketing"],
    ),
    Lesson(
        slug="where-loras-attach",
        title="Where the adapters go",
        summary=(
            "Inside a U-Net: which layers get adapted, why attention is the high-leverage "
            "place to intervene, and what changes when you adapt more."
        ),
        minutes=7,
        knobs=["lora.target_modules"],
        prerequisites=["what-is-a-lora"],
    ),
    Lesson(
        slug="reading-your-results",
        title="Reading your results",
        summary=(
            "How to tell overfitting from undertraining by looking at samples, why the loss "
            "curve will not tell you, and how to pick the right checkpoint."
        ),
        minutes=8,
        knobs=["training.save_every", "training.steps"],
        prerequisites=["the-training-loop"],
    ),
]

_BY_SLUG = {lesson.slug: lesson for lesson in LESSONS}


def list_lessons() -> list[dict]:
    return [lesson.to_dict() for lesson in LESSONS]


def get_lesson(slug: str) -> Lesson:
    if slug not in _BY_SLUG:
        raise KeyError(f"Unknown lesson '{slug}'. Available: {', '.join(_BY_SLUG)}")
    return _BY_SLUG[slug]


@lru_cache(maxsize=1)
def lessons_for_knobs() -> dict[str, str]:
    """Map ``recipe.field.path -> lesson slug`` for in-form deep links.

    Built from the lessons' own ``knobs`` lists, so a lesson claiming a field is
    the single source of that link and the two cannot disagree.
    """
    return {knob: lesson.slug for lesson in LESSONS for knob in lesson.knobs}
