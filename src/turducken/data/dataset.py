"""Reading, validating and reporting on a training dataset.

Dataset problems cause more failed LoRAs than any hyperparameter does, and they
are invisible at training time: a run with three duplicate images and a missing
trigger word looks exactly like a healthy one until you inspect the results
hours later. So this module front-loads the inspection and says plainly what it
found before a single step is taken.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .buckets import Assignment, assign_bucket, generate_buckets, summarize

if TYPE_CHECKING:  # pragma: no cover
    from ..schemas.recipe import DataConfig, Recipe

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}

# Below this, upscaling to training resolution invents detail that is not there.
MIN_SENSIBLE_EDGE = 384


@dataclass
class DatasetItem:
    """One image and its caption."""

    image_path: Path
    caption: str
    width: int
    height: int
    bucket_width: int = 0
    bucket_height: int = 0
    cropped_fraction: float = 0.0
    digest: str = ""
    issues: list[str] = field(default_factory=list)

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    def caption_for_training(self, cfg: DataConfig, *, rng: random.Random | None = None) -> str:
        """Apply the configured caption transforms.

        Order matters: dropout is decided first (a blanked caption skips
        everything else), then the trigger word is prepended, then tags are
        shuffled — with the trigger held in place at the front, since its whole
        purpose is to be the stable anchor.
        """
        rng = rng or random.Random()
        if cfg.caption_dropout and rng.random() < cfg.caption_dropout:
            return ""

        text = self.caption.strip()
        trigger = cfg.trigger_word.strip()

        if cfg.shuffle_captions and "," in text:
            tags = [t.strip() for t in text.split(",") if t.strip()]
            rng.shuffle(tags)
            text = ", ".join(tags)

        if trigger and trigger.lower() not in text.lower():
            text = f"{trigger}, {text}" if text else trigger
        return text

    def to_dict(self) -> dict:
        return {
            "image": self.image_path.name,
            "caption": self.caption,
            "width": self.width,
            "height": self.height,
            "bucket": f"{self.bucket_width}x{self.bucket_height}",
            "cropped_fraction": self.cropped_fraction,
            "issues": self.issues,
        }


@dataclass
class DatasetReport:
    """Everything worth knowing about a dataset before training on it."""

    root: Path
    items: list[DatasetItem] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    bucket_summary: dict = field(default_factory=dict)
    caption_stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "count": self.count,
            "items": [i.to_dict() for i in self.items],
            "rejected": self.rejected,
            "buckets": self.bucket_summary,
            "captions": self.caption_stats,
            "warnings": self.warnings,
        }


def scan_directory(root: Path) -> tuple[list[tuple[Path, Path | None]], list[dict]]:
    """Pair every image with its sidecar caption file.

    The convention is a ``.txt`` file of the same stem — ``cat.png`` pairs with
    ``cat.txt``. It is the format every major LoRA trainer uses, so datasets
    move between tools without conversion.
    """
    if not root.exists():
        return [], [{"path": str(root), "reason": "Dataset directory does not exist."}]
    if not root.is_dir():
        return [], [{"path": str(root), "reason": "Dataset path is a file, not a directory."}]

    pairs, rejected = [], []
    for image in sorted(root.iterdir()):
        if not image.is_file():
            continue
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            if image.suffix.lower() != ".txt":
                rejected.append(
                    {"path": image.name, "reason": f"Unsupported type '{image.suffix}'"}
                )
            continue
        caption = image.with_suffix(".txt")
        pairs.append((image, caption if caption.exists() else None))
    return pairs, rejected


def build_dataset(recipe: Recipe) -> DatasetReport:
    """Scan, validate, bucket and summarise a dataset."""
    cfg = recipe.data
    # Resolved to an absolute path deliberately. A relative path in the recipe
    # is interpreted against the server's working directory, which the user
    # cannot see — so every message about this dataset names the exact place
    # that was searched rather than echoing back "./dataset".
    root = Path(cfg.dataset_path).expanduser().resolve()
    report = DatasetReport(root=root)

    pairs, report.rejected = scan_directory(root)
    if not pairs:
        report.warnings.append(f"No supported images found in {root}.")
        # Keep the empty case the same shape as every other case, so callers
        # never need to special-case "no images" when reading the report.
        report.bucket_summary = summarize([])
        report.caption_stats = analyze_captions([])
        return report

    buckets = generate_buckets(cfg.resolution) if cfg.bucketing else [
        _square(cfg.resolution)
    ]
    assignments: list[Assignment] = []
    seen: dict[str, str] = {}

    for image_path, caption_path in pairs:
        try:
            size = _read_size(image_path)
        except Exception as exc:  # noqa: BLE001 — one bad file must not sink the scan
            report.rejected.append({"path": image_path.name, "reason": f"Unreadable: {exc}"})
            continue

        caption = (
            caption_path.read_text(encoding="utf-8", errors="replace").strip()
            if caption_path
            else ""
        )
        item = DatasetItem(image_path=image_path, caption=caption, width=size[0], height=size[1])

        if caption_path is None:
            item.issues.append("No caption file — this image trains against an empty prompt.")
        elif not caption:
            item.issues.append("Caption file is empty.")

        if min(size) < MIN_SENSIBLE_EDGE:
            item.issues.append(
                f"Small source ({size[0]}x{size[1]}). Upscaling to {cfg.resolution}px will "
                "invent detail that was never captured."
            )

        item.digest = _digest(image_path)
        if item.digest in seen:
            item.issues.append(f"Byte-identical duplicate of {seen[item.digest]}.")
        else:
            seen[item.digest] = image_path.name

        assignment = assign_bucket(size, buckets)
        item.bucket_width = assignment.bucket.width
        item.bucket_height = assignment.bucket.height
        item.cropped_fraction = assignment.cropped_fraction
        if assignment.cropped_fraction > 0.25:
            item.issues.append(
                f"{assignment.cropped_fraction:.0%} of this image is cropped away to fit "
                f"{assignment.bucket}. Crop it yourself to keep the part you care about."
            )

        assignments.append(assignment)
        report.items.append(item)

    report.bucket_summary = summarize(assignments)
    report.caption_stats = analyze_captions(report.items, cfg.trigger_word)
    report.warnings.extend(_dataset_warnings(report, cfg))
    return report


def analyze_captions(items: list[DatasetItem], trigger_word: str = "") -> dict:
    """Summarise caption content.

    The most valuable output here is the frequent-tag list. A tag present in
    nearly every caption is not distinguishing anything — it is being welded to
    your subject, and will then show up in every generation whether you asked
    for it or not.
    """
    if not items:
        return {"count": 0}

    lengths, tags = [], Counter()
    trigger = trigger_word.strip().lower()
    with_trigger = 0

    for item in items:
        text = item.caption.strip()
        lengths.append(len(re.findall(r"\w+", text)))
        for tag in (t.strip().lower() for t in text.split(",")):
            if tag:
                tags[tag] += 1
        if trigger and trigger in text.lower():
            with_trigger += 1

    n = len(items)
    return {
        "count": n,
        "mean_words": round(sum(lengths) / n, 1),
        "empty": sum(1 for i in items if not i.caption.strip()),
        "trigger_word": trigger_word,
        "trigger_coverage": round(with_trigger / n, 3) if trigger else None,
        "frequent_tags": [
            {"tag": t, "count": c, "share": round(c / n, 3)}
            for t, c in tags.most_common(15)
        ],
        "ubiquitous_tags": [t for t, c in tags.items() if c >= n * 0.9 and t != trigger],
    }


def _dataset_warnings(report: DatasetReport, cfg: DataConfig) -> list[str]:
    """Dataset-level observations, as opposed to per-image issues."""
    out: list[str] = []
    n = report.count
    stats = report.caption_stats

    if n < 10:
        out.append(
            f"Only {n} images. Below about 10 it is very hard to avoid overfitting — the "
            "adapter memorises rather than generalises."
        )
    elif n > 300:
        out.append(
            f"{n} images is a large set for a LoRA. Quality and variety matter far more than "
            "count; consider curating down to your best 50-100."
        )

    if stats.get("empty"):
        out.append(
            f"{stats['empty']} image(s) have no caption. They will train against an empty "
            "prompt, which weakens the link between your trigger word and the subject."
        )

    if cfg.trigger_word.strip():
        coverage = stats.get("trigger_coverage") or 0
        if coverage < 0.9:
            out.append(
                f"Only {coverage:.0%} of captions contain the trigger word "
                f"'{cfg.trigger_word}'. It is added automatically at training time, but "
                "putting it in your caption files makes the dataset self-describing."
            )

    for tag in stats.get("ubiquitous_tags", [])[:5]:
        out.append(
            f"The tag '{tag}' appears in nearly every caption. Anything constant across the "
            "dataset gets absorbed into the concept itself rather than staying separately "
            "controllable — expect it in every generation."
        )

    dupes = sum(1 for i in report.items if any("duplicate" in x.lower() for x in i.issues))
    if dupes:
        out.append(
            f"{dupes} duplicate image(s). Duplicates silently weight those examples more "
            "heavily, which is rarely what you intended."
        )

    underfilled = report.bucket_summary.get("underfilled") or []
    if underfilled:
        out.append(
            f"{len(underfilled)} aspect bucket(s) hold a single image "
            f"({', '.join(underfilled[:3])}). "
            "Those images can only ever train in a batch of one, making their gradients noisier "
            "than the rest of the dataset's."
        )
    return out


# ----------------------------------------------------------------------


def _square(resolution: int):
    from .buckets import Bucket  # noqa: PLC0415

    return Bucket(resolution, resolution)


def _read_size(path: Path) -> tuple[int, int]:
    """Read image dimensions without decoding pixel data."""
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as img:
        return img.size


def _digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def load_pixels(item: DatasetItem, resolution: int) -> Any:  # pragma: no cover - needs torch
    """Load an image as a normalised tensor, resized and cropped to its bucket.

    Resize-to-cover then centre-crop, matching what :mod:`.buckets` predicts, so
    the crop percentages shown in the report are the crops actually performed.
    Output is normalised to [-1, 1], which is the range diffusion VAEs expect.
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    target_w = item.bucket_width or resolution
    target_h = item.bucket_height or resolution

    with Image.open(item.image_path) as img:
        img = img.convert("RGB")
        scale = max(target_w / img.width, target_h / img.height)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
        left = (img.width - target_w) // 2
        top = (img.height - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))

        import numpy as np  # noqa: PLC0415

        array = np.asarray(img, dtype="float32") / 127.5 - 1.0

    return torch.from_numpy(array).permute(2, 0, 1)
