"""Dataset scanning, validation and caption analysis."""

from __future__ import annotations

import random

import pytest

from turducken.data.dataset import analyze_captions, build_dataset, scan_directory
from turducken.schemas.recipe import DataConfig, Recipe

PIL = pytest.importorskip("PIL.Image")


def make_image(path, size=(1024, 1024), color=(120, 90, 60)):
    PIL.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def dataset(tmp_path):
    """A small, deliberately imperfect dataset."""
    root = tmp_path / "images"
    root.mkdir()

    make_image(root / "a.png", (1024, 1024))
    (root / "a.txt").write_text("sks_thing, red jacket, kitchen")

    make_image(root / "b.png", (1920, 1080))
    (root / "b.txt").write_text("sks_thing, outdoors, daylight")

    make_image(root / "c.png", (800, 1200))
    # c.txt deliberately missing.

    return root


def recipe_for(root, **data):
    return Recipe.model_validate({"data": {"dataset_path": str(root), **data}})


def test_scan_pairs_images_with_captions(dataset):
    pairs, rejected = scan_directory(dataset)
    assert len(pairs) == 3
    assert not rejected
    assert sum(1 for _, caption in pairs if caption is None) == 1


def test_missing_directory_is_reported_not_raised(tmp_path):
    pairs, rejected = scan_directory(tmp_path / "nope")
    assert not pairs
    assert "does not exist" in rejected[0]["reason"]


def test_unsupported_files_rejected_with_a_reason(dataset):
    (dataset / "notes.pdf").write_bytes(b"%PDF-1.4")
    _, rejected = scan_directory(dataset)
    assert any("Unsupported" in r["reason"] for r in rejected)


def test_build_dataset_assigns_buckets_and_finds_issues(dataset):
    report = build_dataset(recipe_for(dataset, trigger_word="sks_thing"))

    assert report.count == 3
    assert all(item.bucket_width % 64 == 0 for item in report.items)

    missing = next(i for i in report.items if i.image_path.name == "c.png")
    assert any("No caption" in issue for issue in missing.issues)


def test_duplicates_are_detected(dataset):
    make_image(dataset / "a_copy.png", (1024, 1024))
    (dataset / "a_copy.png").write_bytes((dataset / "a.png").read_bytes())
    (dataset / "a_copy.txt").write_text("sks_thing, red jacket, kitchen")

    report = build_dataset(recipe_for(dataset, trigger_word="sks_thing"))
    assert any("duplicate" in i.lower() for item in report.items for i in item.issues)
    assert any("duplicate" in w.lower() for w in report.warnings)


def test_small_images_are_flagged(tmp_path):
    root = tmp_path / "small"
    root.mkdir()
    make_image(root / "tiny.png", (200, 200))
    (root / "tiny.txt").write_text("sks_thing")

    report = build_dataset(recipe_for(root, trigger_word="sks_thing"))
    assert any("Small source" in issue for issue in report.items[0].issues)


def test_small_dataset_warning(dataset):
    report = build_dataset(recipe_for(dataset, trigger_word="sks_thing"))
    assert any("Only 3 images" in w for w in report.warnings)


def test_ubiquitous_tag_warning(tmp_path):
    """A tag in every caption is being welded to the concept."""
    root = tmp_path / "uniform"
    root.mkdir()
    for i in range(12):
        make_image(root / f"{i}.png", (1024, 1024), color=(i * 15, 60, 60))
        (root / f"{i}.txt").write_text("sks_thing, studio lighting, white background")

    report = build_dataset(recipe_for(root, trigger_word="sks_thing"))
    assert "studio lighting" in report.caption_stats["ubiquitous_tags"]
    assert any("studio lighting" in w for w in report.warnings)


def test_bucketing_off_uses_only_squares(dataset):
    report = build_dataset(recipe_for(dataset, bucketing=False, trigger_word="sks_thing"))
    assert all(i.bucket_width == i.bucket_height for i in report.items)
    # The 1920x1080 image should now be losing a lot.
    wide = next(i for i in report.items if i.image_path.name == "b.png")
    assert wide.cropped_fraction > 0.4


def test_empty_directory_reports_cleanly(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    report = build_dataset(recipe_for(root))
    assert report.count == 0
    assert report.warnings
    assert report.to_dict()["count"] == 0


# ── Caption transforms ────────────────────────────────────────────────


def test_trigger_word_is_prepended_when_absent():
    from turducken.data.dataset import DatasetItem

    item = DatasetItem(
        image_path=__import__("pathlib").Path("x.png"),
        caption="red jacket",
        width=1,
        height=1,
    )
    text = item.caption_for_training(DataConfig(trigger_word="sks_x", caption_dropout=0.0))
    assert text.startswith("sks_x")


def test_trigger_word_not_duplicated_when_present():
    from pathlib import Path

    from turducken.data.dataset import DatasetItem

    item = DatasetItem(image_path=Path("x.png"), caption="sks_x, red jacket", width=1, height=1)
    text = item.caption_for_training(DataConfig(trigger_word="sks_x", caption_dropout=0.0))
    assert text.count("sks_x") == 1


def test_caption_dropout_blanks_everything():
    from pathlib import Path

    from turducken.data.dataset import DatasetItem

    item = DatasetItem(image_path=Path("x.png"), caption="sks_x, red jacket", width=1, height=1)
    cfg = DataConfig(trigger_word="sks_x", caption_dropout=1.0)
    assert item.caption_for_training(cfg, rng=random.Random(0)) == ""


def test_shuffle_preserves_the_tag_set():
    from pathlib import Path

    from turducken.data.dataset import DatasetItem

    item = DatasetItem(image_path=Path("x.png"), caption="a, b, c, d", width=1, height=1)
    cfg = DataConfig(trigger_word="", caption_dropout=0.0, shuffle_captions=True)
    result = item.caption_for_training(cfg, rng=random.Random(3))
    assert sorted(t.strip() for t in result.split(",")) == ["a", "b", "c", "d"]


def test_caption_analysis_reports_trigger_coverage():
    from pathlib import Path

    from turducken.data.dataset import DatasetItem

    items = [
        DatasetItem(image_path=Path("1.png"), caption="sks_x, a", width=1, height=1),
        DatasetItem(image_path=Path("2.png"), caption="b", width=1, height=1),
    ]
    stats = analyze_captions(items, "sks_x")
    assert stats["trigger_coverage"] == pytest.approx(0.5)
    assert stats["count"] == 2
