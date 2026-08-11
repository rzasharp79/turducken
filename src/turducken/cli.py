"""Command line entry point.

``turducken serve`` is the main path. The other subcommands exist so the same
capabilities are scriptable and CI-checkable without a browser — and so a user
on a headless GPU box is not forced through a UI to start a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turducken", description="User friendly LoRA training.")
    parser.add_argument("--version", action="version", version=f"turducken {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Start the local web app (the main way to use this).")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8420)
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")

    sub.add_parser("doctor", help="Report what this machine can and cannot do.")

    p_init = sub.add_parser("init", help="Write a starter recipe file.")
    p_init.add_argument("path", nargs="?", default="recipe.json", type=Path)

    p_check = sub.add_parser("check", help="Review a recipe file and its dataset.")
    p_check.add_argument("path", type=Path)

    p_train = sub.add_parser("train", help="Run training from a recipe file.")
    p_train.add_argument("path", type=Path)
    p_train.add_argument("--backend", default=None)

    args = parser.parse_args(argv)
    return {
        "serve": _serve,
        "doctor": _doctor,
        "init": _init,
        "check": _check,
        "train": _train,
    }[args.command](args)


def _serve(args) -> int:
    import uvicorn  # noqa: PLC0415

    from .server.app import create_app  # noqa: PLC0415

    print(f"\n  turducken {__version__}")
    print(f"  → http://{args.host}:{args.port}\n")
    uvicorn.run(
        "turducken.server.app:app" if args.reload else create_app(host=args.host),
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


def _doctor(_args) -> int:
    from .hardware import probe, recommend  # noqa: PLC0415
    from .training.registry import available_backends  # noqa: PLC0415

    report = probe()
    print(f"\nPlatform      {report.platform}")
    print(f"Python        {report.python}")
    print(f"PyTorch       {report.torch_version or 'not installed'}")
    print(f"CUDA          {'yes' if report.cuda_available else 'no'}")
    print(f"Disk free     {report.disk_free_gb} GB")

    if report.gpus:
        print("\nGPUs")
        for gpu in report.gpus:
            bf16 = "bf16 ok" if gpu.supports_bf16 else "fp16 only"
            print(f"  {gpu.name} — {gpu.vram_gb} GB ({bf16})")
        rec = recommend(max(g.vram_gb for g in report.gpus))
        print(
            f"\nSuggested     batch {rec['batch_size']} x accum "
            f"{rec['gradient_accumulation']}, rank {rec['rank']}"
        )
        print(f"              {rec['note']}")

    print("\nBackends")
    for backend in available_backends():
        mark = "✓" if backend["available"] else "✗"
        print(f"  {mark} {backend['name']:<12} {backend['reason']}")

    for note in report.notes:
        print(f"\n! {note}")
    print()
    return 0


def _init(args) -> int:
    from .schemas.recipe import Recipe  # noqa: PLC0415

    path = Path(args.path)
    if path.exists():
        print(f"Refusing to overwrite existing {path}.", file=sys.stderr)
        return 1
    path.write_text(Recipe().model_dump_json(indent=2))
    print(f"Wrote {path}. Edit it, then: turducken check {path}")
    return 0


def _load(path: Path):
    from pydantic import ValidationError  # noqa: PLC0415

    from .schemas.recipe import Recipe  # noqa: PLC0415

    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return None
    try:
        return Recipe.model_validate(json.loads(path.read_text()))
    except ValidationError as exc:
        print(f"Invalid recipe:\n{exc}", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"{path} is not valid JSON: {exc}", file=sys.stderr)
        return None


def _check(args) -> int:
    from .advisor import estimate_vram_gb, review  # noqa: PLC0415
    from .data.dataset import build_dataset  # noqa: PLC0415
    from .hardware import probe  # noqa: PLC0415

    recipe = _load(args.path)
    if recipe is None:
        return 1

    report = build_dataset(recipe)
    print(f"\nDataset       {report.count} images in {report.root}")
    if report.bucket_summary.get("buckets"):
        shapes = ", ".join(
            f"{b['size']}×{b['count']}" for b in report.bucket_summary["buckets"][:5]
        )
        print(f"Buckets       {shapes}")
    for warning in report.warnings:
        print(f"  ! {warning}")

    vram = max((g.vram_gb for g in probe().gpus), default=None)
    advisories = review(recipe, image_count=report.count or None, vram_gb=vram)

    print(f"\nEst. VRAM     {estimate_vram_gb(recipe)} GB" + (f" (have {vram} GB)" if vram else ""))
    print(f"Eff. batch    {recipe.effective_batch_size}")
    print(f"LoRA scale    {recipe.lora_scale:.3f}\n")

    if not advisories:
        print("No advisories — this recipe looks reasonable.\n")
    for advisory in advisories:
        print(f"[{advisory.severity.value.upper()}] {advisory.title}")
        print(f"  {advisory.detail}")
        if advisory.fix:
            print(f"  → {advisory.fix}")
        print()

    return 1 if any(a.severity.value == "blocker" for a in advisories) else 0


def _train(args) -> int:
    from .training.base import EventKind  # noqa: PLC0415
    from .training.registry import default_backend, get_backend  # noqa: PLC0415

    recipe = _load(args.path)
    if recipe is None:
        return 1

    name = args.backend or default_backend(recipe.modality.value)
    backend = get_backend(name)
    usable, reason = backend.availability()
    if not usable:
        print(f"Backend '{name}' unavailable: {reason}", file=sys.stderr)
        return 1

    if backend.is_simulation:
        print(f"\n! Using the '{name}' backend — this is a SIMULATION and writes no adapter.\n")

    failed = False
    on_progress_line = False
    interactive = sys.stdout.isatty()

    def clear_progress() -> None:
        """Wipe the in-place progress line before printing anything else.

        Without the padding, a shorter message leaves the tail of the previous
        step line visible after it. When output is redirected, carriage returns
        do nothing at all, so the progress line is skipped entirely rather than
        producing one enormous concatenated line in the log.
        """
        nonlocal on_progress_line
        if on_progress_line:
            print("\r" + " " * 78 + "\r", end="")
            on_progress_line = False

    for event in backend.train(recipe):
        if event.kind is EventKind.STEP:
            if not interactive:
                continue
            loss = f"{event.loss:.4f}" if event.loss is not None else "—"
            print(
                f"\rstep {event.step}/{event.total_steps}  loss {loss}  "
                f"lr {event.learning_rate:.2e}",
                end="",
                flush=True,
            )
            on_progress_line = True
        else:
            clear_progress()
            print(event.message)
            if event.kind is EventKind.ERROR:
                failed = True
                if hint := event.extra.get("hint"):
                    print(f"  → {hint}")

    clear_progress()
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
