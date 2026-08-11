"""What the machine can actually do.

Probing hardware is kept in one place, and every probe degrades gracefully: the
app must start and be fully explorable on a laptop with no GPU and no PyTorch
installed. Only the training step requires real hardware.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class GpuInfo:
    name: str
    vram_gb: float
    compute_capability: tuple[int, int] | None = None

    @property
    def supports_bf16(self) -> bool:
        """bf16 requires Ampere (compute capability 8.0) or newer."""
        if self.compute_capability is None:
            return False
        return self.compute_capability >= (8, 0)


@dataclass
class HardwareReport:
    platform: str
    python: str
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    gpus: list[GpuInfo]
    disk_free_gb: float
    notes: list[str]

    @property
    def can_train(self) -> bool:
        return self.torch_available and bool(self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["can_train"] = self.can_train
        d["total_vram_gb"] = self.total_vram_gb
        return d


@lru_cache(maxsize=1)
def probe(workspace: str = ".") -> HardwareReport:
    """Inspect the host. Cached — hardware does not change mid-session."""
    notes: list[str] = []
    gpus: list[GpuInfo] = []
    torch_version: str | None = None
    torch_available = False
    cuda_available = False

    try:
        import torch  # noqa: PLC0415 — optional heavy dependency, imported on demand

        torch_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    GpuInfo(
                        name=props.name,
                        vram_gb=round(props.total_memory / 1024**3, 2),
                        compute_capability=(props.major, props.minor),
                    )
                )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            notes.append(
                "Apple Silicon (MPS) detected. Diffusion training on MPS is slow and some "
                "operations fall back to CPU — usable for small experiments, not full runs."
            )
        else:
            notes.append("PyTorch is installed but no GPU was found. Training will not run.")
    except ImportError:
        notes.append(
            "PyTorch is not installed, so training is unavailable. Everything else — recipes, "
            "dataset analysis, and the explainer modules — works without it. "
            "Install with: pip install 'turducken[diffusion]'"
        )

    if gpus and not any(g.supports_bf16 for g in gpus):
        notes.append("This GPU predates bf16 support. Use fp16 precision instead.")

    free_gb = round(shutil.disk_usage(Path(workspace).resolve()).free / 1024**3, 1)
    if free_gb < 20:
        notes.append(
            f"Only {free_gb} GB of disk free. Base model downloads alone are commonly 7-25 GB."
        )

    return HardwareReport(
        platform=f"{platform.system()} {platform.machine()}",
        python=platform.python_version(),
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpus=gpus,
        disk_free_gb=free_gb,
        notes=notes,
    )


def recommend(vram_gb: float, resolution: int = 1024) -> dict:
    """Suggest settings that should fit in a given amount of VRAM.

    Conservative on purpose. An out-of-memory crash forty minutes into a run is
    a far worse outcome than a run that could have been 15% faster.
    """
    pixel_factor = (resolution / 1024) ** 2

    if vram_gb >= 40:
        rec = {
            "batch_size": int(8 / pixel_factor) or 1,
            "gradient_checkpointing": False,
            "rank": 32,
        }
        note = "Plenty of headroom — you can turn off gradient checkpointing for extra speed."
    elif vram_gb >= 24:
        rec = {
            "batch_size": int(4 / pixel_factor) or 1,
            "gradient_checkpointing": False,
            "rank": 32,
        }
        note = "Comfortable for 1024px LoRA training."
    elif vram_gb >= 16:
        rec = {"batch_size": int(2 / pixel_factor) or 1, "gradient_checkpointing": True, "rank": 16}
        note = "Workable at 1024px with gradient checkpointing on."
    elif vram_gb >= 12:
        rec = {"batch_size": 1, "gradient_checkpointing": True, "rank": 16}
        note = "Tight at 1024px. Use gradient accumulation to compensate for the small batch."
    elif vram_gb >= 8:
        rec = {"batch_size": 1, "gradient_checkpointing": True, "rank": 8}
        note = "8 GB is the practical floor. Consider training at 512px, or use SD 1.5."
    else:
        rec = {"batch_size": 1, "gradient_checkpointing": True, "rank": 8}
        note = "Below 8 GB, local diffusion training is not realistic. Consider a rented GPU."

    rec["gradient_accumulation"] = max(1, 4 // max(1, rec["batch_size"]))
    rec["note"] = note
    rec["effective_batch_size"] = rec["batch_size"] * rec["gradient_accumulation"]
    return rec
