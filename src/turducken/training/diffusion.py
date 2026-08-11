"""Real LoRA training for Stable Diffusion / SDXL, via diffusers + peft.

Every heavyweight import is deferred into the methods that need it, so this
module can be imported — and the app can start, and the whole explainer layer
can run — on a machine with no PyTorch installed.

Scope of this first pass: text-to-image LoRA on the U-Net's attention
projections, for SD 1.5-family and SDXL-family checkpoints. Text-encoder
training, Flux/SD3 (which use a DiT rather than a U-Net), and the text-LLM
modality are deliberately not here yet; the registry is structured so each
arrives as a sibling backend rather than as branching inside this one.

STATUS: this path requires a CUDA GPU and has not been executed in CI, which
cannot provide one. Treat it as unverified until it has been run end-to-end on
real hardware.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import EventKind, TrainerBackend, TrainerEvent

if TYPE_CHECKING:  # pragma: no cover
    from ..schemas.recipe import Recipe

REQUIRED = ("torch", "diffusers", "transformers", "peft", "safetensors")


class DiffusionTrainer(TrainerBackend):
    """Trains a real adapter against a real diffusion model."""

    name = "diffusion"
    modality = "image"
    is_simulation = False

    def availability(self) -> tuple[bool, str]:
        import importlib.util  # noqa: PLC0415

        missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
        if missing:
            return False, (
                f"Missing packages: {', '.join(missing)}. "
                "Install with: pip install 'turducken[diffusion]'"
            )

        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False, (
                "No CUDA GPU detected. Diffusion LoRA training needs roughly 8 GB of VRAM at "
                "512px, or 12 GB+ at 1024px."
            )
        return True, f"Ready — {torch.cuda.get_device_name(0)}"

    # ------------------------------------------------------------------
    def train(self, recipe: Recipe, *, cancel: Any = None) -> Iterator[TrainerEvent]:
        ok, reason = self.availability()
        if not ok:
            yield TrainerEvent(kind=EventKind.ERROR, message=reason)
            return

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415, N812

        start = self._clock()

        def ev(kind: EventKind, **kw) -> TrainerEvent:
            return TrainerEvent(kind=kind, seconds_elapsed=round(self._clock() - start, 3), **kw)

        try:
            t = recipe.training
            device = torch.device("cuda")
            dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
                t.precision.value
            ]

            yield ev(EventKind.STATUS, message=f"Loading {recipe.base_model}")
            pipe, is_sdxl = _load_pipeline(recipe.base_model, dtype)
            pipe.to(device)

            # Everything in the base model is frozen. Only the adapter trains —
            # this is what makes the memory footprint tractable.
            pipe.vae.requires_grad_(False)
            pipe.unet.requires_grad_(False)
            pipe.text_encoder.requires_grad_(False)
            if is_sdxl:
                pipe.text_encoder_2.requires_grad_(False)

            yield ev(
                EventKind.LOG,
                message=f"Base model loaded and frozen ({'SDXL' if is_sdxl else 'SD1.x'} "
                f"architecture, {t.precision.value}).",
            )

            # -- attach the adapter ------------------------------------
            from peft import LoraConfig as PeftLoraConfig  # noqa: PLC0415

            pipe.unet.add_adapter(
                PeftLoraConfig(
                    r=recipe.lora.rank,
                    lora_alpha=recipe.lora.alpha,
                    lora_dropout=recipe.lora.dropout,
                    target_modules=list(recipe.lora.target_modules),
                    init_lora_weights="gaussian",
                )
            )
            params = [p for p in pipe.unet.parameters() if p.requires_grad]
            trainable = sum(p.numel() for p in params)
            total = sum(p.numel() for p in pipe.unet.parameters())

            yield ev(
                EventKind.LOG,
                message=f"Adapter attached: {trainable:,} trainable parameters out of "
                f"{total:,} ({trainable / total:.3%}). The other "
                f"{100 * (1 - trainable / total):.2f}% "
                "is frozen and only participates in the forward pass.",
                extra={"trainable_params": trainable, "total_params": total},
            )

            if t.gradient_checkpointing:
                pipe.unet.enable_gradient_checkpointing()

            # -- data --------------------------------------------------
            yield ev(EventKind.STATUS, message="Preparing dataset")
            from ..data.dataset import build_dataset  # noqa: PLC0415

            report = build_dataset(recipe)
            if not report.items:
                yield ev(
                    EventKind.ERROR,
                    message=f"No usable image/caption pairs found in {recipe.data.dataset_path}.",
                )
                return
            yield ev(
                EventKind.LOG,
                message=f"{len(report.items)} image/caption pairs across "
                f"{len(report.bucket_summary.get('buckets', []))} aspect buckets.",
            )

            loader = _build_loader(pipe, recipe, report, device, dtype, is_sdxl)

            # -- optimiser ---------------------------------------------
            optimizer = _build_optimizer(t, params)
            scheduler = _build_scheduler(t, optimizer)
            scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)

            out_dir = Path(recipe.output_dir) / recipe.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "recipe.json").write_text(recipe.model_dump_json(indent=2))

            yield ev(EventKind.STATUS, message=f"Training {t.steps} steps")

            step = 0
            accumulated = 0
            optimizer.zero_grad(set_to_none=True)

            while step < t.steps:
                for batch in loader:
                    if step >= t.steps:
                        break
                    if self._should_stop(cancel):
                        yield ev(EventKind.STATUS, message=f"Cancelled at step {step}.", step=step)
                        return

                    latents = batch["latents"]
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        pipe.scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                    ).long()

                    # The training objective in one line: corrupt a latent by a
                    # random amount, then score the model on recovering the
                    # exact noise that was added.
                    noisy = pipe.scheduler.add_noise(latents, noise, timesteps)

                    pred = pipe.unet(
                        noisy,
                        timesteps,
                        encoder_hidden_states=batch["prompt_embeds"],
                        added_cond_kwargs=batch.get("added_cond_kwargs"),
                    ).sample

                    target = (
                        noise
                        if pipe.scheduler.config.prediction_type == "epsilon"
                        else pipe.scheduler.get_velocity(latents, noise, timesteps)
                    )
                    loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                    scaler.scale(loss / t.gradient_accumulation).backward()
                    accumulated += 1

                    if accumulated < t.gradient_accumulation:
                        continue

                    # A full effective batch has accumulated; now actually step.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    step += 1

                    if step % max(1, t.steps // 200) == 0 or step == t.steps:
                        yield ev(
                            EventKind.STEP,
                            step=step,
                            total_steps=t.steps,
                            loss=round(loss.item(), 5),
                            learning_rate=scheduler.get_last_lr()[0],
                            extra={
                                "vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                            },
                        )

                    if t.save_every and step % t.save_every == 0 and step < t.steps:
                        path = _save_adapter(pipe, out_dir, f"{recipe.name}-{step:06d}")
                        yield ev(
                            EventKind.CHECKPOINT,
                            step=step,
                            path=str(path),
                            message=f"Checkpoint saved at step {step}",
                        )

            final = _save_adapter(pipe, out_dir, recipe.name)
            yield ev(
                EventKind.DONE,
                step=step,
                total_steps=t.steps,
                path=str(final),
                message=f"Training complete. Adapter written to {final}",
            )

        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
            yield ev(
                EventKind.ERROR,
                message=f"{type(exc).__name__}: {exc}",
                extra={"hint": _diagnose(exc)},
            )


# ----------------------------------------------------------------------
# Helpers. Kept module-level so they can be exercised independently once a
# GPU-equipped test environment exists.


def _load_pipeline(model_id: str, dtype: Any) -> tuple[Any, bool]:
    """Load a pipeline, detecting SDXL vs SD 1.x from its config."""
    from diffusers import DiffusionPipeline  # noqa: PLC0415

    kwargs = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False}
    path = Path(model_id)
    if path.suffix in {".safetensors", ".ckpt"}:
        from diffusers import StableDiffusionXLPipeline  # noqa: PLC0415

        pipe = StableDiffusionXLPipeline.from_single_file(model_id, torch_dtype=dtype)
    else:
        try:
            pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
        except TypeError:
            # Not every pipeline class accepts the safety-checker arguments.
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    return pipe, hasattr(pipe, "text_encoder_2")


def _build_optimizer(training: Any, params: list) -> Any:
    """Construct the configured optimiser, falling back when unavailable."""
    import torch  # noqa: PLC0415

    name = training.optimizer.value
    lr = training.learning_rate

    if name == "adamw_8bit":
        try:
            import bitsandbytes as bnb  # noqa: PLC0415

            return bnb.optim.AdamW8bit(params, lr=lr)
        except ImportError:
            # bitsandbytes is optional and platform-specific; plain AdamW is
            # the correct fallback rather than a hard failure.
            return torch.optim.AdamW(params, lr=lr)
    if name == "lion":
        try:
            import bitsandbytes as bnb  # noqa: PLC0415

            return bnb.optim.Lion(params, lr=lr)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def _build_scheduler(training: Any, optimizer: Any) -> Any:
    from diffusers.optimization import get_scheduler  # noqa: PLC0415

    return get_scheduler(
        training.scheduler.value,
        optimizer=optimizer,
        num_warmup_steps=training.warmup_steps,
        num_training_steps=training.steps,
    )


def _build_loader(pipe, recipe, report, device, dtype, is_sdxl):  # pragma: no cover - needs GPU
    """Pre-encode latents and prompt embeddings, then batch them.

    Both the VAE and the text encoders are frozen, so their outputs for a given
    image and caption never change. Computing them once up front removes them
    from the training loop entirely — a large speed win and, because the VAE can
    then be moved off the GPU, a large memory win too.

    The cost is that caption-level augmentation (shuffling, dropout) must be
    applied at encode time rather than per epoch. That is a real limitation and
    the reason this is the first thing to revisit after the GPU pass.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    from ..data.dataset import load_pixels  # noqa: PLC0415

    latents, embeds, pooled, time_ids = [], [], [], []

    with torch.no_grad():
        for item in report.items:
            pixels = load_pixels(item, recipe.data.resolution).to(device, dtype=dtype)
            latent = pipe.vae.encode(pixels.unsqueeze(0)).latent_dist.sample()
            latents.append(latent.squeeze(0) * pipe.vae.config.scaling_factor)

            caption = item.caption_for_training(recipe.data)
            if is_sdxl:
                pe, _, pp, _ = pipe.encode_prompt(
                    prompt=caption,
                    prompt_2=caption,
                    device=device,
                    do_classifier_free_guidance=False,
                )
                embeds.append(pe.squeeze(0))
                pooled.append(pp.squeeze(0))
                # SDXL conditions on original size, crop offset and target size
                # so it can distinguish a genuinely wide image from a cropped one.
                time_ids.append(
                    torch.tensor(
                        [item.width, item.height, 0, 0, item.bucket_width, item.bucket_height],
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                ids = pipe.tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                embeds.append(pipe.text_encoder(ids)[0].squeeze(0))

    # With the caches built, the VAE is dead weight on the GPU.
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    tensors = [torch.stack(latents), torch.stack(embeds)]
    if is_sdxl:
        tensors += [torch.stack(pooled), torch.stack(time_ids)]

    dataset = TensorDataset(*tensors)
    loader = DataLoader(
        dataset,
        batch_size=recipe.training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(recipe.training.seed),
        drop_last=False,
    )

    def collate():
        for batch in loader:
            out = {"latents": batch[0], "prompt_embeds": batch[1]}
            if is_sdxl:
                out["added_cond_kwargs"] = {"text_embeds": batch[2], "time_ids": batch[3]}
            yield out

    class _Reiterable:
        def __iter__(self):
            return collate()

    return _Reiterable()


def _save_adapter(pipe, out_dir: Path, stem: str) -> Path:  # pragma: no cover - needs GPU
    """Write the adapter in the standard diffusers LoRA layout."""
    from diffusers import StableDiffusionXLPipeline  # noqa: PLC0415
    from diffusers.utils import convert_state_dict_to_diffusers  # noqa: PLC0415
    from peft.utils import get_peft_model_state_dict  # noqa: PLC0415

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipe.unet))
    target = out_dir / stem
    target.mkdir(parents=True, exist_ok=True)
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(target),
        unet_lora_layers=state,
        safe_serialization=True,
    )
    (target / "metadata.json").write_text(json.dumps({"stem": stem}, indent=2))
    return target


def _diagnose(exc: Exception) -> str:
    """Turn a raw exception into something a user can act on."""
    text = str(exc).lower()
    if "out of memory" in text:
        return (
            "Out of VRAM. In order of impact: lower batch_size to 1, enable "
            "gradient_checkpointing, drop resolution to 768 or 512, then lower rank."
        )
    if "no such file" in text or "not found" in text:
        return (
            "A path or model id could not be resolved. Check dataset_path, and confirm the "
            "base model id exists on Hugging Face or points at a local file."
        )
    if "target modules" in text:
        return (
            "The target module names do not match this architecture. The defaults suit "
            "SD/SDXL U-Nets; other architectures name their projections differently."
        )
    if "connection" in text or "timeout" in text:
        return "Network problem while downloading the base model. Retry, or use a local path."
    return "Check the full traceback in the job log for details."
