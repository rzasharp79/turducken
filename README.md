# turducken

User friendly, low overhead, custom LoRA training.

A LoRA trainer that refuses to be a wall of unexplained numbers. Every setting
says what it does, why it matters, and what goes wrong when it is set badly.
Every recommendation shows its reasoning. And when you want to know how any of
it actually works, the explainer modules go all the way down to the
mathematics — with demos that run on your machine rather than diagrams that
assert.

None of that gets in your way. Open the app, fill in four fields, press train.

```bash
pip install -e .
turducken serve          # → http://127.0.0.1:8420
```

## What it does

**Build a recipe.** One annotated form. It opens showing only the settings you
genuinely have to decide; two clicks reveals everything else. Each knob carries
a plain-language description, an explanation of what it changes during
training, its trade-off, a rule of thumb, and the symptoms of getting it wrong.

**Check your dataset first.** More LoRAs fail from dataset problems than from
hyperparameters, and those problems are invisible once training starts. The
analyzer reports aspect buckets, per-image crop loss, duplicates, missing and
empty captions, images too small to train at your resolution, and tags so
common they are being welded to your concept instead of staying controllable.

**Get a pre-flight review.** Advisories on what your recipe will probably do:
steps-per-image against your actual dataset size, learning rates outside the
useful range, resolution that does not match your base model, a VRAM estimate
against the GPU you actually have. Every advisory explains its reasoning and
suggests a fix.

Nothing blocks you. Advisories are advice — override any of them and run. A
prediction you watched come true teaches more than a button that refused to
work.

**Train, and watch the loop.** Live loss with a trailing mean, checkpoints, ETA,
and a narrated log. The simulated backend runs anywhere and explains each phase
as a real run would perform it; the diffusion backend does the real thing on a
CUDA GPU.

**Learn, optionally.** Six lessons covering what a LoRA is, what one training
step does, datasets and captions, resolution and buckets, where adapters
attach, and how to read your results. Five demos compute their claims live —
including the SVD energy comparison that shows *why* low rank is sufficient for
weight updates and insufficient for arbitrary matrices.

Every knob in the form links to the lesson that explains it.

## Install

The app itself is small and pure Python. The training engine is a separate
extra, because 6 GB of CUDA wheels should not be a prerequisite for reading a
lesson or checking a dataset.

```bash
pip install -e .                    # app, dataset tools, lessons, simulator
pip install -e '.[diffusion]'       # + real training (needs a CUDA GPU)
pip install -e '.[dev]'             # + pytest and ruff
```

Check what your machine can do:

```bash
turducken doctor
```

## Command line

`serve` is the main path, but everything is scriptable — useful on a headless
GPU box.

```bash
turducken serve [--host H] [--port P]   # the web app
turducken doctor                        # what this machine can do
turducken init recipe.json              # write a starter recipe
turducken check recipe.json             # dataset report + advisories
turducken train recipe.json             # run it
```

## Your dataset

A folder of images, each with a `.txt` caption file of the same name:

```
dataset/
  photo_01.png
  photo_01.txt      →  sks_marla, red jacket, standing in a kitchen
  photo_02.png
  photo_02.txt      →  sks_marla, outdoors, overcast light
```

The rule that explains most of the others: **caption what varies, omit what is
constant.** Anything you caption stays separately controllable; anything you
leave uncaptioned gets absorbed into the concept itself. The
`datasets-and-captions` lesson covers the rest.

## How it is built

```
src/turducken/
  schemas/       Recipe model. Every field declared with its own documentation
                 via knob(), which the API and UI read straight off the schema.
  data/          Bucketing arithmetic, dataset scanning, caption analysis.
  training/      Backend interface, the simulator, the diffusion trainer,
                 the job runner, and LoRA mathematics in plain NumPy.
  explain/       Lesson registry, Markdown content, runnable demos.
  server/        FastAPI app and a dependency-free frontend.
```

Two decisions shape most of the rest:

**Documentation lives on the field.** A knob is declared once, with its
explanation attached. The API serves that metadata; the UI renders it. Adding a
setting makes it appear in the interface, fully documented, with no frontend
change — and there is no second place where docs can drift out of sync. A test
enforces that nothing ships undocumented.

**Trainers are swappable.** Both backends emit the same event stream, so the
UI, charts and logs are written once. The simulator needs no GPU, which is what
lets the whole app be developed, tested and explored on any machine. The
text-LLM backend will arrive as a sibling rather than as branching.

## Status

Everything described above works and is covered by tests, with one exception:
**the diffusion backend has not been executed end-to-end.** It requires a CUDA
GPU, which CI cannot provide. The code is written against the standard
diffusers/peft APIs but should be treated as unverified until it has run on
real hardware.

Also not done yet, deliberately rather than by omission:

- Text-LLM LoRAs. The recipe and registry are shaped for it; no backend yet.
- Flux and SD3, which use a DiT rather than a U-Net.
- Text-encoder training.
- Sample-image generation from checkpoints during a run — the single most
  useful missing feature, since judging by eye is what the lessons keep telling
  you to do.
- Job persistence across restarts.
- Caption augmentation is applied at latent-cache time rather than per epoch,
  so shuffling and dropout are fixed for a run.

## Development

```bash
pip install -e '.[dev]'
pytest          # 94 tests, no GPU required
ruff check .
```

The test suite doubles as documentation of the claims the lessons make: the
parameter savings, the zero-initialisation of B, the alpha/rank scaling, and
the low-rank compressibility of realistic updates are all asserted numerically,
so the teaching material cannot quietly drift from what the code does.
