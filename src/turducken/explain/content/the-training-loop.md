# What happens in one training step

A training run is one small loop repeated a few thousand times. Once you can see the loop, every hyperparameter stops being folklore and becomes a dial on a specific part of it.

## The loop

For each step:

1. **Take a batch of images.** They have already been encoded into *latents* — compressed representations about 8× smaller per side than the image. The VAE that does this is frozen, so this happens once before training rather than every step.

2. **Pick a random timestep.** A number from 0 to ~1000 deciding *how much* noise to add. This is drawn fresh every step, and it matters more than it looks.

3. **Add exactly that much noise.** The noise schedule converts the timestep into a mixing ratio. Near 0, the latent is barely touched. Near 1000, almost nothing of the original survives.

4. **Ask the model to predict the noise.** Given the noisy latent, the timestep, and the caption's text embedding, the U-Net predicts which noise was added.

5. **Measure the error.** Mean squared error between predicted and actual noise. This single number is the loss.

6. **Backpropagate — into the adapter only.** Gradients flow through the whole frozen network but are only *applied* to the LoRA's A and B matrices. Everything else is `requires_grad=False`.

7. **Step the optimiser.** AdamW moves each adapter parameter a little way down its gradient, scaled by the learning rate.

That is it. Repeat a few thousand times and the adapter has learned a correction that makes the model better at denoising images of your subject — which, run in reverse at generation time, means better at *producing* them.

## Why your loss curve looks like static

This surprises everyone: **the diffusion loss curve is nearly useless for judging a run.**

Look at step 2 again. Every step draws a random timestep, so consecutive steps are answering questions of wildly different difficulty. Denoising a nearly-clean latent is easy and scores a low loss. Recovering signal from near-total noise is close to impossible and scores a high one. The difference between two adjacent steps is dominated by which timestep each happened to draw — not by whether your model is learning.

Run the **What the model is asked at each timestep** demo to see the difficulty gradient directly.

The consequence is practical: a jagged loss curve is normal and healthy. Do not tune against it. Judge a run by generating sample images from checkpoints and looking at them. The only loss signals worth acting on are the extremes — a curve that explodes or turns to NaN means your learning rate is too high; a curve that is perfectly flat from the very start means something is not connected.

## The knobs, and where they act

**Learning rate** — how far step 7 moves. The gradient says *which direction* reduces the error; the learning rate decides *how far to go*. Too high and you overshoot and oscillate; too low and you never arrive. LoRAs tolerate rates one to two orders of magnitude above full fine-tuning, because the adapter starts at zero and only adds a correction — there is far less to destroy. `1e-4` is the standard starting point.

**Steps** — how many times the loop runs. Total learning is roughly `steps × learning_rate`, so these two are not independent and should not be tuned as if they were. Halving the rate and doubling the steps lands in a similar place, more slowly and more stably.

**Batch size** — how many images go through step 1 together. Averaging the error over several images gives a less noisy estimate of the right direction. It is also the setting most likely to exhaust your VRAM, because every image in the batch needs its activations held in memory.

**Gradient accumulation** — run steps 1–6 several times, adding up the gradients, and only then do step 7. Four batches of 1 gives you the update quality of a batch of 4 while only ever holding one image's activations. This is how you get large-batch behaviour on a small GPU: more time, not more memory.

**Scheduler** — how the learning rate changes across the run. Cosine decay starts at your configured rate and eases to near zero, so training takes big confident strides early and fine careful ones at the end, when it is settling details rather than finding the general shape.

**Gradient checkpointing** — step 6 needs the intermediate values from the forward pass. You can either keep them in VRAM or discard them and recompute on demand. Recomputing costs roughly 30% more time and saves a great deal of memory. It is usually the difference between training and an out-of-memory error.

## What "converged" means here

There is no clean finish line. The model keeps getting better at denoising your specific images indefinitely — and past a certain point, that stops being what you want. Continue long enough and it reproduces your training photos including backgrounds, poses and JPEG artifacts, while losing the ability to respond to your prompt.

The right stopping point is a judgement made by eye, which is why `save_every` matters. Take checkpoints across the run, generate the same prompt from each, and pick the one that has your subject without dragging your dataset along with it. That checkpoint is frequently not the last one.

---

**Next:** [Datasets and captions](datasets-and-captions) — the input side, which decides more than any of this.
