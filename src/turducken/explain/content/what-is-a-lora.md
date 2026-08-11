# What a LoRA actually is

## The problem it solves

Stable Diffusion XL has about 2.6 billion parameters. Suppose you want it to know your face.

The obvious approach is to fine-tune: keep training the whole model on photos of you. It works, and it is impractical for almost everyone. You need enough VRAM to hold the model *plus* a gradient for every parameter *plus* two optimiser statistics for every parameter — roughly four copies of a 2.6-billion-parameter model. The result is a new 7 GB checkpoint that knows your face, and if you also want your dog, that is another 7 GB.

LoRA — Low-Rank Adaptation — gets you the same capability from a file that is often under 50 MB, trained on a consumer GPU in under an hour.

It does this without approximating or cutting corners on the thing you care about. The saving comes from a genuine mathematical property of fine-tuning updates, not from a quality trade.

## The key insight

Fine-tuning changes a weight matrix `W` into `W + ΔW`. Everything you taught the model lives entirely in `ΔW` — the *change*, not the final weights.

The observation behind LoRA is this: **`W` is complicated, but `ΔW` is not.**

`W` encodes everything the base model knows — anatomy, lighting, composition, thousands of visual concepts — and genuinely needs its millions of numbers. But `ΔW`, the adjustment that teaches it one new face, is a much simpler object. It is one consistent correction applied across many contexts. In the language of linear algebra, it has **low intrinsic rank**: it can be reconstructed accurately from far fewer numbers than its shape suggests.

You do not have to take this on faith. The **Why low rank is enough** demo builds a realistic update, decomposes it, and shows how much of it survives at each rank — next to a random matrix of identical shape, which does not compress at all. That contrast is the entire argument.

## The mechanism

If `ΔW` is low-rank, do not store it as a full matrix. Factor it.

A 1024×1024 matrix of rank 16 can be written exactly as the product of a 1024×16 matrix and a 16×1024 matrix. So instead of learning `ΔW` directly, LoRA learns two thin matrices:

```
ΔW = B @ A

A: (rank × in_features)    — 16 × 1024 = 16,384 numbers
B: (out_features × rank)   — 1024 × 16 = 16,384 numbers
                             ─────────────────────────
                             32,768 numbers total

ΔW: (out_features × in_features)  — 1024 × 1024 = 1,048,576 numbers
```

**32,768 instead of 1,048,576. About 3%.**

At inference the model computes:

```
output = W @ x  +  (alpha / rank) * (B @ A) @ x
         ───────    ─────────────────────────
         frozen         your adapter
```

`W` is never modified. It is loaded read-only, and the adapter's contribution is simply added alongside. This is why LoRAs stack — each one contributes its own `ΔW` to the same frozen base — and why a LoRA file is useless on its own: it is a correction, and a correction needs something to correct.

## Two details that are not arbitrary

### B starts at zero

`A` is initialised randomly. `B` is initialised to **all zeros**.

Therefore `B @ A = 0` at step zero, and the adapter contributes exactly nothing. Attaching an untrained LoRA is a perfect no-op, and training departs from the base model's behaviour gradually rather than starting from a damaged version of it.

Had both matrices started random, the adapter would inject noise into a working model before learning anything, and the first phase of training would be spent repairing that damage. Run the **Why B starts at zero** demo to see the correction's magnitude begin at 0.0 and grow.

### The alpha/rank scaling

The adapter's output is multiplied by `alpha / rank` before being added.

Without this, changing rank would silently change how hard the adapter pushes: doubling the rank roughly doubles the magnitude of `B @ A`, so a rank experiment would also be an unintended learning-rate experiment. Dividing by rank cancels that out, letting you change capacity and strength independently.

The practical advice follows directly: **set alpha equal to rank** for a clean scale of 1.0, and control strength with the learning rate, whose effect does not also depend on rank.

## Choosing a rank

Rank is the width of the bottleneck your correction must pass through. It is the one parameter that genuinely trades expressiveness against overfitting.

| Rank | Suits | Watch for |
|------|-------|-----------|
| 4–8 | One simple, consistent subject | Under-detailed if the subject varies a lot |
| 16 | Most character and object LoRAs | The reliable default |
| 32 | Art styles, complex subjects | Needs more data to justify itself |
| 64+ | Multiple concepts in one adapter | Memorises training images readily |

Higher rank is not "better quality". It is *more capacity*, and capacity you have not got the data to fill gets spent memorising your specific images — backgrounds, watermarks and all. The failure mode is distinctive: outputs that reproduce training photos almost exactly and stop responding to your prompt.

If your result lacks detail, add data before you add rank. Extra capacity fits a flawed dataset more precisely; it does not fix it.

## What to take away

- A LoRA does not modify the model. It learns a correction and adds it.
- The correction compresses well because fine-tuning updates are genuinely low-rank — a measurable property, not an assumption.
- Rank sets capacity; alpha/rank sets strength; keeping them equal keeps those two concerns separate.
- Zero-initialised `B` means training starts from the base model's behaviour, not from noise.

---

**Next:** [What happens in one training step](the-training-loop) — how that correction is actually learned.
