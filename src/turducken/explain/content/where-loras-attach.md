# Where the adapters go

A LoRA does not adapt the whole model. It attaches to specific layers, and the choice of which ones is why a 50 MB file can change what a 7 GB model produces.

## The shape of a U-Net

Stable Diffusion's denoiser is a U-Net: the latent is progressively downsampled through an encoder, passed through a middle block, then upsampled back through a decoder, with skip connections joining matching levels.

Two kinds of block do the work at each level:

- **ResNet blocks** — convolutions that process local spatial structure. Textures, edges, local coherence.
- **Transformer blocks** — attention, which is where the model decides what relates to what.

The transformer blocks are the interesting target, and each contains two kinds of attention:

- **Self-attention** — image regions attending to each other. This is how the model keeps a face coherent across the region it occupies, and how it knows a shadow belongs to an object.
- **Cross-attention** — image regions attending to the *text embedding*. This is where your prompt actually enters the image.

## The four projections

Every attention layer computes four linear projections:

| Module | Role |
|---|---|
| `to_q` | Query — "what am I looking for?" |
| `to_k` | Key — "what do I offer?" |
| `to_v` | Value — "what do I contribute if selected?" |
| `to_out.0` | Output — merges the result back into the stream |

These four names are the default `target_modules`, and they are where LoRA achieves its leverage.

In cross-attention specifically, `to_k` and `to_v` process the *text* embedding. Adapting them changes how the model interprets your prompt — which is precisely the intervention you want when teaching it that a new token means a new face. You are not rebuilding the model's knowledge of faces; you are adding a route from one rare token into the machinery that already knows how to draw one.

That is the whole reason attention-only adaptation is so efficient. The base model already knows how to render skin, hair, cloth and light. It does not need to relearn any of it. It needs to learn one new association, and associations live in attention.

## What happens if you adapt more

**Adding the feed-forward layers** (`ff.net.0.proj`, `ff.net.2`) roughly doubles the adapted parameter count. These layers hold much of the model's factual knowledge rather than its routing, so adapting them gives more raw capacity — and more opportunity to overwrite things you wanted to keep. Worth trying for style LoRAs that need to change *how* things are rendered rather than *what* is rendered. Rarely worth it for a subject.

**Adding the ResNet convolutions** (sometimes marketed as "LoCon" or "conv LoRA") lets the adapter change texture and local structure directly. Genuinely useful for styles with a distinctive surface quality — brush strokes, film grain, halftone. Adds significant size.

**Adapting the text encoder** as well as the U-Net is a separate decision, not covered by `target_modules`. It can strengthen how firmly a trigger word binds, at the cost of a much easier path to damaging the model's general language understanding. Not implemented in this first pass.

## What to actually do

Leave `target_modules` at its default. Attention-only is right for the overwhelming majority of runs, and the alternatives are worth reaching for only when you can articulate what the default failed to capture.

If a subject LoRA is underperforming, the answer is nearly always more or better data — not more adapted layers. Extra capacity fits your existing dataset more precisely, including its flaws.

## Naming caveat

The default module names match SD and SDXL U-Nets. Other architectures name their projections differently — Flux and SD3 use a DiT rather than a U-Net, with a different internal structure entirely. If training reports that no target modules matched, that mismatch is why.
