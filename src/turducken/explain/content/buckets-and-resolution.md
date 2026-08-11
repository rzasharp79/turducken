# Resolution and aspect buckets

## Train at the model's native resolution

Every diffusion model has a resolution it was trained at, and it is genuinely better at that size than at any other.

| Model family | Native |
|---|---|
| Stable Diffusion 1.5 | 512 |
| Stable Diffusion 2.1 | 768 |
| SDXL | 1024 |
| Flux | 1024 |
| SD3 | 1024 |

Train away from native and you spend part of the run teaching the model a scale it does not natively work in. The result is soft or subtly malformed detail — not a dramatic failure, just a run that is quietly worse than it should have been.

Going *above* native is the more expensive direction. VRAM and step time scale with the **square** of resolution: moving from 512 to 1024 is four times the pixels, not twice. If you are out of memory, this is the setting with the most leverage.

Dimensions must be multiples of 64. This is a hard requirement of the architecture, not a convention: the VAE downsamples by 8, and the U-Net halves resolution repeatedly after that. Sizes that are not multiples of 64 cannot survive that round trip cleanly.

## Aspect buckets

Your images are not all square. The model needs fixed shapes per batch. Something has to give.

**Without bucketing**, every image is centre-cropped to a square. A 1920×1080 photo loses 44% of its width. A portrait loses the top and bottom — which, on a photo of a person, routinely means part of the head.

**With bucketing**, the trainer defines a set of allowed shapes with roughly equal pixel counts and routes each image to the closest one:

```
1024×1024   (1:1)
1152×896    (9:7)
1216×832    (3:2)
1344×768    (16:9)
896×1152    (7:9)
832×1216    (2:3)
768×1344    (9:16)
```

A 1920×1080 photo goes to 1344×768 and loses almost nothing. A portrait trains as a portrait.

Every bucket holds approximately the same number of pixels, so VRAM and step time are consistent regardless of which bucket an image lands in — a wide image costs no more than a square one.

Run the **What bucketing saves** demo to see the crop percentages side by side, with and without.

There is no real reason to turn bucketing off. If every image is already square it changes nothing; otherwise it saves data you would rather keep.

## What the report is telling you

**Cropped fraction, per image.** Images cropped by more than about 25% get flagged. Bucketing minimises cropping but cannot eliminate it — an extreme panorama still has to lose something. When you see a large crop, crop it yourself first, so you choose which part survives instead of letting the centre win by default.

**Underfilled buckets.** A bucket holding a single image can only ever train in a batch of one, making that image's gradients noisier than the rest of the dataset's. Either crop it toward a more populated shape, or find a few more images that share it.

**Small sources.** Images whose shortest edge is well below your training resolution get upscaled, and upscaling invents detail that was never captured. The model learns the invention along with everything else.

## Practical advice

- Set resolution to your base model's native size and leave it there.
- Leave bucketing on.
- Crop deliberately before training when an image would lose a lot — you know what matters in it, the centre-crop does not.
- Shoot or source at or above training resolution. Downscaling is free; upscaling is not.
