# Dataset Pruning by Loss Profiling

A method for finding the images in a training set that are actively costing you
quality, rather than guessing from thumbnails.

The core trick is that **the noise tensor is shared across all images at each
timestep**. That removes the noise draw as a confound entirely and makes losses
directly comparable image to image. Without it, per-image loss differences are
mostly the luck of the sample and the whole exercise is noise.

## The probe

```python
TIMESTEPS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]   # flow-matching sigma
# ONE noise tensor per timestep, reused for every image
noise = {t: torch.randn(latent_shape, generator=torch.Generator().manual_seed(1000+i))
         for i, t in enumerate(TIMESTEPS)}

def profile(model, dataset):          # batch_size = 1
    rows = []
    for img, caption in dataset:
        z = vae.encode(img)
        losses = []
        for t in TIMESTEPS:
            zt     = (1 - t) * z + t * noise[t]
            target = noise[t] - z
            pred   = model(zt, t, text_encode(caption))
            losses.append(mse(pred, target).item())
        rows.append((img.name, mean(losses), std(losses)))
    return rows
```

Run it three times:

1. Probe with the LoRA **disabled** → `L_base`
2. Train **one epoch**
3. Probe again, identical settings, with the LoRA **active** → `L_after`

The two working numbers are:

- **Δ = L_base − L_after** — how much the model actually absorbed from that image
- **σ_t** — the spread of loss across noise levels for that image

On a 4090 this is roughly 1200 forward passes for a 100-image set. Call it 10
minutes.

## Pretend results, n=100

Distribution: `L_base` median 0.412 (IQR 0.371–0.458), Δ median 0.087
(IQR 0.061–0.112), σ_t median 0.052 (top decile above 0.115).

| img | L_base | Δ | σ_t | read |
|---|---|---|---|---|
| 007 | 0.598 | +0.194 | 0.061 | core signal |
| 041 | 0.571 | +0.182 | 0.055 | core signal |
| 050 | 0.455 | +0.093 | 0.044 | solid keep |
| 062 | 0.412 | +0.087 | 0.049 | solid keep |
| 012 | 0.489 | +0.088 | 0.121 | inspect (σ) |
| 019 | 0.318 | +0.021 | 0.038 | redundant |
| 088 | 0.302 | +0.014 | 0.033 | redundant |
| 023 | 0.294 | +0.013 | 0.031 | redundant |
| 055 | 0.784 | +0.015 | 0.198 | resistant |
| 071 | 0.831 | +0.009 | 0.214 | resistant |
| 034 | 0.643 | −0.028 | 0.147 | conflicting |
| 096 | 0.612 | −0.043 | 0.163 | conflicting |

## How to read it

Split on **your own medians**, never on fixed thresholds. Absolute loss scale is
meaningless across models.

- **High `L_base`, high Δ** (007, 041) — hard but learnable. This is the dataset
  earning its keep. Never prune these, even though they look like your worst
  images by raw loss.
- **Low `L_base`, low Δ** (019, 088, 023) — the model already knew this.
  Redundant, but do not blanket-delete. Cross-reference against your embedding
  clusters and cut only where a cluster is overrepresented, or you will delete
  your cleanest canonical shots.
- **High `L_base`, low Δ** (055, 071) — resisting learning. Highest-value prune.
  Inspect these and you will typically find caption mismatch, wrong subject,
  heavy compression, extreme crop, or a latent sitting outside VAE range. These
  consume gradient budget and return nothing.
- **Negative Δ** (034, 096) — loss went *up* after training on the set. That
  image conflicts with the rest of your data. This is the hardest reject signal
  the method produces, and it is nearly always a true positive.

### The σ_t override

Anything in the **top decile of σ_t** goes to the inspection queue regardless of
quadrant — which is why 012 is flagged despite a healthy Δ. High dispersion
across noise levels means easy at some noise scales and hard at others, which is
the fingerprint of a sharpness or resolution mismatch. These are your actual
loss-spike generators.

## Prune order and stopping

1. Negative Δ
2. High `L_base` / low Δ
3. Top-decile σ_t
4. Redundant **and** duplicated

On 100 images that lands around 22–25 removed. **Stop there on the first pass.**
Deeper cuts at this dataset size cost more in coverage than they buy in
stability.

Then re-profile the survivors. You want median σ_t down and the Δ distribution
tightened. If median Δ jumps hard *and* σ collapses, you over-pruned into
homogeneity — that is the memorization failure mode wearing a nice loss curve.

## Caveat on interpretation

Δ is measured on images the LoRA trained on, so it conflates learning with
memorization. That is fine for deciding what to prune, but do not read it as a
generalization measure.
