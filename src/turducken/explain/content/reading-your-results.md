# Reading your results

Training finishes and you have eight checkpoints. Now what?

## The loss curve will not help you

Say it once more, because it is the most common wasted effort in LoRA training: **the diffusion loss curve does not tell you whether your LoRA is good.**

Each step samples a random timestep, so consecutive losses are answering questions of very different difficulty. A jagged curve is normal. A curve that flattens is not evidence of convergence. A slightly lower final loss does not mean a better adapter.

The only loss signals worth acting on are the extremes:

- **NaN or exploding** — learning rate far too high, or fp16 instability.
- **Perfectly flat from step 1** — something is not connected. Check that target modules matched.

Everything else: ignore it and look at images.

## The actual method

Generate the same set of prompts from every checkpoint, with a fixed seed.

Fixing the seed is what makes the comparison meaningful — without it you are comparing random variation between generations, not the difference between checkpoints. Use four or five prompts covering different situations:

1. **The plain trigger.** `sks_marla` alone. Does the subject appear at all?
2. **A prompted variation.** `sks_marla wearing a spacesuit on Mars`. Does it still respond to your prompt?
3. **A style shift.** `sks_marla, oil painting`. Is the concept detachable from photographic rendering?
4. **A control prompt with no trigger.** `a woman in a park`. Has the LoRA leaked into everything?

That fourth one catches damage the first three hide.

## Reading what you see

**Undertrained** — the subject is vaguely suggested but never locks in. Features drift between generations. It looks like the model is trying to remember someone it met once. *Fix: more steps, or a higher learning rate.*

**Overtrained** — the subject is exact, and so is everything around it. Backgrounds from your dataset appear uninvited. Poses repeat. Prompts stop working: ask for a spacesuit and get the same shirt as photo 12. In the worst case, near-copies of training images. *Fix: an earlier checkpoint. If even the earliest is overfit, fewer steps or a lower learning rate — and probably more data.*

**Leaking** — your control prompt is contaminated: unrelated subjects drift toward yours. *Fix: a rarer trigger word, and caption more of what varies.*

**Fried** — oversaturated, high-contrast, noisy, burnt. *Fix: lower the learning rate, or check that alpha/rank is not much greater than 1.*

**Style-locked** — the subject is right but always rendered the same way. Usually a dataset with uniform lighting or a uniform source. *Fix: more visual variety in the data.*

## Adapter strength at inference

When you load a LoRA you choose a weight, typically `0.0`–`1.5`. This multiplies the whole correction at generation time and is a genuinely useful dial — you tune it *after* training, without retraining anything.

If your LoRA only looks right below about 0.5, it is overtrained: you are compensating by turning it down. If it needs more than about 1.2 to show up at all, it is undertrained. A healthy adapter works around 0.7–1.0.

This is also why the strongest checkpoint is not the best one. A slightly undertrained adapter that responds well to prompts and can be pushed to 1.0 is more useful than a maximally strong one that has to be held at 0.4.

## Picking the winner

Choose the **earliest** checkpoint that reliably produces your subject.

Not the most exact one. The earliest reliable one. Later checkpoints buy fidelity to your training images with flexibility everywhere else, and flexibility is what you will actually want the first time you prompt something the dataset never contained.

## When to retrain

Change one thing at a time, keeping the seed fixed, or you will not know what helped.

In descending order of expected impact:

1. **Data** — more variety, better captions, remove duplicates. Almost always the real answer.
2. **Steps** — the cheapest knob and often sufficient.
3. **Learning rate** — halve or double; smaller adjustments rarely show.
4. **Rank** — only after the first three, and only when there is data to justify the extra capacity.

The instinct to reach for rank first is strong and almost always wrong.
