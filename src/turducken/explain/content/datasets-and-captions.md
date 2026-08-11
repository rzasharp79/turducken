# Datasets and captions

Most failed LoRAs are dataset failures wearing a hyperparameter costume. People re-run with a different rank and learning rate a dozen times when the actual problem was fifteen photos taken in the same room.

## How many images

Fewer than you think, and better than you have.

- **10–20** — workable for a single consistent subject.
- **20–50** — the sweet spot for characters and objects.
- **50–150** — appropriate for styles, which need more variety to generalise.
- **300+** — usually a sign of scraping rather than curating.

Twenty varied images beat two hundred near-duplicates every time. The model learns what is *consistent* across your dataset and treats it as the concept. If every photo shares the same lighting, that lighting becomes part of your subject and you cannot prompt it away.

## Variety is the actual requirement

Vary everything that is *not* your subject:

- **Angles** — front, three-quarter, profile, above, below.
- **Distances** — close-up, mid, full body or full object.
- **Lighting** — daylight, indoor, warm, cool, harsh, soft.
- **Backgrounds** — as many different ones as you can manage.
- **Expression, pose, clothing** — for people, unless a specific one is the point.

Keep constant only the thing you are training. Everything else that stays constant gets absorbed into the concept.

This is exactly what the dataset report's "ubiquitous tags" warning is looking for: a tag in nearly every caption is not distinguishing anything, it is being welded to your subject.

## What ruins a dataset

- **Duplicates.** Silently weight those examples more heavily. The report flags byte-identical files, but visually near-identical frames pulled from a video are just as damaging and no scanner can catch them.
- **Low resolution.** Upscaling a 200×200 image to 1024 does not recover detail; it invents it, and then teaches the model that invention.
- **Compression artifacts.** JPEG blocking is a visual pattern like any other. The model will learn it.
- **Watermarks and text.** Learned readily and reproduced faithfully.
- **Motion blur and soft focus.** Teaches the model your subject is blurry.
- **Other people or objects in frame.** Without captions distinguishing them, they become part of the concept.

## Captions

Each image needs a `.txt` file with the same stem: `photo_01.png` pairs with `photo_01.txt`. This is the convention every major LoRA trainer uses, so datasets move between tools unchanged.

The rule that explains all the others: **caption what varies, omit what is constant.**

The model binds your concept to whatever is *consistently present but unexplained*. Anything you caption becomes separately controllable; anything you leave uncaptioned gets folded into the concept itself.

So for a character LoRA:

```
sks_marla, wearing a red jacket, standing in a kitchen, side lighting
```

You caption the jacket, the kitchen and the lighting **because you want them to remain optional**. You do *not* caption her face, hair colour or build — those are the thing you are training, and naming them would let the model treat them as detachable.

Beginners routinely get this backwards, describing the subject in exhaustive detail and omitting the setting. The result is a LoRA that produces your subject's kitchen reliably and their face inconsistently.

## The trigger word

A rare token that means "the thing I trained".

```
sks_marla, ohwx_castle, zxc_style
```

Without one, your subject binds to whatever common words appear in the captions. Caption a woman with "woman" and you are not adding a concept — you are overwriting the model's existing understanding of "woman". Every woman you generate afterwards drifts toward your subject, in prompts that have nothing to do with her.

A rare token gives the concept somewhere new to live and leaves the model's vocabulary intact.

Choose something the tokenizer will not recognise as ordinary English. Avoid real names — they already carry strong associations you would be fighting.

The exception: if you *want* to overwrite an existing concept, leave the trigger word empty. That is occasionally right for a style and almost never right for a subject.

## Caption style

Match the base model. SDXL and Flux were trained on natural language and respond to sentences. Anime models were largely trained on comma-separated Danbooru tags and respond to those. Using the wrong style means fighting the text encoder for the whole run.

Keep it consistent within a dataset. Mixed prose and tags is worse than either alone.

## Two settings worth understanding

**Caption dropout** blanks a caption entirely with some probability, so the image trains with no text at all. This teaches the model what your subject looks like unconditionally, strengthening the trigger word and improving behaviour under classifier-free guidance. `0.05` is a good default; above `0.2` the concept starts detaching from your trigger.

**Caption shuffling** reorders comma-separated tags each time a caption is used. Text encoders weight earlier tokens more heavily, so without shuffling the model learns your captions' arbitrary word order as if it meant something. Useful for tag-style captions, pointless for prose.

---

**Next:** [Resolution and aspect buckets](buckets-and-resolution) — how those images are actually fed in.
