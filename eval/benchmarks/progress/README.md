# Progress estimation (200 questions, 4 frames each)

Given four chronological frames of a robot episode and the task text, the model must estimate the overall task progress at the last
frame as a number in [0, 1]. Metric: **MAE** against the reference progress (lower is better); an unparsable or missing reply
counts as absolute error 1.

> The 798 frames are included in the Hugging Face archive `progress.tar.gz` (not in this git repository); `python prepare.py --from-hub --sets progress`
> downloads, verifies and extracts them to `data/progress/images/`. Frame sources keep their upstream licenses (see `LICENSES.md`).

> The score measures agreement with the reference progress definition, not physical task completion.
> Differences smaller than about 0.02 MAE should be treated as ties.

## Data (`data/progress/`)

* `questions.jsonl` - 200 items in the common LLaVA row format; image paths `images/<sha256>.jpg` are relative to `data/progress/` and resolve after the hub download.
* `labels.jsonl` - reference `process` per question (+ source, progress bin 0-4 with 40 questions each).
* `image_manifest.jsonl` - 800 rows (798 distinct frames): path, sha256 and size of the original encoded bytes, size in pixels, and where the
  frame comes from (`source`, `camera`, `episode`, `frame_id` in the original robot dataset). 40 sources: 28 Table30 tasks, BridgeData V2
  variants, RT-1/fractal variants, LIBERO, DobbE, FMB, ToTo, Berkeley AutoLab UR5 and two tabletop collections; one episode per question.
No internal paths, ids or private information are included; the original frames were extracted by the dataset builders and keep
their original encoded bytes, which is why the manifest carries sha256 values.

## Getting the frames

They come from the hub archive (`prepare.py --from-hub --sets progress` in the eval root); no separate build step is needed:

```bash
python benchmarks/progress/prepare.py --verify                              # check data/progress/images/ against the manifest sha256
python benchmarks/progress/prepare.py --images-dir /path/to/your/frames     # optional fallback: rebuild from your own copy
```

For the fallback, `--images-dir` may hold files named `<sha256>.jpg`, or `<source>/<camera>/<episode>/<frame_id:06d>.jpg`. Frames you re-extract from
the original datasets are re-encoded and will normally not match the sha256; the script then stops, and `--allow-hash-mismatch`
copies them anyway (results are then **not comparable** with published numbers). Questions are exact benchmark inputs only when
all 798 frames verify (the hub archive passes: 800/800 manifest rows ok).

## Protocol

Prompt per question: four `<image>` tags, then
`Task: <task text>` / `The images are in chronological order. Estimate overall task progress at the last image. Return only JSON with a numeric "process" between 0 and 1.`
Decoding: greedy, thinking off, max 64 new tokens, pixels 65,536..16,777,216 per image, up to 4 images, no system prompt (`config.json`).

```bash
python benchmarks/progress/eval.py --model /path/to/hf_checkpoint --out results/progress/mymodel
python benchmarks/progress/score.py --pred results/progress/mymodel/predictions.jsonl --json results/progress/mymodel/score.json
```

Parsing (`score.py`): the last JSON object in the reply (after any `</think>`) with a numeric `process`, else a bare number already in
[0, 1]; rounded to 2 decimals and clipped. Reported: official MAE (invalid = 1) and valid-only MAE with coverage, on all 200 questions
and on the 197-question subset that drops three questions (ids 0064, 0075, 0141) containing a frame with source index 320.
Reference points computed by the same scorer: constant 0.5 gives 0.2528.

## Reproduction check

The paper model was not run on this benchmark. Parity was checked with a saved run of the base 9B model (the public
Qwen3.5-9B checkpoint; predictions plus a recorded score): the release scorer gives official MAE **0.2571** (0.257066), valid-only
0.2458, coverage 0.985 (3 unparsed), s197 official 0.2587, identical to the recorded 0.2570660282. Other saved runs reproduce
too (e.g. a fine-tuned 9B variant 0.1529). The frames used by those runs are the original files; the manifest sha256 values were verified
against them.
