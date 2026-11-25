# EgoPlan-Bench (v1 validation) - build-from-source release

**No frames and no questions are shipped.** This directory contains scripts that rebuild our exact evaluation set from the
official sources, plus a manifest (`data/egoplan/MANIFEST.json`, `data/egoplan/ids.json`) that lets you check you got the
same set. The paper reports **74.54** for the model it evaluates on this set (3,343 items).

## What the benchmark is
EgoPlan-Bench (Chen et al., arXiv 2312.06722): four-option multiple choice, "given the task goal, the progress so far and the
current observation, what is the next action?". Videos come from EPIC-KITCHENS-100 (CC BY-NC 4.0) and Ego4D (Ego4D license
agreement). The official code repository (ChenYi99/EgoPlan) is BSD-3-Clause; the dataset card of lmms-lab/EgoPlan-Bench is gated
(login and acceptance of terms required) and we did not find a licence field there. You are responsible for complying with the
EPIC-KITCHENS and Ego4D terms; nothing of them is redistributed here.

## Our protocol (differs from the official one - please read)
* Source: the **lmms-lab/EgoPlan-Bench** validation parquet and its per-sample mp4 clips (`<sample_id>.mp4`), not the original
  frames. Official pinned references: lmms-eval `tasks/egoplan` @ `08affba1`, ChenYi99/EgoPlan @ `1510afc1`.
* **8 images per question**: 7 task-progress frames + the current observation (the reference code uses `n_frms=8`).
  Frames are re-sampled from the clip (long side 448, JPEG q90, OpenCV `INTER_AREA`).
* The lmms-lab clips contain the union of all annotated action segments (gaps removed) and are *not* cut at the current
  observation: ~400 clips include frames from the future. We map every clip position to its original frame index and sample
  progress frames only strictly before the current observation (see `prepare.py` docstring). 15 clips are shorter than their
  metadata predicts (their frame indices are recorded as null; the same formula is applied on raw clip positions); 502
  questions have no usable progress and repeat the current observation 8 times.
* Prompt = lmms-eval `egoplan_doc_to_text`, word for word, with the 8 `<image>` tags in front.
* Generation: greedy, non-thinking, `max_tokens` 1024, image pixels 65,536..16,777,216 (`config.json`).
* Scoring (`score.py`): headline = accuracy of `parse_choice(reply) == gold letter` (the last standalone option letter of the
  reply; this is what the paper table uses). Secondary, also printed: the lmms-eval first-letter extractor, which takes the first
  capital A-D anywhere in the reply (e.g. the "B" of "Based"); the two agree for replies that are just a letter.
* Not comparable with the official leaderboard (different frames, 3,343 instead of 3,355 items, different scoring of verbose replies).

## 3,343 vs 3,355 items
The official validation set has 3,355 questions. The lmms-lab parquet carries an `is_valid` flag and 12 sample ids are not
`is_valid=True`; **this is the set we used** (we kept the `is_valid=True` rows only):

`247 370 564 651 704 963 1036 1753 1764 1909 2119 2346`

We do **not** know why the flag is False for these 12. The lmms-eval task and the official repository do not use or mention
`is_valid` (we checked both pinned clones), the dataset card is gated and was not available to us, and all 3,355 clips exist in the zip
shards. We did not inspect those 12 questions for annotation problems. If you prefer the full 3,355, drop the id filter in
`prepare.py` (`--ids` accepts any id list); numbers on that set are not those in the paper.

## Reproduce
```bash
bash benchmarks/egoplan/download.sh data/egoplan/raw --with-clips      # login + accept terms first; clips are ~120 GB
python benchmarks/egoplan/prepare.py --parquet data/egoplan/raw/hf --zips-dir data/egoplan/raw/hf --out data/egoplan --workers 8
#   (or --clips-dir DIR with extracted <sample_id>.mp4;  needs: pip install -e ".[build]")
python benchmarks/egoplan/eval.py --model /path/to/model --out results/egoplan/mymodel      # vllm backend
python benchmarks/egoplan/score.py --pred results/egoplan/mymodel/predictions.jsonl
```
`prepare.py` ends with a PASS/FAIL verification: sha256 of the built `eval.jsonl` against `data/egoplan/MANIFEST.json`
(must be byte-identical), the row-id list, and the set of referenced image paths. `--verify-only --out DIR` re-checks a build.

### Honest notes on reproducibility
* The question file (`eval.jsonl`) is deterministic given the parquet and the frame selection, and is verified byte-for-byte.
  Its image *paths* come from the same selection. The frame *bytes* depend on the video decoder and JPEG encoder (OpenCV/FFmpeg
  build); `data/egoplan/ids.json` holds the sha256 of every image from our build for information, and `prepare.py` reports how many
  match. A mismatch in image bytes does not change the verdict, and in practice does not change scores meaningfully, but we have not
  measured that.
* The original build inputs (`val.json`, zip member index) no longer exist. What was verified: (a) the row assembly in `prepare.py`
  re-derives our `eval.jsonl` byte-for-byte from the per-sample parquet fields and `frame_provenance.jsonl`
  (checked offline against `frame_provenance.jsonl`); (b) the frame extractor on synthetic
  clips. The end-to-end rebuild from the raw zip shards is a separate check (see the release notes; not part of this directory).
* `prepare.py` expects the parquet columns `sample_id, task_goal, choice_a..d, answer, golden_choice_idx, video_id, video_source,
  current_observation_frame, task_progress_metadata` (JSON string or list of `{start_frame, stop_frame}`) and optionally `is_valid`.

## Files
`download.sh`, `prepare.py`, `eval.py`, `score.py`, `config.json`; `data/egoplan/{MANIFEST.json, ids.json, NOTICE.md}`.
