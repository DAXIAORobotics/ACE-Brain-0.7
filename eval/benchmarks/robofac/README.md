# RoboFAC (test split, 8-frame rebuild)

Robot failure analysis benchmark built from the **test split** of RoboFAC (arXiv:2505.12224, dataset `MINT-SJTU/RoboFAC-dataset`,
MIT license; code https://github.com/MINT-SJTU/RoboFAC). Five question families over 8 frames of a robot video:

| family | questions | answer | scoring |
|---|---|---|---|
| `task_identification` | 1,204 | free text | LLM judge, Yes/No vs reference (local protocol) |
| `task_planning` | 1,204 | numbered subtasks | **shipped, but not in the paper average**; judged only with `--include-planning` |
| `failure_detection` | 1,204 | yes / no | rule |
| `failure_identification_mc` | 960 | letter A-F | rule |
| `failure_locating_mc` | 960 | letter A-D | rule |

Headline `fac` = **macro average of 4 family accuracies**: `failure_detection`, `failure_identification_mc`, `failure_locating_mc`, `task_identification`
(the families used in the paper). Family accuracy = 100 * correct / n.

## Contents (`data/robofac/`, 241 MB)

`test.jsonl` (5,532 questions, LLaVA-style, `image` = 8 frame paths relative to `data/robofac/`), `images/<video id>/0000..0007.jpg` (9,632 frames, long side 448 px, JPEG q90, only
the frames the test questions reference), `SHA256SUMS`, `MANIFEST.json` (build summary of the test split), `LICENSE-NOTICE.md` (MIT text and attribution). No training files are included.

## Our modifications (please read)

* **Rebuilt, not the official release.** The questions are not the upstream files: the test split was rebuilt from the upstream dataset as an 8-frame jsonl (frames sampled from
  each video, re-encoded at long side 448 / JPEG q90), with one prompt per family. Results are therefore not directly comparable to numbers computed on the original release or its video input.
* **Failure-locating answer corrections.** While building the failure-locating questions, some annotations were *dropped* (annotated answers that do not occur in the video's
  subtask plan, 319 in the full build) and some were *rewritten* (e.g. "Grasp for the green cube." normalised to "Grasp the green cube."). The full list of corrections, with counts for the
  whole build, is `locating_answer_fixes` in `MANIFEST.json`. Among the 960 shipped test locating questions, 80 StackCube questions had the reference "Reach for the cube." rewritten to
  "Reach for the red cube." (the original text is kept in `orig_answer`); per-test counts of dropped annotations were not recorded. The "Grasp for -> Grasp" normalisation concerns task families that are not in the test split.
* Upstream failure-explanation and correction questions (generated with a closed model upstream) are not included.
* **Judge protocol is local.** The official GPT-judge prompt of RoboFAC is not available to us, so `task_identification` (and optionally `task_planning`) is judged with the prompts `P_ID` / `P_PLAN` in
  `score.py`, a local protocol, not the official one. Do not report these numbers as official RoboFAC scores.

Data is hosted on the Hugging Face Hub: `HF_REPO=ACERobotics/ACE-Brain-0.7-Bench python prepare.py --from-hub --sets robofac` downloads it into `data/` and verifies `SHA256SUMS` (local copies work as before).

## Running

```bash
python benchmarks/robofac/prepare.py                      # verify the packaged data (counts + sha256)
python benchmarks/robofac/eval.py --model <HF checkpoint dir> --out results/robofac/mymodel
export JUDGE_BASE_URL=<OpenAI-compatible endpoint> JUDGE_API_KEY=... JUDGE_MODEL=qwen3.8-flash
python benchmarks/robofac/score.py --pred results/robofac/mymodel/predictions.jsonl --json results/robofac/mymodel/score.json
```

* Generation (`config.json`): greedy, thinking off, `max_tokens` 1024, per-image pixels 65,536 .. 16,777,216, 8 frames per question.
* Judge: one user message (text only), temperature 0, `max_tokens` 16, thinking off. Attempt 0 plus up to 3 retries (`--retries`) until the reply starts with Yes/No.
  `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` (default `qwen3.8-flash`), `JUDGE_CONCURRENCY`: see `taskbench/judge.py`. Hosted judges drift and temperature 0 is not bit-exact.
* **Unscored items.** An item whose verdict is still unparsable after the retries is *unscored*: it is neither right nor wrong. The family accuracy (and `fac`) is then `None`, the unscored count is printed prominently,
  and the mean over the scored items only is reported next to it (`scored_only_accuracy`, `fac_scored_only`). (Unlike RoboBench there is no "count as correct" rule here.)
* `--judgments FILE` re-scores from a saved judge journal (jsonl of `{id, verdict, ...}`) without any judge call; `--include-planning` also judges `task_planning`.
  The journal is resumable (`--judgments-out`).

## Reproduction check

Re-scoring the saved predictions and judge journal of the reference model without judge calls reproduces the board values exactly:
fac 69.1764 (task_identification 86.2126, failure_detection 83.3056, failure_identification_mc 58.6458, failure_locating_mc 48.5417; task_planning 78.8206 with `--include-planning`, not in `fac`); all journal prompt hashes matched.

## License

Code: Apache-2.0. Data: MIT (RoboFAC authors), modified, see `data/robofac/LICENSE-NOTICE.md`.
