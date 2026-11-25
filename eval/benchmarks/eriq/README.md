# ERIQ (task-understanding subsets)

3,905 multiple-choice / yes-no questions from the 11 task-understanding subsets of ERIQ (Embodied Reasoning Intelligence Quotient;
arXiv 2512.24125, Hugging Face `KineMind/ERIQ`). The 4 spatial-perception subsets (2,147 questions) are excluded. Metric: micro
accuracy in percent over the 3,905 questions (higher is better); the subsets are not reported separately in the headline.

| dimension | subsets | n |
|---|---|---|
| planning_monitoring | ACTION_UNDERSTANDING 600, SUCCESS_DETECTION 800, SUBTASK_PLANNING 600, FINE_GRAINED_PLAN 112, TRAJ_UNDERSTANDING 505, TASK_PROGRESS 281 | 2,898 |
| human_intent | HUMAN_INTENTION 225, HUMAN_INTERACTION 227 | 452 |
| error_detect_recover | MISTAKE_EXISTENCE 334, MISTAKE_CLASSIFY 116, MISTAKE_RECOVERY 105 | 555 |

## Data

`data/eriq/items.jsonl` (LLaVA-style rows, relative image paths), `data/eriq/images/` (10,441 JPEGs, 3,138,092,008 bytes; up to 10
images per question), `data/eriq/subsets_11.json`, `data/eriq/SHA256SUMS`. **The data folder is CC BY-SA 4.0 (share-alike), with
attribution to the dataset authors: see `data/eriq/LICENSE-NOTICE.md`.** To rebuild from the original download:
`python benchmarks/eriq/prepare.py --download <dir> --out <out>` (needs `huggingface_hub`) or `--src <local copy>`.

Data is hosted on the Hugging Face Hub: `HF_REPO=ACERobotics/ACE-Brain-0.7-Bench python prepare.py --from-hub --sets eriq` downloads it into `data/` and verifies `SHA256SUMS` (local copies work as before).

## Run

```bash
python benchmarks/eriq/eval.py --model /path/to/hf_checkpoint --out results/eriq/mymodel                  # all 3,905
python benchmarks/eriq/eval.py --model /path/to/hf_checkpoint --shard 0/4 --out results/eriq/mymodel      # one process per GPU, i = 0..3
python benchmarks/eriq/score.py --pred results/eriq/mymodel --json results/eriq/mymodel/score.json       # accepts shard files / a directory
```

Settings (`config.json`): greedy, thinking off (chat template `enable_thinking=False`), max 1024 new tokens, image pixels
65,536..16,777,216 per image, `max_model_len` 32768, up to 10 images per prompt, no system prompt. The prompt is the dataset's
human turn plus one fixed answer-format line (the dataset defines none).

## Scoring

`score.py` is the rule-based scorer used for the paper numbers: a reply is parsed to a letter (or Yes/No for MISTAKE_EXISTENCE):
leading letter, a lone final-line letter, the last "answer/option/choice is X", else the single distinct standalone letter.
Missing, empty or unparseable replies are wrong and reported. It also prints per-dimension and per-subset accuracy, macro over
subsets, and the random / always-A baselines (39.31 / 31.86).

## Reproduction check

The four saved prediction shards of the paper model, scored with the release `score.py` against the packaged `items.jsonl`:
**85.07** micro (n = 3,905, 0 failed, 0 unparsed), equal to the paper table value 85.07 (macro over subsets 81.56).
The packaged `items.jsonl` is identical to the items file used for that run.
