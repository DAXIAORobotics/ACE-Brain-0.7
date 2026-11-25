# taskbench: task-understanding benchmarks, exact protocol

Evaluation code of the paper *ACE-Brain-0.7: Learning Execution-Grounded Reasoning for Embodied Agents* (ACE-Brain Team). This repository holds the **code only**; the data is hosted separately as the Hugging Face dataset `ACERobotics/ACE-Brain-0.7-Bench` (see Quickstart).

This repository lets you evaluate a vision-language model on the seven task-understanding benchmarks of our paper with exactly the protocol the paper used: same items, same prompts, same decoding settings, same parsers and scorers.

| Benchmark | Items | Metric (0-100 unless noted) | LLM judge | Data (Hugging Face dataset, not in this repo) | Source / license of the data |
|---|---|---|---|---|---|
| EgoPlan-Bench (`egoplan`) | 3,343 | accuracy (4-way, letter parsed from the answer) | no | **not shipped**: download + deterministic build script, verified by sha256 | EgoPlan-Bench (lmms-lab release); video from EPIC-KITCHENS (CC BY-NC 4.0) and Ego4D (data-use agreement) |
| LADDER (`ladder`) | 3,409 queries (145 decomposition, 562 verification, 2,142 state, 560 next sub-goal) | mean of decomposition exact match, verification balanced accuracy, state all-correct, next-subgoal validity | no | **not shipped**: download + deterministic build script, verified by sha256 | LADDER offline VLM benchmark, CC BY 4.0 (the RoboLab-Long frames show RoboLab assets, CC BY-NC-SA 4.0) |
| RoboFAC (`robofac`) | 5,532 test questions (4 families in the average: 4,328) | macro mean of failure detection, failure identification, failure locating, task identification | task identification | test split, 8 frames per question | RoboFAC dataset, MIT |
| Cosmos-Reason1, undistorted (`cosmos`) | 510 questions x 32 frames, 3 seeds | accuracy, mean of 3 seeds | no | questions + generatively rewritten frames | Cosmos-Reason1-Benchmark, CC BY 4.0 (+ RoboVQA Apache-2.0, RoboFail MIT) |
| RoboBench planning, natural-language version (`robobench`) | 3,979 (Q1 1,945, Q2 842, Q3 1,192) | Q1/Q2 judge score 0-100, Q3 yes/no accuracy; mean over all questions | Q1, Q2 | converted questions/references, `rewrite_mapping.json`, rewritten images | RoboBench, CC BY 4.0 (modified) |
| ERIQ (`eriq`) | 3,905 (11 task-understanding subsets) | accuracy | no | questions + images of the 11 subsets | ERIQ, CC BY-SA 4.0 (that folder is share-alike) |
| Progress estimation (`progress`) | 200 (one per episode) | mean absolute error in [0,1] (lower is better) | no | questions, labels and 798 frames (in the HF archive) | frames from public robot datasets; licenses per source (see `LICENSES.md`) |

Paper numbers (ours and the closed-model rows) are in [RESULTS.md](RESULTS.md). Per-benchmark details, including every deviation from the upstream benchmark, are in `benchmarks/<name>/README.md`.

## Before comparing numbers

- Several sets are modified derivatives (RoboBench in natural language, RoboFAC corrected locating answers, rewritten Cosmos/RoboBench images); scores are not comparable with the original leaderboards.
- RoboBench Q1/Q2 and RoboFAC task identification use an LLM judge (paper: `qwen3.8-flash`, the default `JUDGE_MODEL`); re-scores can drift slightly. Unscorable RoboBench judge items count as correct, as in the paper.
- Open-source numbers: vLLM, greedy decoding, thinking disabled, 65,536 to 16,777,216 pixels per image.

## Quickstart

```bash
pip install -e . huggingface_hub          # extras: .[vllm] .[hf] .[build]
HF_REPO=ACERobotics/ACE-Brain-0.7-Bench python prepare.py --from-hub     # robofac robobench eriq cosmos progress (~12 GB; --sets to pick)
bash benchmarks/ladder/download.sh && python benchmarks/ladder/prepare.py   # EgoPlan / LADDER: build from official sources
bash benchmarks/egoplan/download.sh && python benchmarks/egoplan/prepare.py
python benchmarks/eriq/eval.py  --model /path/to/hf_checkpoint --out results/eriq
python benchmarks/eriq/score.py --pred results/eriq/predictions.jsonl
bash scripts/run_all.sh /path/to/hf_checkpoint results/mymodel          # GPUS=8 to shard, LIMIT=5 smoke test
```

API models: `--backend openai --model <name>` with `MODEL_BASE_URL` / `MODEL_API_KEY`. Judge: `JUDGE_BASE_URL`, `JUDGE_API_KEY`, optional `JUDGE_MODEL`, `JUDGE_CONCURRENCY`.

## Licenses

Code: Apache-2.0 (`LICENSE`). Data keeps the license of its source (see `LICENSES.md` and `data/<n>/LICENSE-NOTICE.md`); ERIQ is CC BY-SA 4.0 (share-alike), Cosmos and RoboBench are CC BY 4.0 and modified, RoboFAC is MIT.
