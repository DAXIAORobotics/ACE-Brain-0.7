# Results

Scores only. These are the numbers reported in the paper, produced with exactly the protocol in this repository (greedy decoding, thinking off, one run; Cosmos is the mean of 3 seeds). RoboBench and RoboFAC task-identification use an LLM judge (see `README.md`), so a re-run with a different judge back-end can move them slightly. Per-item predictions and judgments are not shipped.

RoboBench here is the natural-language version (Q1/Q2 plans written as numbered natural-language steps, 3,979 questions); the numbers are **not comparable** with the original RoboBench leaderboard. Cosmos-Reason1 uses the undistorted images (generative rewrite, see `benchmarks/cosmos/README.md`). Avg6 is the unweighted mean of the six benchmark columns.

| Model | EgoPlan | Cosmos-Reason1 | RoboBench (NL) | LADDER | RoboFAC | ERIQ | Avg6 | Progress MAE (lower is better) |
|---|---|---|---|---|---|---|---|---|
| Ours (Qwen3.5-9B fine-tuned for task understanding) | 74.5 | 83.5 | 67.4 | 79.0 | 69.2 | 85.1 | 76.5 | 0.087 |
| **Closed models** (as reported in the paper) | | | | | | | | |
| GPT-6 Astra | 75.9 | 77.5 | 66.0 | 75.7 | 53.2 | 87.9 | 72.7 | 0.125 |
| Claude Opus 5.5 | 70.1 | 76.5 | 68.9 | 79.5 | 57.7 | 88.4 | 73.5 | 0.118 |
| **General-purpose open models** | | | | | | | | |
| Qwen3.5-9B (base) | 48.8 | 62.0 | 55.6 | 68.8 | 35.9 | 71.4 | 57.1 | 0.257 |
| Qwen3.5-27B | 56.1 | 66.7 | 60.2 | 74.3 | 43.7 | 77.6 | 63.1 | 0.215 |
| **Open embodied models** | | | | | | | | |
| RoboBrain 2.5-8B | 44.6 | 57.6 | 55.2 | 64.2 | 42.7 | 73.0 | 56.2 | 0.190 |
| RynnBrain 1.1-9B | 56.1 | 63.7 | 40.0 | 62.3 | 40.9 | 74.3 | 56.2 | 0.324 |
| SenseNova-SI 1.1-8B | 40.1 | 52.2 | 52.9 | 68.9 | 44.8 | 67.4 | 54.4 | 0.216 |
| PhysBrain 1.5-8B | 66.9 | 72.7 | 51.4 | 48.5 | 38.6 | 72.1 | 58.4 | 0.218 |
| Embodied-R1.5-8B | 60.5 | 68.6 | 50.4 | 59.6 | 43.6 | 67.6 | 58.4 | 0.280 |
| ACE-Brain-0.5-8B | 31.1 | 47.3 | 45.7 | 45.4 | 41.5 | 61.4 | 45.4 | 0.333 |
| Cosmos-Reason1-7B | 33.3 | 61.7 | 52.4 | 50.9 | 38.1 | 63.9 | 50.1 | 0.274 |
| MiMo-Embodied-7B | 42.1 | 60.3 | 54.7 | 70.2 | 34.7 | 67.2 | 54.9 | 0.239 |
| Hy-Embodied-VLM-1.0 | 41.1 | 59.5 | 54.6 | 64.2 | 55.3 | 68.4 | 57.2 | 0.190 |
| RynnBrain-30B-A3B | 57.1 | 64.6 | 31.6 | 63.0 | 45.5 | 68.7 | 55.1 | 0.327 |
| RoboBrain 2.0-32B | 51.4 | 53.5 | 56.1 | 65.0 | 39.5 | 67.8 | 55.6 | 0.267 |
| Cosmos-Reason2-32B | 44.3 | 73.9 | 55.0 | 60.0 | 48.2 | 71.9 | 58.9 | 0.220 |

## Closed-model rows

The GPT-6 Astra and Claude Opus 5.5 rows were computed once with the same prompts and scorers (open-source models: letter extracted from the end of the answer; closed models: the same scorers, with the EgoPlan letter extraction of the public lmms-eval toolkit) and are reproduced here as reported. They are not re-scored by this repository because that would require access to those APIs; `taskbench/backends.py` provides an OpenAI-compatible client if you want to re-run them.
