#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RoboFAC (test split, 8-frame rebuild): greedy generation for all 5,532 questions.

  python benchmarks/robofac/eval.py --model <HF dir> [--backend vllm|hf|openai] [--out DIR] [--data-dir DIR] [--limit N]

Writes <out>/predictions.jsonl ({id, pred, finish_reason, n_tokens}); score with benchmarks/robofac/score.py.
Settings (greedy, thinking off, max_tokens 1024, image pixels 65536..16777216) are in config.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.infer import main  # noqa: E402

if __name__ == "__main__":
    main(Path(__file__).parent)
