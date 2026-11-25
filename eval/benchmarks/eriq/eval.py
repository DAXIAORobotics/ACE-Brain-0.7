#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ERIQ (11 task-understanding subsets, 3,905 questions): greedy, non-thinking, 1024 new tokens, pixels 65536..16777216.

  python benchmarks/eriq/eval.py --model <HF dir | api model> [--backend vllm|hf|openai] [--shard I/N] [--limit N]

Run one process per GPU with --shard i/N and score the shard files together (score.py accepts several --pred).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.infer import main

if __name__ == "__main__":
    main(Path(__file__).parent)
