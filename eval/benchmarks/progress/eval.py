#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Progress estimation (200 questions, 4 chronological frames each): greedy, non-thinking, 64 new tokens.

  python benchmarks/progress/eval.py --model <HF dir | api model> [--backend vllm|hf|openai] [--data-dir D] [--limit N]

The frames come with the Hugging Face archive: run `python prepare.py --from-hub --sets progress` first (check with
`python benchmarks/progress/prepare.py --verify`). The mock backend needs no images.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.infer import main

if __name__ == "__main__":
    main(Path(__file__).parent)
