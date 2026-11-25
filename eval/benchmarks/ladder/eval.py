# SPDX-License-Identifier: Apache-2.0
"""LADDER generation. Rows carry a per-row "system" prompt (official prompts); see README.md.
  python benchmarks/ladder/eval.py --model <hf dir> [--backend vllm|hf|openai|mock] [--out DIR] [--data-dir D] [--limit N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.infer import main

if __name__ == "__main__":
    main(Path(__file__).parent)
