# SPDX-License-Identifier: Apache-2.0
"""EgoPlan-Bench generation (8 frames per question); see README.md.
  python benchmarks/egoplan/eval.py --model <hf dir> [--backend vllm|hf|openai|mock] [--out DIR] [--data-dir D] [--limit N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.infer import main

if __name__ == "__main__":
    main(Path(__file__).parent)
