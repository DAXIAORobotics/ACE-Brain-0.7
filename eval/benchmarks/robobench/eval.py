#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RoboBench (natural-language version): greedy generation for the 3,979 eligible items.

  python benchmarks/robobench/eval.py --model <HF dir> [--backend vllm|hf|openai] [--out DIR] [--data-dir DIR] [--limit N]

The 28 Q1 items whose converted reference is unverified (answer null) are skipped. Questions and the optional system prompt
come from data/robobench/items.jsonl; images are passed in order, one `<image>` tag per image. Writes
<out>/predictions.jsonl ({id, pred, finish_reason, n_tokens}); score with benchmarks/robobench/score.py.
Settings (greedy, thinking off, max_tokens 1024, per-image pixels 65536..16777216) are in config.json.
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from taskbench.backends import make_backend  # noqa: E402
from taskbench.infer import REPO, common_args, data_dir, gen_from, infer_rows, load_config  # noqa: E402
from taskbench.io import read_jsonl  # noqa: E402


def eligible(rows):
    return [r for r in rows if r.get("answer") is not None and r.get("answer_status") != "needs_review"]


def main():
    cfg = load_config(HERE)
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args()
    d = data_dir(cfg["data_subdir"], a.data_dir)
    rows = eligible(read_jsonl(d / cfg["items"]))
    if len(rows) != cfg["eligible_items"] and not a.limit:
        print(f"WARNING: {len(rows)} eligible items, expected {cfg['eligible_items']}", file=sys.stderr)
    out = Path(a.out or REPO / "results" / cfg["benchmark"] / Path(a.model.rstrip("/")).name)
    name = f"predictions.shard{a.shard.replace('/', 'of')}.jsonl" if a.shard else "predictions.jsonl"
    infer_rows(rows, d, make_backend(a.backend, a.model, gen_from(cfg, a)), out / name, a.limit, a.shard)
    print("wrote", out)


if __name__ == "__main__":
    main()
