#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the packaged RoboFAC test data (nothing is downloaded; the data ships in data/robofac/).

  python benchmarks/robofac/prepare.py [--data-dir DIR] [--no-hash]

Checks: question count per family, 8 existing frames per question, every shipped image listed in SHA256SUMS (sha256 verified
unless --no-hash), and that questions.jsonl hash equals the one recorded in MANIFEST.json.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.infer import data_dir  # noqa: E402
from taskbench.io import read_jsonl, sha256_file  # noqa: E402

EXPECT = {"task_identification": 1204, "task_planning": 1204, "failure_detection": 1204, "failure_identification_mc": 960, "failure_locating_mc": 960}


def verify(d, hashes=True):
    d = Path(d)
    rows = read_jsonl(d / "test.jsonl")
    errs = []
    if dict(Counter(r["family"] for r in rows)) != EXPECT:
        errs.append(f"family counts {dict(Counter(r['family'] for r in rows))} != {EXPECT}")
    man = json.load(open(d / "MANIFEST.json"))
    if sha256_file(d / "test.jsonl") != man["shipped_files"]["questions_sha256"]:
        errs.append("test.jsonl hash differs from MANIFEST.json")
    sums = {}
    for line in open(d / "SHA256SUMS"):
        h, p = line.rstrip("\n").split("  ", 1)
        sums[p] = h
    imgs = sorted({i for r in rows for i in r["image"]})
    if any(len(r["image"]) != 8 for r in rows):
        errs.append("a question does not have 8 frames")
    for i in imgs:
        if i not in sums:
            errs.append("not in SHA256SUMS: " + i)
        elif not (d / i).is_file():
            errs.append("missing image " + i)
        elif hashes and sha256_file(d / i) != sums[i]:
            errs.append("hash mismatch " + i)
    return len(rows), len(imgs), errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()
    n, ni, errs = verify(data_dir("robofac", a.data_dir), not a.no_hash)
    print(f"robofac: {n} questions, {ni} images,", "OK" if not errs else f"{len(errs)} problems")
    for e in errs[:20]:
        print(" ", e)
    sys.exit(1 if errs else 0)
