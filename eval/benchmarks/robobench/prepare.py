#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the packaged RoboBench (natural-language) data (nothing is downloaded; the data ships in data/robobench/).

  python benchmarks/robobench/prepare.py [--data-dir DIR] [--no-hash]

Checks: 4,007 items (Q1 1,973 / Q2 842 / Q3 1,192), 3,979 eligible, every referenced image present and listed in SHA256SUMS
(sha256 verified unless --no-hash), a gt DAG for every eligible Q1, and the `<image>` tag count of every question.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.infer import data_dir  # noqa: E402
from taskbench.io import read_jsonl, sha256_file  # noqa: E402


def verify(d, hashes=True):
    d = Path(d)
    rows = read_jsonl(d / "items.jsonl")
    errs = []
    c = Counter(r["split"] for r in rows)
    if dict(c) != {"Q1": 1973, "Q2": 842, "Q3": 1192}:
        errs.append(f"split counts {dict(c)}")
    elig = [r for r in rows if r["answer"] is not None]
    if len(elig) != 3979:
        errs.append(f"{len(elig)} eligible items != 3979")
    dags = json.load(open(d / "q1_dags.json"))
    for r in elig:
        if r["split"] == "Q1" and r["id"] not in dags:
            errs.append("no DAG for " + r["id"])
    sums = {}
    for line in open(d / "SHA256SUMS"):
        h, p = line.rstrip("\n").split("  ", 1)
        sums[p] = h
    imgs = sorted({i for r in rows for i in r["image"]})
    for r in rows:
        if r["conversations"][0]["value"].count("<image>") != len(r["image"]):
            errs.append("image tag mismatch " + r["id"])
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
    n, ni, errs = verify(data_dir("robobench", a.data_dir), not a.no_hash)
    print(f"robobench: {n} items, {ni} images,", "OK" if not errs else f"{len(errs)} problems")
    for e in errs[:20]:
        print(" ", e)
    sys.exit(1 if errs else 0)
