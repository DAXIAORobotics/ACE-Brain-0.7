#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Top-level data setup.

  HF_REPO=ACERobotics/ACE-Brain-0.7-Bench python prepare.py --from-hub [--sets robofac eriq ...] [--data-dir DIR]

Downloads the per-subset archives (RoboFAC test, RoboBench NL, ERIQ, Cosmos undistorted, progress; optionally the EgoPlan /
LADDER id lists via --sets egoplan ladder) from the Hugging Face dataset repo named by HF_REPO (or --repo), checks each
archive against the repo's SHA256SUMS, extracts it into data/ (dataset/<Subset>/... -> data/<benchmark>/...) and verifies
every extracted file against the folder's own SHA256SUMS.
Without --from-hub it only verifies the local copy in data/ (or --data-dir / $TASKBENCH_DATA).
EgoPlan and LADDER data are NOT hosted (only their id lists): see benchmarks/egoplan and benchmarks/ladder.
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taskbench.data_hub import ARCHIVES, PACKAGED, download, verify_sums  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-hub", action="store_true", help="download from the HF dataset repo in HF_REPO first")
    ap.add_argument("--repo", default=None, help="override HF_REPO (default ACERobotics/ACE-Brain-0.7-Bench)")
    ap.add_argument("--sets", nargs="+", default=list(PACKAGED), choices=sorted(ARCHIVES))
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    if a.from_hub:
        res = download(a.repo, a.sets, a.data_dir)
    else:
        base = Path(a.data_dir or os.environ.get("TASKBENCH_DATA") or Path(__file__).resolve().parent / "data")
        res = {s: verify_sums(base / s) for s in a.sets}
    bad = 0
    for s, errs in res.items():
        print(f"{s}: {'OK' if not errs else str(len(errs)) + ' problems'}")
        for e in errs[:10]:
            print("  ", e)
        bad += bool(errs)
    sys.exit(1 if bad else 0)
