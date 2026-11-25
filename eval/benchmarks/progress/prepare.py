#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify data/progress/images/ against the frame manifest (or rebuild it from frames you supply).

The 798 frames ship inside the Hugging Face archive progress.tar.gz: `python prepare.py --from-hub --sets progress` extracts them to
data/progress/images/. Use this script with --verify to check them; --images-dir is only a fallback to rebuild the folder from your own copy.

The 200 questions use 798 distinct frames (four per question, 448x448 and similar, encoded JPEG) taken from public robot
datasets (see data/progress/image_manifest.jsonl: fields source, camera, episode, frame_id; sources are Table30 tasks,
BridgeData V2, RT-1/fractal, LIBERO, DobbE, FMB, ToTo, Berkeley AutoLab UR5 and two tabletop collections). The file names
in questions.jsonl are the sha256 of the original encoded bytes, so a rebuilt file only counts as the exact benchmark input
when its sha256 matches. Frames re-extracted from videos are re-encoded and will generally NOT match byte for byte; this
script then refuses to continue unless --allow-hash-mismatch is given, and scores obtained that way are not comparable.

  python benchmarks/progress/prepare.py --images-dir <dir> [--data-dir D] [--allow-hash-mismatch]
  python benchmarks/progress/prepare.py --verify [--data-dir D]       # check an existing data/progress/images/

For every manifest entry the script looks in --images-dir for (first hit wins):
  <sha256>.jpg | images/<sha256>.jpg | <source>/<camera>/<episode>/<frame_id:06d>.jpg (or .png)
and copies it to <data-dir>/progress/images/<sha256>.jpg after checking the sha256.
"""
import argparse, json, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from taskbench.io import read_jsonl, sha256_file  # noqa: E402


def candidates(root, e):
    sha = e["sha256"]
    yield root / f"{sha}.jpg"
    yield root / "images" / f"{sha}.jpg"
    for ext in ("jpg", "jpeg", "png"):
        yield root / e["source"] / e["camera"] / e["episode"] / f"{e['frame_id']:06d}.{ext}"


def unique_entries(manifest):
    seen, out = set(), []
    for e in manifest:
        if e["path"] not in seen:
            seen.add(e["path"]); out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", help="directory with the frames you obtained from the original datasets")
    ap.add_argument("--data-dir", default=None, help="root holding progress/ (default ./data)")
    ap.add_argument("--verify", action="store_true", help="only verify data/progress/images against the manifest")
    ap.add_argument("--allow-hash-mismatch", action="store_true", help="copy frames whose sha256 differs (scores then not comparable)")
    a = ap.parse_args()
    d = Path(a.data_dir or REPO / "data") / "progress"
    entries = unique_entries(read_jsonl(d / "image_manifest.jsonl"))
    out = d / "images"
    ok = missing = mismatch = 0
    for e in entries:
        dst = d / e["path"]
        if a.verify:
            if not dst.exists(): missing += 1
            elif sha256_file(dst) == e["sha256"]: ok += 1
            else: mismatch += 1
            continue
        if not a.images_dir:
            ap.error("give --images-dir (or --verify)")
        src = next((c for c in candidates(Path(a.images_dir), e) if c.is_file()), None)
        if src is None:
            missing += 1; continue
        good = sha256_file(src) == e["sha256"]
        if not good and not a.allow_hash_mismatch:
            mismatch += 1; continue
        if not good:
            mismatch += 1
        else:
            ok += 1
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"{len(entries)} frames: sha256-verified {ok}, missing {missing}, sha256 mismatch {mismatch}")
    if missing or (mismatch and not a.allow_hash_mismatch):
        print("incomplete: the benchmark inputs are exact only when every frame is present and verified", file=sys.stderr)
        sys.exit(1)
    if mismatch:
        print("WARNING: some frames do not match the original bytes; results are NOT comparable with published numbers", file=sys.stderr)


if __name__ == "__main__":
    main()
