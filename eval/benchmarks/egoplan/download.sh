#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the official EgoPlan-Bench sources. Nothing is redistributed by this repository.
#   usage: bash download.sh [DEST] [--with-clips]      (default DEST=data/egoplan/raw)
#   default            validation parquet + dataset card (small) and the official GitHub repo at the pinned commit
#   --with-clips       also the per-sample mp4 zip shards (roughly 120 GB; you may instead extract only what you need)
# The Hugging Face dataset may be gated: log in (`huggingface-cli login`) and accept the terms on the dataset page first.
# The underlying videos are EPIC-KITCHENS-100 (CC BY-NC 4.0) and Ego4D (Ego4D license agreement); check that your use is allowed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; SUB=data
DEST="${1:-$ROOT/$SUB/egoplan/raw}"
WITH_CLIPS="${2:-}"
HF_REPO="lmms-lab/EgoPlan-Bench"
GH_REPO="https://github.com/ChenYi99/EgoPlan"
GH_REV="1510afc12bc832b50d6285d83503ecbdca093bf4"
mkdir -p "$DEST"
if [ "$WITH_CLIPS" = "--with-clips" ]; then
  huggingface-cli download "$HF_REPO" --repo-type dataset --local-dir "$DEST/hf"
else
  huggingface-cli download "$HF_REPO" --repo-type dataset --local-dir "$DEST/hf" --include "*.parquet" "*.md" "*.json"
fi
if [ ! -d "$DEST/EgoPlan/.git" ]; then git clone --no-checkout "$GH_REPO" "$DEST/EgoPlan"; fi
git -C "$DEST/EgoPlan" checkout --detach "$GH_REV"
echo "parquet dir: $DEST/hf     zips: $DEST/hf (when --with-clips)"
echo "next: python benchmarks/egoplan/prepare.py --parquet $DEST/hf --zips-dir $DEST/hf --out data/egoplan"
