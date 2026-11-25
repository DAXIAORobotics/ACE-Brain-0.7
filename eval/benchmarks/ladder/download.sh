#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the official LADDER sources at the revisions used for the paper numbers. Nothing is redistributed by this repo.
#   usage: bash download.sh [DEST]      (default DEST=data/ladder/raw; ~330 MB of parquet)
# Needs: git, and `huggingface-cli` (pip install huggingface_hub).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; SUB=data
DEST="${1:-$ROOT/$SUB/ladder/raw}"
HF_REPO="INSAIT-Institute/LADDER-Offline-VLM-Benchmark"
HF_REV="1a1cb958d9e0e47d4b4497bb79a594dc66a3be2a"
GH_REPO="https://github.com/insait-institute/ladder"
GH_REV="8703b9d02bfe5e20b7d544bd5ecd71234586246a"
mkdir -p "$DEST"
huggingface-cli download "$HF_REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$DEST/hf" \
  --include "data/decomposition/*" "data/subtask_states/*" "README.md"
if [ ! -d "$DEST/repo/.git" ]; then git clone --no-checkout "$GH_REPO" "$DEST/repo"; fi
git -C "$DEST/repo" checkout --detach "$GH_REV"
echo "parquet: $DEST/hf/data/{decomposition,subtask_states}/train-00000-of-00001.parquet"
echo "repo:    $DEST/repo @ $(git -C "$DEST/repo" rev-parse HEAD)"
echo "next:    python benchmarks/ladder/prepare.py --raw-dir $DEST --out data/ladder"
