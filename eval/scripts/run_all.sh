#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run inference + scoring for every benchmark with one model.
#   bash scripts/run_all.sh MODEL_DIR [OUT_DIR]
# Env: BACKEND=vllm|hf|openai (default vllm)   GPUS=N data-parallel shards per benchmark (default 1)
#      LIMIT=N  only the first N items per benchmark (smoke test)   SEEDS=3|1  Cosmos seeds (default 3 = paper)
#      BENCHES="egoplan ladder ..."   subset   PYTHON=python
#      JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL  needed to score RoboBench and RoboFAC (see README)
set -euo pipefail
MODEL=${1:?usage: run_all.sh MODEL_DIR [OUT_DIR]}
OUT=${2:-results/$(basename "$MODEL")}
PY=${PYTHON:-python}; BACKEND=${BACKEND:-vllm}; GPUS=${GPUS:-1}; SEEDS=${SEEDS:-3}
BENCHES=${BENCHES:-"egoplan ladder robofac cosmos robobench eriq progress"}
cd "$(dirname "$0")/.."
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
for b in $BENCHES; do
  d=benchmarks/$b; o=$OUT/$b; mkdir -p "$o"
  echo "=== $b: inference"
  if [ "$b" = cosmos ]; then   # one vLLM process per seed, handled inside eval.py
    $PY $d/eval.py --model "$MODEL" --out "$o" --seeds "$SEEDS" "${LIM[@]}"
    PRED=("$o"/*/predictions.jsonl)
  else
    if [ "$GPUS" -gt 1 ] && [ "$BACKEND" != openai ]; then
      for i in $(seq 0 $((GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$i $PY $d/eval.py --model "$MODEL" --backend "$BACKEND" --out "$o" --shard $i/$GPUS "${LIM[@]}" > "$o/infer.$i.log" 2>&1 &
      done
      wait
      cat "$o"/predictions.shard*of*.jsonl > "$o/predictions.jsonl"
    else
      $PY $d/eval.py --model "$MODEL" --backend "$BACKEND" --out "$o" "${LIM[@]}"
    fi
    PRED=("$o/predictions.jsonl")
  fi
  echo "=== $b: scoring"
  PART=(); [ -n "${LIMIT:-}" ] && grep -q allow-partial $d/score.py && PART=(--allow-partial)
  $PY $d/score.py --pred "${PRED[@]}" --json "$o/score.json" "${PART[@]}" || echo "scoring of $b failed (judge env set?)"
done
echo "done: per-benchmark scores in $OUT/*/score.json"
