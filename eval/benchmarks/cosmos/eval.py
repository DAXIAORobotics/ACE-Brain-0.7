#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Reason1 benchmark (undistorted frames), self-contained port of the paper protocol.

  python benchmarks/cosmos/eval.py --model <HF checkpoint dir> [--data-dir DIR] [--out DIR]
                                   [--seeds 3407,3408,3409 | --seeds 1] [--limit N]

Protocol (EmbodiedEvalKit eval_cosmos.py, logic vendored under vendor/): each question = 32 frames passed as one
video through qwen_vl_utils, system prompt "You are a helpful assistant.", prompt = question + choices + fixed answer
sentence; vLLM greedy (temperature 0, top_p 1, top_k -1), repetition_penalty 1.05, max_tokens 4096, max_model_len 20000,
one engine per seed (3407, 3408, 3409; the three repeats of the 510 questions = 1,530 inferences). Qwen3.5 checkpoints
(model_type qwen3_5) are rendered with enable_thinking=False, i.e. with the empty think block.
`--seeds 1` is the quick mode (first seed only, 510 inferences); `--limit N` keeps the first N questions.
Each seed runs in its own process (as in the paper runs). Output: <out>/<seed>/predictions.jsonl with
{id, pred, finish_reason, n_tokens}. Needs a GPU, vllm, transformers, qwen_vl_utils, pyarrow, pillow; the parquet
frames of one chunk (default 16 questions) are decoded at a time, so RAM stays small.
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from taskbench.infer import data_dir, load_config  # noqa: E402


def parse_seeds(spec, default):
    """'3407,3408' -> [3407, 3408]; a bare small integer N (e.g. '1') -> the first N default seeds."""
    parts = [p for p in str(spec).split(",") if p.strip()]
    if len(parts) == 1 and int(parts[0]) < 1000:
        return list(default[:int(parts[0])])
    return [int(p) for p in parts]


def iter_questions(parquet_dir, limit=0):
    """Yield dict(id, frames_bytes, question, choices, answer) in shard order, 32 JPEG/PNG byte strings per question."""
    import pyarrow.parquet as pq
    files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no *.parquet shards in {parquet_dir}")
    n = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=4):
            for r in batch.to_pylist():
                yield {"id": r["id"], "frames": [v["bytes"] for v in r["videos"]], "question": r["question"],
                       "choices": r["choices"], "answer": r["answer"]}
                n += 1
                if limit and n >= limit:
                    return


def run_seed(a, cfg, seed):
    import io
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM
    from vendor.eek_cosmos import build_prompt
    from vendor.eek_vllm_input import make_chat_template_fn, make_sampling_params, prepare_messages, prepare_vllm_input
    g = cfg["generation"]
    model_type = json.load(open(Path(a.model) / "config.json")).get("model_type")
    processor = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=g["tensor_parallel_size"], gpu_memory_utilization=g["gpu_memory_utilization"],
              trust_remote_code=True, max_model_len=g["max_model_len"],
              limit_mm_per_prompt={"image": g["max_images"], "video": g["max_videos"]}, seed=seed)
    render = make_chat_template_fn(processor, model_type)
    sp = make_sampling_params(g["max_tokens"], g["temperature"], g["top_p"], g["top_k"], g["repetition_penalty"], g["presence_penalty"])
    out = Path(a.out) / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {json.loads(l)["id"] for l in open(pred_path) if l.strip()} if pred_path.exists() else set()
    chunk, B = [], g["batch_size"]

    def flush(f):
        if not chunk:
            return
        inputs = []
        for q in chunk:
            frames = [Image.open(io.BytesIO(b)) for b in q["frames"]]
            for im in frames:
                im.load()
            sample = {"question": build_prompt(q["question"], q["choices"]), "video": [frames]}
            inputs.append(prepare_vllm_input(prepare_messages(sample), render))
        res = llm.generate(inputs, sampling_params=sp)
        for q, o in zip(chunk, res):
            c = o.outputs[0]
            f.write(json.dumps({"id": q["id"], "pred": c.text, "finish_reason": c.finish_reason,
                                "n_tokens": len(c.token_ids)}, ensure_ascii=False) + "\n")
        f.flush()
        print(f"INFERENCE_PROGRESS seed={seed} +{len(chunk)}", flush=True)
        chunk.clear()

    with open(pred_path, "a", encoding="utf-8") as f:
        for q in iter_questions(data_dir(cfg["data_subdir"], a.data_dir) / "data", a.limit):
            if q["id"] in done:
                continue
            chunk.append(q)
            if len(chunk) >= B:
                flush(f)
        flush(f)
    print("rendered prompts with empty think block:", render.seen["n"], flush=True)


def main():
    cfg = load_config(HERE)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local HF checkpoint directory")
    ap.add_argument("--data-dir", default=None, help="root holding data/cosmos/ (default $TASKBENCH_DATA or ./data)")
    ap.add_argument("--out", default=None, help="output dir (default results/cosmos/<model basename>)")
    ap.add_argument("--seeds", default="3407,3408,3409", help="comma list of seeds, or a count N (first N default seeds; 1 = quick mode)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds, cfg["seeds"])
    a.out = a.out or str(REPO / "results" / "cosmos" / Path(a.model.rstrip("/")).name)
    if a.worker:
        run_seed(a, cfg, seeds[0])
        return
    for s in seeds:  # one process (one vLLM engine) per seed
        cmd = [sys.executable, __file__, "--model", a.model, "--out", a.out, "--seeds", str(s), "--limit", str(a.limit), "--worker"]
        if a.data_dir:
            cmd += ["--data-dir", a.data_dir]
        print("RUN seed", s, flush=True)
        subprocess.run(cmd, check=True)
    print("wrote", a.out, "- score with: python benchmarks/cosmos/score.py --pred", a.out)


if __name__ == "__main__":
    main()
