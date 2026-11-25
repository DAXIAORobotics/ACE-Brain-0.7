"""Generic inference driver for LLaVA-style item files. Benchmark eval.py scripts are thin wrappers around `main`.

  python benchmarks/<name>/eval.py --model <hf dir | api model> [--backend vllm|hf|openai] [--out DIR]
                                   [--data-dir DIR] [--limit N] [--shard I/N] [--max-tokens N] [--thinking]

Writes <out>/predictions.jsonl with one record per item: {"id", "pred", "finish_reason", "n_tokens"}.
Resumable: ids already present are skipped. Generation settings live in benchmarks/<name>/config.json.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

from .backends import GenConfig, make_backend
from .io import item_messages, read_jsonl

REPO = Path(__file__).resolve().parent.parent


def data_dir(name, override=None):
    base = Path(override or os.environ.get("TASKBENCH_DATA") or REPO / "data")
    return base / name


def load_config(bench_dir):
    return json.load(open(Path(bench_dir) / "config.json"))


def gen_from(cfg, args):
    g = dict(cfg.get("generation", {}))
    if getattr(args, "max_tokens", None):
        g["max_tokens"] = args.max_tokens
    if getattr(args, "thinking", False):
        g["thinking"] = True
    return GenConfig(**g)


def common_args(ap):
    ap.add_argument("--model", required=True, help="local HF checkpoint dir (vllm/hf) or API model name (openai)")
    ap.add_argument("--backend", choices=["vllm", "hf", "openai", "mock"], default="vllm")
    ap.add_argument("--out", default=None, help="output dir (default results/<bench>/<model basename>)")
    ap.add_argument("--data-dir", default=None, help="root holding data/<bench>/ (default $TASKBENCH_DATA or ./data)")
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke tests)")
    ap.add_argument("--shard", default=None, help="I/N: handle items[I::N] (run one process per GPU)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--thinking", action="store_true", help="enable the chat template's thinking mode (paper: off)")
    return ap


def infer_rows(rows, root, backend, out_path, limit=0, shard=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if limit:
        rows = rows[:limit]
    if shard:
        i, n = map(int, shard.split("/")); rows = rows[i::n]
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path) if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} items, {len(done)} already done, {len(todo)} to run", flush=True)
    B = 256
    with open(out_path, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), B):
            chunk = todo[s:s + B]
            res = backend.generate([item_messages(r, root) for r in chunk])
            if len(res) != len(chunk):
                raise RuntimeError("backend returned a different number of results")
            for r, o in zip(chunk, res):
                f.write(json.dumps({"id": r["id"], **o}, ensure_ascii=False) + "\n")
            f.flush()
    return out_path


def main(bench_dir, items_name=None):
    """Standard eval.py entry for benchmarks whose data/<bench>/<items file> is in LLaVA item format."""
    bench_dir = Path(bench_dir)
    cfg = load_config(bench_dir)
    ap = common_args(argparse.ArgumentParser(description=cfg.get("benchmark")))
    a = ap.parse_args()
    d = data_dir(cfg["data_subdir"], a.data_dir)
    rows = read_jsonl(d / (items_name or cfg["items"]))
    gen = gen_from(cfg, a)
    out = Path(a.out or REPO / "results" / cfg["benchmark"] / Path(a.model.rstrip("/")).name)
    be = make_backend(a.backend, a.model, gen)
    infer_rows(rows, d / cfg.get("image_root", "."), be, out / (f"predictions.shard{a.shard.replace('/', 'of')}.jsonl" if a.shard else "predictions.jsonl"), a.limit, a.shard)
    print("wrote", out)
