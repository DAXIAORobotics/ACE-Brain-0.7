#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score Cosmos-Reason1 (undistorted) predictions: per-seed accuracy with the EmbodiedEvalKit answer extractor, then the
mean over seeds (the board number). Also per-domain accuracy and the count of replies with no letter at all.

  python benchmarks/cosmos/score.py --pred results/cosmos/<model> [--data-dir DIR] [--json out.json]

--pred accepts, repeatable / in any mix:
  * a run directory containing one sub-directory per seed, each with predictions.jsonl ({id, pred}) (output of eval.py),
  * a single predictions.jsonl (one seed),
  * a saved EmbodiedEvalKit result file (.json with results[].question_id / raw_output / ground_truth), one per seed.
Gold answers come from the parquet shards in <data-dir>/cosmos/data (id, answer); without them the ground_truth stored in
EmbodiedEvalKit result files is used. Questions without a prediction count as wrong (and are reported).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from vendor.eek_cosmos import domain_of, extract_answer_from_text  # noqa: E402


def load_gold(data_dir):
    import pyarrow.parquet as pq
    gold = {}
    for fp in sorted((Path(data_dir) / "data").glob("*.parquet")):
        t = pq.read_table(fp, columns=["id", "answer"])
        gold.update(dict(zip(t.column("id").to_pylist(), t.column("answer").to_pylist())))
    return gold


def read_pred_file(path):
    """-> (name, {id: raw text}, {id: gold or None})."""
    path = Path(path)
    if path.suffix == ".json":
        d = json.load(open(path, encoding="utf-8"))
        rows = d["results"]
        return (path.parent.parent.parent.name if path.name.startswith("COSMOS") else path.stem,
                {r["question_id"]: r.get("raw_output", "") for r in rows},
                {r["question_id"]: r.get("ground_truth") for r in rows})
    preds = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["id"] in preds:
                raise ValueError(f"duplicate id {r['id']} in {path}")
            preds[r["id"]] = r.get("pred", "")
    return path.parent.name, preds, {}


def collect_runs(paths):
    runs = []
    for p in map(Path, paths):
        if p.is_dir():
            subs = sorted(x for x in p.iterdir() if x.is_dir() and (x / "predictions.jsonl").exists())
            if subs:
                runs += [read_pred_file(x / "predictions.jsonl") for x in subs]
            elif (p / "predictions.jsonl").exists():
                runs.append(read_pred_file(p / "predictions.jsonl"))
            else:
                raise SystemExit(f"{p}: no seed sub-directories with predictions.jsonl")
        else:
            runs.append(read_pred_file(p))
    return runs


def score_seed(preds, gold, allow_partial=False):
    ids = [i for i in gold if i in preds] if allow_partial else list(gold)
    rows, dom = [], defaultdict(list)
    for i in ids:
        pc = extract_answer_from_text(preds[i]) if i in preds else None
        ok = 1.0 if (i in preds and pc == gold[i]) else 0.0
        rows.append((i, pc, ok))
        dom[domain_of(i)].append(ok)
    n = len(rows)
    return dict(n=n, acc=100.0 * sum(r[2] for r in rows) / n if n else None,
                no_letter=sum(1 for i, pc, _ in rows if i in preds and pc is None),
                missing=sum(1 for i in ids if i not in preds),
                domain={d: dict(n=len(v), acc=100.0 * sum(v) / len(v)) for d, v in sorted(dom.items())})


def score_runs(runs, gold, allow_partial=False):
    seeds = []
    for name, preds, _ in runs:
        s = score_seed(preds, gold, allow_partial)
        s["name"] = name
        seeds.append(s)
    out = dict(benchmark="cosmos", n_seeds=len(seeds), seeds=seeds,
               mean_acc=sum(s["acc"] for s in seeds) / len(seeds) if seeds else None)
    doms = sorted({d for s in seeds for d in s["domain"]})
    out["domain_mean"] = {d: sum(s["domain"][d]["acc"] for s in seeds if d in s["domain"]) / sum(1 for s in seeds if d in s["domain"]) for d in doms}
    out["no_letter_total"] = sum(s["no_letter"] for s in seeds)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--data-dir", default=None, help="root holding cosmos/ (default ./data)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="score only the questions that have a prediction (smoke runs)")
    a = ap.parse_args()
    runs = collect_runs(a.pred)
    gold, src = {}, None
    base = Path(a.data_dir) if a.data_dir else REPO / "data"
    if (base / "cosmos" / "data").is_dir():
        gold, src = load_gold(base / "cosmos"), "parquet"
    else:
        for _, _, g in runs:
            gold.update({k: v for k, v in g.items() if v is not None})
        src = "ground_truth stored in the result files (no parquet found)"
        if not gold:
            raise SystemExit("no gold answers: give --data-dir with cosmos/data/*.parquet")
    mism = sum(1 for _, _, g in runs for k, v in g.items() if k in gold and v is not None and gold[k] != v)
    res = score_runs(runs, gold, a.allow_partial)
    res["gold_source"], res["gold_mismatch_with_result_files"] = src, mism
    print(f"Cosmos-Reason1 (undistorted): {res['mean_acc']:.2f}  = mean of {res['n_seeds']} seed(s); gold: {src}; n={res['seeds'][0]['n']}")
    for s in res["seeds"]:
        print(f"  {s['name']:>12}: acc {s['acc']:.4f} (n={s['n']}, no-letter {s['no_letter']}, missing {s['missing']})")
    print(f"  exact mean {res['mean_acc']!r}; gold mismatches vs result files: {mism}")
    print("  domain (mean over seeds): " + ", ".join(f"{d} {v:.2f}" for d, v in res["domain_mean"].items()))
    print(f"  no-letter replies (all seeds): {res['no_letter_total']}")
    if res["seeds"][0]["n"] != 510 and not a.allow_partial:
        print("  WARNING: expected 510 questions per seed")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
