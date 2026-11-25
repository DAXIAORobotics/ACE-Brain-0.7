# SPDX-License-Identifier: Apache-2.0
"""EgoPlan-Bench scorer.

Headline (what the paper table uses): accuracy of parse_choice(prediction) == gold letter, over all items.
Secondary: the lmms-eval first-letter extractor (extract_characters_regex, commit 08affba133b97b19351897f4d42081ead2b3412e).
The two differ on verbose replies (the lmms-eval extractor takes the first capital A-D anywhere, e.g. the "B" of "Based").

usage: python benchmarks/egoplan/score.py --pred predictions.jsonl [--data-dir D] [--items FILE] [--json out.json]
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taskbench.parsers import parse_choice


def lmms_egoplan_extract(s: str) -> str:
    """Port of lmms-eval lmms_eval/tasks/egoplan/utils.py::extract_characters_regex (quirks included)."""
    s = s.strip()
    answer_prefixes = ["The best answer is", "The correct answer is", "The answer is", "The answer",
                       "The best option isThe correct option is", "Best answer:Best option:"]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def score(rows, preds):
    n = c = c_lmms = unparsed = unparsed_lmms = missing = 0
    by_split = {}
    for r in rows:
        gold = r["conversations"][-1]["value"][:1]
        p = preds.get(r["id"])
        if p is None:
            missing += 1
        text = (p or {}).get("pred") or ""
        a, b = parse_choice(text), lmms_egoplan_extract(text)
        n += 1; c += a == gold; c_lmms += b == gold
        unparsed += a is None; unparsed_lmms += not b
        s = by_split.setdefault(r.get("split", "all"), [0, 0]); s[0] += a == gold; s[1] += 1
    return {"n": n, "accuracy": 100 * c / n if n else None, "n_correct": c, "unparsed": unparsed, "missing_predictions": missing,
            "accuracy_lmms_first_letter": 100 * c_lmms / n if n else None, "unparsed_lmms": unparsed_lmms,
            "by_split": {k: {"accuracy": 100 * v[0] / v[1], "n": v[1]} for k, v in by_split.items()}}


def main():
    from taskbench.infer import data_dir
    from taskbench.io import read_jsonl, load_predictions
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--items", default=None, help="explicit eval.jsonl (overrides --data-dir)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = read_jsonl(a.items or data_dir("egoplan", a.data_dir) / "eval.jsonl")
    res = score(rows, load_predictions(a.pred))
    print(f"EgoPlan-Bench accuracy (parse_choice) = {res['accuracy']:.2f}  ({res['n_correct']}/{res['n']}; unparsed {res['unparsed']}, missing predictions {res['missing_predictions']})")
    print(f"  secondary, lmms-eval first-letter extractor = {res['accuracy_lmms_first_letter']:.2f}  (unanswered {res['unparsed_lmms']})")
    for k, v in res["by_split"].items():
        print(f"  {k}: {v['accuracy']:.2f} (n={v['n']})")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
