# SPDX-License-Identifier: Apache-2.0
"""LADDER offline-VLM-benchmark scorer (ported from the evaluation code used for the paper numbers).

Metrics follow the official score_decomposition.py / score_subtask_states.py (loaded from the official repo clone via
ladder_official.py, never restated):
  decomposition   exact match over scenes (multiset of head-noun goals == reference)
  verification    balanced accuracy (mean of TP rate and TN rate); unanswered replies are excluded from tp/tn, reported separately
  state_tracking  all_correct = fraction of frames whose every distinct-subtask query is answered and right
  next_subtask    valid = fraction of replies classified "valid" by the official classifier
Headline = mean of the four (a missing/None component counts as 0).

usage: python benchmarks/ladder/score.py --pred predictions.jsonl [--data-dir D] [--items FILE] [--repo-dir R] [--json out.json]
"""
import argparse, collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import ladder_official as L

O = None


def _init(repo_dir=None):
    global O
    if O is None:
        O = L.load(repo_dir)
    return O


def gold_pred(row):
    """a prediction string the official parsers should score as fully correct"""
    f, m = row["family"], row["meta"]
    if f == "ladder_decomp":
        return json.dumps({"subtasks": m["subtasks"], "ordered": m["ordered"]})
    if f in ("ladder_verify", "ladder_state"):
        return json.dumps({"count": m["required"] if m["label"] else max(0, m["required"] - 1)})
    return row["conversations"][-1]["value"]


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def score(rows, preds):
    R = collections.defaultdict(list)
    for r in rows:
        p = preds.get(r["id"])
        R[r["family"]].append((r, p))
    out = {}
    # ---- decomposition
    res = []
    for r, p in R["ladder_decomp"]:
        m = r["meta"]; txt = p["pred"] if p else ""
        subs, ordered = L.decomposition_plan(O, txt, m["instruction"])
        sc = O.plan.score_plan(subs, m["subtasks"])
        row = {"predicted": subs, "instruction": m["instruction"], "error": "" if p else "missing", "exact": sc["exact"],
               "benchmark": m["benchmark"]}
        parsed_json = O.json_plan(txt) is not None
        res.append((row, sc, ordered == m["ordered"], parsed_json, p))
    tot = lambda k: [sum(x[1][k][i] for x in res) for i in (0, 1)]
    d = {"n": len(res), "json_parse_rate": pct(sum(x[3] for x in res), len(res)), "no_plan": sum(1 for x in res if O.no_plan(x[0])),
         "truncated": sum(1 for x in res if x[4] and x[4].get("finish_reason") == "length")}
    for k in ("exact", "goals_missed", "goals_invented", "register", "length"):
        a, b = tot(k); d[k] = pct(a, b)
    d["ordered"] = pct(sum(x[2] for x in res), len(res))
    d["by_benchmark"] = {b: pct(sum(x[1]["exact"][0] for x in res if x[0]["benchmark"] == b), sum(1 for x in res if x[0]["benchmark"] == b))
                         for b in sorted({x[0]["benchmark"] for x in res})}
    out["decomposition"] = d
    # ---- verification
    tp = fp = tn = fn = un = 0; trunc = 0; per = collections.defaultdict(lambda: [0, 0])
    for r, p in R["ladder_verify"]:
        m = r["meta"]; c = L.verifier_count(O, p["pred"]) if p else None
        trunc += bool(p and p.get("finish_reason") == "length")
        if c is None:
            un += 1; continue
        v = c >= m["required"]
        tp += m["label"] and v; fn += m["label"] and not v; fp += (not m["label"]) and v; tn += (not m["label"]) and not v
        per[m["benchmark"]][0] += v == m["label"]; per[m["benchmark"]][1] += 1
    n = len(R["ladder_verify"]); ans = tp + fp + tn + fn
    out["verification"] = {"n": n, "unanswered": un, "parse_rate": pct(ans, n), "truncated": trunc, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                           "TP_rate": pct(tp, tp + fn), "TN_rate": pct(tn, tn + fp), "accuracy_answered": pct(tp + tn, ans),
                           "accuracy_strict": pct(tp + tn, n), "balanced_acc": pct(tp / (tp + fn) + tn / (tn + fp), 2) if tp + fn and tn + fp else None,
                           "always_not_done_acc": pct(sum(1 for r, _ in R["ladder_verify"] if not r["meta"]["label"]), n)}
    # ---- state tracking: sample correct iff every distinct-subtask query answered and right
    S = collections.defaultdict(lambda: {"ok": 0, "n": 0, "un": 0, "bm": None})
    trunc = 0
    for r, p in R["ladder_state"]:
        m = r["meta"]; s = S[m["sample"]]; s["bm"] = m["benchmark"]; s["n"] += 1
        c = L.verifier_count(O, p["pred"]) if p else None
        trunc += bool(p and p.get("finish_reason") == "length")
        if c is None:
            s["un"] += 1
        else:
            s["ok"] += (c >= m["required"]) == m["label"]
    ex = sum(1 for s in S.values() if s["ok"] == s["n"] and not s["un"])
    gq = sum(s["n"] for s in S.values()); gok = sum(s["ok"] for s in S.values()); gun = sum(s["un"] for s in S.values())
    out["state_tracking"] = {"n_frames": len(S), "n_goal_queries": gq, "unanswered_queries": gun, "parse_rate": pct(gq - gun, gq), "truncated": trunc,
                             "all_correct": pct(ex, len(S)), "goal_accuracy": pct(gok, gq),
                             "by_benchmark": {b: pct(sum(1 for s in S.values() if s["bm"] == b and s["ok"] == s["n"] and not s["un"]), sum(1 for s in S.values() if s["bm"] == b))
                                              for b in sorted({s["bm"] for s in S.values()})}}
    # ---- next subtask
    cnt = collections.Counter(); reg = 0; empty = 0; ref_unres = 0; trunc = 0
    for r, p in R["ladder_next"]:
        m = r["meta"]; txt = p["pred"] if p else ""
        cmd = O.extract_command(txt)
        trunc += bool(p and p.get("finish_reason") == "length")
        ref_unres += "<" in cmd and ">" in cmd
        v = O.classify_next(cmd, m["subtasks"], m["achieved"], m["ordered"]) if cmd else "invalid"
        cnt[v] += 1; reg += bool(cmd and O.plan.in_register(cmd)); empty += not cmd
    n = len(R["ladder_next"])
    out["next_subtask"] = {"n": n, "valid": pct(cnt["valid"], n), "redundant": pct(cnt["redundant"], n), "invalid": pct(cnt["invalid"], n),
                           "register": pct(reg, n), "empty_reply": empty, "angle_bracket_refs(unresolved)": ref_unres, "truncated": trunc}
    return out


def composite(r):
    comps = [r["decomposition"]["exact"], r["verification"]["balanced_acc"], r["state_tracking"]["all_correct"], r["next_subtask"]["valid"]]
    none_idx = [i for i, c in enumerate(comps) if c is None]
    comps = [0.0 if c is None else c for c in comps]
    return {"mean": round(sum(comps) / 4, 4), "comps": dict(decomposition_exact=comps[0], verification_balanced_acc=comps[1],
                                                           state_all_correct=comps[2], next_valid=comps[3]), "none_as_zero": none_idx}


def main():
    from taskbench.infer import data_dir
    from taskbench.io import read_jsonl, load_predictions
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data-dir", default=None, help="root holding data/ladder/ (default $TASKBENCH_DATA or ./data)")
    ap.add_argument("--items", default=None, help="explicit ladder_all.jsonl (overrides --data-dir)")
    ap.add_argument("--repo-dir", default=None, help="clone of the official repo (default $LADDER_REPO_DIR)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    _init(a.repo_dir)
    items = a.items or str(data_dir("ladder", a.data_dir) / "ladder_all.jsonl")
    rows, preds = read_jsonl(items), load_predictions(a.pred)
    res = score(rows, preds)
    res["n_missing_predictions"] = sum(1 for r in rows if r["id"] not in preds)
    c = composite(res)
    res["composite"] = c
    d, v, s, n = res["decomposition"], res["verification"], res["state_tracking"], res["next_subtask"]
    print(f"LADDER composite = {c['mean']:.3f}  (rows {len(rows)}, predictions {len(preds)}, missing {res['n_missing_predictions']})")
    print(f"  decomposition exact        {d['exact']}   (n={d['n']}, no_plan={d['no_plan']})")
    print(f"  verification balanced acc  {v['balanced_acc']}   (n={v['n']}, unanswered={v['unanswered']})")
    print(f"  state all-correct          {s['all_correct']}   (frames={s['n_frames']}, unanswered queries={s['unanswered_queries']})")
    print(f"  next-subgoal valid         {n['valid']}   (n={n['n']}, empty replies={n['empty_reply']})")
    if c["none_as_zero"]:
        print("  note: components counted as 0 because they were None:", c["none_as_zero"])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
