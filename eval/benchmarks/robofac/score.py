#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score RoboFAC (test split, 8-frame rebuild) predictions.

  python benchmarks/robofac/score.py --pred <predictions.jsonl> [--data-dir DIR] [--json out.json]
                                     [--judgments FILE] [--judgments-out FILE] [--include-planning]
                                     [--skip-judge] [--allow-partial]

Families and scoring (faithful port of the scorer used for the paper numbers):
  failure_detection            yes/no, rule  (first yes/no/cannot-tell in the reply, after a last "Answer:" if any)
  failure_identification_mc    letter A-F, parse_choice(letters="ABCDEF") == gold letter
  failure_locating_mc          letter A-D, parse_choice == gold letter
  task_identification          LLM judge, binary Yes/No against the reference (LOCAL protocol, see below)
  task_planning                shipped, scored by the LLM judge ONLY with --include-planning; it is NOT part of the
                               headline, which is the paper's 4-family macro average:
  headline  fac = mean(failure_detection, failure_identification_mc, failure_locating_mc, task_identification)
  family accuracy = 100 * sum(correct) / n.

LLM judge (task_identification; task_planning with --include-planning): the official RoboFAC GPT-judge prompt is not
available, so the prompts P_ID / P_PLAN below are a local protocol (kept verbatim from the protocol used for the paper).
One user message, temperature 0, max_tokens 16, thinking off, OpenAI-compatible endpoint configured through
JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL (default qwen3.8-flash) / JUDGE_CONCURRENCY (see taskbench/judge.py).
Retry semantics: attempt 0, then up to --retries (3) more attempts, stopping at the first reply that parses as yes/no
(regex `^\\W*(yes|no)\\b`, case-insensitive). An item still unparsed after the retries is UNSCORED: the family accuracy is
then None (and so is fac); the unscored count is printed prominently and a `*_scored_only` mean over the scored items is
reported next to it. Nothing is silently counted as right or wrong.

--judgments FILE re-scores from a saved judge journal (jsonl of {id, verdict, ...}; verdict true/false/null) WITHOUT calling
any judge. Without it, the judge is called and the journal is written to --judgments-out (default: next to --json, or next to
--pred) and is resumable (ids already journalled are not asked again). If the journal records `input_hash` (sha256 of the
canonical JSON of the judge messages) it is checked against the prompts rebuilt here; a mismatch is an error unless
--ignore-input-hash is given.

Judge back-ends are hosted services; temperature 0 is not guaranteed bit-exact, so a fresh judging run can differ slightly
from a saved journal.
"""
import argparse, hashlib, json, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from taskbench.infer import data_dir  # noqa: E402
from taskbench.io import read_jsonl  # noqa: E402
from taskbench.parsers import parse_choice  # noqa: E402

FAMILIES = ("task_identification", "task_planning", "failure_detection", "failure_identification_mc", "failure_locating_mc")
HEADLINE = ("failure_detection", "failure_identification_mc", "failure_locating_mc", "task_identification")
JUDGED = ("task_identification", "task_planning")
PROTOCOL = "robofac-local-five-families-v1"
JUDGE_PROTOCOL = "robofac-open-judge-v1"

# Judge prompts: local protocol, kept verbatim from the protocol used for the paper.
P_ID = """You are grading a robot-vision assistant's answer against a reference answer.
Task: the assistant was shown frames of a robot video and asked: "{q}"
Reference answer (ground truth): {gold}
Assistant answer: {pred}

Does the assistant answer describe the same task as the reference, i.e. the same action and the same target objects/places (wording and extra detail may differ, but a different action, object or goal is wrong)?
Reply with exactly one word: Yes or No."""
P_PLAN = """You are grading a robot-vision assistant's answer against a reference answer.
Task: the assistant was shown frames of a robot video and asked: "{q}"
Reference subtask sequence (ground truth):
{gold}
Assistant subtask sequence:
{pred}

Is the assistant sequence an equivalent plan for the same task: same overall goal and objects, the key reference subtasks present in the same order, and no wrong or contradicting step (finer or coarser step granularity and different wording are acceptable)?
Reply with exactly one word: Yes or No."""


def packed(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return hashlib.sha256(packed(x).encode()).hexdigest()


def rf_message(row, pred):
    """Judge messages for one task_identification / task_planning item."""
    q = re.sub(r"<image>\s*", "", row["conversations"][0]["value"]).strip().split("the last one is the current view.", 1)[-1].strip()
    template = P_ID if row["family"] == "task_identification" else P_PLAN
    return [{"role": "user", "content": template.format(q=q, gold=row["conversations"][-1]["value"].strip(), pred=pred.strip() or "(empty)")}]


def rf_parse(text):
    """True / False for a leading Yes / No, None when the reply is not a verdict."""
    m = re.match(r"\W*(yes|no)\b", (text or "").strip(), re.I)
    return None if not m else m[1].lower() == "yes"


def _after_answer(p):
    """Text after the last 'Answer:' when a reply reasons first; the whole reply otherwise."""
    m = list(re.finditer(r"answer\s*:", p, flags=re.I))
    return p[m[-1].end():] if m else p


def rf_closed(row, pred):
    """Rule scoring of the closed families -> {family, parsed, correct}."""
    fam = row["family"]
    gold = row["conversations"][-1]["value"].strip()
    p = pred.strip()
    out = {"family": fam}
    if fam == "failure_identification_mc":
        c = parse_choice(pred, "ABCDEF")
        out.update(parsed=c is not None, correct=float(c == gold[:1]))
    elif fam.endswith("_mc"):
        c = parse_choice(p)
        out.update(parsed=c is not None, correct=float(c == gold[:1]))
    elif gold.lower() in ("yes", "no", "cannot tell"):
        m = re.search(r"\b(yes|no|cannot tell)\b", _after_answer(p).lower())
        out.update(parsed=m is not None, correct=float(bool(m) and m.group(1) == gold.lower()))
    else:
        raise ValueError(f"no closed scorer for family {fam}")
    return out


# --------------------------------------------------------------------------------------------- judging
def read_journal(path):
    """-> {id: record}; the last record of an id wins."""
    out = {}
    if path and Path(path).exists():
        for r in read_jsonl(path):
            out[r["id"]] = r
    return out


def judge_one(client, rid, messages, retries):
    """Attempt 0 plus up to `retries` more, stopping at the first parsable verdict (board semantics)."""
    attempts, verdict = [], None
    for k in range(retries + 1):
        try:
            raw = client.chat(messages, "rf" + str(k))
            verdict = rf_parse(raw)
            attempts.append(dict(attempt=k, raw=raw, parsed=verdict))
            if verdict is not None:
                break
        except Exception as e:  # noqa: BLE001  (message not recorded: it may carry endpoint details)
            attempts.append(dict(attempt=k, error_type=type(e).__name__))
    return dict(id=rid, input_hash=digest(messages), verdict=verdict, attempts=attempts)


def run_judge(jobs, client, journal_path, retries=3, workers=8, rejudge_unscored=False, log=print):
    """jobs: {id: messages}. Appends one record per item to journal_path (resumable). Returns {id: record}."""
    done = read_journal(journal_path)
    todo = [i for i in jobs if i not in done or (rejudge_unscored and done[i].get("verdict") is None)]
    if todo:
        Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
        with open(journal_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(judge_one, client, i, jobs[i], retries) for i in todo]
            for n, fu in enumerate(as_completed(futs), 1):
                r = fu.result()
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done[r["id"]] = r
                if n % 50 == 0 or n == len(todo):
                    log(f"RoboFAC judge {n}/{len(todo)}")
    return done


# --------------------------------------------------------------------------------------------- scoring
def score_rows(rows, preds, judgments, include_planning=False):
    """-> (items, result). judgments: {id: {verdict}}; ids without a verdict are unscored."""
    judged = set(JUDGED) if include_planning else {"task_identification"}
    items = []
    for row in rows:
        rid, fam, pred = row["id"], row["family"], preds[row["id"]]["pred"]
        if fam in JUDGED:
            if fam not in judged:
                continue
            v = (judgments.get(rid) or {}).get("verdict")
            items.append(dict(id=rid, family=fam, parsed=v is not None, correct=None if v is None else float(v)))
        else:
            sc = rf_closed(row, pred)
            items.append(dict(id=rid, **sc))
    by = defaultdict(list)
    for it in items:
        by[it["family"]].append(it)
    metrics = {}
    for f in FAMILIES:
        n_all = sum(1 for r in rows if r["family"] == f)
        if f not in by:
            metrics[f] = dict(n=n_all, scored=0, unscored=0, unparsed=0, accuracy=None, scored_only_accuracy=None,
                              judged=False, note="not judged (pass --include-planning)" if f == "task_planning" else "no items")
            continue
        rs = by[f]
        scored = [r for r in rs if r["correct"] is not None]
        metrics[f] = dict(
            n=len(rs), scored=len(scored), unscored=len(rs) - len(scored), unparsed=sum(not r["parsed"] for r in rs),
            accuracy=100 * sum(r["correct"] for r in scored) / len(rs) if len(scored) == len(rs) else None,
            scored_only_accuracy=100 * sum(r["correct"] for r in scored) / len(scored) if scored else None)
    accs = [metrics[f]["accuracy"] for f in HEADLINE]
    so = [metrics[f]["scored_only_accuracy"] for f in HEADLINE]
    unscored = sum(metrics[f]["unscored"] for f in HEADLINE) + (metrics["task_planning"]["unscored"] if include_planning else 0)
    result = dict(
        benchmark="robofac", protocol=PROTOCOL, judge_protocol=JUDGE_PROTOCOL, n=len(rows), n_scored_families_items=len(items),
        complete=unscored == 0,
        fac=sum(accs) / 4 if all(a is not None for a in accs) else None,
        fac_scored_only=sum(so) / 4 if all(a is not None for a in so) else None,
        unscored=unscored, metrics=metrics, include_planning=include_planning,
        headline_families=list(HEADLINE),
        note="fac = mean of the 4 family accuracies (failure_detection, failure_identification_mc, failure_locating_mc, "
             "task_identification); task_planning is not part of it. Judged families use a local LLM-judge protocol "
             "(official RoboFAC judge prompt unavailable). accuracy is None when any item of the family is unscored.")
    return items, result


def print_report(result):
    m = result["metrics"]
    print(f"RoboFAC  n={result['n']}  protocol={result['protocol']}")
    for f in FAMILIES:
        x = m[f]
        if x.get("judged") is False:
            print(f"  {f:28s} {'not judged':>9s}  n={x['n']} (shipped, not in headline; --include-planning judges it)")
            continue
        acc = "n/a" if x["accuracy"] is None else f"{x['accuracy']:.4f}"
        so = "" if x["scored_only_accuracy"] is None or x["accuracy"] is not None else f"  scored-only {x['scored_only_accuracy']:.4f} over {x['scored']}"
        tag = " (not in headline)" if f not in HEADLINE else ""
        print(f"  {f:28s} {acc:>9s}  n={x['n']} unscored={x['unscored']} unparsed={x['unparsed']}{so}{tag}")
    fac = "None (unscored items)" if result["fac"] is None else f"{result['fac']:.4f}"
    print(f"FAC (mean of 4 families) = {fac}")
    if result["unscored"]:
        print("!" * 70)
        print(f"!! UNSCORED ITEMS: {result['unscored']} (judge verdict missing after retries) -> accuracy/fac are None")
        so = result["fac_scored_only"]
        print(f"!! scored-only mean over the scored items: fac_scored_only = {'n/a' if so is None else f'{so:.4f}'}")
        print("!" * 70)
    else:
        print("unscored items: 0")


def load_rows(d):
    rows = read_jsonl(Path(d) / "test.jsonl")
    for r in rows:
        if r["family"] not in FAMILIES:
            raise ValueError("unexpected family " + r["family"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions.jsonl ({id, pred, ...})")
    ap.add_argument("--data-dir", default=None, help="root holding data/robofac/ (default $TASKBENCH_DATA or ./data)")
    ap.add_argument("--json", default=None, help="write the result json here")
    ap.add_argument("--items-out", default=None, help="optional per-item scored jsonl")
    ap.add_argument("--judgments", default=None, help="saved judge journal: re-score from it, make NO judge calls")
    ap.add_argument("--judgments-out", default=None, help="where a fresh judging run writes/resumes its journal")
    ap.add_argument("--include-planning", action="store_true", help="also judge task_planning (not part of the headline)")
    ap.add_argument("--skip-judge", action="store_true", help="do not judge; judged families stay unscored")
    ap.add_argument("--allow-partial", action="store_true", help="score only the ids present in --pred (smoke tests)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None, help="judge threads (default $JUDGE_CONCURRENCY or 8)")
    ap.add_argument("--rejudge-unscored", action="store_true", help="on resume, ask again for journalled items without a verdict")
    ap.add_argument("--ignore-input-hash", action="store_true")
    a = ap.parse_args(argv)

    rows = load_rows(data_dir("robofac", a.data_dir))
    preds = {}
    for r in read_jsonl(a.pred):
        if r["id"] in preds:
            raise SystemExit(f"duplicate prediction id {r['id']}")
        preds[r["id"]] = r
    missing = [r["id"] for r in rows if r["id"] not in preds]
    if missing and not a.allow_partial:
        raise SystemExit(f"{len(missing)} of {len(rows)} questions have no prediction (use --allow-partial for a subset)")
    rows = [r for r in rows if r["id"] in preds]
    judged_fams = set(JUDGED) if a.include_planning else {"task_identification"}
    jobs = {r["id"]: rf_message(r, preds[r["id"]]["pred"]) for r in rows if r["family"] in judged_fams}

    if a.judgments:
        journal = read_journal(a.judgments)
        bad = [i for i in jobs if i in journal and journal[i].get("input_hash") not in (None, digest(jobs[i]))]
        if bad and not a.ignore_input_hash:
            raise SystemExit(f"{len(bad)} journal records were produced for a different judge prompt (input_hash); "
                             "use --ignore-input-hash to score them anyway")
        print(f"re-scoring from journal {Path(a.judgments).name}: {sum(i in journal for i in jobs)}/{len(jobs)} judged items found; no judge calls")
    elif a.skip_judge:
        journal = {}
    else:
        from taskbench.judge import JudgeClient
        client = JudgeClient(max_tokens=16, temperature=0.0, workers=a.workers)
        jpath = a.judgments_out or (str(Path(a.json).with_suffix("")) + ".judgments.jsonl" if a.json
                                    else str(Path(a.pred).with_name("robofac_judgments.jsonl")))
        journal = run_judge(jobs, client, jpath, a.retries, client.workers, a.rejudge_unscored)
        print("judge journal:", jpath)

    items, result = score_rows(rows, preds, journal, a.include_planning)
    result["partial"] = bool(missing)
    result["judge"] = dict(source="journal" if a.judgments else ("skipped" if a.skip_judge else "live"),
                           max_tokens=16, temperature=0, thinking=False, retries=a.retries)
    print_report(result)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(a.json, "w"), indent=1, ensure_ascii=False)
        print("wrote", a.json)
    if a.items_out:
        with open(a.items_out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    main()
