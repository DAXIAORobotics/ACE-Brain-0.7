#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score RoboBench (natural-language version, undistorted images) predictions.

  python benchmarks/robobench/score.py --pred <predictions.jsonl> [--data-dir DIR] [--json out.json]
                                       [--judgments FILE] [--judgments-out FILE] [--skip-judge] [--allow-partial]

NOT comparable to the original RoboBench leaderboard: questions were converted to natural language (see README).

Items and scoring (3,979 eligible items = 4,007 minus 28 Q1 items whose converted reference is unverified):
  Q1 task decomposition (1,945)  LLM judge, image of the first scene frame + reference + gt DAG; two integer scores 0..10;
                                 score = (node_correctness + task_completion) / 20 * 100
  Q2 next subtask (842)          LLM judge in two stages: EXTRACT the actions stated by the candidate (no reference given),
                                 then COMPARE with the reference; score = (S + O + P') / 3 * 100 with the dependency rules
                                 of rules.py. An empty reply scores 0 without a judge call.
  Q3 completion check (1,192)    rule: yes/no (case-insensitive, optional "Answer:" prefix and final punctuation) == reference,
                                 score 100 or 0; unparsable replies score 0.
  Overall (the headline)         mean of the item scores over ALL 3,979 eligible items (not a mean of the split means);
                                 the per-split means are reported as well.

THE UNSCORED RULE (identical to the paper's board): the judge is called with strict, per-stage validation (attempt 0 plus
`--retries` (3) more attempts, temperature 0, max_tokens 4096, thinking off). An item whose judge calls are exhausted is then
re-judged in up to `--rejudge-rounds` (6) further rounds (the rb_rejudge procedure of the paper runs). An item that is STILL
unscored after that counts as CORRECT (100) in the headline `score_0_100`; their number is reported as
`unscored_forced_correct` right next to the score, and `score_without_rule` (mean over the scored items only) is reported
as a secondary figure. Items never sent to a judge (no journal record, e.g. with --skip-judge) are not covered by this rule:
the headline is then None and `not_judged` is reported. In the runs behind the reported numbers the rule applied to at most one item (at most 0.03 points).

--judgments FILE re-scores from a saved judge journal (jsonl of {id, response, prompt_sha256, ...}) without calling any
judge; the journal's prompt_sha256 is checked against the prompts rebuilt here (needs the images; --no-verify-prompts skips
this). Judge: OpenAI-compatible endpoint from JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL (default qwen3.8-flash) /
JUDGE_CONCURRENCY (taskbench/judge.py); the Q1 judge message carries the first scene frame as an image, so the judge model
must accept images.
"""
import argparse, hashlib, json, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from taskbench.infer import data_dir  # noqa: E402
from taskbench.io import read_jsonl  # noqa: E402
import rules  # noqa: E402

PROTOCOL = "robobench-nl-v1-judge100"
TASK_TYPE = {"Q1": "task_decomposition", "Q2": "next_action", "Q3": "completion_verification"}
SPLITS = ("Q1", "Q2", "Q3")


# ---------------------------------------------------------------------------------------- helpers (verbatim port)
def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def strip_thinking(text):
    # A missing closing tag is an incomplete response, not a candidate answer.
    if "<think>" in text and "</think>" not in text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_yes_no(text):
    text = strip_thinking(text).strip(" \n\t*`")
    m = re.fullmatch(r"(?:(?:final\s+)?answer\s*[:\uff1a]\s*)?(yes|no)[.!\u3002]?", text, re.I)
    return m[1].lower() if m else None


def excluded(r):
    return r.get("answer_status") == "needs_review" or r["answer"] is None


def route(r):
    if excluded(r):
        return "excluded_gt_review"
    return {"Q1": "semantic_plan_100", "Q2": "semantic_next_100", "Q3": "yes_no_accuracy"}[r["split"]]


def load_rows(root):
    """items.jsonl -> board-style rows (question = joined human turns, media = absolute image paths)."""
    root = Path(root)
    rows = []
    for n in read_jsonl(root / "items.jsonl"):
        question = "\n".join(c["value"] for c in n["conversations"] if c["from"] == "human")
        answer = next(c["value"] for c in n["conversations"] if c["from"] == "gpt")
        rows.append(dict(id=n["id"], split=n["split"], task_type=TASK_TYPE[n["split"]], question=question,
                         system=n.get("system", ""), answer=answer, answer_status=n.get("answer_status"),
                         media=[{"path": str(root / p)} for p in n["image"]], rewrite=n.get("rewrite")))
        if question.count("<image>") != len(n["image"]):
            raise ValueError("image placeholder mismatch: " + n["id"])
    return rows


def first_stage_messages(row, response, dags):
    """The judge request whose hash identifies a journal record (Q1: the only stage; Q2: the extraction stage)."""
    return rules.judge_prompt(row, strip_thinking(response), dags)


# ---------------------------------------------------------------------------------------- per-item scoring
def score_response(row, response, judge_raw=None):
    """-> dict(method, score, parsed, status). Same logic as the board scorer."""
    result = dict(method=route(row), score=None, parsed=None, status="scored")
    if excluded(row):
        return dict(result, status="excluded_gt_review")
    text = strip_thinking(response)
    if result["method"] == "yes_no_accuracy":
        pred = parse_yes_no(text)
        return dict(result, score=100 * int(pred == row["answer"].lower()), parsed=pred, status="scored" if pred else "invalid_answer")
    if not text and row["task_type"] == "next_action":
        return dict(result, score=0, status="invalid_answer", reason="empty final response")
    if judge_raw is None:
        return dict(result, status="needs_judge")
    return dict(result, **rules.score_judgment(row, strip_thinking(judge_raw)))


def needs_judge(row, response):
    """Rows sent to the judge: every eligible Q1, and every Q2 whose final reply is non-empty."""
    return route(row).startswith("semantic_") and (row["task_type"] == "task_decomposition" or bool(strip_thinking(response)))


# ---------------------------------------------------------------------------------------- judging
def evaluate_one(row, response, client, retries, dags, first_messages=None):
    """Per-stage validated judging with retries (port of the paper's pool judge). Returns {response, attempts}."""
    attempts = []

    def call_valid(messages, stage, validator):
        for attempt in range(retries + 1):
            try:
                raw = client.chat(messages, "rbnl:" + stage + ":" + str(attempt))
                parsed = validator(raw)
                attempts.append(dict(stage=stage, attempt=attempt, prompt_sha256=digest(messages), response=raw, valid=parsed is not None))
                if parsed is not None:
                    return raw, parsed
            except Exception as e:  # noqa: BLE001  (do not record the message: it may carry endpoint details)
                attempts.append(dict(stage=stage, attempt=attempt, error_type=type(e).__name__))
        return None, None

    first = first_messages or first_stage_messages(row, response, dags)
    if row["task_type"] == "task_decomposition":
        raw, _ = call_valid(first, "q1", lambda t: rules.score_judgment(row, t) if rules.score_judgment(row, t)["score"] is not None else None)
        return dict(response=raw or "", attempts=attempts)
    extract_raw, ext = call_valid(first, "q2-extract", rules.extraction_parse)
    if ext is None:
        return dict(response="", attempts=attempts)
    messages = rules.compare_prompt(row, strip_thinking(response), ext)

    def valid(raw):
        result = rules.score_judgment(row, packed(dict(extraction=extract_raw, comparison=raw)))
        return result if result["score"] is not None else None

    compare_raw, _ = call_valid(messages, "q2-compare", valid)
    return dict(response=packed(dict(extraction=extract_raw, comparison=compare_raw)) if compare_raw else "", attempts=attempts)


def read_journal(path):
    out = {}
    if path and Path(path).exists():
        for r in read_jsonl(path):
            out[r["id"]] = r  # last record of an id wins
    return out


def run_judge(rows, preds, client, dags, journal_path, retries=3, workers=8, rounds=6, backoff=30, log=print):
    """Judge every row that needs it (resumable journal), then re-judge exhausted items for up to `rounds` rounds."""
    todo_rows = {r["id"]: r for r in rows if not excluded(r) and needs_judge(r, preds[r["id"]]["pred"])}
    journal = read_journal(journal_path)
    Path(journal_path).parent.mkdir(parents=True, exist_ok=True)

    def one_pass(ids, label):
        if not ids:
            return
        with open(journal_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            futs = {}
            for i in ids:
                r = todo_rows[i]
                first = first_stage_messages(r, preds[i]["pred"], dags)
                futs[ex.submit(evaluate_one, r, preds[i]["pred"], client, retries, dags, first)] = (i, first)
            for n, fu in enumerate(as_completed(futs), 1):
                i, first = futs[fu]
                v = fu.result()
                v.update(id=i, prompt_sha256=digest(first), finished_unix=time.time())
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
                f.flush()
                journal[i] = v
                if n % 50 == 0 or n == len(ids):
                    log(f"RoboBench judge [{label}] {n}/{len(ids)}")

    one_pass([i for i in todo_rows if i not in journal], "pass")
    for rnd in range(1, rounds + 1):
        bad = [i for i in todo_rows if i in journal and not journal[i].get("response")]
        log(f"REJUDGE round {rnd}: {len(bad)} exhausted of {len(todo_rows)}")
        if not bad:
            break
        one_pass(bad, f"rejudge {rnd}")
        if any(not journal[i].get("response") for i in bad) and rnd < rounds and backoff:
            time.sleep(min(backoff * rnd, 120))
    return journal


# ---------------------------------------------------------------------------------------- aggregation
def aggregate(rows, preds, journal):
    """Per-item results + the headline with the unscored-forced-correct rule. See module docstring."""
    items = []
    for r in rows:
        if excluded(r):
            continue
        pred = preds[r["id"]]["pred"]
        j = journal.get(r["id"])
        sc = score_response(r, pred, j["response"] if j else None)
        if sc["score"] is not None:
            state = "scored"
        elif needs_judge(r, pred) and j is not None:
            state = "judge_exhausted"        # judged, but no valid verdict after all retries/rounds
        else:
            state = "not_judged"
        items.append(dict(id=r["id"], split=r["split"], status=sc["status"], state=state, score=sc["score"]))

    def stats(its):
        sc = [x["score"] for x in its if x["state"] == "scored"]
        ex = sum(x["state"] == "judge_exhausted" for x in its)
        nj = sum(x["state"] == "not_judged" for x in its)
        n = len(its)
        return dict(n=n, scored=len(sc), unscored_forced_correct=ex, not_judged=nj,
                    score_0_100=None if nj or not n else (sum(sc) + 100 * ex) / n,
                    score_without_rule=sum(sc) / len(sc) if sc else None)

    out = dict(overall=stats(items), splits={s: stats([x for x in items if x["split"] == s]) for s in SPLITS})
    return items, out


def build_result(rows, preds, journal):
    items, agg = aggregate(rows, preds, journal)
    o = agg["overall"]
    return items, dict(
        benchmark="robobench", protocol=PROTOCOL, n_items=len(rows), eligible=o["n"], excluded_gt_review=sum(excluded(r) for r in rows),
        score_0_100=o["score_0_100"], unscored_forced_correct=o["unscored_forced_correct"], not_judged=o["not_judged"],
        score_without_rule=o["score_without_rule"], splits=agg["splits"],
        complete=o["not_judged"] == 0 and o["unscored_forced_correct"] == 0,
        note="Overall = mean of item scores over all eligible items; items whose judge verdict is still missing after retries "
             "and re-judge rounds count as 100 (unscored_forced_correct); score_without_rule is the mean over scored items only. "
             "Natural-language conversion of RoboBench: not comparable to the original leaderboard.")


def print_report(res):
    f = lambda v: "None" if v is None else f"{v:.4f}"
    print(f"RoboBench-NL  eligible={res['eligible']} (+{res['excluded_gt_review']} excluded)  protocol={res['protocol']}")
    for s in SPLITS:
        x = res["splits"][s]
        print(f"  {s} {TASK_TYPE[s]:24s} {f(x['score_0_100']):>10s}  n={x['n']} scored={x['scored']} "
              f"unscored_forced_correct={x['unscored_forced_correct']} not_judged={x['not_judged']}  without_rule={f(x['score_without_rule'])}")
    print(f"ROBOBENCH score_0_100 (unscored counted correct) = {f(res['score_0_100'])}   "
          f"[unscored_forced_correct = {res['unscored_forced_correct']}]")
    print(f"  secondary: score_without_rule (scored items only) = {f(res['score_without_rule'])}")
    if res["unscored_forced_correct"] or res["not_judged"]:
        print("!" * 72)
        if res["unscored_forced_correct"]:
            print(f"!! {res['unscored_forced_correct']} items were unscored after retries and re-judging and are counted as CORRECT (100)")
        if res["not_judged"]:
            print(f"!! {res['not_judged']} items were never judged: headline score is None")
        print("!" * 72)


def verify_prompts(rows, preds, journal, dags):
    """Number of journal records whose prompt_sha256 differs from the prompt rebuilt from data + prediction."""
    bad = checked = 0
    for r in rows:
        if excluded(r) or r["id"] not in journal or not needs_judge(r, preds[r["id"]]["pred"]):
            continue
        h = journal[r["id"]].get("prompt_sha256")
        if h is None:
            continue
        checked += 1
        bad += h != digest(first_stage_messages(r, preds[r["id"]]["pred"], dags))
    return checked, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions.jsonl ({id, pred, ...})")
    ap.add_argument("--data-dir", default=None, help="root holding data/robobench/ (default $TASKBENCH_DATA or ./data)")
    ap.add_argument("--json", default=None, help="write the result json here")
    ap.add_argument("--items-out", default=None, help="optional per-item scored jsonl")
    ap.add_argument("--judgments", default=None, help="saved judge journal: re-score from it, make NO judge calls")
    ap.add_argument("--judgments-out", default=None, help="where a fresh judging run writes/resumes its journal")
    ap.add_argument("--skip-judge", action="store_true", help="do not judge (Q1/Q2 stay unjudged)")
    ap.add_argument("--allow-partial", action="store_true", help="score only the ids present in --pred (smoke tests)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--rejudge-rounds", type=int, default=6)
    ap.add_argument("--rejudge-backoff", type=int, default=30, help="seconds x round between re-judge rounds (0 = none)")
    ap.add_argument("--workers", type=int, default=None, help="judge threads (default $JUDGE_CONCURRENCY or 8)")
    ap.add_argument("--no-verify-prompts", action="store_true", help="skip the journal prompt-hash check (needs no images then)")
    a = ap.parse_args(argv)

    root = data_dir("robobench", a.data_dir)
    rows = load_rows(root)
    dags = rules.load_dags(root / "q1_dags.json")
    preds = {}
    for r in read_jsonl(a.pred):
        if r["id"] in preds:
            raise SystemExit(f"duplicate prediction id {r['id']}")
        preds[r["id"]] = r
    elig = [r for r in rows if not excluded(r)]
    missing = [r["id"] for r in elig if r["id"] not in preds]
    if missing and not a.allow_partial:
        raise SystemExit(f"{len(missing)} of {len(elig)} eligible items have no prediction (use --allow-partial for a subset)")
    rows = [r for r in rows if excluded(r) or r["id"] in preds]
    for r in rows:
        if not excluded(r):
            preds[r["id"]].setdefault("pred", "")
            preds[r["id"]]["pred"] = preds[r["id"]]["pred"] or ""

    if a.judgments:
        journal = read_journal(a.judgments)
        print(f"re-scoring from journal {Path(a.judgments).name}: {len(journal)} records; no judge calls")
        if not a.no_verify_prompts:
            checked, bad = verify_prompts(rows, preds, journal, dags)
            print(f"journal prompt check: {checked} records checked, {bad} differ")
            if bad:
                raise SystemExit("journal was produced for different judge prompts/predictions (use --no-verify-prompts to ignore)")
    elif a.skip_judge:
        journal = {}
    else:
        from taskbench.judge import JudgeClient
        client = JudgeClient(max_tokens=4096, temperature=0.0, workers=a.workers)
        jpath = a.judgments_out or (str(Path(a.json).with_suffix("")) + ".judgments.jsonl" if a.json
                                    else str(Path(a.pred).with_name("robobench_judgments.jsonl")))
        journal = run_judge(rows, preds, client, dags, jpath, a.retries, client.workers, a.rejudge_rounds, a.rejudge_backoff)
        print("judge journal:", jpath)

    items, res = build_result(rows, preds, journal)
    res["partial"] = bool(missing)
    res["judge"] = dict(source="journal" if a.judgments else ("skipped" if a.skip_judge else "live"),
                        max_tokens=4096, temperature=0, thinking=False, retries=a.retries, rejudge_rounds=a.rejudge_rounds)
    print_report(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)
        print("wrote", a.json)
    if a.items_out:
        with open(a.items_out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return res


if __name__ == "__main__":
    main()
