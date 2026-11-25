# SPDX-License-Identifier: Apache-2.0
"""Build the EgoPlan-Bench evaluation set from the official lmms-lab/EgoPlan-Bench release and verify it against the
shipped manifest. No question or frame is redistributed by this repository; run download.sh first.

  # build from extracted clips (<sample_id>.mp4) or straight from the original zip shards, then verify
  python benchmarks/egoplan/prepare.py --parquet <validation parquet> --clips-dir CLIPS --out data/egoplan
  python benchmarks/egoplan/prepare.py --parquet <validation parquet> --zips-dir ZIPS  --out data/egoplan --workers 8
  python benchmarks/egoplan/prepare.py --out data/egoplan --verify-only        # verify an existing build

Steps. (1) read the validation parquet; keep the 3,343 sample ids of the shipped ids.json (these are exactly the rows with
is_valid=True in the parquet we used; see README); (2) for each sample read <sample_id>.mp4; (3) pick 7 progress frames +
the current observation (frame selection below), resize to a long side of 448 and write JPEG q90; (4) assemble eval.jsonl
(LLaVA rows; the prompt is the lmms-eval egoplan_doc_to_text, word for word).

Clip model (verified on the package): <sample_id>.mp4 holds the frames of the union of ALL annotated action segments
[start_frame, stop_frame] (sorted, gaps removed, NOT cut at the current observation, so ~400 clips contain frames after it),
followed by the current-observation frame appended last when it is not already the last frame. Every clip position is mapped
to its original frame index and only positions with original index < current_observation_frame are used for progress frames;
the last clip frame is the current observation. 15 clips have fewer decoded frames than this model predicts; for them the
same formula is applied to the raw clip positions (frame_orig_idx = null). Rows without progress repeat the current
observation 8 times.

Verification (PASS/FAIL): sha256(eval.jsonl) must equal MANIFEST.json, row ids must equal ids.json, and the set of image
paths must equal the shipped list. JPEG bytes can differ across opencv/ffmpeg builds; they are compared only informationally.
"""
import argparse, collections, hashlib, json, os, sys, tempfile, time, traceback, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "egoplan"
N_PROGRESS = 7   # reference implementation: n_frms=8 -> 7 progress frames + current observation
LONG = 448
N_IMG = 8


# ----------------------------------------------------------------------------- prompt (lmms-eval, verbatim logic)
def egoplan_doc_to_text(doc):
    """Port of lmms-eval lmms_eval/tasks/egoplan/utils.py::egoplan_doc_to_text (commit 08affba)."""
    task_goal = doc["task_goal"]
    if "goal" in task_goal:
        task_goal = task_goal.split("to", 1)[1].strip()
    words = task_goal.split()
    if words[0].endswith("ing"):
        question_pattern = (
            "I am tasked with {}. "
            "The task's progress is demonstrated in the provided video. "
            "My current field of view is shown in the provided image. "
            "What should be my next action? "
            "Please output the most reasonable action you think, expressed in a short phrase."
        )
    else:
        question_pattern = (
            "My current task is to {}. "
            "The task's progress is demonstrated in the provided video. "
            "My current field of view is shown in the provided image. "
            "What should be my next action? "
            "Please output the most reasonable action you think, expressed in a short phrase."
        )
    question = question_pattern.format(task_goal)
    for choice_idx in ["A", "B", "C", "D"]:
        question += "\n" + f"{choice_idx}. " + (doc[f"choice_{choice_idx.lower()}"])
    post_prompt = "\nAnswer with the option's letter from the given choices"
    return f"{question}{post_prompt}"


# ----------------------------------------------------------------------------- frame selection
def progress_metadata(doc):
    m = doc.get("task_progress_metadata") or []
    if isinstance(m, str):
        m = json.loads(m)
    return [{"start_frame": int(x["start_frame"]), "stop_frame": int(x["stop_frame"])} for x in m]


def clip_orig_frames(doc):
    cur = int(doc["current_observation_frame"])
    fr = set()
    for x in progress_metadata(doc):
        fr.update(range(x["start_frame"], x["stop_frame"] + 1))
    U = sorted(fr)
    seq = U + ([cur] if (not U or U[-1] != cur) else [])
    return U, seq


def plan(doc):
    """-> (seq, sel, check_pos): predicted original frame index of every clip position, the clip positions chosen as progress
    frames, and (when the current observation also occurs inside the action union) the position used for an identity check."""
    import numpy as np
    cur = int(doc["current_observation_frame"])
    U, seq = clip_orig_frames(doc)
    prog = [i for i, f in enumerate(seq[:-1]) if f < cur]
    L = len(prog)
    sel = [prog[i] for i in np.arange(0, L, L / N_PROGRESS).astype(int).tolist()[:N_PROGRESS]] if L else []
    cur_in_union = bool(U) and (cur in set(U)) and U[-1] != cur
    return seq, sel, (U.index(cur) if cur_in_union else None)


def raw_select(n):
    """Fallback for clips with fewer frames than predicted: same formula on raw clip positions 0..n-2."""
    import numpy as np
    L = n - 1
    return [int(i) for i in np.arange(0, L, L / N_PROGRESS).astype(int).tolist()[:N_PROGRESS]] if L > 0 else []


def resize_write(cv2, img, path):
    h, w = img.shape[:2]
    s = LONG / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, path
    return list(img.shape[:2])


def _decode(cv2, clip, want):
    cap = cv2.VideoCapture(str(clip))
    prop_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    got, i = {}, 0
    while cap.grab():   # sequential decode (exact positions); colour-convert only the frames we keep
        if i in want:
            ok, f = cap.retrieve()
            if not ok:
                break
            got[i] = f
        i += 1
    cap.release()
    return i, prop_n, got


def process_clip(doc, clip, out):
    """Decode one clip and write its frames. Returns the provenance record (no timing fields)."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    import cv2, numpy as np
    cv2.setNumThreads(1)
    sid, vid = int(doc["sample_id"]), doc["video_id"]
    cur = int(doc["current_observation_frame"])
    vdir = Path(out) / "images" / vid
    vdir.mkdir(parents=True, exist_ok=True)
    rel = lambda k: f"images/{vid}/{sid}_{k}.jpg"
    seq, sel, check_pos = plan(doc)
    want = set(sel) | ({check_pos} if check_pos is not None else set()) | {len(seq) - 1}
    n, prop_n, got = _decode(cv2, clip, want)
    res = {"sample_id": sid, "n_decoded": n, "n_prop": prop_n, "n_model": len(seq), "status": "ok"}
    if n == 0:
        raise RuntimeError(f"sample {sid}: clip could not be decoded")
    if n == len(seq):
        last = got[n - 1]
        res["clip_h"], res["clip_w"] = int(last.shape[0]), int(last.shape[1])
        imgs, hws, origs = [], {}, []
        for k, p in enumerate(sel):
            hws[f"{sid}_{k}"] = resize_write(cv2, got[p], vdir / f"{sid}_{k}.jpg"); imgs.append(rel(k)); origs.append(seq[p])
        hws[f"{sid}_7"] = resize_write(cv2, last, vdir / f"{sid}_7.jpg")
        if sel:
            imgs.append(rel(7)); origs.append(seq[-1])
        else:  # no progress frames: the reference code feeds the current observation n_frms times
            imgs = [rel(7)] * (N_PROGRESS + 1); origs = [seq[-1]] * (N_PROGRESS + 1)
        res.update(images=imgs, hw=hws, frame_orig_idx=origs, clip_pos=sel + [n - 1],
                   n_progress_available=sum(1 for f in seq[:-1] if f < cur), n_future_in_clip=sum(1 for f in seq[:-1] if f > cur))
        if check_pos is not None:
            a = cv2.resize(got[check_pos], (160, 90)).astype(np.float32); b = cv2.resize(last, (160, 90)).astype(np.float32)
            res["cur_check_diff"] = round(float(np.abs(a - b).mean()), 2)
        return res
    # clip shorter than the metadata model: all actions must end at or before the current observation
    assert max(x["stop_frame"] for x in progress_metadata(doc)) <= cur, sid
    sel = raw_select(n)
    n2, _, got = _decode(cv2, clip, set(sel) | {n - 1})
    assert n2 == n
    last = got[n - 1]
    res["clip_h"], res["clip_w"] = int(last.shape[0]), int(last.shape[1])
    imgs, hws = [], {}
    for k, p in enumerate(sel):
        hws[f"{sid}_{k}"] = resize_write(cv2, got[p], vdir / f"{sid}_{k}.jpg"); imgs.append(rel(k))
    hws[f"{sid}_7"] = resize_write(cv2, last, vdir / f"{sid}_7.jpg")
    imgs = imgs + [rel(7)] if sel else [rel(7)] * (N_PROGRESS + 1)
    res.update(fallback="clip_count_mismatch_uniform_over_raw_clip", images=imgs, hw=hws, frame_orig_idx=[None] * len(imgs),
               clip_pos=sel + [n - 1], n_progress_available=n - 1, n_future_in_clip=0)
    return res


# ----------------------------------------------------------------------------- assemble (pure; tested offline)
def assemble_rows(docs, prov, check=True):
    """docs: parquet rows (dicts); prov: {sample_id: provenance record}. -> rows in eval.jsonl order (sorted by sample_id)."""
    rows = []
    for r in sorted(docs, key=lambda r: int(r["sample_id"])):
        sid = int(r["sample_id"])
        res = prov[sid]
        assert res["status"] == "ok", sid
        cur = int(r["current_observation_frame"])
        if check and "task_progress_metadata" in r and not res.get("fallback"):
            assert all(f is not None and f <= cur for f in res["frame_orig_idx"]), sid
            assert res["frame_orig_idx"][-1] == cur
            assert all(f < cur for f in res["frame_orig_idx"][:-1]) or len(set(res["images"])) == 1, sid
        text = egoplan_doc_to_text(r)
        gold = r["golden_choice_idx"]
        assert r[f"choice_{gold.lower()}"] == r["answer"]
        split = {"EpicKitchens": "epic", "Ego4D": "ego4d"}[r["video_source"]]
        rows.append({
            "id": str(sid), "sample_id": sid,
            "image": res["images"],
            "conversations": [{"from": "human", "value": "<image>" * len(res["images"]) + text},
                              {"from": "gpt", "value": gold}],
            "family": "egoplan_mc", "qa_type": "egoplan_mc", "split": split, "control": "full",
            "task_goal": r["task_goal"], "video_id": r["video_id"], "video_source": r["video_source"],
            "answer_text": r["answer"], "options": [r["choice_a"], r["choice_b"], r["choice_c"], r["choice_d"]],
            "current_observation_frame": cur, "frame_orig_idx": res["frame_orig_idx"],
            "n_progress_frames": 0 if len(set(res["images"])) == 1 else len(res["images"]) - 1,
        })
    return rows


def write_rows(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def image_paths(rows):
    return sorted({p for r in rows for p in r["image"]})


# ----------------------------------------------------------------------------- input access
def load_docs(parquet, keep_ids):
    import pyarrow.parquet as pq
    parquet = Path(parquet)
    if parquet.is_dir():   # download.sh layout: pick the validation parquet
        cands = sorted(p for p in parquet.rglob("*.parquet") if "validation" in str(p.relative_to(parquet)).lower()) or sorted(parquet.rglob("*.parquet"))
        if not cands:
            raise SystemExit(f"no parquet under {parquet}")
        parquet = cands[0]
    t = pq.read_table(parquet)
    need = {"sample_id", "task_goal", "choice_a", "choice_b", "choice_c", "choice_d", "answer", "golden_choice_idx",
            "video_id", "video_source", "current_observation_frame", "task_progress_metadata"}
    miss = need - set(t.column_names)
    if miss:
        raise SystemExit(f"parquet lacks columns {sorted(miss)}; found {t.column_names}")
    docs = t.to_pylist()
    if "is_valid" in t.column_names:
        valid = {int(d["sample_id"]) for d in docs if d["is_valid"]}
        if valid != set(keep_ids):
            print(f"[warn] is_valid=True rows ({len(valid)}) differ from the shipped id list ({len(keep_ids)}): "
                  f"only-in-parquet {sorted(valid - set(keep_ids))[:20]}, only-in-list {sorted(set(keep_ids) - valid)[:20]}")
    docs = [d for d in docs if int(d["sample_id"]) in set(keep_ids)]
    missing = set(keep_ids) - {int(d["sample_id"]) for d in docs}
    if missing:
        raise SystemExit(f"{len(missing)} shipped ids not in the parquet, e.g. {sorted(missing)[:10]}")
    return docs


class ClipSource:
    """Yields a local path to <sample_id>.mp4 from an extracted directory or from the original zip shards."""
    def __init__(self, clips_dir=None, zips_dir=None):
        self.clips_dir, self.index = clips_dir, {}
        if zips_dir:
            for z in sorted(Path(zips_dir).rglob("*.zip")):
                with zipfile.ZipFile(z) as zf:
                    for n in zf.namelist():
                        if n.endswith(".mp4"):
                            self.index[int(Path(n).stem)] = (str(z), n)
        if not clips_dir and not self.index:
            raise SystemExit("no clips: give --clips-dir (<id>.mp4 files) or --zips-dir (HF zip shards)")

    def get(self, sid, tmpdir):
        if self.clips_dir:
            for cand in (Path(self.clips_dir) / f"{sid}.mp4", *Path(self.clips_dir).rglob(f"{sid}.mp4")):
                if cand.exists():
                    return cand, False
        if sid in self.index:
            z, n = self.index[sid]
            dst = Path(tmpdir) / f"{sid}.mp4"
            with zipfile.ZipFile(z) as zf, zf.open(n) as src, open(dst, "wb") as f:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            return dst, True
        raise FileNotFoundError(f"clip for sample {sid} not found")


def _work(args):
    doc, clips_dir, zips_dir, out, tmp = args
    src = _work.cache.get((clips_dir, zips_dir))
    if src is None:
        src = _work.cache[(clips_dir, zips_dir)] = ClipSource(clips_dir, zips_dir)
    try:
        p, temp = src.get(int(doc["sample_id"]), tmp)
        try:
            return process_clip(doc, p, out)
        finally:
            if temp and Path(p).exists():
                Path(p).unlink()
    except Exception as e:
        return {"sample_id": int(doc["sample_id"]), "status": "error", "error": repr(e), "tb": traceback.format_exc()[-800:]}
_work.cache = {}


# ----------------------------------------------------------------------------- verify
def verify(out, manifest_path=None, ids_path=None):
    out = Path(out)
    man = json.load(open(manifest_path or DATA / "MANIFEST.json"))
    ids = json.load(open(ids_path or DATA / "ids.json"))
    ok = True
    def rep(flag, msg):
        nonlocal ok
        print(f"[{'PASS' if flag else 'FAIL'}] {msg}"); ok &= bool(flag)
    got = sha256_file(out / "eval.jsonl")
    rep(got == man["eval_jsonl_sha256"], f"eval.jsonl sha256 {got[:16]}.. vs manifest {man['eval_jsonl_sha256'][:16]}..")
    rows = [json.loads(l) for l in open(out / "eval.jsonl", encoding="utf-8") if l.strip()]
    rep(len(rows) == man["rows"] and dict(collections.Counter(r["split"] for r in rows)) == man["rows_by_split"], f"rows {len(rows)} {dict(collections.Counter(r['split'] for r in rows))}")
    rep([int(r["id"]) for r in rows] == ids["ids"], "row id list/order == shipped ids.json")
    paths = image_paths(rows)
    rep(paths == ids["image_paths"], f"referenced image-path set ({len(paths)}) == shipped list")
    missing = [p for p in paths if not (out / p).exists()]
    rep(not missing, f"all image files present ({len(missing)} missing)")
    shas = ids.get("image_sha256")
    if shas and not missing:
        same = sum(1 for p in paths if sha256_file(out / p) == shas.get(p))
        print(f"[INFO] image bytes identical to the reference build for {same}/{len(paths)} files "
              f"(differences are expected across opencv/ffmpeg builds and do not affect the verdict)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", help="validation parquet (file, or the download.sh directory) from download.sh")
    ap.add_argument("--clips-dir", help="directory with extracted <sample_id>.mp4")
    ap.add_argument("--zips-dir", help="directory with the original HF zip shards (read with zipfile, no extraction)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=str(DATA / "ids.json"))
    ap.add_argument("--manifest", default=str(DATA / "MANIFEST.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated sample ids (debug); verification is skipped")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.verify_only:
        sys.exit(0 if verify(out, a.manifest, a.ids) else 1)
    if not a.parquet:
        raise SystemExit("--parquet required")
    ids = json.load(open(a.ids))["ids"]
    docs = load_docs(a.parquet, ids)
    only = {int(x) for x in a.only.split(",")} if a.only else None
    todo = [d for d in docs if only is None or int(d["sample_id"]) in only]
    out.mkdir(parents=True, exist_ok=True)
    prov, t0 = {}, time.time()
    with tempfile.TemporaryDirectory(prefix="egoplan_clip_") as tmp:
        jobs = [(d, a.clips_dir, a.zips_dir, str(out), tmp) for d in todo]
        if a.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            it = ProcessPoolExecutor(a.workers).map(_work, jobs, chunksize=4)
        else:
            it = map(_work, jobs)
        for k, res in enumerate(it, 1):
            prov[res["sample_id"]] = res
            if res["status"] != "ok":
                print("ERR", res["sample_id"], res.get("error"), flush=True)
            if k % 100 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    bad = [s for s, r in prov.items() if r["status"] != "ok"]
    if bad:
        raise SystemExit(f"{len(bad)} samples failed: {sorted(bad)[:20]}")
    with open(out / "frame_provenance.jsonl", "w") as f:
        for s in sorted(prov):
            f.write(json.dumps(prov[s]) + "\n")
    rows = assemble_rows(todo, prov)
    write_rows(rows, out / "eval.jsonl")
    print(f"wrote {len(rows)} rows ->", out / "eval.jsonl")
    if not (a.no_verify or only):
        sys.exit(0 if verify(out, a.manifest, a.ids) else 1)


if __name__ == "__main__":
    main()
