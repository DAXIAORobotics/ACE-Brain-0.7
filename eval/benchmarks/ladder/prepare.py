# SPDX-License-Identifier: Apache-2.0
"""Build the LADDER evaluation set (ladder_all.jsonl + images/) from the official sources and verify it against the shipped
manifest. No question or image is redistributed by this repository; run download.sh first.

  python benchmarks/ladder/prepare.py --raw-dir data/ladder/raw --out data/ladder          # build, then verify
  python benchmarks/ladder/prepare.py --out data/ladder --verify-only                      # verify an existing build

Families (rows): ladder_decomp (plan decomposition), ladder_verify (one subtask-completion query per frame),
ladder_state (one query per distinct subtask of the plan; scored per frame), ladder_next (next-subtask; all-done frames skipped).
All prompts, view order and thresholds come from the official repo via ladder_official.py (AST extraction, never restated).
Images are re-encoded as JPEG q85 (as the official client does); RoboLab decomposition frames are letterboxed to the
published size first (official _as_published).

Verification (PASS/FAIL): sha256 of the built ladder_all.jsonl == MANIFEST.json (must be byte-identical), and the set of image
paths == shipped list. Image BYTES are compared only informationally (JPEG encoders differ between PIL/libjpeg builds).
"""
import argparse, collections, hashlib, io, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import ladder_official as L

DEFAULT_MANIFEST = HERE.parents[1] / "data" / "ladder" / "MANIFEST.json"
DEFAULT_IDS = HERE.parents[1] / "data" / "ladder" / "ids.json"


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def read_parquet(raw, cfg):
    import pyarrow.parquet as pq
    for base in (Path(raw) / "hf" / "data" / cfg, Path(raw) / "data" / cfg, Path(raw) / cfg):
        p = base / "train-00000-of-00001.parquet"
        if p.exists():
            return pq.read_table(p).to_pylist()
    raise SystemExit(f"parquet for '{cfg}' not found under {raw} (run download.sh; needs pyarrow >= 20)")


def save(img_bytes, path: Path, O, letterbox_for=None):
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"), dtype=np.uint8)
    if letterbox_for:
        im = O.as_published(im, letterbox_for)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(im).save(path, format="JPEG", quality=85)


def build_rows(scenes, states, O, save_image):
    """Pure row construction. save_image(bytes, rel_path, letterbox_for_or_None) writes one image."""
    ref = {r["scene"]: r for r in scenes}
    rows = []
    sysp = O.DECOMP_PROMPT.format(object_naming=O.OBJECT_NAMING[False])
    for r in scenes:
        rel = f"decomp/{r['id'].replace('/', '__')}.jpg"
        save_image(r["image_main"]["bytes"], rel, r["benchmark"])
        rows.append({"id": "decomp/" + r["id"], "family": "ladder_decomp", "system": sysp, "image": [rel],
                     "conversations": [{"from": "human", "value": f"Task: {r['instruction']}<image>"},
                                       {"from": "gpt", "value": json.dumps({"subtasks": r["subtasks"], "ordered": bool(r["ordered"])})}],
                     "meta": {"benchmark": r["benchmark"], "instruction": r["instruction"], "subtasks": r["subtasks"],
                              "ordered": bool(r["ordered"]), "n_goals": r["n_goals"]}})
    for s in states:
        assert s["scene"] in ref, s["id"]
        s["revisits"] = bool(ref[s["scene"]]["revisits"])
        views = [(c, s["image_" + c]["bytes"]) for c in ("main", "wrist") if s.get("image_" + c) and s["image_" + c].get("bytes")]
        rels = []
        for c, b in views:
            rel = f"states/{s['id'].replace('/', '__')}_{c}.jpg"
            save_image(b, rel, None); rels.append((c, rel))
        s["_views"] = rels

    def verify_row(fid, s, subtask, required, gold_done, family, extra):
        prompt = O.VERIFY_PROMPT + O.CONTAINER_SUFFIX
        txt = f"Subtask: {subtask}"
        if len(s["_views"]) > 1:   # multi-view suffix, labelled views in order main, wrist
            prompt += O.MULTIVIEW_SUFFIX
            val = txt + "".join(f"{c} view:<image>" for c, _ in s["_views"])
        else:
            val = txt + "<image>"
        return {"id": fid, "family": family, "system": prompt, "image": [r for _, r in s["_views"]],
                "conversations": [{"from": "human", "value": val}, {"from": "gpt", "value": "true" if gold_done else "false"}],
                "meta": {"benchmark": s["benchmark"], "tags": s["tags"], "required": required, "label": bool(gold_done), **extra}}

    for s in states:   # verification: one query per sample (query_idx)
        sub = s["subtasks"][s["query_idx"]]
        copies = s["subtasks"][: s["query_idx"] + 1].count(sub)
        rows.append(verify_row("verify/" + s["id"], s, sub, O.required(s, copies), s["label"], "ladder_verify", {"sample": s["id"]}))
    for s in states:   # state tracking: one query per DISTINCT subtask of the plan
        for k, (sub, done, copies) in enumerate(O.distinct_queries(s)):
            rows.append(verify_row(f"state/{s['id']}/{k}", s, sub, O.required(s, copies), done, "ladder_state",
                                   {"sample": s["id"], "q": k}))
    for s in states:   # next subtask: skip all-done frames
        if all(s["achieved"]):
            continue
        main = [r for c, r in s["_views"] if c == "main"][0]
        rows.append({"id": "next/" + s["id"], "family": "ladder_next", "system": O.NEXT_SYSTEM, "image": [main],
                     "conversations": [{"from": "human", "value": "<image>" + O.NEXT_TEMPLATE.format(task=s["instruction"])},
                                       {"from": "gpt", "value": next(x for x, d in zip(s["subtasks"], s["achieved"]) if not d)}],
                     "meta": {"benchmark": s["benchmark"], "tags": s["tags"], "subtasks": s["subtasks"], "achieved": s["achieved"],
                              "ordered": bool(s["ordered"]), "sample": s["id"]}})
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    return rows


def write_rows(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def image_paths(rows):
    return sorted({p for r in rows for p in r["image"]})


def verify(out, manifest_path, ids_path=None):
    """-> True on PASS. Compares byte-exact question file and the image path set; image bytes are informational."""
    out = Path(out)
    man = json.load(open(manifest_path))
    ok = True
    qf = out / "ladder_all.jsonl"
    got = sha256_file(qf)
    same = got == man["ladder_all_jsonl_sha256"]
    print(f"[{'PASS' if same else 'FAIL'}] ladder_all.jsonl sha256 {got[:16]}.. vs manifest {man['ladder_all_jsonl_sha256'][:16]}..")
    ok &= same
    rows = [json.loads(l) for l in open(qf, encoding="utf-8") if l.strip()]
    cnt = dict(collections.Counter(r["family"] for r in rows))
    same = cnt == man["rows_per_family"] and len(rows) == man["rows"]
    print(f"[{'PASS' if same else 'FAIL'}] row counts {cnt} total {len(rows)}")
    ok &= same
    ids = json.load(open(ids_path or DEFAULT_IDS))
    same = [r["id"] for r in rows] == ids["ids"]
    print(f"[{'PASS' if same else 'FAIL'}] row id list/order == shipped ids.json")
    ok &= same
    paths = image_paths(rows)
    same = paths == ids["image_paths"]
    print(f"[{'PASS' if same else 'FAIL'}] referenced image-path set ({len(paths)}) == shipped list")
    ok &= same
    missing = [p for p in paths if not (out / "images" / p).exists()]
    same = not missing
    print(f"[{'PASS' if same else 'FAIL'}] all image files present ({len(missing)} missing)")
    ok &= same
    shas = man.get("image_sha256")
    if shas and not missing:
        diff = sum(1 for p in paths if sha256_file(out / "images" / p) != shas.get(p))
        print(f"[INFO] image bytes identical to the reference build for {len(paths) - diff}/{len(paths)} files "
              f"(differences are expected across PIL/libjpeg versions and do not affect the verdict)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=None, help="download.sh destination (holds hf/ and repo/)")
    ap.add_argument("--repo-dir", default=None, help="official repo clone (default <raw-dir>/repo or $LADDER_REPO_DIR)")
    ap.add_argument("--parquet-dir", default=None, help="dir holding decomposition/ and subtask_states/ parquet dirs (default <raw-dir>/hf/data)")
    ap.add_argument("--out", required=True, help="output dir (ladder_all.jsonl + images/)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--ids", default=str(DEFAULT_IDS))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--write-manifest", default=None, help="(maintainers) write MANIFEST.json + ids.json into this dir from the build")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.verify_only:
        raw = a.raw_dir
        repo = a.repo_dir or (str(Path(raw) / "repo") if raw else None)
        O = L.load(repo)
        pq_root = a.parquet_dir or raw
        if not pq_root:
            raise SystemExit("give --raw-dir or --parquet-dir")
        scenes, states = read_parquet(pq_root, "decomposition"), read_parquet(pq_root, "subtask_states")
        out.mkdir(parents=True, exist_ok=True)
        rows = build_rows(scenes, states, O, lambda b, rel, lb: save(b, out / "images" / rel, O, lb))
        write_rows(rows, out / "ladder_all.jsonl")
        print(collections.Counter(r["family"] for r in rows), "->", out / "ladder_all.jsonl")
        if a.write_manifest:
            md = Path(a.write_manifest); md.mkdir(parents=True, exist_ok=True)
            paths = image_paths(rows)
            man = {"benchmark": "ladder", "ladder_all_jsonl_sha256": sha256_file(out / "ladder_all.jsonl"), "rows": len(rows),
                   "rows_per_family": dict(collections.Counter(r["family"] for r in rows)),
                   "state_frames": len({r["meta"]["sample"] for r in rows if r["family"] == "ladder_state"}),
                   "images": len(paths),
                   "image_paths_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
                   "image_sha256": {p: sha256_file(out / "images" / p) for p in paths},
                   "image_sha256_note": "informational only: JPEG bytes depend on the PIL/libjpeg build; the verdict uses the question file and the path set",
                   "sources": {"hf_dataset": "INSAIT-Institute/LADDER-Offline-VLM-Benchmark", "hf_revision": "1a1cb958d9e0e47d4b4497bb79a594dc66a3be2a",
                               "github": "insait-institute/ladder", "github_commit": "8703b9d02bfe5e20b7d544bd5ecd71234586246a"},
                   "parquet_sha256": {"decomposition/train-00000-of-00001.parquet": "21f8f3f4b5a5e981a0b1805b736f3a4348b73446a85e941ea65ebcefeebee597",
                                      "subtask_states/train-00000-of-00001.parquet": "8dafbf23c982c352d16a295a8b9e8d7b7c68aef22e80e43f471f3a74f63ec118"}}
            json.dump(man, open(md / "MANIFEST.json", "w"), indent=1)
            json.dump({"ids": [r["id"] for r in rows], "image_paths": paths}, open(md / "ids.json", "w"))
            print("wrote manifest to", md)
    if a.no_verify or a.write_manifest:
        return
    sys.exit(0 if verify(out, a.manifest, a.ids) else 1)


if __name__ == "__main__":
    main()
