#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build data/eriq/items.jsonl (the 11 task-understanding subsets, 3,905 questions) from the ERIQ dataset.

ERIQ (Embodied Reasoning Intelligence Quotient) is released by its authors on the Hugging Face hub
(repo id KineMind/ERIQ, CC BY-SA 4.0). The packaged copy shipped in data/eriq/ was produced with this script.
Use it only if you want to rebuild from the original download:

  python benchmarks/eriq/prepare.py --src <dir with QA_*.json and images/> --out <output dir> [--copy-images]
  python benchmarks/eriq/prepare.py --download <dir> ...      # fetch the dataset with huggingface_hub first

The prompt is the original human turn (options inline) plus one fixed answer-format line (the dataset gives no
prompt of its own): "Answer with the option's letter from the given choices directly." (Yes/No subset:
"Answer with Yes or No directly."). The four spatial-perception subsets are excluded.
"""
import argparse, collections, glob, json, os, shutil, sys
from pathlib import Path

SPATIAL = {"SCENE_UNDERSTANDING", "DUALVIEW_MATCHING", "TASK_GROUNDING", "RELATIVE_POS_GROUNDING"}
DIM = {"ACTION_UNDERSTANDING": "planning_monitoring", "SUCCESS_DETECTION": "planning_monitoring",
       "SUBTASK_PLANNING": "planning_monitoring", "FINE_GRAINED_PLAN": "planning_monitoring",
       "TRAJ_UNDERSTANDING": "planning_monitoring", "TASK_PROGRESS": "planning_monitoring",
       "HUMAN_INTENTION": "human_intent", "HUMAN_INTERACTION": "human_intent",
       "MISTAKE_EXISTENCE": "error_detect_recover", "MISTAKE_CLASSIFY": "error_detect_recover",
       "MISTAKE_RECOVERY": "error_detect_recover"}
SUF = "\nAnswer with the option's letter from the given choices directly."
SUF_YN = "\nAnswer with Yes or No directly."
HF_REPO = "KineMind/ERIQ"


def build(src, out, copy_images=False, check_images=True):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"subsets": {}, "excluded": sorted(SPATIAL), "missing_images": 0}
    n, miss, used = 0, [], set()
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for p in sorted(glob.glob(str(src / "QA_*.json"))):
            sub = os.path.basename(p)[3:-5]
            rows = json.load(open(p, encoding="utf-8"))
            rep["subsets"][sub] = {"n": len(rows), "kept": sub not in SPATIAL}
            if sub in SPATIAL:
                continue
            for r in rows:
                imgs = r["image"] if isinstance(r["image"], list) else [r["image"]]
                h, g = r["conversations"][0]["value"], r["conversations"][1]["value"].strip()
                assert h.count("<image>") == len(imgs), (r["id"], h.count("<image>"), len(imgs))
                for im in imgs:
                    used.add(im)
                    if check_images and not (src / "images" / im).exists():
                        miss.append(im)
                f.write(json.dumps({
                    "id": r["id"], "image": imgs,
                    "conversations": [{"from": "human", "value": h.rstrip() + (SUF if g in "ABCDEF" else SUF_YN)},
                                      {"from": "gpt", "value": g}],
                    "subset": sub, "dim": DIM.get(sub, "spatial"), "answer": g,
                    "n_options": sum(1 for l in h.splitlines() if l[:2] in ("A.", "B.", "C.", "D.", "E.", "F."))},
                    ensure_ascii=False) + "\n")
                n += 1
    rep.update(n_items=n, n_images=len(used), missing_images=len(miss), missing_examples=miss[:5])
    if copy_images:
        for im in sorted(used):
            d = out / "images" / im
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / "images" / im, d)
    json.dump(rep, open(out / "prep_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "subsets"}))
    if miss:
        sys.exit("missing images")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="dataset directory (QA_*.json + images/)")
    ap.add_argument("--download", metavar="DIR", help="download the dataset into DIR with huggingface_hub, then use it as --src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-images", action="store_true", help="also copy the images of the kept subsets into <out>/images")
    ap.add_argument("--no-image-check", action="store_true")
    a = ap.parse_args()
    src = a.src
    if a.download:
        from huggingface_hub import snapshot_download
        src = snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=a.download)
    if not src:
        ap.error("give --src or --download")
    build(src, a.out, a.copy_images, not a.no_image_check)


if __name__ == "__main__":
    main()
