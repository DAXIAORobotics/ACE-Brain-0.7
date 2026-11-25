#!/usr/bin/env python3
"""Regenerates the script-loadable copies of the data files.

The JSON files in data/ are the single source of truth. Browsers block fetch()
of local JSON under file://, so each JSON file is mirrored as data/<name>.js
(assigning to window.ACE_DATA.<name>). Run after editing any JSON file:

    python3 tools/build_data.py          # write
    python3 tools/build_data.py --check  # exit 1 if a mirror is out of date
"""
import csv, io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NAMES = ["results", "cases", "figures"]


def mirror(name):
    obj = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return f"window.ACE_DATA=window.ACE_DATA||{{}};window.ACE_DATA.{name}={body};\n"


def csv_text():
    res = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for key in ("task", "grounding"):
        t = res[key]
        w.writerow([f"# {key}"])
        w.writerow(["Model", "Group"] + t["columns"])
        for m in t["models"]:
            w.writerow([m["name"], m["group"]] + ["" if v is None else v for v in m["scores"]])
        w.writerow([])
    return out.getvalue()


def main():
    check = "--check" in sys.argv
    targets = {DATA / f"{n}.js": mirror(n) for n in NAMES}
    targets[DATA / "benchmarks.csv"] = csv_text()
    stale = []
    for path, text in targets.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
    if check and stale:
        print("out of date:", ", ".join(stale))
        sys.exit(1)
    print("ok", "checked" if check else "written", len(targets), "files")


if __name__ == "__main__":
    main()
