# SPDX-License-Identifier: Apache-2.0
"""Download the packaged benchmark data from a Hugging Face dataset repo, verify it and extract it into a data directory.

The hub repo holds ONE ARCHIVE PER SUBSET (``RoboFAC.tar.gz``, ``RoboBench-NL.tar.gz``, ``ERIQ.tar.gz``, ``Cosmos.tar.gz``,
``progress.tar.gz`` and ``ids.tar.gz`` with the EgoPlan / LADDER id lists) plus a ``SHA256SUMS`` file for the archives.
Every archive extracts to ``dataset/<Subset>/...``; here that is mapped to ``<data dir>/<benchmark>/...`` (the layout the
benchmark scripts expect). The repo id defaults to DEFAULT_REPO: pass it as ``repo`` or set the environment variable HF_REPO (default: DEFAULT_REPO).
"""
from __future__ import annotations
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from .io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
# benchmark (= local folder under the data dir) -> (archive file in the hub repo, folder inside the archive under dataset/)
ARCHIVES = {
    "robofac": ("RoboFAC.tar.gz", "RoboFAC"),
    "robobench": ("RoboBench-NL.tar.gz", "RoboBench-NL"),
    "eriq": ("ERIQ.tar.gz", "ERIQ"),
    "cosmos": ("Cosmos.tar.gz", "Cosmos"),
    "progress": ("progress.tar.gz", "progress"),
    "egoplan": ("ids.tar.gz", "EgoPlan"),   # id lists + sha256 only, the data itself is built with benchmarks/egoplan
    "ladder": ("ids.tar.gz", "LADDER"),     # id lists + sha256 only, the data itself is built with benchmarks/ladder
}
PACKAGED = ("robofac", "robobench", "eriq", "cosmos", "progress")  # default sets of `prepare.py --from-hub`
HAS_SHA256SUMS = ("robofac", "robobench", "eriq", "cosmos")        # folders that carry a per-file SHA256SUMS
SUMS_NAME = "SHA256SUMS"
DEFAULT_REPO = "ACERobotics/ACE-Brain-0.7-Bench"


def verify_sums(set_dir) -> list[str]:
    """Check every entry of <set_dir>/SHA256SUMS; returns a list of problems (empty = OK).
    Folders without a per-file SHA256SUMS (progress, egoplan, ladder) are checked only through the archive hash."""
    set_dir = Path(set_dir)
    sums = set_dir / SUMS_NAME
    if not sums.is_file():
        return [] if set_dir.name not in HAS_SHA256SUMS and set_dir.is_dir() else [f"{sums} missing"]
    errs = []
    for line in open(sums, encoding="utf-8"):
        if not line.strip():
            continue
        h, p = line.rstrip("\n").split("  ", 1)
        f = set_dir / p
        if not f.is_file():
            errs.append("missing " + p)
        elif sha256_file(f) != h:
            errs.append("hash mismatch " + p)
    return errs


def parse_sums(path) -> dict[str, str]:
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            h, p = line.rstrip("\n").split("  ", 1)
            out[p.lstrip("*")] = h
    return out


def extract_archive(archive, dest, mapping: dict[str, str]) -> int:
    """Extract ``dataset/<folder>/<rest>`` of a tar.gz into ``dest/<local>/<rest>`` for every (folder -> local) in `mapping`.
    Only regular files are written; absolute paths, '..' and links are rejected. Returns the number of files written."""
    dest = Path(dest).resolve()
    n = 0
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            parts = PurePosixPath(m.name).parts
            if m.isdir():
                continue
            if not m.isfile() or ".." in parts or m.name.startswith("/"):
                raise ValueError(f"unsafe or unsupported member in {archive}: {m.name!r}")
            if len(parts) < 3 or parts[0] != "dataset" or parts[1] not in mapping:
                continue
            target = (dest / mapping[parts[1]]).joinpath(*parts[2:]).resolve()
            if dest not in target.parents:
                raise ValueError(f"member escapes the data dir: {m.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            n += 1
    return n


def download(repo: str | None = None, sets=PACKAGED, dest=None, revision=None, cache_dir=None) -> dict:
    """Fetch SHA256SUMS + the archives of `sets` from the dataset repo, verify each archive's sha256, extract into `dest`
    (default $TASKBENCH_DATA or <repo>/data) and verify the per-file SHA256SUMS inside. Returns {set: [problems]}."""
    repo = repo or os.environ.get("HF_REPO") or DEFAULT_REPO
    unknown = [s for s in sets if s not in ARCHIVES]
    if unknown:
        raise SystemExit(f"unknown set(s): {unknown}; choose from {sorted(ARCHIVES)}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    dest = Path(dest or os.environ.get("TASKBENCH_DATA") or REPO_ROOT / "data")
    dest.mkdir(parents=True, exist_ok=True)
    kw = dict(repo_id=repo, repo_type="dataset", revision=revision, cache_dir=cache_dir)
    expected = parse_sums(hf_hub_download(filename=SUMS_NAME, **kw))
    by_archive: dict[str, dict[str, str]] = {}
    for s in sets:
        arc, folder = ARCHIVES[s]
        by_archive.setdefault(arc, {})[folder] = s
    res: dict[str, list[str]] = {}
    for arc, mapping in by_archive.items():
        if arc not in expected:
            for s in mapping.values():
                res[s] = [f"{arc} not listed in {SUMS_NAME} of the hub repo"]
            continue
        path = hf_hub_download(filename=arc, **kw)
        if sha256_file(path) != expected[arc]:
            for s in mapping.values():
                res[s] = [f"archive hash mismatch {arc} (corrupt download? delete it from the HF cache and retry)"]
            continue
        extract_archive(path, dest, mapping)
        for s in mapping.values():
            res[s] = verify_sums(dest / s)
    return {s: res[s] for s in sets}
