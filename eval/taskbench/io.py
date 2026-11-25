"""Small shared helpers: JSONL I/O, sha256, the LLaVA-style item format used by most benchmarks."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def load_predictions(path):
    """predictions.jsonl -> {id: record}. Rejects duplicate ids."""
    out = {}
    for r in read_jsonl(path):
        if r["id"] in out:
            raise ValueError(f"duplicate prediction id {r['id']}")
        out[r["id"]] = r
    return out


def item_messages(row: dict, root: str | os.PathLike):
    """LLaVA item -> chat messages with lazy image references.

    Item format: {"id", "image": [relative paths], "system"?: str,
                  "conversations": [{"from": "human", "value": "...<image>..."}, {"from": "gpt", "value": gold}]}.
    The human turn is split at every "<image>" tag; the k-th tag is replaced by the k-th image.
    Returns a list of messages whose content is a list of {"type": "text", "text"} / {"type": "image", "path"}.
    """
    parts = row["conversations"][0]["value"].split("<image>")
    images = row.get("image") or []
    if len(parts) - 1 != len(images):
        raise ValueError(f"{row['id']}: {len(parts) - 1} <image> tags but {len(images)} images")
    content = []
    for i, part in enumerate(parts):
        if i > 0:
            content.append({"type": "image", "path": str(Path(root) / images[i - 1])})
        if part:
            content.append({"type": "text", "text": part})
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": content})
    return msgs
