# data/

The datasets are **not stored in this git repository**. They live in the Hugging Face dataset `ACERobotics/ACE-Brain-0.7-Bench`, one
archive per subset, and are fetched into this folder with

```bash
HF_REPO=ACERobotics/ACE-Brain-0.7-Bench python prepare.py --from-hub [--sets robofac robobench eriq cosmos progress] [--data-dir DIR]
```

which checks every archive against the hub's `SHA256SUMS`, extracts `dataset/<Subset>/...` to `data/<benchmark>/...` and re-verifies the files
against the per-folder `SHA256SUMS`. Set `TASKBENCH_DATA=/path/to/data` (or `--data-dir`) to use a folder outside the repository.

| Folder | Hub archive | Content | Size (extracted) |
|---|---|---|---|
| `robofac/` | `RoboFAC.tar.gz` | `test.jsonl` (5,532), `images/` (9,632), `SHA256SUMS`, `MANIFEST.json` | 241 MB |
| `robobench/` | `RoboBench-NL.tar.gz` | `items.jsonl` (4,007 incl. 28 excluded), `images/` (14,038), `rewrite_mapping.json`, `q1_dags.json`, `SHA256SUMS` | 2.5 GB |
| `eriq/` | `ERIQ.tar.gz` | `items.jsonl` (3,905), `images/` (10,441), `subsets_11.json`, `SHA256SUMS` | 3.0 GB |
| `cosmos/` | `Cosmos.tar.gz` | 16 parquet shards in `data/` (510 questions x 32 frames), `SHA256SUMS`, notices | 6.1 GB |
| `progress/` | `progress.tar.gz` | `questions.jsonl`, `labels.jsonl`, `image_manifest.jsonl`, `images/` (798 frames) | 15 MB |
| `egoplan/`, `ladder/` | `ids.tar.gz` | id lists, sha256 manifests, notices only (the data is built locally by `benchmarks/<name>/prepare.py`) | 3.5 MB |

This git repository keeps only the small files the code needs without a download: `egoplan/` and `ladder/` (ids, `MANIFEST.json`, notices),
`progress/` (questions, labels, frame manifest; the frames come from the hub), `cosmos/SHA256SUMS`, `robofac/MANIFEST.json`, `eriq/subsets_11.json`.
Licenses of the data differ per subset; see `LICENSES.md` and the dataset card.
