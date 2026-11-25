# LADDER offline VLM benchmark - build-from-source release

**No frames and no questions are shipped.** `download.sh` fetches the official data and code at pinned revisions,
`prepare.py` rebuilds our exact evaluation file, and `data/ladder/{MANIFEST.json, ids.json}` let you verify it.

## Sources (pinned)
* Data: Hugging Face `INSAIT-Institute/LADDER-Offline-VLM-Benchmark`, revision `1a1cb958d9e0e47d4b4497bb79a594dc66a3be2a`
  (config `decomposition` and `subtask_states`, one parquet each). Licence **CC BY 4.0**. **Caveat:** the RoboLab-Long portion
  shows RoboLab assets, which are **CC BY-NC-SA 4.0** (non-commercial, share-alike); rows with `meta.benchmark` starting
  with `robolab` are affected. The other sources in the data are SimplerEnv- and RobotArena-based; check their terms too.
* Code: https://github.com/insait-institute/ladder (MIT), commit `8703b9d02bfe5e20b7d544bd5ecd71234586246a`.
  Prompts, reply parsers, thresholds and the scoring functions are **not restated here**: `ladder_official.py` extracts the
  named top-level definitions from your clone with Python's `ast` module, so you do not need the official bazel environment.

## Protocol
Four item families built from the official sources (3,409 rows, 883 images; JPEG q85 as the official client re-encodes them;
RoboLab decomposition frames are letterboxed to the published size first):

| family | rows | what is asked | metric |
|---|---|---|---|
| `ladder_decomp` | 145 | decompose the task into subtasks from the image | exact match of the multiset of head-noun goals |
| `ladder_verify` | 562 | has the subtask been completed (count reply) | balanced accuracy (mean of TP rate and TN rate) |
| `ladder_state` | 2,142 queries on 562 frames | one verification query per distinct subtask | frame counts as correct only if all its queries are right |
| `ladder_next` | 560 | next subtask (all-done frames are skipped) | official classifier says "valid" |

Headline = mean of the four (decomposition exact, verification balanced accuracy, state all-correct, next valid); a missing
component counts as 0. Rows carry a per-row `"system"` prompt (the official system prompts); `taskbench.io.item_messages`
renders it as a system message. Generation: greedy, non-thinking, **max_tokens 8192**, pixels 65,536..16,777,216
(`config.json`). Scores are not comparable with the official LADDER leaderboard where its settings differ (we run offline, single
greedy sample per item, no filtering of items). The paper reports **79.025** (93.1 / 78.6 / 67.6 / 76.8) for the model it evaluates.

## Reproduce
```bash
bash benchmarks/ladder/download.sh data/ladder/raw            # parquet (~330 MB) + git clone at the pinned commit
python benchmarks/ladder/prepare.py --raw-dir data/ladder/raw --out data/ladder     # needs pyarrow>=20, pillow, numpy, pyyaml
python benchmarks/ladder/eval.py --model /path/to/model --out results/ladder/mymodel
python benchmarks/ladder/score.py --pred results/ladder/mymodel/predictions.jsonl --repo-dir data/ladder/raw/repo
```
`prepare.py` finishes with a PASS/FAIL check: sha256 of the built `ladder_all.jsonl` must equal `MANIFEST.json`
(byte-identical), the row-id list/order and the set of image paths must equal the shipped lists, and all images must exist.
Image byte equality is reported as INFO only: JPEG bytes can change with the PIL/libjpeg build. In our check, a rebuild in a
different environment than the original produced the identical question file and identical bytes for all 883 images, but we do not
promise that for other library versions.
Older pyarrow (19) cannot read these parquet files ("Repetition level histogram size mismatch"); use pyarrow >= 20.

## Notes
* `score.py` also needs the official clone (`--repo-dir` or `$LADDER_REPO_DIR`).
* Verification replies that contain no usable count are excluded from TP/TN (reported as `unanswered`), as in the official summary.
* Files: `download.sh`, `prepare.py`, `ladder_official.py`, `eval.py`, `score.py`, `config.json`; `data/ladder/{MANIFEST.json, ids.json, NOTICE.md}`.
