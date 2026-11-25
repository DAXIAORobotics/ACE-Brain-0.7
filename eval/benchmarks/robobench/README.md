# RoboBench (natural-language version, undistorted images)

A natural-language conversion of the planning part of **RoboBench** (arXiv:2510.17801, https://github.com/yulin-luo/RoboBench,
CC BY 4.0). Models answer in plain English instead of calling a fixed skill library, and answers are scored by an LLM judge
(Q1, Q2) or a yes/no rule (Q3).

> **The scores are NOT comparable to the original RoboBench leaderboard.** Questions, references, judging and images were all
> modified (see "What was changed"). Only compare numbers produced by this exact release.

## Contents (`data/robobench/`, 2.5 GB)

| file | what |
|---|---|
| `items.jsonl` | 4,007 questions: Q1 task decomposition 1,973, Q2 next subtask 842, Q3 completion check 1,192. LLaVA-style (`image` list, `conversations`, optional `system`), plus `answer`, `answer_status`, and for Q1/Q2 the `rewrite` record (original function-call reference used by the scorer). Image paths are relative to `data/robobench/`. |
| `images/` | 14,038 frames (PNG/JPEG as in the source), `SHA256SUMS` lists their sha256. |
| `q1_dags.json` | ground-truth dependency graph of every eligible Q1 item (copied from the original benchmark), used in the Q1 judge prompt. |
| `rewrite_mapping.json` | per converted item: id, row in the source table, question type, `answer_status` (`converted_from_source` or `needs_review`). |
| `LICENSE-NOTICE.md` | CC BY 4.0 attribution and the list of modifications. |

Eligible items: **3,979** (the 28 Q1 items with `answer: null` / `needs_review` are shipped but excluded from inference and scoring).

## What was changed relative to the original RoboBench

1. **Questions (Q1, Q2).** The original prompts ask for function calls from a skill library (`move_to`, `grasp`, ...) with in-prompt
   examples. Here the skill library and examples are removed and the model is asked for a numbered list of short natural-language
   steps (Q1) or one natural-language next subtask (Q2).
2. **References (Q1, Q2).** The original function-call answers were converted to sentences by a **fixed, deterministic template**
   (`answer_status: converted_from_source`; e.g. `move_to(none, X)` -> "Move the empty gripper to X."). The original calls are kept in
   `rewrite.parsed_steps` and are what the Q2 judge compares against. 28 Q1 references could not be verified after conversion and are null.
3. **Q3** (completion yes/no) is verbatim from the original, including its in-prompt skill library and examples.
4. **190 navigation questions** were removed.
5. **Images.** 595 images (used by 133 questions) were replaced by **generative image-model rewrites** meant to remove lens distortion.
   No lens calibration was used, and consistency of the rewritten images with the labels was **not re-validated**. The remaining
   13,443 images are byte-identical to the source frames.
6. **Scoring** is a local, LLM-judge adaptation of the original rubric (see below), run with a different judge than the original.

**The conversion script was not found.** What ships is the converted data and the item-level mapping (`rewrite_mapping.json`), not
the converter; the conversion cannot be re-run from this repository.

Upstream data licenses: the frames come from third-party robot datasets (judging from the ids: Open X-Embodiment sub-datasets such as DROID,
RH20T-derived frames, RoboMIND-style data and others). Each frame keeps the license of its upstream source; released as part of ACE-Brain-0.7-Bench.

Data is hosted on the Hugging Face Hub: `HF_REPO=ACERobotics/ACE-Brain-0.7-Bench python prepare.py --from-hub --sets robobench` downloads it into `data/` and verifies `SHA256SUMS` (local copies work as before).

## Running

```bash
python benchmarks/robobench/prepare.py                      # verify the packaged data (counts + sha256)
python benchmarks/robobench/eval.py --model <HF checkpoint dir> --out results/robobench/mymodel
export JUDGE_BASE_URL=<OpenAI-compatible endpoint that accepts images> JUDGE_API_KEY=... JUDGE_MODEL=qwen3.8-flash
python benchmarks/robobench/score.py --pred results/robobench/mymodel/predictions.jsonl --json results/robobench/mymodel/score.json
```

* Generation (`config.json`): the 3,979 eligible items, greedy, thinking off, `max_tokens` 1024, per-image pixels 65,536 .. 16,777,216, up to 40 images per
  question (Q1 questions carry the scene image, Q2/Q3 up to 16 frames), vLLM by default (`--backend hf|openai` also available).
* The judge is configured through `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` (default `qwen3.8-flash`, the judge of the paper),
  `JUDGE_CONCURRENCY` (default 8), see `taskbench/judge.py`. The Q1 judge request contains the first scene frame, so the judge must be a vision-language model.
  Hosted judge back-ends can change over time and temperature 0 is not bit-exact, so a fresh run can differ slightly from the paper numbers.
* The judge journal (`<json stem>.judgments.jsonl`) is resumable. `--judgments FILE` re-scores from a saved journal **without any judge call**
  (it also checks the journal's prompt hashes against the prompts rebuilt from data and predictions).

## Scoring

* **Q1 (1,945 scored items).** Judge sees the first scene frame, the reference (text + original actions), the gt DAG and the candidate plan and returns two integers 0..10
  (node correctness, task completion): `score = (node + completion) / 20 * 100`. The original Q1 judge prompt is in `assets/original_q1_prompt.txt`; the
  natural-language adaptation is appended in `rules.py` (`ADAPT_Q1`).
* **Q2 (842).** Two judge stages: EXTRACT the actions the candidate states (without seeing the reference), then COMPARE with the reference. Scores S in {0,1}, O and P in {0,.5,1};
  code enforces the dependency rules (S forced to 0 if the extraction is not a single step; P forced to 0 if S = 0 or O = 0); `score = (S + O + P) / 3 * 100`. An empty reply scores 0 without a judge call.
* **Q3 (1,192).** Yes/no rule: case-insensitive `yes` / `no`, optional `Answer:` prefix and final punctuation, thinking blocks removed; 100 if equal to the reference, else 0 (unparsable = 0).
* Judge calls: temperature 0, thinking off, `max_tokens` 4096, strict JSON validation of every reply (duplicate keys, wrong types or out-of-range values are invalid),
  per stage attempt 0 plus 3 retries.
* **Overall (headline `score_0_100`) = mean of the item scores over all 3,979 eligible items** (not the mean of the three split means). Per-split means are reported too.

### Items the judge cannot score (rule kept identical to the paper's board)

If an item still has no valid judge verdict after the retries and after up to 6 re-judge rounds (`--rejudge-rounds`), it is counted **as correct (score 100)**
in the headline. This is a deliberate rule applied uniformly to every model. The number of such items is printed right next to the score
(`unscored_forced_correct`) and stored in the json; the score **without** this rule (mean over scored items only, `score_without_rule`) is printed as a secondary line.
Items that were never sent to a judge (for example with `--skip-judge`) are *not* covered by the rule: the headline is then `None` and `not_judged` is reported.
In the paper's runs this rule affected at most one item (at most 0.03 points).

## Reproduction check

Re-scoring the saved predictions and judge journal of the reference model (no judge calls): `score_0_100 = 67.41727402194856`,
`unscored_forced_correct = 0` (Q1 67.7224, Q2 52.0586, Q3 77.7685), identical to the board value; all 2,787 journal prompt hashes matched the prompts rebuilt
by this code (so prompts and images are byte-identical to the ones the judge saw).

## License

Code: Apache-2.0. Data: CC BY 4.0 (RoboBench authors), modified, see `data/robobench/LICENSE-NOTICE.md`.
