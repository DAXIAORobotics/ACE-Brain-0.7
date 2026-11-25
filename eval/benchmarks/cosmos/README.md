# Cosmos-Reason1 (embodied multiple choice), undistorted frames

510 multiple-choice questions (domains: agibot 100, bridgev2 100, holoassist 100, robofail 100, robovqa 110), each with 32 video
frames, from the Cosmos-Reason1 benchmark (NVIDIA, arXiv 2503.15558). Headline metric: accuracy in percent, **mean of three
seeds** (3407, 3408, 3409; 1,530 inferences). Higher is better.

## Important: what "undistorted" means here

For 40 questions (1,280 frames, all in the `agibot` domain) the frames are **generative rewrites** produced by a generic
image-generation model asked to remove fisheye distortion. It is **not** a calibrated geometric undistortion (no lens parameters
were used), details may be altered, and the labels, answers and scores were **not re-validated** against the rewritten frames.
The other 470 questions use the original frames. Details and the list of affected question ids: `data/cosmos/UNDISTORT_NOTE.md`.
Absolute numbers are therefore not comparable with numbers on the original frames; all models in the paper use this same set.

## Data

`data/cosmos/data/*.parquet` (16 shards, 6.47 GB = 6.0 GiB, checksums in `data/cosmos/SHA256SUMS`). Columns: `id`, `videos`
(32 encoded frames), `question`, `choices`, `answer`. Licenses and attribution: `data/cosmos/LICENSE-NOTICE.md`
(CC BY 4.0 for the NVIDIA benchmark; RoboVQA Apache-2.0; RoboFail MIT).
Frames keep the licenses of their upstream datasets (BridgeData V2, AgiBot World, HoloAssist); released as part of ACE-Brain-0.7-Bench.

Data is hosted on the Hugging Face Hub: `HF_REPO=ACERobotics/ACE-Brain-0.7-Bench python prepare.py --from-hub --sets cosmos` downloads it into `data/` and verifies `SHA256SUMS` (local copies work as before).

## Run

```bash
python benchmarks/cosmos/eval.py --model /path/to/hf_checkpoint --out results/cosmos/mymodel      # 3 seeds, 1,530 inferences
python benchmarks/cosmos/eval.py --model /path/to/hf_checkpoint --seeds 1                          # quick mode: first seed only (510)
python benchmarks/cosmos/eval.py --model /path/to/hf_checkpoint --limit 20 --seeds 1               # smoke test
python benchmarks/cosmos/score.py --pred results/cosmos/mymodel --json results/cosmos/mymodel/score.json
```

Requirements: one GPU, `vllm>=0.18`, `transformers`, `qwen_vl_utils`, `pyarrow`, `pillow`. Every seed starts its own process
and vLLM engine. The frames are decoded 16 questions at a time (`batch_size` in `config.json`), so host memory stays small
(the original kit holds all 510 x 32 frames in RAM).

## Protocol (what is reproduced)

Port of EmbodiedEvalKit (MIT, https://github.com/pickxiguapi/EmbodiedEvalKit, commit 5ec3a60e03894ea5b9215127b47afaf39d969dae)
`eval_cosmos.py`, vendored in `vendor/` (license + NOTICE there; no local clone of the kit is needed):
* prompt = question, choices (one per line), then `Please answer with the letter of your choice (A, B, C, or D).`;
* system prompt `You are a helpful assistant.`; the 32 frames are one video input processed with `qwen_vl_utils`
  (`image_patch_size=16`, video metadata); `limit_mm_per_prompt` image 16 / video 1;
* vLLM, tensor parallel 1, `max_model_len` 20000, `gpu_memory_utilization` 0.8, greedy (temperature 0, top_p 1, top_k -1),
  repetition_penalty 1.05, presence_penalty 0, max_tokens 4096, engine seed = run seed;
* Qwen3.5 checkpoints (`model_type: qwen3_5`) use the chat template with `enable_thinking=False` (empty think block, asserted for
  every prompt); other model types use their template unchanged; no answer-tag extraction;
* answer extractor = the kit's `extract_answer_from_text` (last `Answer: (X)` pattern, else the last `(X)`, else the first
  standalone letter); a reply without any letter is wrong and counted in `no-letter`.
`score.py` reports per-seed accuracy, the mean over seeds (the number to quote), per-domain means and the no-letter count. It also
accepts EmbodiedEvalKit result files (`COSMOS_suite.json`, `results[].question_id/raw_output/ground_truth`) for parity checks.

## Reproduction check

Scoring the saved result files of the paper model with the release scorer gives seed accuracies 83.3333 / 83.5294 / 83.5294 and
**mean 83.4641** (paper table 83.46); gold answers from the shipped parquet agree with the saved ground truths (0 mismatches).
The end-to-end inference path of `eval.py` (parquet decoding, prompt, `qwen_vl_utils` input) was exercised on CPU
(32 frames -> tensor `[32, 3, H, W]`); a full GPU run of `eval.py` was not repeated for this release, so small differences from
batching/engine nondeterminism between the original kit run and this port are possible (the seeds already differ by about 0.1-0.2 points).
