# Licenses

| Component | License |
|---|---|
| Code of this repository (`taskbench/`, `benchmarks/*/*.py`, `scripts/`) | Apache-2.0 (`LICENSE`) |
| Vendored code (EmbodiedEvalKit port in `benchmarks/cosmos/vendor/`, official-LADDER definitions loaded from the upstream clone, RoboBench judge prompt in `benchmarks/robobench/assets/`) | upstream licenses, see the headers and `NOTICE` files next to each |
| `data/robofac/` | MIT (RoboFAC authors); test split only |
| `data/cosmos/` | CC BY 4.0 (Cosmos-Reason1-Benchmark) with RoboVQA (Apache-2.0) and RoboFail (MIT) parts; frames are **modified** (generative rewrite) |
| `data/robobench/` | CC BY 4.0 (RoboBench); questions/references **converted**, 595 images **rewritten**; upstream image sources may carry stricter terms |
| `data/eriq/` | CC BY-SA 4.0 (ERIQ); this folder is share-alike |
| `data/progress/` | question/label files and an image manifest in git; the 798 frames come with the Hugging Face archive and keep their source licenses (BridgeData V2/FMB CC BY 4.0, RT-1 Apache-2.0, LIBERO/DobbE MIT, OXE sub-datasets per card; Table30 and tabletop collections released as part of the benchmark) |
| EgoPlan, LADDER | nothing redistributed; you download the official data yourself under its license (EgoPlan videos: EPIC-KITCHENS-100 CC BY-NC 4.0 and Ego4D agreement; LADDER CC BY 4.0, RoboLab portion CC BY-NC-SA 4.0) |

Please cite the original benchmark papers when you use their data (see each `benchmarks/<name>/README.md`).
