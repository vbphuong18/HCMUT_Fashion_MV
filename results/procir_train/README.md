# ProCIR training runs at reduced scale (lab server, 9-10 Oct 2026)

Two training runs of our ProCIR reimplementation (`src/procir_train`) on 25% of FashionMV, plus three
reference evaluations on the same images and the full validation split. Everything here was copied
from `runs/` on the server (`procir_results.tgz`); checkpoints and exported models are not in the repo.

## How much data these runs trained on

**25% of the training pool = 41,235 triplets (21.9% of all 188,015 FashionMV train triplets).**
Both runs use the same triplets (`<run>/data_stats.json`, `train_fraction: 0.25` in `<run>/config.yaml`).

| Source | All train triplets | Dev (5% of products) | Dropped (link train and dev) | Train pool | **Used = 25% of pool** | Share of all |
|---|---|---|---|---|---|---|
| DeepFashion | 16,399 | 392 | 1,678 | 14,329 | **3,582** | 21.8% |
| Fashion200K | 98,800 | 2,306 | 9,834 | 86,660 | **21,665** | 21.9% |
| FashionGen | 72,816 | 1,705 | 7,157 | 63,954 | **15,988** | 22.0% |
| **Total** | **188,015** | 4,403 | 18,669 | 164,943 | **41,235** | **21.9%** |

- One epoch at batch 64 is 644 steps, so 41,216 triplets were actually seen (the last 19 do not fill a batch).
- The 25% is drawn per source, so the three sources keep their proportions.
- Evaluation is on 100% of the validation split: 32,718 triplets (5,188 / 18,499 / 9,031).
- The whole pool (`train_fraction=1.0`) would be 164,943 triplets, 2,577 steps.

## Setup

- GPU: one MIG `1g.35gb` slice of an NVIDIA H200 NVL (sm90), shared, about 11 GiB free.
- torch 2.12.0+cu130, transformers 5.19.0, peft 0.21.2, flash-linear-attention 0.5.2 with Triton 3.7.1;
  `causal_conv1d` not installed (PyTorch reference path). Full list: `*/env.json`.
- Code: training at commit `5978800`, reference evaluations at `e4fe819` (`git_commit.txt`).
- Images: `data/images_official_hr` (DeepFashion high-res, Fashion200K cropped, FashionGen); every
  triplet has images. Train 188,015 triplets, val 32,718.
- Config: `configs/procir_fast.yaml` with `train_fraction=0.25` (41,235 triplets: 3,582 DeepFashion,
  21,665 Fashion200K, 15,988 FashionGen; 644 steps at batch 64, one epoch), LoRA r=16, 3 views at
  336^2, `chunk_size=4`, gradient checkpointing, seed 42. Dev = 5% of train products.

## How each file was produced

`queue.sh` holds the exact commands of the two runs and `ref_queue.sh` those of the references;
`logs/queue.log` and `logs/ref_queue.log` hold their start and exit times (UTC).

| Path | Produced by |
|---|---|
| `<run>/config.yaml`, `env.json`, `data_stats.json`, `train_log.jsonl` | `procir_train.train` (loss every 10 steps, dev eval every 100) |
| `<run>/STATUS` | `procir_train.pipeline` |
| `fast_mt_align_s42/eval_upstream/eval_results.json` | unmodified `external/FashionMV/evaluate.py` on the merged export of `ckpt/best.pt` |
| `fast_single_s42/eval_upstream.json` | `procir_train.retrieval_eval --protocol upstream` (upstream `evaluate.py` has no single-turn query) |
| `<run>/eval_train.json` | `procir_train.retrieval_eval --protocol train` (3 views, 336^2) |
| `ref_procir_released/eval_upstream/eval_results.json` | upstream `evaluate.py` on the authors' checkpoint `yuandaxia/ProCIR` |
| `ref_clip/eval.json`, `ref_fashionclip/eval.json` | `src/baselines/clip_val_eval.py` (late fusion, alpha sweep) |
| `speed.json`, `pilot_c*_gc*/` | the notebook's speed pilot (12 steps per setting) |
| `pilot/`, `pilot_summary.json` | the first 20-step pilot; `pilot/` itself is an aborted second attempt |
| `logs/<name>.log` | stdout of that job |

## Results

Upstream protocol (up to 5 views, 512^2, source kept in the gallery), full val: 5,188 / 18,499 / 9,031
queries, galleries of 2,791 / 10,720 / 5,292 products. R@1 / R@5 / R@10.

| Model | DeepFashion | Fashion200K | FashionGen | Mean R@5 |
|---|---|---|---|---|
| Authors' checkpoint (paper: R@5 89.2 / 77.6 / 75.0) | 51.04 / 89.26 / 95.10 | 30.92 / 67.71 / 76.97 | 39.59 / 75.82 / 86.05 | 77.60 |
| Ours, MT+Align (`best.pt` = step 400) | 29.36 / 80.47 / 90.79 | 22.36 / 66.22 / 77.34 | 29.39 / 70.72 / 81.76 | 72.47 |
| Ours, single-turn (`best.pt` = step 644) | 26.41 / 78.41 / 89.51 | 23.11 / 66.18 / 77.48 | 27.76 / 69.19 / 80.67 | 71.26 |
| FashionCLIP, alpha 0.25 | 25.00 / 59.83 / 71.59 | 11.76 / 30.09 / 38.27 | 14.39 / 39.01 / 50.45 | 42.98 |
| FashionCLIP, alpha 0.5 | 0.58 / 48.46 / 65.56 | 0.28 / 26.34 / 36.75 | 0.69 / 37.24 / 51.33 | 37.35 |
| CLIP ViT-B/32, alpha 0.5 | 0.89 / 27.49 / 41.79 | 0.36 / 10.43 / 16.14 | 0.49 / 14.17 / 22.36 | 17.36 |

Train protocol (3 views, 336^2), same tool for both runs, R@1 / R@5 / R@10 with the source kept:

| Model | DeepFashion | Fashion200K | FashionGen |
|---|---|---|---|
| Ours, MT+Align | 28.64 / 76.39 / 86.51 | 23.14 / 64.04 / 75.10 | 31.16 / 70.96 / 81.87 |
| Ours, single-turn | 21.92 / 73.03 / 85.33 | 23.23 / 64.00 / 75.06 | 29.39 / 69.57 / 80.53 |

Source-excluded recalls are in the JSON files of our tools and the CLIP script.

## Timings

| Job | Wall time |
|---|---|
| MT+Align training | median 40.8 s/step (about 7.3 h of steps for 644; stopped once for the speed pilot and resumed from step 43) |
| Single-turn training | median 22.7 s/step; 5 h 17 min from start to the last checkpoint, dev evals included |
| Single-turn export + both evals | 4 h 38 min |
| Authors' checkpoint, upstream `evaluate.py` | 67 min including a 3 min download |
| CLIP, FashionCLIP | 12 min, 11 min |

## Caveats

- One seed per configuration.
- The two runs were scored at different steps: dev selection (mean R@5 with source, at most 500
  queries per source) picked step 400 for MT+Align and step 644 for single-turn. Dev R@5 is near its
  ceiling on these small galleries, so the choice is noisy.
- The authors' checkpoint reproduces the paper on DeepFashion and FashionGen with these images but
  is about 10 points lower on Fashion200K, so our Fashion200K images (crops) are not the paper's.
- Upstream numbers of the two runs come from two tools; the train protocol uses one tool for both.
