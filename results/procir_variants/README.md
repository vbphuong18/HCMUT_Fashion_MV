# ProCIR training on the corrected data and the Fashion200K image variants (lab server, 11-14 Oct 2026)

Five training runs of our ProCIR reimplementation (`src/procir_train`, commit `2d2e4a5`) and the
authors' checkpoint, on the same 25% of FashionMV as `results/procir_train/` (41,235 triplets:
3,582 DeepFashion, 21,665 Fashion200K, 15,988 FashionGen; 644 steps at batch 64; LoRA r=16; 3 views
at 336^2; seed 42) but with correct Fashion200K images. Every run is scored at its last step (644).
Copied from `runs/` on the server (`variants_results.tgz`); checkpoints and exports are not in the repo.

## Fashion200K image variants

An image is either in the Fashion200K release (a label line, 58% of FashionMV's Fashion200K images) or not, and
each group is either cropped to its garment or left as the photo (`results/data/f200k_crop_fix/`).
DeepFashion and FashionGen are identical in every run.

| | Other images cropped | Other images as photos |
|---|---|---|
| Release images cropped | `crop` | `mixed` |
| Release images as photos | `mixed_other` | `photo` (the paper's images) |

`queue_f200k_variants.sh` is the script that ran (built the variants, checked each with the authors'
checkpoint, trained, scored); `logs/variants_queue.log` has its timeline.

## Main result: MT+Align against single-turn on `photo`

Same tool for both (`procir_train.retrieval_eval`, upstream protocol: up to 5 views at 512^2, source
kept in the gallery), full val. R@1 / R@5 / R@10:

| Run | DeepFashion | Fashion200K | FashionGen | Mean R@5 |
|---|---|---|---|---|
| MT+Align (`photo_mt_align_s42`) | 37.09 / 83.87 / 92.75 | 27.45 / 72.51 / 83.62 | 34.27 / 71.54 / 82.08 | 75.97 |
| Single-turn (`photo_single_s42`) | 27.72 / 81.55 / 91.46 | 20.64 / 69.86 / 81.77 | 31.36 / 70.04 / 81.04 | 73.82 |
| Difference, R@5 | +2.32 | +2.65 | +1.50 | +2.16 |
| Paper (pretrained init), R@5 difference | +2.5 | +0.4 | +3.0 | +2.0 |

The MT+Align run scored with the unmodified upstream `evaluate.py` gives 83.87 / 72.48 / 71.60
(R@5), within 0.06 of our tool, so the two tools agree. Source excluded from the ranking, R@5:
MT+Align 86.93 / 75.40 / 74.12, single-turn 84.68 / 73.17 / 72.71. Train protocol (3 views at
336^2), R@5: MT+Align 78.26 / 67.07 / 71.44, single-turn 73.28 / 64.88 / 69.77.

The authors' checkpoint on the same images: 89.26 / 78.29 / 75.82 (R@5; paper 89.2 / 77.6 / 75.0).

## Image variants

Fashion200K R@5, upstream `evaluate.py`, full val (18,499 queries, gallery 10,720):

| Model | on its own images | on `photo` |
|---|---|---|
| Authors' checkpoint, `photo` | 78.29 | 78.29 |
| Authors' checkpoint, `mixed_other` | 76.04 | |
| Authors' checkpoint, `mixed` | 75.72 | |
| Authors' checkpoint, `crop` | 74.86 | |
| MT+Align trained on `photo` | 72.48 | 72.48 |
| MT+Align trained on `mixed_other` | 72.69 | 73.28 |
| MT+Align trained on `mixed` | 72.14 | 72.62 |
| MT+Align trained on `crop` | 73.07 | 70.84 |

All three datasets for the four MT+Align runs, R@5 on their own images:

| Trained on | DeepFashion | Fashion200K | FashionGen | Mean |
|---|---|---|---|---|
| `photo` | 83.87 | 72.48 | 71.60 | 75.98 |
| `mixed_other` | 84.60 | 72.69 | 71.96 | 76.42 |
| `mixed` | 83.40 | 72.14 | 70.45 | 75.33 |
| `crop` | 83.94 | 73.07 | 72.10 | 76.37 |

How to read it:

- The DeepFashion and FashionGen images are the same in all four runs, yet their R@5 spreads over
  1.2 and 1.65 points: that is the run-to-run variation of this setup, one seed each. The
  Fashion200K differences between the four runs on their own images (0.9 points) are within it.
- What stands out is a mismatch between training and test images: the authors' checkpoint (trained
  on photos) loses 2.3 to 3.4 points on cropped variants, and our `crop` model loses 2.2 points
  when scored on the photos. Losing the most when both groups are cropped (74.86) rather than one
  (75.72, 76.04) is consistent with that.

## Training

Median seconds per step: MT+Align 42.0 to 43.0, single-turn 26.1. Dev score (mean R@5 on the dev
split) rises to the last evaluation in every run (MT+Align on `photo`: 90.87 at step 100, 95.15 at
step 644), unlike the run on the wrongly cropped data in `results/procir_train/`, whose best dev step
was 400.

## Files

Per run: `config.yaml`, `data_stats.json`, `env.json`, `train_log.jsonl`, `STATUS`,
`eval_upstream/eval_results.json` (upstream `evaluate.py`), `eval_photos/` (variant models on the
photos), `eval_upstream_ourtool.json` and `eval_train.json` (our tool). `check_<variant>/` holds the
authors' checkpoint on each variant. `logs/` has every job's output. `git_commit_variants.txt` is the
commit the server ran.
