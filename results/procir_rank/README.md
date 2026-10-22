# Rank metrics and per-query rankings on the photos (lab server, 21-22 Oct 2026)

Four scorings of the full FashionMV val set on the Fashion200K photos (`data/images_official_hr`,
the paper's images), each keeping every query's target rank, source rank and top-51 list, so that
all metrics below are computed on CPU from the same files.

| Name | Model | Scored with |
|---|---|---|
| `released` | the authors' ProCIR checkpoint | `eval_metrics.upstream_dump` (the steps of upstream `evaluate.py`) |
| `mt_align` | `photo_mt_align_s42`, step 644 | our `procir_train.retrieval_eval`, upstream protocol |
| `single` | `photo_single_s42`, step 644 | our `procir_train.retrieval_eval`, upstream protocol |
| `mt_align_upstream` | `photo_mt_align_s42` export | `eval_metrics.upstream_dump`, as a cross-check of the two tools |

Server code: commit `8502ee8` (`server/git_commit.txt`), queue `src/server/queue_rank_eval.sh`,
log `server/rank_eval.log`. Tables: `python -m eval_metrics.report` (command below) on the
downloaded dumps, which stay outside git (`data/rank_eval_dl/`, ~0.4 GB).

## Checks

- **Recall is unchanged.** For all four scorings, R@1/5/10 equal the earlier evaluations of the
  same model on the same images to the second decimal (`server/rank_eval.log`: largest difference
  0.00 points each), and the report recomputes R@1/5/10 from the stored ranks and stops on a mismatch.
- **The two tools agree per query.** `mt_align` against `mt_align_upstream`: no metric differs by
  more than 0.13 points, every 95% CI contains 0, and McNemar p >= 0.14 on every dataset. The ranks
  of 88-94% of the queries are identical; the rest differ by tiny score differences between the tools.
- Times: `released` 1 h 30, `mt_align` 3 h 43, `single` 3 h 33, `mt_align_upstream` 1 h 31. Our
  tool is ~2.4x slower than upstream's DataLoader even with image prefetching.

## Main results (mean over the three datasets, source kept as in the paper)

| Model | R@1 | R@5 | R@10 | R@50 | MRR | NDCG@10 | RLR@1 |
|---|---|---|---|---|---|---|---|
| released | 43.90 | 81.12 | 89.46 | 98.37 | 60.36 | 67.07 | 35.49 |
| mt_align | 32.94 | 75.97 | 86.15 | 97.60 | 52.13 | 59.92 | 49.72 |
| single | 26.57 | 73.82 | 84.76 | 97.23 | 47.54 | 56.05 | 57.56 |

MT+Align minus single-turn (paired bootstrap over queries, 1,000 resamples within each dataset):

| | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|
| source kept | +6.36 [+5.98, +6.80] | +2.16 [+1.83, +2.47] | +1.39 [+1.16, +1.64] | +4.59 [+4.37, +4.83] | +3.86 [+3.67, +4.06] |
| source removed | +3.60 [+3.21, +3.97] | +1.97 [+1.67, +2.25] | +1.26 [+1.03, +1.51] | +2.88 [+2.65, +3.11] | +2.54 [+2.35, +2.73] |

Per dataset every difference is significant: no bootstrap resample reaches 0 (p < 0.002, the
resolution of 1,000 resamples) and McNemar p < 1e-4 at R@5 and R@10. Per query,
MT+Align ranks the target higher than single-turn about twice as often as lower (Fashion200K 6,415
against 3,260, the rest equal).

How to read it:

- **The gain is largest at the top.** R@1 gains +6.4 but R@10 only +1.4, and MRR (+4.6) moves more
  than R@5. Most of MT+Align's gain is moving the target from rank 2-3 to rank 1.
- **Much of that is the model no longer returning the query's own product.** With the source kept,
  the source is the first result for 57.6% of single-turn queries, 49.7% for MT+Align and 35.5% for
  the authors' checkpoint (RLR@1). Removing the source halves the R@1 gain (+6.4 to +3.6) while the
  R@5 gain barely changes (+2.2 to +2.0). Reading the source and the text together (MT+Align's
  second turn) seems to keep the query from collapsing onto the source image.
- **Depth.** When the target is missed, it is rarely far down: the median query has 0-0.04% of
  the gallery above its target, under 0.2% of queries are below rank 500, and practically none is
  in the bottom half of the list. The deepest are listed per model in `examples.json`
  (`deepest_targets`), with the text and the top-5; for our two models the source is the first
  result in 52 of those 60 queries.
- **Views.** Among the groups of 100+ queries, modifications involving back+side views have the
  lowest R@5 for every model and dataset (tied with back+front+side for single-turn on FashionGen;
  e.g. Fashion200K MT+Align 62.9 against 72.2 for back+front), but those groups are small (105-256
  queries) and have no CI here.
- Against the authors' checkpoint, our MT+Align is lower on every metric (mean R@5 -5.1, MRR -8.3,
  same protocol); the gap is about as large on DeepFashion and FashionGen, whose images are the
  same as the authors', which points at our reduced training (25% of the data, LoRA, 644 steps).

## Limits

- One training seed (42) per configuration. The bootstrap intervals cover query sampling only, not
  training randomness, which earlier runs put at 1.2-1.7 R@5 points on unchanged data; so the
  +2.16 R@5 gain is certain for these two models, not yet for the method.
- One target per query: mAP@K equals MRR@K, and NDCG@K is 1/log2(rank + 1) inside the top K, a
  different weighting of the same rank, not extra information. Graded relevance is not available.

## Files

`metrics.json` (per model and dataset: both protocols, leak rates, rank histogram, rank bands,
depth, views groups), `compare.json` (deltas, CIs, p, McNemar, per-query rank changes),
`examples.json` (deepest targets per model; largest gains and losses per pair, with top-5 lists),
`tables.md` (all of it as tables), `server/` (queue log, per-dataset meta with timings, the
scorings' own result files).

```
python -m eval_metrics.report --data_dir data/FashionMV_hf/data/data --out results/procir_rank \
  --model released=data/rank_eval_dl/runs/rank_eval/released --model mt_align=data/rank_eval_dl/runs/rank_eval/mt_align \
  --model single=data/rank_eval_dl/runs/rank_eval/single --model mt_align_upstream=data/rank_eval_dl/runs/rank_eval/mt_align_upstream \
  --compare mt_align:single --compare mt_align_upstream:released --compare mt_align:mt_align_upstream
```
