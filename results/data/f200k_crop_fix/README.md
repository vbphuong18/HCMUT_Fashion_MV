# Fashion200K cropping bug (found and fixed 2026-10-10)

`src/data/crop_f200k.py` rebuilds the released Fashion200K crops from the photos at their URLs. For
images without a line in the release's `labels/*_detect_all.txt` it used the top-scored detection.
That is another garment (the trousers in a jacket photo, the shoes under a skirt) for about 55% of
those images. Fixed in commit `c84c47d`: an unlabelled image now takes the best-scored detection of
its product category's class.

## Numbers

| | |
|---|---|
| Labelled images in the release | 201,824 |
| Release box is of the product's category class | 201,618 (99.9%) |
| Release box is the top-scored detection | 133,193 (66.0%) |
| New rule (best-scored detection of the category class) gives the release box | 197,108 (97.7%) |
| FashionMV Fashion200K images without a label line | train 79,268 of 190,558 (41.6%), val 14,885 of 35,834 (41.5%) |
| Files re-cropped (box changed) | 58,102 in 31,260 of 67,680 products |
| Val images that evaluation reads and that were wrong | 9,244 of 35,814 (25.8%), in 4,951 of 10,720 val products |

Category to detection class, from the labelled images (`class_map.txt`): dresses → dress,
jackets → outerwear, pants → pants, skirts → skirt, tops → top.

`label_check.txt` counts 51,446 unlabelled images whose top detection is of another class while
accepting `shorts` for the pants category; 58,102 files changed because the fixed rule takes `pants`
only, as the release does.

## Which images did the authors use? (tests of 2026-10-10)

Fixing the crops was not the end of it. The authors' checkpoint (`yuandaxia/ProCIR`), scored with the
unmodified upstream `evaluate.py` on the full Fashion200K val split (18,499 queries, gallery 10,720),
with only the image files changed:

| Fashion200K val images | R@1 | R@5 | R@10 |
|---|---|---|---|
| Unlabelled images cropped to the top-scored detection (the bug) | 30.92 | 67.71 | 76.97 |
| Unlabelled images cropped to the category garment (the fix above) | 35.02 | 74.70 | 84.17 |
| Release crops for labelled images, unlabelled images uncropped | 36.33 | 75.51 | 85.23 |
| **Every image uncropped (the photos at the URLs)** | **41.08** | **78.29** | **87.22** |
| Paper | | 77.6 | 86.6 |

So the authors used the uncropped photos, not the release's crops, and cropping at all was the
mistake. Other models on the same images, R@5 with the source kept (late fusion alpha 0.5 for CLIP):

| Model | Buggy crops | Fixed crops | Uncropped |
|---|---|---|---|
| Our MT+Align run (trained on the buggy crops) | 66.22 | 69.30 | not measured |
| FashionCLIP | 26.34 | 29.96 | 34.17 |
| CLIP ViT-B/32 | 10.43 | 12.06 | 14.36 |

These numbers were read from the server's output on 2026-10-10 (`runs/f200kfix_*`,
`runs/f200k_uncropped_released`, `runs/f200k_fullphoto_*`); the result files are to be copied here.
Since commit `d9ef4b2`, `scripts/setup_data.py` downloads the photos into `f200k/` and does not crop.

## The four Fashion200K image variants

All four are built from the same photos (`data/images_official_hr/f200k`) and differ only in what is
done to each image; DeepFashion and FashionGen are identical across them.

| Variant | Images in the release (label line) | Other images (41%) | Built with | Authors' checkpoint R@5 |
|---|---|---|---|---|
| `photo` | uncropped | uncropped | nothing: the downloaded photos | 78.29 |
| `mixed` | cropped as released | uncropped | `crop_f200k.py --unlabelled keep` | 75.51 |
| `crop` | cropped as released | cropped to the category garment | `crop_f200k.py --unlabelled category` | 74.70 |
| `top_crop` | cropped as released | cropped to the top-scored detection | `crop_f200k.py --unlabelled top` | 67.71 |

Crops rebuilt from the photos have the size of the crops evaluated above on all 1,300 images of a
400-product val sample (within 3 px). `src/server/queue_f200k_variants.sh` builds `mixed` and `crop`
on the server, checks each image set with the authors' checkpoint and trains on `photo` (MT+Align and
single-turn), `mixed` and `crop` (MT+Align); the `top_crop` run is `results/procir_train`.

## Consequence

Every Fashion200K number in `results/procir_train/` was measured with the wrong crops and both runs
there were trained on them. DeepFashion and FashionGen are unaffected. The Fashion200K numbers must
be re-measured on the uncropped photos and both configurations retrained on them.

## Files

- `label_check.txt`, `class_map.txt`: output of `scripts/f200k_label_check.py` and
  `scripts/f200k_class_map.py` (they read only the detection and label files in the release zip).
- `val_stats.json`: output of `scripts/f200k_crop_check.py`, which also builds a before/after page
  of 100 random affected val images (kept locally; the images cannot be redistributed).
- `scripts/f200k_recrop.py`: re-crops the files whose box changed and lists them.

Run the scripts from the repository root, e.g. `python results/data/f200k_crop_fix/scripts/f200k_label_check.py`.
