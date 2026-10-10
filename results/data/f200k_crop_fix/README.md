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

## Consequence

Every Fashion200K number in `results/procir_train/` was measured with the wrong crops and both runs
there were trained on them. DeepFashion and FashionGen are unaffected. The Fashion200K numbers must
be re-measured on the fixed crops and both configurations retrained.

## Files

- `label_check.txt`, `class_map.txt`: output of `scripts/f200k_label_check.py` and
  `scripts/f200k_class_map.py` (they read only the detection and label files in the release zip).
- `val_stats.json`: output of `scripts/f200k_crop_check.py`, which also builds a before/after page
  of 100 random affected val images (kept locally; the images cannot be redistributed).
- `scripts/f200k_recrop.py`: re-crops the files whose box changed and lists them.

Run the scripts from the repository root, e.g. `python results/data/f200k_crop_fix/scripts/f200k_label_check.py`.
