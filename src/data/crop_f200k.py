"""Rebuild the released (cropped) Fashion200K images from the full images at their URLs.

The Fashion200K release (xthan/fashion-200k; the Marqo/fashion200k mirror is a copy) ships each image
cropped to one of the boxes in fashion-200k/detection/*_detection.txt
("<label>_<score>_<x1>_<x2>_<y1>_<y2>", coordinates normalised to [0, 1]): the box of the product's
own garment, see load_boxes. image_urls.txt points to the uncropped photos. Cropping the URL images
with that box reproduces the released crops (checked against the mirror: same region, mean absolute
difference ~3/255 from JPEG noise), so train and val come from one source.

Files that are already released crops (byte-identical to --mirror) are copied as they are; every
other image is cropped and written as JPEG (quality 95). The source folder is left untouched.

  python src/data/crop_f200k.py --src data/images_official_hr/f200k_source --dst data/images_official_hr/f200k
"""
import argparse
import io
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "raw" / "fashion-200k-20261003T154803Z-1-001.zip"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# women/<category>/... -> the detection class the release crops that category to
CATEGORY_CLASS = {"dresses": "dress", "jackets": "outerwear", "pants": "pants", "skirts": "skirt", "tops": "top"}


def _lines(z, prefix):
    for name in z.namelist():
        if name.startswith(prefix) and name.endswith(".txt"):
            for line in io.TextIOWrapper(z.open(name), encoding="utf-8"):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > 1:
                    yield parts


def load_boxes(zip_path):
    """{image stem (e.g. '91352269_0'): (x1, x2, y1, y2)} of the box each released image is cropped to.

    labels/*_detect_all.txt records, per image, the score of the detection the release used; the box
    is the detection with that score. On the 201,824 labelled images that box is of the product's
    category garment (CATEGORY_CLASS) 99.9% of the time but the top-scored detection only 66% of the
    time, so the 41% of FashionMV's images without a label line take the best-scored detection of
    their category class (the labelled box on 97.7% of labelled images), not the top one: the top one
    is another garment, e.g. the trousers in a jacket photo, for about 55% of them. Only an image
    with no detection of its class falls back to the top-scored detection."""
    with zipfile.ZipFile(zip_path) as z:
        label_score = {Path(p[0]).stem: float(p[1]) for p in _lines(z, "fashion-200k/labels/")}
        boxes = {}
        for path, *dets in _lines(z, "fashion-200k/detection/"):
            stem = Path(path).stem
            parsed = [d.split("_") for d in dets]
            garment = CATEGORY_CLASS.get(Path(path).parts[1])
            same_class = [d for d in parsed if "_".join(d[:-5]) == garment]
            chosen = parsed[0]
            if stem in label_score:
                chosen = min(parsed, key=lambda f: abs(float(f[-5]) - label_score[stem]))
            elif same_class:
                chosen = max(same_class, key=lambda f: float(f[-5]))
            boxes[stem] = tuple(min(1.0, max(0.0, float(v))) for v in chosen[-4:])
    return boxes


def pixel_box(box, size):
    x1, x2, y1, y2 = box
    w, h = size
    left, right = sorted((round(x1 * w), round(x2 * w)))
    top, bottom = sorted((round(y1 * h), round(y2 * h)))
    left, top = min(left, w - 1), min(top, h - 1)
    return left, top, max(right, left + 1), max(bottom, top + 1)


def _is_released(path, mirror_dir):
    if mirror_dir is None:
        return False
    m = Path(mirror_dir) / path.name
    return m.exists() and m.stat().st_size == path.stat().st_size


def crop_product(src_dir, dst_dir, boxes, mirror_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    counts = {"cropped": 0, "released": 0, "no_box": 0}
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        dst = dst_dir / f.name
        if dst.exists():
            continue
        tmp = dst.with_name(dst.name + ".part")
        if _is_released(f, mirror_dir):
            shutil.copy2(f, tmp)
            counts["released"] += 1
        elif f.stem in boxes:
            with Image.open(f) as im:
                im.convert("RGB").crop(pixel_box(boxes[f.stem], im.size)).save(tmp, "JPEG", quality=95)
            counts["cropped"] += 1
        else:
            shutil.copy2(f, tmp)
            counts["no_box"] += 1
        os.replace(tmp, dst)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="product folders with the downloaded (uncropped) images")
    ap.add_argument("--dst", required=True, help="output product folders (released-style crops)")
    ap.add_argument("--zip", default=str(ZIP), help="Fashion200K zip holding fashion-200k/detection/")
    ap.add_argument("--mirror", default=str(ROOT / "data" / "images" / "f200k"),
                    help="Marqo mirror folders: files identical to these are released crops already")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    boxes = load_boxes(args.zip)
    print(f"detection boxes: {len(boxes)}", flush=True)
    mirror = Path(args.mirror) if Path(args.mirror).is_dir() else None
    products = sorted(p.name for p in Path(args.src).iterdir() if p.is_dir())

    def one(pid):
        return crop_product(Path(args.src) / pid, Path(args.dst) / pid, boxes, mirror / pid if mirror else None)

    total = {"cropped": 0, "released": 0, "no_box": 0}
    with ThreadPoolExecutor(args.workers) as ex:
        for n, counts in enumerate(ex.map(one, products), 1):
            for k, v in counts.items():
                total[k] += v
            if n % 5000 == 0:
                print(f"{n}/{len(products)} products {total}", flush=True)
    print(f"DONE {len(products)} products {total}", flush=True)


if __name__ == "__main__":
    main()
