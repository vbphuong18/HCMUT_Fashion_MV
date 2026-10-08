"""Regroup MMLab DeepFashion In-shop img_highres into FashionMV product folders.

img_highres keeps every colorway of an item in one folder (id_XXXX/02_1_front.jpg, 03_1_front.jpg);
FashionMV product ids end with the colorway (WOMEN/Dresses/id_XXXX/02), so each colorway becomes
its own folder: <out>/WOMEN/Dresses/id_XXXX/02/02_1_front.jpg.
"""
import argparse
import os
import shutil
from pathlib import Path

from parquet_extract import wanted_ids

ROOT = Path(__file__).resolve().parents[2]  # repository root
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # same set as procir_train.data.IMAGE_EXTS
DEFAULT_OUT = ROOT / "data" / "images_official_hr" / "deepfashion"


def regroup(src_root, out_root, wanted, symlink=False):
    n_written, found = 0, set()
    for pid in sorted(wanted):
        item_dir, colorway = os.path.split(pid)
        src_dir = os.path.join(src_root, item_dir)
        if not os.path.isdir(src_dir):
            continue
        files = sorted(
            f for f in os.listdir(src_dir)
            if f.startswith(f"{colorway}_")
            and os.path.splitext(f)[1].lower() in IMAGE_EXTS
            and os.path.isfile(os.path.join(src_dir, f))
        )
        if not files:
            continue
        dst_dir = os.path.join(out_root, pid)
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            dst = os.path.join(dst_dir, f)
            if os.path.lexists(dst):
                continue
            if symlink:
                try:
                    os.symlink(os.path.abspath(os.path.join(src_dir, f)), dst)
                except OSError:
                    raise SystemExit("symlinks not permitted here; rerun without --symlink "
                                     "(copies) or enable Developer Mode")
            else:
                tmp = dst + ".tmp"
                shutil.copy2(os.path.join(src_dir, f), tmp)
                os.replace(tmp, dst)
            n_written += 1
        found.add(pid)
    return n_written, found


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="img_highres folder that contains MEN/ and WOMEN/")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--triplets", action="append", required=True, help="repeat for train and val")
    ap.add_argument("--symlink", action="store_true")
    args = ap.parse_args(argv)

    wanted = set()
    for path in args.triplets:
        wanted |= wanted_ids(path, "deepfashion")
    n_written, found = regroup(args.src, args.out, wanted, args.symlink)
    print(f"Written images: {n_written}")
    print(f"Products covered: {len(found)} / {len(wanted)}")
    missing = sorted(wanted - found)
    if missing:
        print(f"Missing: {len(missing)}, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
