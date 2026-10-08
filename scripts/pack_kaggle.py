"""Prepare the two private Kaggle datasets that src/kaggle/train/procir_train.ipynb reads.

  python scripts/pack_kaggle.py code   [--out dist/kaggle/procir-code] [--owner vbichphuong]
  python scripts/pack_kaggle.py images [--src data/images_official_hr] [--owner vbichphuong]

`code` copies the training package, the pinned upstream eval code, configs and scripts into --out;
`images` only writes dataset-metadata.json next to the image folders (nothing is copied).
Then upload (first time `create`, later `version`):
  kaggle datasets create  -p dist/kaggle/procir-code --dir-mode zip
  kaggle datasets version -p dist/kaggle/procir-code --dir-mode zip -m "update code"
  kaggle datasets create  -p data/images_official_hr --dir-mode zip
Datasets created by the CLI are private unless --public is passed.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ("src/procir_train", "external/FashionMV/procir", "external/FashionMV/evaluate.py",
              "configs", "scripts", "requirements-train.txt")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")


def write_metadata(folder, owner, slug, title):
    meta = {"title": title, "id": f"{owner}/{slug}", "licenses": [{"name": "other"}]}
    (Path(folder) / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def pack_code(out, owner):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for rel in CODE_PATHS:
        src, dst = ROOT / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
    write_metadata(out, owner, "procir-code", "procir-code")
    print(f"code bundle in {out}")


def pack_images(src, owner):
    src = Path(src)
    if not src.is_dir():
        raise SystemExit(f"{src} is not a folder")
    write_metadata(src, owner, "procir-images", "procir-images")
    print(f"metadata written; folders to upload: {sorted(p.name for p in src.iterdir() if p.is_dir())}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("code", "images"))
    ap.add_argument("--owner", default="vbichphuong", help="Kaggle username that owns the datasets")
    ap.add_argument("--out", default=str(ROOT / "dist" / "kaggle" / "procir-code"))
    ap.add_argument("--src", default=str(ROOT / "data" / "images_official_hr"))
    args = ap.parse_args(argv)
    if args.what == "code":
        pack_code(args.out, args.owner)
    else:
        pack_images(args.src, args.owner)


if __name__ == "__main__":
    main()
