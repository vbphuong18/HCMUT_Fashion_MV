"""Zip the prepared training data for another machine (e.g. upload to Google Drive, then
scripts/fetch_data.py on the server).

Only product folders referenced by train/val triplets are packed (FashionGen h5 files hold extra
products). Images are stored uncompressed (JPEG does not shrink) and parts larger than --max-gb are
split, so each upload/download stays manageable. Paths inside the zips start at data/, so
extracting into the repository root restores the layout the configs expect.

  python scripts/pack_data.py                 # -> dist/data/procir_*.zip + manifest.json
  python scripts/pack_data.py --parts f200k --max-gb 2
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = {"annotations": None, "deepfashion": ["deepfashion"], "f200k": ["f200k"],
         "fashiongen": ["fashiongen_train", "fashiongen_val"]}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _ann_dir(data):
    return Path(data) / "FashionMV_hf" / "data" / "data"


def referenced_products(data):
    """{image_dataset: {product_id}} over train and val triplets."""
    out = {}
    for split in ("train", "val"):
        with open(_ann_dir(data) / f"{split}_triplets.jsonl", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                out.setdefault(t["dataset"], set()).update([str(t["source_id"]), str(t["target_id"])])
    return out


def part_files(data, part, products):
    data = Path(data)
    if PARTS[part] is None:
        return sorted(_ann_dir(data).glob("*.jsonl"))
    files = []
    for ds in PARTS[part]:
        for pid in sorted(products.get(ds, ())):
            folder = data / "images_official_hr" / ds / pid
            if folder.is_dir():
                files.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    return files


def _chunks(files, max_bytes):
    chunk, size = [], 0
    for f in files:
        n = f.stat().st_size
        if chunk and size + n > max_bytes:
            yield chunk
            chunk, size = [], 0
        chunk.append(f)
        size += n
    if chunk:
        yield chunk


def pack(data, out, parts=tuple(PARTS), max_bytes=4 * 10**9):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    products = referenced_products(data)
    base = data.parent  # arcnames start at "data/"
    manifest = {}
    for part in parts:
        chunks = list(_chunks(part_files(data, part, products), max_bytes))
        for i, chunk in enumerate(chunks, 1):
            name = f"procir_{part}.zip" if len(chunks) == 1 else f"procir_{part}_{i}.zip"
            tmp = out / (name + ".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                for f in chunk:
                    z.write(f, f.relative_to(base).as_posix())
            tmp.replace(out / name)
            manifest[name] = {"files": len(chunk), "bytes": sum(f.stat().st_size for f in chunk)}
            print(f"{name}: {len(chunk)} files, {manifest[name]['bytes'] / 1e9:.2f} GB", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "dist" / "data"))
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    ap.add_argument("--max-gb", type=float, default=4.0, help="split a part into zips of at most this size")
    args = ap.parse_args(argv)
    pack(args.data, args.out, args.parts, int(args.max_gb * 1e9))


if __name__ == "__main__":
    main()
