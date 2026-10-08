"""Statistics of the FashionMV data as prepared locally (all three sources), for the report.

  python scripts/dataset_stats.py      # -> results/data/dataset_stats.json and dataset_stats.md

Per source and split: triplets, products, views per product (all images on disk and the first 5
that evaluate.py reads), image size (sampled), modification-text and caption lengths in words,
and, with the training config, the product-level dev split and the train sizes at 10% / 25%.
"""
import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCES = ("deepfashion", "f200k", "fashiongen")
DISPLAY = {"deepfashion": "DeepFashion", "f200k": "Fashion200K", "fashiongen": "FashionGen"}


def split_name(source, split):
    return f"fashiongen_{split}" if source == "fashiongen" else source


def _words(texts):
    counts = [len(t.split()) for t in texts if t]
    if not counts:
        return None
    return {"mean": round(statistics.mean(counts), 1), "median": statistics.median(counts)}


def _images(folder):
    try:
        return [e for e in os.scandir(folder) if Path(e.name).suffix.lower() in IMAGE_EXTS]
    except FileNotFoundError:
        return []


def split_stats(rows, image_root, image_ds, sample_images=300):
    products = sorted({str(r[k]) for r in rows for k in ("source_id", "target_id")})
    sources = {str(r["source_id"]) for r in rows}
    hist, views, n_disk, n_bytes = Counter(), [], 0, 0
    for p in products:
        files = _images(Path(image_root) / image_ds / p)
        n = len(files)
        views.append(n)
        hist[str(n) if n < 5 else "5+"] += 1
        n_disk += n
        n_bytes += sum(e.stat().st_size for e in files)
    sizes = []
    for p in random.Random(0).sample(products, min(sample_images, len(products))):
        files = sorted(_images(Path(image_root) / image_ds / p), key=lambda e: e.name)
        if files:
            with Image.open(files[0].path) as im:
                sizes.append(im.size)
    return {
        "triplets": len(rows),
        "products": len(products),
        "views_per_product": dict(sorted(hist.items())),
        "views_mean": round(statistics.mean(views), 2) if views else 0,
        "images_on_disk": n_disk,
        "images_used_max5": sum(min(v, 5) for v in views),
        "bytes_on_disk": n_bytes,
        "targets_per_source_mean": round(len(rows) / len(sources), 2) if sources else 0,
        "mod_short_words": _words(r.get("modification_text_short") for r in rows),
        "mod_long_words": _words(r.get("modification_text_long") for r in rows),
        "image_size_median": ([int(statistics.median(s[0] for s in sizes)), int(statistics.median(s[1] for s in sizes))]
                              if sizes else None),
    }


def _read(path, dataset):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if r["dataset"] == dataset]


def caption_stats(rows):
    return {"products": len(rows), "short_words": _words(r.get("short_caption") for r in rows),
            "long_words": _words(r.get("long_caption") for r in rows)}


def config_split(ann, image_root, sources):
    """Product-level dev split and subsample sizes exactly as procir_train builds them (seed 42)."""
    sys.path.insert(0, str(ROOT / "src"))
    from procir_train.config import TrainConfig
    from procir_train.data import build_records, stratified_subsample

    cfg = TrainConfig(data_dir=str(ann), image_root=str(image_root), train_fraction=1.0,
                      datasets=[split_name(s, "train") for s in sources])
    train_pool, dev, stats = build_records(cfg)
    for frac in (0.10, 0.25):
        sub = stratified_subsample(train_pool, frac, cfg.seed)
        stats[f"train_{int(frac * 100)}pct"] = {"triplets": len(sub), "steps_at_b64": len(sub) // 64}
    stats["dev_products"] = len({r[k] for r in dev for k in ("source_key", "target_key")})
    return stats


def markdown(result, sources):
    def fmt(x):
        return f"{x:,}".replace(",", ".") if isinstance(x, int) else str(x)

    def w(d):
        return f"{d['mean']} / {d['median']}" if d else "–"

    lines = ["| Bộ | Split | Triplet | Sản phẩm | Ảnh (≤5 góc) | Góc/sp TB | Phân bố số góc | Kích thước ảnh (trung vị) "
             "| Văn bản sửa ngắn (từ, TB/trung vị) | Văn bản sửa dài |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    tot = Counter()
    for s in sources:
        for split in ("train", "val"):
            d = result[s][split]
            tot[f"{split}_t"] += d["triplets"]
            tot[f"{split}_p"] += d["products"]
            tot[f"{split}_i"] += d["images_used_max5"]
            lines.append(f"| {DISPLAY[s]} | {split} | {fmt(d['triplets'])} | {fmt(d['products'])} | "
                         f"{fmt(d['images_used_max5'])} | {d['views_mean']} | {d['views_per_product']} | "
                         f"{d['image_size_median']} | {w(d['mod_short_words'])} | {w(d['mod_long_words'])} |")
    for split in ("train", "val"):
        lines.append(f"| **Tổng** | {split} | **{fmt(tot[split + '_t'])}** | **{fmt(tot[split + '_p'])}** | "
                     f"**{fmt(tot[split + '_i'])}** | | | | | |")
    lines += ["", "| Bộ | Caption train: sp | ngắn (từ) | dài (từ) | Caption val: sp | ngắn | dài |",
              "|---|---|---|---|---|---|---|"]
    for s in sources:
        c = result[s]["captions"]
        lines.append(f"| {DISPLAY[s]} | {fmt(c['train']['products'])} | {w(c['train']['short_words'])} | "
                     f"{w(c['train']['long_words'])} | {fmt(c['val']['products'])} | {w(c['val']['short_words'])} | "
                     f"{w(c['val']['long_words'])} |")
    split = result.get("config_split")
    if split:
        lines += ["", "Chia tập theo code huấn luyện (dev = 5% sản phẩm, seed 42):", "",
                  "| Bộ | Triplet có ảnh | Train pool | Dev | Bỏ (nối train–dev) |", "|---|---|---|---|---|"]
        for s in sources:
            d = split[split_name(s, "train")]
            lines.append(f"| {DISPLAY[s]} | {fmt(d['usable'])} | {fmt(d['train_pool'])} | {fmt(d['dev'])} | "
                         f"{fmt(d['dropped'])} |")
        lines += ["", f"Dev: {fmt(split['dev_products'])} sản phẩm. "
                  f"Train 10%: {fmt(split['train_10pct']['triplets'])} triplet ({split['train_10pct']['steps_at_b64']} bước, B=64); "
                  f"25%: {fmt(split['train_25pct']['triplets'])} triplet ({split['train_25pct']['steps_at_b64']} bước)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", default=str(ROOT / "data" / "FashionMV_hf" / "data" / "data"))
    ap.add_argument("--image-root", default=str(ROOT / "data" / "images_official_hr"))
    ap.add_argument("--out", default=str(ROOT / "results" / "data"))
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--sample-images", type=int, default=300)
    ap.add_argument("--no-config-split", action="store_true")
    args = ap.parse_args(argv)

    ann = Path(args.ann)
    result = {}
    for s in args.sources:
        result[s] = {}
        for split in ("train", "val"):
            ds = split_name(s, split)
            result[s][split] = split_stats(_read(ann / f"{split}_triplets.jsonl", ds), args.image_root, ds,
                                           args.sample_images)
        result[s]["captions"] = {split: caption_stats(_read(ann / f"{split}_captions.jsonl", split_name(s, split)))
                                 for split in ("train", "val")}
        print(f"{DISPLAY[s]}: done", flush=True)
    if not args.no_config_split:
        result["config_split"] = config_split(ann, args.image_root, args.sources)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(result, args.sources)
    (out / "dataset_stats.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
