"""Training records: FashionMV triplets with images, product-level dev split, stratified subsample."""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def product_key(dataset, product_id):
    return f"{dataset}/{product_id}"


def list_images(img_dir, max_views):
    if not os.path.isdir(img_dir):
        return []
    files = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if Path(f).suffix.lower() in IMAGE_EXTS)
    return files[:max_views]


def load_triplets(path, datasets):
    keep = set(datasets)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["dataset"] in keep:
                out.append(item)
    return out


def load_captions(path):
    caps = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            key = product_key(item["dataset"], str(item["product_id"]))
            caps[key] = {"long": item.get("long_caption", ""), "short": item["short_caption"]}
    return caps


def attach_images(triplets, image_root, max_views):
    cache = {}

    def images_of(dataset, pid):
        key = product_key(dataset, pid)
        if key not in cache:
            cache[key] = list_images(os.path.join(image_root, dataset, pid), max_views)
        return cache[key]

    # Records of the same product share ONE image-path list (from `cache`): treat as read-only.
    records = []
    for t in triplets:
        ds, sid, tid = t["dataset"], str(t["source_id"]), str(t["target_id"])
        s_imgs, t_imgs = images_of(ds, sid), images_of(ds, tid)
        if not s_imgs or not t_imgs:
            continue
        records.append({
            "dataset": ds, "source_id": sid, "target_id": tid,
            "source_key": product_key(ds, sid), "target_key": product_key(ds, tid),
            "source_images": s_imgs, "target_images": t_imgs,
            # val_triplets.jsonl has no long text; only training reads mod_long
            "mod_short": t["modification_text_short"], "mod_long": t.get("modification_text_long"),
        })
    return records


def _by_dataset(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["dataset"]].append(r)
    return groups


def split_dev(records, dev_fraction, seed):
    """Product-level split: no product appears in both train and dev triplets."""
    train, dev = [], []
    for ds, rs in sorted(_by_dataset(records).items()):
        products = {r[k] for r in rs for k in ("source_key", "target_key")}
        quota = max(1, round(dev_fraction * len(products)))
        targets_of = defaultdict(set)
        for r in rs:
            targets_of[r["source_key"]].add(r["target_key"])
        seeds = sorted(targets_of)
        random.Random(f"{seed}-dev-{ds}").shuffle(seeds)
        dev_products = set()
        for s in seeds:
            if len(dev_products) >= quota:
                break
            dev_products.add(s)
            dev_products |= targets_of[s]
        ds_dev_start = len(dev)
        for r in rs:
            in_s, in_t = r["source_key"] in dev_products, r["target_key"] in dev_products
            if in_s and in_t:
                dev.append(r)
            elif not in_s and not in_t:
                train.append(r)
        # triplets bridging train and dev products are dropped
        # decorrelate order from source grouping so dev[:n] is representative
        ds_dev = dev[ds_dev_start:]
        random.Random(f"{seed}-devorder-{ds}").shuffle(ds_dev)
        dev[ds_dev_start:] = ds_dev
    return train, dev


def stratified_subsample(records, fraction, seed):
    out = []
    for ds, rs in sorted(_by_dataset(records).items()):
        n = round(fraction * len(rs))
        out.extend(random.Random(f"{seed}-sub-{ds}").sample(rs, n))
    random.Random(f"{seed}-suborder").shuffle(out)
    return out


def build_records(cfg):
    triplets = load_triplets(Path(cfg.data_dir) / "train_triplets.jsonl", cfg.datasets)
    usable = attach_images(triplets, cfg.image_root, cfg.max_views)
    for ds in cfg.datasets:
        if not any(r["dataset"] == ds for r in usable):
            n = sum(t["dataset"] == ds for t in triplets)
            raise ValueError(f"{ds}: 0 of {n} train triplets have images under {cfg.image_root}"
                             " - extract train images first")
    train_pool, dev = split_dev(usable, cfg.dev_product_fraction, cfg.seed)
    train = stratified_subsample(train_pool, cfg.train_fraction, cfg.seed)
    stats = {}
    for ds in cfg.datasets:
        stats[ds] = {
            "triplets": sum(t["dataset"] == ds for t in triplets),
            "usable": sum(r["dataset"] == ds for r in usable),
            "train_pool": sum(r["dataset"] == ds for r in train_pool),
            "train": sum(r["dataset"] == ds for r in train),
            "dev": sum(r["dataset"] == ds for r in dev),
        }
        stats[ds]["dropped"] = stats[ds]["usable"] - stats[ds]["train_pool"] - stats[ds]["dev"]
    return train, dev, stats
