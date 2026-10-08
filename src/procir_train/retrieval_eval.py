"""Retrieval evaluation following the upstream protocol (external/FashionMV/evaluate.py):
one gallery per dataset, short modification text, source kept in the gallery.

- evaluate_records: dev split during training (gallery = products of the given records).
- evaluate_val: published val set exactly like upstream ProductValDataset / CIRValDataset
  (gallery = every product of val_triplets.jsonl that has images, up to 5 views, 128^2-512^2 px).
  Needed for multi_turn=false variants: upstream evaluate.py always builds the two-turn query.
  --protocol train keeps that gallery but uses the run's own max_views / pixel range, to measure
  the train-eval mismatch (e.g. 3 views at 336^2 in training vs 5 views at 512^2 upstream).
Both also report Recall with the source removed from each query's ranking.

Usage: PYTHONPATH=src:external/FashionMV python -m procir_train.retrieval_eval --config RUN/config.yaml \
         --checkpoint RUN/ckpt/latest.pt --data_dir DIR --image_root DIR --datasets deepfashion f200k --out OUT.json          [--protocol upstream|train]
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import attach_images, list_images, load_triplets, product_key
from .dataset import load_images
from .prompts import doc_text, multiturn_query_text, process_visual, singleturn_query_text

UPSTREAM_MAX_VIEWS = 5
UPSTREAM_PIXELS = (128 * 128, 512 * 512)
PROTOCOLS = ("upstream", "train")


def protocol_settings(protocol, cfg):
    """(max_views, (min_pixels, max_pixels)) for an eval protocol."""
    if protocol == "upstream":
        return UPSTREAM_MAX_VIEWS, UPSTREAM_PIXELS
    if protocol == "train":
        return cfg.max_views, (cfg.image_min_pixels, cfg.image_max_pixels)
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def recall_at_k(q, g, gt, ks=(1, 5, 10), exclude=None):
    sim = F.normalize(q.float(), dim=-1) @ F.normalize(g.float(), dim=-1).T
    if exclude is not None:
        sim[torch.arange(len(q)), torch.tensor(exclude)] = float("-inf")
    gt_t = torch.tensor(gt)
    out = {}
    for k in ks:
        _, topk = sim.topk(min(k, sim.shape[1]), dim=1)
        out[f"R@{k}"] = round((topk == gt_t.unsqueeze(1)).any(dim=1).float().mean().item() * 100, 2)
    return out


def _score(encoder, processor, products, queries, multi_turn, px, batch_size):
    keys = list(products)
    g_rows = []
    for i in range(0, len(keys), batch_size):
        inputs = []
        for k in keys[i:i + batch_size]:
            imgs = load_images(products[k])
            inputs.append(process_visual(processor, doc_text(processor, len(imgs)), imgs, *px))
        g_rows.append(encoder.encode_single(inputs).float().cpu())

    build = multiturn_query_text if multi_turn else singleturn_query_text
    q_rows = []
    for i in range(0, len(queries), batch_size):
        inputs = []
        for r in queries[i:i + batch_size]:
            imgs = load_images(r["source_images"])
            inputs.append(process_visual(processor, build(processor, len(imgs), r["mod_short"]), imgs, *px))
        if multi_turn:
            q_rows.append(encoder.encode_multiturn(inputs)[1].float().cpu())
        else:
            q_rows.append(encoder.encode_single(inputs).float().cpu())

    g, q = torch.cat(g_rows), torch.cat(q_rows)
    idx = {k: i for i, k in enumerate(keys)}
    gt = [idx[r["target_key"]] for r in queries]
    src = [idx[r["source_key"]] for r in queries]
    res = {"n_queries": len(queries), "n_gallery": len(keys)}
    res.update(recall_at_k(q, g, gt))
    res.update({f"{k}_src_excluded": v for k, v in recall_at_k(q, g, gt, exclude=src).items()})
    return res


def _in_eval_mode(encoder, fn):
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            return fn()
    finally:
        encoder.train(was_training)


def evaluate_records(encoder, processor, records, cfg, max_queries=None, batch_size=16):
    """Dev eval. Gallery = products of the first `max_queries` records per dataset, so dev R@K
    depends on max_queries and is not comparable to val.
    max_queries=None or 0 means all queries."""
    px = (cfg.image_min_pixels, cfg.image_max_pixels)

    def run():
        by_ds = defaultdict(list)
        for r in records:
            by_ds[r["dataset"]].append(r)
        out = {}
        for ds, rs in sorted(by_ds.items()):
            rs = rs[:max_queries] if max_queries else rs
            products = {}
            for r in rs:
                products.setdefault(r["source_key"], r["source_images"])
                products.setdefault(r["target_key"], r["target_images"])
            out[ds] = _score(encoder, processor, products, rs, cfg.multi_turn, px, batch_size)
        return out

    return _in_eval_mode(encoder, run)


def val_gallery_and_queries(data_dir, image_root, dataset, max_views=UPSTREAM_MAX_VIEWS):
    triplets = load_triplets(Path(data_dir) / "val_triplets.jsonl", [dataset])
    products = {}
    for t in triplets:
        for k in ("source_id", "target_id"):
            pid = str(t[k])
            key = product_key(dataset, pid)
            if key not in products:
                products[key] = list_images(os.path.join(image_root, dataset, pid), max_views)
    products = {k: v for k, v in products.items() if v}
    return products, attach_images(triplets, str(image_root), max_views)


def evaluate_val(encoder, processor, data_dir, image_root, datasets, multi_turn, batch_size=16,
                 max_views=UPSTREAM_MAX_VIEWS, pixels=UPSTREAM_PIXELS):
    # build and validate every dataset first, so a bad one does not waste earlier compute
    built = {}
    for ds in datasets:
        products, queries = val_gallery_and_queries(data_dir, image_root, ds, max_views)
        if not products or not queries:
            raise ValueError(f"no queries/products for dataset {ds}")
        built[ds] = (products, queries)

    def run():
        return {ds: _score(encoder, processor, products, queries, multi_turn, pixels, batch_size)
                for ds, (products, queries) in built.items()}

    return _in_eval_mode(encoder, run)


def main(argv=None):
    from .config import load_config
    from .encoder import load_encoder
    from .train import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True,
                    choices=["deepfashion", "f200k", "fashiongen_val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="upstream",
                    help="upstream: 5 views, 128^2-512^2 px (reported numbers); "
                         "train: the run's max_views and pixel range (diagnostic)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    encoder, processor = load_encoder(cfg, torch.device("cuda"))
    load_checkpoint(args.checkpoint, encoder)
    max_views, pixels = protocol_settings(args.protocol, cfg)
    res = evaluate_val(encoder, processor, args.data_dir, args.image_root, args.datasets,
                       cfg.multi_turn, args.batch_size, max_views=max_views, pixels=pixels)
    res["protocol"] = {"name": args.protocol, "max_views": max_views, "pixels": list(pixels)}
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
