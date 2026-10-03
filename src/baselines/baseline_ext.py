"""Extended zero-shot CLIP / FashionCLIP baseline analysis for FashionMV CIR.

Re-uses the late-fusion protocol of baseline_eval.py, but:
  * caches per-image embeddings so several settings can be evaluated from one pass;
  * evaluates Fashion200K on exactly the 1,500-triplet subsample used by the ProCIR
    Kaggle run (gallery = products of those triplets), plus the earlier 2,000 subsample;
  * sweeps the fusion weight alpha and reports R@K with / without the source product
    in the ranking (the official evaluate.py keeps the source in the gallery).

Usage:
  python baseline_ext.py --model openai/clip-vit-base-patch32
  python baseline_ext.py --model patrickjohncyh/fashion-clip
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[2]  # repository root
DATA_DIR = str(ROOT / "data" / "FashionMV_hf" / "data" / "data")
IMAGE_ROOT = str(ROOT / "data" / "images")
OUT_DIR = str(ROOT / "results" / "baselines")
CACHE_DIR = str(ROOT / "results" / "baselines" / "cache")
MAX_VIEWS = 5
IMG_BATCH = 32
TXT_BATCH = 64
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not os.path.isdir(img_dir):
        return []
    return sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                  if os.path.splitext(f)[1].lower() in exts)[:MAX_VIEWS]


def load_usable(dataset):
    items = []
    with open(f"{DATA_DIR}/val_triplets.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["dataset"] != dataset:
                continue
            sub = os.path.join(IMAGE_ROOT, dataset)
            if list_images(os.path.join(sub, str(item["source_id"]))) and \
               list_images(os.path.join(sub, str(item["target_id"]))):
                items.append(item)
    return items


def emb(out):
    e = out.pooler_output if hasattr(out, "pooler_output") else out
    return F.normalize(e, dim=-1)


@torch.no_grad()
def product_embeddings(model, processor, dataset, pids, cache_path):
    cache = torch.load(cache_path) if os.path.exists(cache_path) else {}
    todo = [p for p in pids if p not in cache]
    flat, owner = [], []
    for p in todo:
        for path in list_images(os.path.join(IMAGE_ROOT, dataset, p)):
            flat.append(path)
            owner.append(p)
    print(f"[{dataset}] {len(todo)} new products, {len(flat)} images to embed", flush=True)
    t0 = time.time()
    embs = []
    for i in range(0, len(flat), IMG_BATCH):
        imgs = [Image.open(x).convert("RGB") for x in flat[i:i + IMG_BATCH]]
        inputs = processor(images=imgs, return_tensors="pt")
        embs.append(emb(model.get_image_features(**inputs)))
    if embs:
        embs = torch.cat(embs)
        sums, cnt = {}, {}
        for e, p in zip(embs, owner):
            sums[p] = sums.get(p, 0) + e
            cnt[p] = cnt.get(p, 0) + 1
        for p in sums:
            cache[p] = F.normalize(sums[p] / cnt[p], dim=0)
        torch.save(cache, cache_path)
    print(f"[{dataset}] image embedding done in {time.time() - t0:.1f}s", flush=True)
    return cache


@torch.no_grad()
def text_embeddings(model, processor, texts):
    out = []
    for i in range(0, len(texts), TXT_BATCH):
        inputs = processor(text=texts[i:i + TXT_BATCH], return_tensors="pt",
                           padding=True, truncation=True)
        out.append(emb(model.get_text_features(**inputs)))
    return torch.cat(out)


def recall(sim, gt, ks=(1, 5, 10)):
    gt_t = torch.tensor(gt)
    res = {}
    for k in ks:
        _, topk = sim.topk(min(k, sim.shape[1]), dim=1)
        res[f"R@{k}"] = round((topk == gt_t.unsqueeze(1)).any(dim=1).float().mean().item() * 100, 2)
    return res


def evaluate(triplets, prod_emb, txt_emb):
    pids = sorted({str(t["source_id"]) for t in triplets} | {str(t["target_id"]) for t in triplets})
    idx = {p: i for i, p in enumerate(pids)}
    gallery = torch.stack([prod_emb[p] for p in pids])
    src = torch.stack([prod_emb[str(t["source_id"])] for t in triplets])
    gt = [idx[str(t["target_id"])] for t in triplets]
    src_idx = torch.tensor([idx[str(t["source_id"])] for t in triplets])
    out = {"n_queries": len(triplets), "n_gallery": len(pids), "alpha": {}}
    for a in ALPHAS:
        q = F.normalize(a * src + (1 - a) * txt_emb, dim=-1)
        sim = q @ gallery.T
        r_in = recall(sim, gt)
        sim_ex = sim.clone()
        sim_ex[torch.arange(len(triplets)), src_idx] = -1e4
        r_ex = recall(sim_ex, gt)
        # how often the source product itself is ranked first
        src_top1 = round((sim.argmax(dim=1) == src_idx).float().mean().item() * 100, 2)
        out["alpha"][str(a)] = {"with_source": r_in, "source_excluded": r_ex, "source_top1_pct": src_top1}
        print(f"  alpha={a:<4} with_src={r_in}  src_excluded={r_ex}  src@1={src_top1}%", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    tag = args.model.replace("/", "_")
    print(f"Loading {args.model} on cpu...", flush=True)
    model = CLIPModel.from_pretrained(args.model).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    results = {}
    settings = []
    df = load_usable("deepfashion")
    settings.append(("deepfashion_full", "deepfashion", df))
    f2 = load_usable("f200k")
    for n in (1500, 2000):
        random.seed(42)
        settings.append((f"f200k_sample{n}", "f200k", random.sample(f2, n)))

    for name, ds, triplets in settings:
        print(f"== {name}: {len(triplets)} triplets", flush=True)
        pids = sorted({str(t["source_id"]) for t in triplets} | {str(t["target_id"]) for t in triplets})
        prod = product_embeddings(model, processor, ds, pids, f"{CACHE_DIR}/cache_{tag}_{ds}.pt")
        txt = text_embeddings(model, processor, [t["modification_text_short"] for t in triplets])
        results[name] = evaluate(triplets, prod, txt)

    path = f"{OUT_DIR}/baseline_ext_{tag}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {path}", flush=True)


if __name__ == "__main__":
    main()
