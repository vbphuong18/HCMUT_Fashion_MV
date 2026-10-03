"""Zero-shot CLIP / FashionCLIP baseline for FashionMV CIR evaluation.

Late-fusion baseline: query embedding = normalize(alpha*image_emb + (1-alpha)*text_emb),
compared against a gallery of normalized product image embeddings (cosine similarity).
Source-image embeddings are reused from the gallery (every source_id is also a gallery
product), so images are only ever embedded once.

Usage:
  python baseline_eval.py --model openai/clip-vit-base-patch32 --datasets deepfashion
  python baseline_eval.py --model patrickjohncyh/fashion-clip --datasets f200k
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
MAX_VIEWS = 5
F200K_SAMPLE_SIZE = 2000  # match the subsample used for the ProCIR f200k-v3 Kaggle run
IMG_BATCH = 32
TXT_BATCH = 64


def list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not os.path.isdir(img_dir):
        return []
    return sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])[:MAX_VIEWS]


def load_triplets(dataset):
    lines = []
    with open(f"{DATA_DIR}/val_triplets.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["dataset"] == dataset:
                lines.append(item)
    return lines


def prepare_triplets(dataset):
    items = load_triplets(dataset)
    img_subdir = os.path.join(IMAGE_ROOT, dataset)

    usable = []
    for item in items:
        src_dir = os.path.join(img_subdir, str(item["source_id"]))
        tgt_dir = os.path.join(img_subdir, str(item["target_id"]))
        if list_images(src_dir) and list_images(tgt_dir):
            usable.append(item)

    print(f"[{dataset}] usable triplets (images available): {len(usable)} / {len(items)}", flush=True)

    if dataset == "f200k" and len(usable) > F200K_SAMPLE_SIZE:
        random.seed(42)
        usable = random.sample(usable, F200K_SAMPLE_SIZE)
        print(f"[{dataset}] subsampled to {len(usable)} (seed=42, matches ProCIR f200k-v3 run)", flush=True)

    return usable


def get_image_embeds(model, processor, device, out):
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    return F.normalize(emb, dim=-1)


@torch.no_grad()
def embed_images_batched(model, processor, device, img_paths):
    """Embed a flat list of image paths in batches. Returns (N, D) tensor."""
    all_embs = []
    for i in range(0, len(img_paths), IMG_BATCH):
        chunk = img_paths[i:i + IMG_BATCH]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        all_embs.append(get_image_embeds(model, processor, device, out).cpu())
    return torch.cat(all_embs, dim=0) if all_embs else torch.zeros(0, 512)


@torch.no_grad()
def embed_texts_batched(model, processor, device, texts):
    all_embs = []
    for i in range(0, len(texts), TXT_BATCH):
        chunk = texts[i:i + TXT_BATCH]
        inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        all_embs.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(all_embs, dim=0) if all_embs else torch.zeros(0, 512)


def build_gallery(model, processor, device, dataset, triplets):
    """Returns dict pid -> averaged, normalized multi-view image embedding."""
    img_subdir = os.path.join(IMAGE_ROOT, dataset)
    product_ids = set()
    for item in triplets:
        product_ids.add(str(item["source_id"]))
        product_ids.add(str(item["target_id"]))
    pid_list = sorted(product_ids)

    # flatten (pid, path) pairs so all images across all products are batched together
    flat_paths, owner_idx = [], []
    pid_img_counts = []
    for pi, pid in enumerate(pid_list):
        imgs = list_images(os.path.join(img_subdir, pid))
        pid_img_counts.append(len(imgs))
        for p in imgs:
            flat_paths.append(p)
            owner_idx.append(pi)

    print(f"[{dataset}] embedding {len(flat_paths)} images for {len(pid_list)} products...", flush=True)
    t0 = time.time()
    flat_embs = embed_images_batched(model, processor, device, flat_paths)
    print(f"[{dataset}] image embedding done in {time.time()-t0:.1f}s", flush=True)

    # average per product
    sums = torch.zeros(len(pid_list), flat_embs.shape[1] if len(flat_embs) else 512)
    counts = torch.zeros(len(pid_list))
    for j, pi in enumerate(owner_idx):
        sums[pi] += flat_embs[j]
        counts[pi] += 1
    counts = counts.clamp(min=1).unsqueeze(1)
    avg = F.normalize(sums / counts, dim=-1)

    return {pid: avg[i] for i, pid in enumerate(pid_list)}, avg, pid_list


def compute_recall(q_embs, gallery_embs, gt_indices, ks=(1, 5, 10)):
    sim = q_embs @ gallery_embs.T
    gt_t = torch.tensor(gt_indices)
    results = {}
    for k in ks:
        _, topk = sim.topk(min(k, sim.shape[1]), dim=1)
        r = (topk == gt_t.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"R@{k}"] = round(r * 100, 2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         help="e.g. openai/clip-vit-base-patch32 or patrickjohncyh/fashion-clip")
    parser.add_argument("--datasets", nargs="+", default=["deepfashion", "f200k"])
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="weight for image embedding in query fusion (1-alpha for text)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}...", flush=True)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    all_results = {}
    for dataset in args.datasets:
        triplets = prepare_triplets(dataset)

        gallery_dict, gallery_t, pid_list = build_gallery(model, processor, device, dataset, triplets)
        pid_to_idx = {p: i for i, p in enumerate(pid_list)}
        print(f"[{dataset}] gallery built: {len(pid_list)} products", flush=True)

        print(f"[{dataset}] embedding {len(triplets)} modification texts...", flush=True)
        t0 = time.time()
        texts = [item["modification_text_short"] for item in triplets]
        text_embs = embed_texts_batched(model, processor, device, texts)
        print(f"[{dataset}] text embedding done in {time.time()-t0:.1f}s", flush=True)

        q_embs, gt = [], []
        for item, txt_emb in zip(triplets, text_embs):
            img_emb = gallery_dict[str(item["source_id"])]
            q = F.normalize(args.alpha * img_emb + (1 - args.alpha) * txt_emb, dim=0)
            q_embs.append(q)
            gt.append(pid_to_idx[str(item["target_id"])])

        q_t = torch.stack(q_embs)
        r = compute_recall(q_t, gallery_t, gt)
        print(f"[{dataset}] R@1={r['R@1']} R@5={r['R@5']} R@10={r['R@10']} "
              f"(queries={len(q_embs)}, gallery={len(pid_list)})", flush=True)
        all_results[dataset] = {**r, "n_queries": len(q_embs), "n_gallery": len(pid_list)}

    out_path = args.output or f"{OUT_DIR}/baseline_{args.model.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
