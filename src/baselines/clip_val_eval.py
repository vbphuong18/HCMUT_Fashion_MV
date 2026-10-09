"""Zero-shot CLIP / FashionCLIP late-fusion baseline on the full FashionMV val split.

Same gallery and views as the upstream evaluate.py, so the numbers sit in one table with ProCIR:
per dataset, the gallery is every product named by that dataset's val triplets (sources included),
a product is the mean of its first 5 images (sorted file names), and the query is
normalize(alpha * source + (1 - alpha) * text) with the short modification text.
R@K is reported with the source kept in the gallery (as upstream) and with it excluded.

  python src/baselines/clip_val_eval.py --model openai/clip-vit-base-patch32 --out runs/ref_clip/eval.json
  python src/baselines/clip_val_eval.py --model patrickjohncyh/fashion-clip --out runs/ref_fashionclip/eval.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MAX_VIEWS = 5  # upstream procir/datasets.py
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTS)[:MAX_VIEWS]


def load_val(data_dir, dataset):
    with open(Path(data_dir) / "val_triplets.jsonl", encoding="utf-8") as f:
        return [t for t in map(json.loads, f) if t["dataset"] == dataset]


def gallery_ids(triplets):
    return sorted({str(t["source_id"]) for t in triplets} | {str(t["target_id"]) for t in triplets})


def recall(sim, gt, ks=(1, 5, 10)):
    gt = torch.as_tensor(gt).unsqueeze(1)
    return {f"R@{k}": round((sim.topk(min(k, sim.shape[1]), dim=1).indices == gt).any(dim=1).float().mean().item()
                            * 100, 2) for k in ks}


def score(triplets, prod, txt, alphas=ALPHAS):
    """prod: {product_id: unit embedding}; txt: (n, d) unit text embeddings in triplet order."""
    pids = [p for p in gallery_ids(triplets) if p in prod]
    idx = {p: i for i, p in enumerate(pids)}
    keep = [i for i, t in enumerate(triplets) if str(t["source_id"]) in idx and str(t["target_id"]) in idx]
    gallery = torch.stack([prod[p] for p in pids])
    src_idx = torch.tensor([idx[str(triplets[i]["source_id"])] for i in keep])
    gt = [idx[str(triplets[i]["target_id"])] for i in keep]
    out = {"n_queries": len(keep), "n_gallery": len(pids), "alpha": {}}
    for a in alphas:
        q = F.normalize(a * gallery[src_idx] + (1 - a) * txt[keep], dim=-1)
        sim = q @ gallery.T
        excluded = sim.clone()
        excluded[torch.arange(len(keep)), src_idx] = float("-inf")
        out["alpha"][str(a)] = {"with_source": recall(sim, gt), "source_excluded": recall(excluded, gt)}
    return out


def _unit(out):
    return F.normalize(getattr(out, "pooler_output", out), dim=-1)


@torch.no_grad()
def embed_products(model, processor, device, image_root, dataset, pids, batch=64):
    from PIL import Image

    flat, owner = [], []
    for p in pids:
        for path in list_images(os.path.join(image_root, dataset, p)):
            flat.append(path)
            owner.append(p)
    sums, counts = {}, {}
    for i in range(0, len(flat), batch):
        imgs = [Image.open(x).convert("RGB") for x in flat[i:i + batch]]
        e = _unit(model.get_image_features(**processor(images=imgs, return_tensors="pt").to(device))).cpu()
        for v, p in zip(e, owner[i:i + batch]):
            sums[p] = sums.get(p, 0) + v
            counts[p] = counts.get(p, 0) + 1
    return {p: F.normalize(sums[p] / counts[p], dim=0) for p in sums}, len(flat)


@torch.no_grad()
def embed_texts(model, processor, device, texts, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        inputs = processor(text=texts[i:i + batch], return_tensors="pt", padding=True, truncation=True).to(device)
        out.append(_unit(model.get_text_features(**inputs)).cpu())
    return torch.cat(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "FashionMV_hf" / "data" / "data"))
    ap.add_argument("--image-root", default=str(ROOT / "data" / "images_official_hr"))
    ap.add_argument("--datasets", nargs="+", default=["deepfashion", "f200k", "fashiongen_val"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    results = {"model": args.model, "image_root": args.image_root}
    for ds in args.datasets:
        t0 = time.time()
        triplets = load_val(args.data_dir, ds)
        prod, n_img = embed_products(model, processor, device, args.image_root, ds, gallery_ids(triplets))
        txt = embed_texts(model, processor, device, [t["modification_text_short"] for t in triplets])
        results[ds] = score(triplets, prod, txt)
        r = results[ds]["alpha"]["0.5"]
        print(f"[{ds}] {results[ds]['n_queries']} queries, gallery {results[ds]['n_gallery']} ({n_img} images), "
              f"alpha=0.5 with source {r['with_source']} | source excluded {r['source_excluded']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
