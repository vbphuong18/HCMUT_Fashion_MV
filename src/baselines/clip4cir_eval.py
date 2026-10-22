"""CLIP4Cir baseline on the FashionMV val split (paper Table 2, "CLIP4Cir", MeanPool and MaxSim).

No training on FashionMV: the authors' released checkpoints (github.com/ABaldrati/CLIP4Cir, release
"weights", default FashionIQ/RN50x4_fullft: fine-tuned CLIP RN50x4 + Combiner). Gallery and queries
are those of procir_train.retrieval_eval (upstream protocol): per dataset, every product of
val_triplets.jsonl with images, first 5 views by sorted file name, source kept in the gallery.

CLIP4Cir takes one reference image, so every view is encoded alone (as the paper's single-image
baselines):
  gallery view  CLIP image feature, L2-normalised
  query view    Combiner(CLIP feature of that source view, CLIP feature of the modification text)
and the views are scored as meanpool (cosine of the normalised mean of the per-view embeddings) and
maxsim (max cosine over all query-view x gallery-view pairs), with the scoring and dump format of
baselines.qwen3vl_emb_eval. Image features are cached in OUT/cache/ (also partially), and datasets
whose dumps exist are skipped, so re-running the same command continues where it stopped.

Preprocessing as the authors' validate.py: TargetPad (ratio 1.25), bicubic resize to the model's
input size (288 for RN50x4), center crop, CLIP normalisation; the CLIP model runs in float32 here
(open_clip builds it from the OpenAI-format state dict, the same architecture as the openai/CLIP
package the authors use).

Usage: PYTHONPATH=src:external/FashionMV python -m baselines.clip4cir_eval --data_dir DIR \
         --image_root DIR --out_dir DIR --clip_path FILE --combiner_path FILE [--datasets ...]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from baselines.qwen3vl_emb_eval import (cosine_scorer, embed, group_views, maxsim_scorer, mean_pool,
                                        rank_scores, write_dump, write_eval)
from procir_train.retrieval_eval import UPSTREAM_MAX_VIEWS, val_gallery_and_queries

DATASETS = ("deepfashion", "f200k", "fashiongen_val")
STRATEGIES = ("meanpool", "maxsim")
TARGET_RATIO = 1.25  # validate.py --target-ratio default
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ---- model (as CLIP4Cir src/combiner.py and src/data_utils.py) ----


class Combiner(nn.Module):
    """CLIP4Cir Combiner, same parameter names so the released state dicts load as is."""

    def __init__(self, clip_feature_dim, projection_dim, hidden_dim):
        super().__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)
        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def combine_features(self, image_features, text_features):
        """Raw (not normalised) CLIP features in, normalised query embedding out."""
        t = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        i = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        raw = torch.cat((t, i), -1)
        combined = self.dropout3(F.relu(self.combiner_layer(raw)))
        alpha = self.dynamic_scalar(raw)
        out = self.output_layer(combined) + alpha * text_features + (1 - alpha) * image_features
        return F.normalize(out, dim=-1)


def target_pad(image, target_ratio=TARGET_RATIO):
    """Zero-pad the shorter side when the aspect ratio is at least target_ratio (TargetPad)."""
    import torchvision.transforms.functional as TF

    w, h = image.size
    if max(w, h) / min(w, h) < target_ratio:
        return image
    scaled = max(w, h) / target_ratio
    hp, vp = max(int((scaled - w) / 2), 0), max(int((scaled - h) / 2), 0)
    return TF.pad(image, [hp, vp, hp, vp], 0, "constant")


def make_preprocess(size, target_ratio=TARGET_RATIO):
    from PIL import Image
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

    return Compose([lambda im: target_pad(im, target_ratio), Resize(size, interpolation=Image.BICUBIC),
                    CenterCrop(size), lambda im: im.convert("RGB"), ToTensor(), Normalize(CLIP_MEAN, CLIP_STD)])


def load_clip4cir(clip_path, combiner_path, device):
    """(clip model, combiner, tokenizer, input size). Combiner sizes are read from its state dict
    (RN50x4: 2560/5120, the README's --projection-dim/--hidden-dim)."""
    import open_clip
    from open_clip.model import build_model_from_openai_state_dict

    clip_sd = torch.load(clip_path, map_location="cpu")["CLIP"]
    model = build_model_from_openai_state_dict(clip_sd).float().eval().to(device)
    comb_sd = torch.load(combiner_path, map_location="cpu")["Combiner"]
    feat = comb_sd["output_layer.weight"].shape[0]
    combiner = Combiner(feat, comb_sd["text_projection_layer.weight"].shape[0],
                        comb_sd["combiner_layer.weight"].shape[0])
    combiner.load_state_dict(comb_sd)
    combiner = combiner.eval().to(device)
    size = model.visual.image_size
    size = size[0] if isinstance(size, (tuple, list)) else size
    return model, combiner, open_clip.get_tokenizer("RN50"), size  # every OpenAI CLIP uses the same BPE


def caption(text):
    """One modification text formatted as the authors' FashionIQ captions (utils.py)."""
    return text.strip(".?, ").capitalize()


# ---- per dataset ----


def dataset_done(out_dir, ds):
    return all((Path(out_dir) / s / f"{ds}.{ext}").exists() for s in STRATEGIES for ext in ("npz", "json"))


def run_dataset(out_dir, ds, products, queries, encode_images, encode_texts, combine, batch_size, meta,
                device="cpu", save_every=200):
    """Encode (or load cached) the view features of one dataset, score both strategies and write
    the dumps. Returns {strategy: recall dict}."""
    out_dir = Path(out_dir)
    keys = list(products)
    idx = {k: i for i, k in enumerate(keys)}
    gt = [idx[r["target_key"]] for r in queries]
    src = [idx[r["source_key"]] for r in queries]
    ids = ([r["source_id"] for r in queries], [r["target_id"] for r in queries])

    # every image once: gallery views first, then any query view not among them
    paths = [p for k in keys for p in products[k]]
    seen = set(paths)
    for r in queries:
        for p in r["source_images"]:
            if p not in seen:
                seen.add(p)
                paths.append(p)
    feats, t_img = embed([([p], None) for p in paths], encode_images[0], encode_images[1], batch_size,
                         out_dir / "cache" / f"{ds}_images.pt", save_every, f"{ds} images")
    row = {p: i for i, p in enumerate(paths)}
    t0 = time.perf_counter()
    texts = encode_texts([caption(r["mod_short"]) for r in queries])

    g_owner = [i for i, k in enumerate(keys) for _ in products[k]]
    g_views = feats[[row[p] for k in keys for p in products[k]]]
    q_owner = [i for i, r in enumerate(queries) for _ in r["source_images"]]
    q_img = feats[[row[p] for r in queries for p in r["source_images"]]]
    q_views = combine(q_img, texts[q_owner])
    t_q = time.perf_counter() - t0
    print(f"[{ds}] encoded {len(paths)} images in {t_img:.0f}s, {len(queries)} texts + {len(q_views)} "
          f"query views in {t_q:.0f}s", flush=True)

    gv, gm = group_views(g_views, g_owner, len(keys))
    qv, qm = group_views(q_views, q_owner, len(queries))
    q_mean, g_mean = mean_pool(qv, qm).numpy(), mean_pool(gv, gm).numpy()
    base = dict(meta, dataset=ds, n_images=len(paths), n_query_views=len(q_views),
                seconds_gallery=round(t_img, 1), seconds_queries=round(t_q, 1))
    strategies = {"meanpool": cosine_scorer(q_mean, g_mean), "maxsim": maxsim_scorer(qv, qm, gv, gm, device)}
    results = {}
    for name, scorer in strategies.items():
        t0 = time.perf_counter()
        ranks, recall = rank_scores(scorer, len(queries), gt, src)
        res = {"n_queries": len(queries), "n_gallery": len(keys)}
        res.update(recall)
        m = dict(base, strategy=name, recall=res, seconds_scoring=round(time.perf_counter() - t0, 1))
        if name == "maxsim":
            m["q_g_note"] = ("q and g are the mean-pooled per-view embeddings (as meanpool); ranks and "
                             "top lists come from the maxsim scores, not from q @ g")
        write_dump(out_dir / name / f"{ds}.npz", q_mean, g_mean, keys, *ids, gt, src, ranks, m)
        results[name] = res
        print(f"[{ds} {name}] " + " ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clip_path", required=True, help="e.g. FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt")
    ap.add_argument("--combiner_path", required=True, help="e.g. FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--batch_size", type=int, default=64, help="images per forward pass")
    ap.add_argument("--max_queries", type=int, default=0, help="first N queries per dataset (0 = all; smoke test)")
    ap.add_argument("--save_every", type=int, default=200, help="batches between partial cache saves")
    args = ap.parse_args(argv)

    built = {}
    for ds in args.datasets:
        products, queries = val_gallery_and_queries(args.data_dir, args.image_root, ds, UPSTREAM_MAX_VIEWS)
        if not products or not queries:
            raise ValueError(f"no queries/products for dataset {ds}")
        if args.max_queries:
            queries = queries[:args.max_queries]
        built[ds] = (products, queries)
    todo = [ds for ds in args.datasets if not dataset_done(args.out_dir, ds)]
    for ds in args.datasets:
        if ds not in todo:
            print(f"[{ds}] dumps exist, skipped", flush=True)

    if todo:
        import open_clip

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, combiner, tokenizer, size = load_clip4cir(args.clip_path, args.combiner_path, device)
        preprocess = make_preprocess(size)

        def run_images(x):
            with torch.no_grad():
                return model.encode_image(x.to(device)).float().cpu()

        def encode_texts(texts, bs=256):
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    out.append(model.encode_text(tokenizer(texts[i:i + bs]).to(device)).float().cpu())
            return torch.cat(out)

        def combine(img, txt, bs=4096):
            with torch.no_grad():
                return torch.cat([combiner.combine_features(img[i:i + bs].to(device), txt[i:i + bs].to(device)).cpu()
                                  for i in range(0, len(img), bs)])

        build = lambda _, imgs: torch.stack([preprocess(ims[0]) for ims in imgs])
        meta = {"tool": "baselines.clip4cir_eval", "clip_path": str(args.clip_path),
                "combiner_path": str(args.combiner_path), "zero_shot_on_fashionmv": True,
                "caption_format": "mod_short.strip('.?, ').capitalize()", "input_size": size,
                "target_ratio": TARGET_RATIO, "max_views": UPSTREAM_MAX_VIEWS, "batch_size": args.batch_size,
                "dtype": "float32", "max_queries": args.max_queries, "torch": torch.__version__,
                "open_clip": open_clip.__version__}
        for ds in todo:
            products, queries = built[ds]
            print(f"[{ds}] {len(queries)} queries, gallery {len(products)}", flush=True)
            run_dataset(args.out_dir, ds, products, queries, (build, run_images), encode_texts, combine,
                        args.batch_size, meta, device=device, save_every=args.save_every)
    summary = {s: write_eval(args.out_dir, s, DATASETS) for s in STRATEGIES}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
