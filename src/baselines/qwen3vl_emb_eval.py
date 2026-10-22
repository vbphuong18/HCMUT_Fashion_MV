"""Zero-shot Qwen3-VL-Embedding-2B baseline on the FashionMV val split (paper Table 2, "Qwen3-V-2B").

No training on FashionMV: the released embedding model is used as is. Gallery and queries are those
of procir_train.retrieval_eval (upstream protocol): per dataset, every product of val_triplets.jsonl
with images, first 5 views by sorted file name, 128^2-512^2 px, source kept in the gallery.

Input format of the official helper (Qwen3VLEmbedder, without qwen_vl_utils): messages
[system: instruction], [user: images..., text], chat template with add_generation_prompt=True, right
padding, embedding = last-layer hidden state of the last non-pad token, L2-normalised (dim 2048).

Modes (strategies written):
  joint    all views of a product in one input (the paper's Joint encoding)        -> joint
  perview  each view encoded alone (query view = that view + the text), scored as
           meanpool: cosine of the normalised mean of the per-view embeddings       -> meanpool
           maxsim:   max cosine over all query-view x gallery-view pairs             -> maxsim

Writes OUT/<strategy>/<dataset>.npz/.json (eval_metrics.ranking dump format, readable by
eval_metrics.report) and OUT/<strategy>/eval.json. Embeddings are cached in OUT/cache/ (also
partially, every --save_every batches), and datasets whose dumps exist are skipped, so re-running
the same command after a crash continues where it stopped.
Assumptions are numbered within this module (the paper does not describe its baseline setup).

Usage: PYTHONPATH=src:external/FashionMV python -m baselines.qwen3vl_emb_eval --data_dir DIR \
         --image_root DIR --out_dir DIR [--datasets ...] [--modes joint perview] [--batch_size 8] \
         [--model Qwen/Qwen3-VL-Embedding-2B]
"""
import argparse
import hashlib
import json
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_metrics.ranking import K_STORE, _rank_of, _topk, normalize
from procir_train.dataset import load_images
from procir_train.retrieval_eval import UPSTREAM_MAX_VIEWS, UPSTREAM_PIXELS, val_gallery_and_queries

MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DATASETS = ("deepfashion", "f200k", "fashiongen_val")
MODES = {"joint": ("joint",), "perview": ("meanpool", "maxsim")}
# Assumption 1: the paper gives no instructions for its Qwen3-VL-Embedding baseline. Products use the
# model's default instruction (config_sentence_transformers.json); queries a CIR instruction
# (--query_instruction). Both are written to each dump's meta.
DOC_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = ("Retrieve the fashion product that matches the reference product after applying "
                     "the modification described in the text.")
KS = (1, 5, 10)
SCORE_CHUNK = 64  # queries per similarity block; maxsim never builds the full Q x N x views matrix


# ---- prompts ----


def instruction_text(instruction):
    """As the official format_model_input: strip, and end with punctuation."""
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


def messages(instruction, n_images, text=None):
    """[system: instruction], [user: n_images images, then the text]. Image entries carry no pixels:
    the images go to the processor in the same order."""
    content = [{"type": "image"} for _ in range(n_images)]
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": "NULL"})  # official placeholder for an empty input
    return [{"role": "system", "content": [{"type": "text", "text": instruction_text(instruction)}]},
            {"role": "user", "content": content}]


def processor_inputs(processor, convs, images, pixels):
    """One padded batch. images: per conversation, its PIL images in message order.
    Assumption 2: images are resized by the processor itself (min_pixels/max_pixels override its
    size {shortest_edge, longest_edge} in transformers 5.x, smart_resize with factor 32), instead of
    qwen_vl_utils.fetch_image + do_resize=False as in the official helper; both use factor 32 and the
    same pixel bounds, only the resampling code differs. Assumption 3: no truncation (the official
    helper truncates at 8192 tokens; 5 views at 512^2 are ~1.3k tokens)."""
    texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
    flat = [im for ims in images for im in ims]
    return processor(text=texts, images=flat or None, padding=True, return_tensors="pt",
                     min_pixels=pixels[0], max_pixels=pixels[1])


# ---- model ----


def pool_last(hidden, mask):
    """Hidden state of the last position whose mask is 1 (official _pooling_last)."""
    col = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), col]


def load_model(name, device):
    """The checkpoint stores model.visual.* and model.language_model.* (no lm_head; it is tied), i.e.
    the Qwen3VLModel inside Qwen3VLForConditionalGeneration (its config.json architecture), so that
    class loads every weight; only its .model is run."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(name)
    processor.tokenizer.padding_side = "right"
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
    vlm.config.use_cache = False
    return vlm.to(device).eval().model, processor


def make_runner(model, device):
    """inputs -> (B, 2048) pooled last-token states (bf16, not yet normalised)."""
    def run(inputs):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        return pool_last(out.last_hidden_state, inputs["attention_mask"]).cpu()

    return run


# ---- encoding with prefetch and resume ----


def _batches(items, batch_size, build, io_threads=4):
    """As retrieval_eval._prepared, but build() gets the whole batch (one padded processor call).
    items: (image paths, conversation). The next batch is read and processed in a worker thread
    while the caller encodes the current one; order is kept."""
    chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    with ThreadPoolExecutor(max_workers=io_threads) as io, ThreadPoolExecutor(max_workers=1) as worker:
        def prepare(chunk):
            imgs = list(io.map(load_images, [paths for paths, _ in chunk]))
            return build([conv for _, conv in chunk], imgs)

        fut = worker.submit(prepare, chunks[0]) if chunks else None
        for i in range(len(chunks)):
            inputs = fut.result()
            fut = worker.submit(prepare, chunks[i + 1]) if i + 1 < len(chunks) else None
            yield inputs


def _signature(items):
    return hashlib.sha1(json.dumps(items, sort_keys=True).encode("utf-8")).hexdigest()


def embed(items, build, run, batch_size, cache_path, save_every=200, desc=""):
    """Encode items in order; returns ((n, D) tensor, seconds). The result is cached at cache_path;
    a partial result is saved every save_every batches and resumed from on the next call.
    Seconds accumulate over resumed runs."""
    from tqdm import tqdm

    cache_path = Path(cache_path)
    partial = cache_path.with_name(cache_path.stem + ".partial.pt")
    sig = _signature(items)
    for p in (cache_path, partial):
        if p.exists():
            c = torch.load(p, map_location="cpu")
            if c["sig"] != sig:
                raise ValueError(f"{p} was made for other inputs; delete it to re-encode")
            if p == cache_path:
                return c["emb"], c["seconds"]
            rows, done, seconds = [c["emb"]], c["n_done"], c["seconds"]
            break
    else:
        rows, done, seconds = [], 0, 0.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_batches = 0
    with tqdm(total=len(items), initial=done, desc=desc, unit="in") as bar:
        for inputs in _batches(items[done:], batch_size, build):
            e = run(inputs)
            rows.append(e)
            done += len(e)
            n_batches += 1
            bar.update(len(e))
            if n_batches % save_every == 0 and done < len(items):
                rows = [torch.cat(rows)]
                torch.save({"sig": sig, "emb": rows[0], "n_done": done,
                            "seconds": seconds + time.perf_counter() - t0}, partial)
    seconds += time.perf_counter() - t0
    emb = torch.cat(rows) if rows else torch.zeros(0)
    if len(emb) != len(items):
        raise RuntimeError(f"encoded {len(emb)} of {len(items)} inputs")
    torch.save({"sig": sig, "emb": emb, "seconds": seconds}, cache_path)
    partial.unlink(missing_ok=True)
    return emb, seconds


def joint_items(products, keys, queries, query_instruction):
    gallery = [(products[k], messages(DOC_INSTRUCTION, len(products[k]))) for k in keys]
    qs = [(r["source_images"], messages(query_instruction, len(r["source_images"]), r["mod_short"]))
          for r in queries]
    return gallery, qs


def perview_items(products, keys, queries, query_instruction):
    """One input per view. Also returns the owner (product or query index) of every input.
    Assumption 5: a per-view query is that one source view plus the full modification text (the
    paper's per-view baseline gives no query format); a gallery view is that view alone."""
    gallery, g_owner, qs, q_owner = [], [], [], []
    for i, k in enumerate(keys):
        for p in products[k]:
            gallery.append(([p], messages(DOC_INSTRUCTION, 1)))
            g_owner.append(i)
    for i, r in enumerate(queries):
        for p in r["source_images"]:
            qs.append(([p], messages(query_instruction, 1, r["mod_short"])))
            q_owner.append(i)
    return gallery, g_owner, qs, q_owner


# ---- scoring ----


def group_views(emb, owner, n):
    """(total views, D) + owner index -> (n, Vmax, D) unit vectors and (n, Vmax) mask."""
    emb = F.normalize(emb.float(), dim=-1)
    owner = torch.as_tensor(owner)
    counts = torch.bincount(owner, minlength=n)
    if (counts == 0).any():
        raise ValueError("an item has no views")
    order = torch.argsort(owner, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(len(owner)) - starts[owner[order]]
    out = emb.new_zeros(n, int(counts.max()), emb.shape[1])
    mask = torch.zeros(n, int(counts.max()), dtype=torch.bool)
    out[owner[order], slot] = emb[order]
    mask[owner[order], slot] = True
    return out, mask


def mean_pool(views, mask):
    """Paper Eq.: normalise the mean of the (unit) per-view embeddings."""
    s = (views * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return F.normalize(s, dim=-1)


def maxsim_block(qv, qm, gv, gm):
    """(nq, N): max cosine over every query-view x gallery-view pair of nq queries against the whole
    gallery. qv (nq, A, D), qm (nq, A), gv (N, B, D), gm (N, B); padded views are masked out."""
    nq, a, d = qv.shape
    n, b, _ = gv.shape
    sim = (qv.reshape(nq * a, d) @ gv.reshape(n * b, d).T).view(nq, a, n, b)
    sim = sim.masked_fill(~(qm[:, :, None, None] & gm[None, None]), float("-inf"))
    return sim.amax(dim=(1, 3))


def cosine_scorer(q, g):
    """Cosine blocks exactly as ranking.rank_queries computes them (float32 numpy), so the ranks
    equal those save_dump would compute from the stored embeddings."""
    q, g = normalize(q), normalize(g)
    return lambda i0, i1: q[i0:i1] @ g.T


def maxsim_scorer(qv, qm, gv, gm, device="cpu"):
    qv, qm, gv, gm = qv.to(device), qm.to(device), gv.to(device), gm.to(device)
    return lambda i0, i1: maxsim_block(qv[i0:i1], qm[i0:i1], gv, gm).cpu().numpy()


def rank_scores(score, n_queries, gt, src, chunk=SCORE_CHUNK, k_store=K_STORE):
    """ranking.rank_queries from a similarity function score(i0, i1) -> (i1 - i0, N) float32 numpy
    array instead of embeddings (same keys, tie rule and dtypes). Also returns Recall@K computed
    independently with top-k on the same blocks, as retrieval_eval.recall_at_k does."""
    gt, src = np.asarray(gt, dtype=np.int64), np.asarray(src, dtype=np.int64)
    out = {"rank_keep": [], "rank_excl": [], "rank_src": [], "top_idx": [], "top_score": []}
    hits = {f"R@{k}": 0 for k in KS}
    hits.update({f"R@{k}_src_excluded": 0 for k in KS})
    for i in range(0, n_queries, chunk):
        sim = np.asarray(score(i, min(i + chunk, n_queries)), dtype=np.float32)
        t, s = gt[i:i + chunk], src[i:i + chunk]
        rows = np.arange(len(sim))
        for k in KS:
            hits[f"R@{k}"] += int((torch.from_numpy(sim).topk(min(k, sim.shape[1]), dim=1).indices.numpy()
                                   == t[:, None]).any(axis=1).sum())
        out["rank_keep"].append(_rank_of(sim, t))
        out["rank_src"].append(_rank_of(sim, s))
        idx, val = _topk(sim, k_store)
        out["top_idx"].append(idx.astype(np.int32))
        out["top_score"].append(val.astype(np.float16))
        sim[rows, s] = -np.inf
        for k in KS:
            hits[f"R@{k}_src_excluded"] += int((torch.from_numpy(sim).topk(min(k, sim.shape[1]), dim=1).indices.numpy()
                                                == t[:, None]).any(axis=1).sum())
        out["rank_excl"].append(_rank_of(sim, t))
    recall = {k: round(100 * v / n_queries, 2) for k, v in hits.items()}
    return {k: np.concatenate(v) for k, v in out.items()}, recall


def write_dump(path, q, g, gallery_keys, source_ids, target_ids, gt, src, ranks, meta):
    """ranking.save_dump with the ranks given instead of computed from q @ g: same npz keys, dtypes
    and meta fields. For joint/meanpool the ranks come from cosine_scorer and so equal save_dump's;
    for maxsim q, g are only stored (report.py reads ids, ranks, top lists and meta)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q, g = normalize(q), normalize(g)
    np.savez_compressed(
        path, q=q.astype(np.float16), g=g.astype(np.float16),
        gallery_keys=np.array(gallery_keys), source_ids=np.array(source_ids),
        target_ids=np.array(target_ids), gt=np.asarray(gt, dtype=np.int32),
        src=np.asarray(src, dtype=np.int32), **ranks)
    m = dict(meta)
    m.update({"n_queries": int(len(q)), "n_gallery": int(len(g)), "k_store": K_STORE})
    path.with_suffix(".json").write_text(json.dumps(m, indent=2), encoding="utf-8")


# ---- per dataset ----


def dataset_done(out_dir, mode, ds):
    return all((Path(out_dir) / s / f"{ds}.{ext}").exists() for s in MODES[mode] for ext in ("npz", "json"))


def run_dataset(out_dir, mode, ds, products, queries, build, run, batch_size, meta, device="cpu",
                save_every=200):
    """Encode (or load cached) embeddings for one mode and dataset, score and write the dumps.
    Returns {strategy: recall dict}."""
    out_dir = Path(out_dir)
    keys = list(products)
    idx = {k: i for i, k in enumerate(keys)}
    gt = [idx[r["target_key"]] for r in queries]
    src = [idx[r["source_key"]] for r in queries]
    ids = ([r["source_id"] for r in queries], [r["target_id"] for r in queries])
    qi = meta["query_instruction"]
    cache = out_dir / "cache"
    if mode == "joint":
        g_items, q_items = joint_items(products, keys, queries, qi)
    else:
        g_items, g_owner, q_items, q_owner = perview_items(products, keys, queries, qi)
    g, t_g = embed(g_items, build, run, batch_size, cache / f"{mode}_{ds}_gallery.pt", save_every, f"{ds} {mode} gallery")
    q, t_q = embed(q_items, build, run, batch_size, cache / f"{mode}_{ds}_queries.pt", save_every, f"{ds} {mode} queries")
    print(f"[{ds} {mode}] encoded {len(g_items)} gallery inputs in {t_g:.0f}s, {len(q_items)} query inputs "
          f"in {t_q:.0f}s", flush=True)
    base = dict(meta, mode=mode, dataset=ds, n_gallery_inputs=len(g_items), n_query_inputs=len(q_items),
                seconds_gallery=round(t_g, 1), seconds_queries=round(t_q, 1))

    if mode == "joint":
        q, g = q.float().numpy(), g.float().numpy()
        strategies = {"joint": (q, g, cosine_scorer(q, g))}
    else:
        gv, gm = group_views(g, g_owner, len(keys))
        qv, qm = group_views(q, q_owner, len(queries))
        q_mean, g_mean = mean_pool(qv, qm).numpy(), mean_pool(gv, gm).numpy()
        strategies = {"meanpool": (q_mean, g_mean, cosine_scorer(q_mean, g_mean)),
                      "maxsim": (q_mean, g_mean, maxsim_scorer(qv, qm, gv, gm, device))}
    results = {}
    for name, (qe, ge, scorer) in strategies.items():
        t0 = time.perf_counter()
        ranks, recall = rank_scores(scorer, len(queries), gt, src)
        res = {"n_queries": len(queries), "n_gallery": len(keys)}
        res.update(recall)
        m = dict(base, strategy=name, recall=res, seconds_scoring=round(time.perf_counter() - t0, 1))
        if name == "maxsim":
            m["q_g_note"] = ("q and g are the mean-pooled per-view embeddings (as meanpool); ranks and "
                             "top lists come from the maxsim scores, not from q @ g")
        write_dump(out_dir / name / f"{ds}.npz", qe, ge, keys, *ids, gt, src, ranks, m)
        results[name] = res
        print(f"[{ds} {name}] " + " ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    return results


def write_eval(out_dir, strategy, datasets):
    """OUT/<strategy>/eval.json from the dump metas present (so a resumed run lists every dataset)."""
    d = Path(out_dir) / strategy
    res = {}
    for ds in datasets:
        p = d / f"{ds}.json"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8"))
            res[ds] = dict(meta["recall"], seconds_gallery=meta.get("seconds_gallery"),
                           seconds_queries=meta.get("seconds_queries"))
    if res:
        (d / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--batch_size", type=int, default=8, help="inputs per forward pass (products or views)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--query_instruction", default=QUERY_INSTRUCTION)
    ap.add_argument("--save_every", type=int, default=200, help="batches between partial cache saves")
    args = ap.parse_args(argv)

    # build and validate every dataset first, so a bad one does not waste earlier compute
    built = {}
    for ds in args.datasets:
        products, queries = val_gallery_and_queries(args.data_dir, args.image_root, ds, UPSTREAM_MAX_VIEWS)
        if not products or not queries:
            raise ValueError(f"no queries/products for dataset {ds}")
        built[ds] = (products, queries)
    todo = [(m, ds) for m in args.modes for ds in args.datasets if not dataset_done(args.out_dir, m, ds)]
    for m, ds in [(m, ds) for m in args.modes for ds in args.datasets if (m, ds) not in todo]:
        print(f"[{ds} {m}] dumps exist, skipped", flush=True)

    if todo:
        import transformers

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = load_model(args.model, device)
        # Assumption 4: the ProCIR eval cap (128^2-512^2 px) instead of the model's own default
        # (4096 to 1310720 px), so that both models see the same images.
        build = lambda convs, imgs: processor_inputs(processor, convs, imgs, UPSTREAM_PIXELS)
        run = make_runner(model, device)
        meta = {"tool": "baselines.qwen3vl_emb_eval", "model": args.model, "zero_shot": True,
                "doc_instruction": instruction_text(DOC_INSTRUCTION),
                "query_instruction": instruction_text(args.query_instruction),
                "pooling": "last token, L2-normalised", "max_views": UPSTREAM_MAX_VIEWS,
                "pixels": list(UPSTREAM_PIXELS), "batch_size": args.batch_size, "dtype": "bfloat16",
                "attn_implementation": "sdpa", "torch": torch.__version__,
                "transformers": transformers.__version__}
        for m, ds in todo:
            products, queries = built[ds]
            print(f"[{ds} {m}] {len(queries)} queries, gallery {len(products)}", flush=True)
            run_dataset(args.out_dir, m, ds, products, queries, build, run, args.batch_size, meta,
                        device=device, save_every=args.save_every)
    summary = {s: write_eval(args.out_dir, s, DATASETS) for m in args.modes for s in MODES[m]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
