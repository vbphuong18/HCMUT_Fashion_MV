"""The authors' evaluation (external/FashionMV/evaluate.py, single GPU) that also saves per-query
ranks and top-K lists, for models that load with upstream code: the released ProCIR checkpoint and
our exported runs (runs/<run>/export). Multi-turn queries only, like evaluate.py.

encode() follows evaluate.py's phases 1-2 (gallery, two-turn queries, source embeddings reused for
products missing from the gallery) and score() its phase 3, writing the same eval_results.json, so
the Recall numbers can be checked against an earlier evaluate.py run of the same model.

Usage: PYTHONPATH=src:external/FashionMV python -m eval_metrics.upstream_dump --model_path DIR \
         --image_root DIR --data_dir DIR --out_dir DIR [--datasets ...] [--batch_size 10]
"""
import argparse
import importlib.util
import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch

from .ranking import save_dump

UPSTREAM = Path(__file__).resolve().parents[2] / "external" / "FashionMV" / "evaluate.py"


def load_upstream():
    spec = importlib.util.spec_from_file_location("upstream_evaluate", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encode(up, model_path, data_dir, image_root, datasets, batch_size, device):
    from torch.utils.data import DataLoader, SequentialSampler
    from tqdm import tqdm

    from procir.collators import CIRQueryCollator, DocCollator
    from procir.datasets import CIRValDataset, ProductValDataset

    model, processor, emb_token_id = up.setup_model(model_path, device)
    t0 = time.perf_counter()
    product_ds = ProductValDataset(data_dir, image_root, datasets=datasets)
    loader = DataLoader(product_ds, batch_size=batch_size, sampler=SequentialSampler(product_ds),
                        collate_fn=DocCollator(processor, emb_token_id), num_workers=2, pin_memory=False)
    doc_embs, doc_pids, doc_dsets = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding gallery"):
            for j, emb in enumerate(model.forward_visual_batch(batch["doc_visual_inputs"], device)):
                doc_embs.append(emb.cpu())
                doc_pids.append(batch["batch_meta"][j]["product_id"])
                doc_dsets.append(batch["batch_meta"][j]["dataset"])
    doc_emb, doc_dset = OrderedDict(), {}
    for e, pid, ds in zip(doc_embs, doc_pids, doc_dsets):
        if pid not in doc_emb:
            doc_emb[pid], doc_dset[pid] = e, ds
    t_gallery = time.perf_counter() - t0

    cir_ds = CIRValDataset(data_dir, image_root, datasets=datasets)
    loader = DataLoader(cir_ds, batch_size=batch_size, sampler=SequentialSampler(cir_ds),
                        collate_fn=CIRQueryCollator(processor, emb_token_id), num_workers=2, pin_memory=False)
    q_embs, s_embs, tids, sids, q_dsets = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding queries"):
            s_list, q_list = model.forward_visual_batch_multiturn(batch["query_visual_inputs"], device)
            for j, meta in enumerate(batch["batch_meta"]):
                q_embs.append(q_list[j].cpu())
                s_embs.append(s_list[j].cpu())
                tids.append(meta["target_id"])
                sids.append(meta["source_id"])
                q_dsets.append(meta["dataset"])
    for s, sid, ds in zip(s_embs, sids, q_dsets):  # evaluate.py's "source reuse"
        if sid not in doc_emb:
            doc_emb[sid], doc_dset[sid] = s, ds
    timing = {"seconds_gallery": round(t_gallery, 1),
              "seconds_queries": round(time.perf_counter() - t0 - t_gallery, 1)}
    return doc_emb, doc_dset, torch.stack(q_embs), tids, sids, q_dsets, timing


def score(up, doc_emb, doc_dset, all_q, tids, sids, q_dsets, out_dir, meta=None):
    """evaluate.py phase 3, plus one dump per dataset. Returns the eval_results.json dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for i, ds in enumerate(q_dsets):
        groups[ds].append(i)
    results, r5, r10 = {}, [], []
    for ds in sorted(groups):
        name = up.DS_DISPLAY.get(ds, ds)
        pids = [p for p, d in doc_dset.items() if d == ds]
        gallery = torch.stack([doc_emb[p] for p in pids])
        pid_to_idx = {p: i for i, p in enumerate(pids)}
        valid = [qi for qi in groups[ds] if tids[qi] in pid_to_idx]
        gt = [pid_to_idx[tids[qi]] for qi in valid]
        src = [pid_to_idx[sids[qi]] for qi in valid]
        r = up.compute_recall(all_q[valid], gallery, gt)
        r5.append(r["R@5"])
        r10.append(r["R@10"])
        results.update({f"{name}_{k}": v for k, v in r.items()})
        m = dict(meta or {})
        m.update({"tool": "eval_metrics.upstream_dump (evaluate.py protocol)", "recall": r})
        save_dump(out_dir / f"{ds}.npz", all_q[valid].float().numpy(), gallery.float().numpy(), pids,
                  [sids[qi] for qi in valid], [tids[qi] for qi in valid], gt, src, meta=m)
        print(f"  {name:15s}  R@1={r['R@1']:6.2f}  R@5={r['R@5']:6.2f}  R@10={r['R@10']:6.2f}  "
              f"(queries={len(valid)}, gallery={len(pids)})")
    results["Average_R@5"] = round(sum(r5) / len(r5), 2)
    results["Average_R@10"] = round(sum(r10) / len(r10), 2)
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=None,
                    choices=["deepfashion", "f200k", "fashiongen_val"])
    args = ap.parse_args(argv)
    up = load_upstream()
    datasets = set(args.datasets) if args.datasets else None
    doc_emb, doc_dset, all_q, tids, sids, q_dsets, timing = encode(
        up, args.model_path, args.data_dir, args.image_root, datasets, args.batch_size, torch.device("cuda"))
    meta = {"model_path": args.model_path, "image_root": args.image_root, "batch_size": args.batch_size}
    meta.update(timing)
    score(up, doc_emb, doc_dset, all_q, tids, sids, q_dsets, args.out_dir, meta)


if __name__ == "__main__":
    main()
