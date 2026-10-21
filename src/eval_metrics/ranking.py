"""Per-query ranks and top-K lists from query and gallery embeddings, saved as one .npz per dataset.

Similarity is the cosine of L2-normalised float32 embeddings, as in upstream evaluate.py. Ranks are
1-based. Ties are broken by gallery index (a lower index ranks first), the order a stable descending
sort gives; with float embeddings exact ties do not occur in practice, so Recall@K from these ranks
equals the torch.topk-based Recall@K of the evaluation scripts.

Two protocols per query:
- keep: the source product stays in the gallery (upstream evaluate.py, the paper's numbers);
- excl: the source product is removed from that query's ranking.
"""
import json
from pathlib import Path

import numpy as np

K_STORE = 51  # top-50 with the source removed is always inside the top-51 with it kept
CHUNK = 1024


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _rank_of(sim, col):
    """1-based rank of column col[i] in row i of sim, ties broken by lower column index first."""
    rows = np.arange(len(sim))
    s = sim[rows, col][:, None]
    higher = (sim > s).sum(axis=1)
    idx = np.arange(sim.shape[1])[None, :]
    tied_before = ((sim == s) & (idx < col[:, None])).sum(axis=1)
    return higher + tied_before + 1


def _topk(sim, k):
    k = min(k, sim.shape[1])
    part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(sim, part, axis=1)
    order = np.lexsort((part, -vals), axis=1)  # by score descending, then index ascending
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def rank_queries(q, g, gt, src, k_store=K_STORE, chunk=CHUNK):
    """Ranks of the target (source kept / removed), rank of the source, and the top-k_store list
    with the source kept. q: (Q, D), g: (N, D), gt and src: (Q,) gallery indices."""
    q, g = normalize(q), normalize(g)
    gt, src = np.asarray(gt, dtype=np.int64), np.asarray(src, dtype=np.int64)
    out = {"rank_keep": [], "rank_excl": [], "rank_src": [], "top_idx": [], "top_score": []}
    for i in range(0, len(q), chunk):
        sim = q[i:i + chunk] @ g.T
        t, s = gt[i:i + chunk], src[i:i + chunk]
        out["rank_keep"].append(_rank_of(sim, t))
        out["rank_src"].append(_rank_of(sim, s))
        idx, val = _topk(sim, k_store)
        out["top_idx"].append(idx.astype(np.int32))
        out["top_score"].append(val.astype(np.float16))
        sim[np.arange(len(sim)), s] = -np.inf
        out["rank_excl"].append(_rank_of(sim, t))
    return {k: np.concatenate(v) for k, v in out.items()}


def save_dump(path, q, g, gallery_keys, source_ids, target_ids, gt, src, meta=None):
    """Write one dataset's embeddings (float16), ids, ranks and top-K lists to path (.npz) and
    meta to path with .json. Ranks are computed here from the float32 embeddings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q, g = normalize(q), normalize(g)
    r = rank_queries(q, g, gt, src)
    np.savez_compressed(
        path, q=q.astype(np.float16), g=g.astype(np.float16),
        gallery_keys=np.array(gallery_keys), source_ids=np.array(source_ids),
        target_ids=np.array(target_ids), gt=np.asarray(gt, dtype=np.int32),
        src=np.asarray(src, dtype=np.int32), **r)
    m = dict(meta or {})
    m.update({"n_queries": int(len(q)), "n_gallery": int(len(g)), "k_store": K_STORE})
    path.with_suffix(".json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return r


def load_dump(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta_path = path.with_suffix(".json")
    d["meta"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return d
