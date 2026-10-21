"""Metrics from per-query ranks (1-based rank of the single target of each FashionMV query).

Every per-query function returns an array of shape (Q,); summaries take the mean. With one target
per query, mAP@K equals MRR@K and NDCG@K reduces to 1/log2(rank + 1) inside the top K, so both are
functions of the target's rank only (see the report's notes).
"""
import math
from collections import defaultdict

import numpy as np

KS = (1, 5, 10, 50)
VIEW_NAMES = ("front", "back", "side")


def hits(rank, k):
    return (np.asarray(rank) <= k).astype(np.float64)


def reciprocal_rank(rank, k=None):
    rank = np.asarray(rank, dtype=np.float64)
    rr = 1.0 / rank
    return rr if k is None else np.where(rank <= k, rr, 0.0)


def ndcg(rank, k):
    rank = np.asarray(rank, dtype=np.float64)
    return np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0)


def summary(rank, ks=KS):
    """Recall@K, MRR, MRR@10, NDCG@5/10 in percent; median and mean rank."""
    rank = np.asarray(rank)
    out = {f"R@{k}": 100 * hits(rank, k).mean() for k in ks}
    out["MRR"] = 100 * reciprocal_rank(rank).mean()
    out["MRR@10"] = 100 * reciprocal_rank(rank, 10).mean()
    out["NDCG@5"] = 100 * ndcg(rank, 5).mean()
    out["NDCG@10"] = 100 * ndcg(rank, 10).mean()
    out["MedR"] = float(np.median(rank))
    out["MeanR"] = float(rank.mean())
    return {k: round(float(v), 2) for k, v in out.items()}


def leak_rates(rank_src, ks=(1, 5, 10)):
    """Reference Leak Rate: % of queries whose own source product is in the top K (source kept)."""
    return {f"RLR@{k}": round(100 * float(hits(rank_src, k).mean()), 2) for k in ks}


def rank_histogram(rank, k=10):
    """Share of queries (%) whose target sits at rank 1..k, and beyond k."""
    rank = np.asarray(rank)
    n = len(rank)
    out = {str(r): round(100 * float((rank == r).sum()) / n, 2) for r in range(1, k + 1)}
    out[f">{k}"] = round(100 * float((rank > k).sum()) / n, 2)
    return out


def canonical_views(views):
    """Group key of a triplet's views_involved: the front/back/side names it mentions, sorted;
    any other view (detail, sleeve, ...) adds 'other'; an empty list gives 'none'."""
    names = set()
    for v in views:
        v = v.lower()
        found = [n for n in VIEW_NAMES if n in v]
        names.update(found if found else ["other"])
    return "+".join(sorted(names)) if names else "none"


def by_group(rank, groups, ks=(1, 5, 10)):
    """Recall@K and MRR per group label; groups: (Q,) labels aligned with rank."""
    idx = defaultdict(list)
    for i, gname in enumerate(groups):
        idx[gname].append(i)
    rank = np.asarray(rank)
    out = {}
    for gname, ii in sorted(idx.items(), key=lambda kv: -len(kv[1])):
        r = rank[ii]
        row = {"n": len(ii)}
        row.update({f"R@{k}": round(100 * float(hits(r, k).mean()), 2) for k in ks})
        row["MRR"] = round(100 * float(reciprocal_rank(r).mean()), 2)
        out[gname] = row
    return out


def paired_bootstrap(a_by_ds, b_by_ds, n_boot=1000, seed=0):
    """Paired bootstrap of mean(a) - mean(b), resampling queries within each dataset and averaging
    the per-dataset means (the paper's 'Average' over datasets). a_by_ds, b_by_ds: lists of per-query
    arrays, one per dataset, in the same order. Returns (delta, 95% CI, two-sided p), in the units
    of a and b."""
    rng = np.random.default_rng(seed)
    diffs = [np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
             for a, b in zip(a_by_ds, b_by_ds)]
    delta = float(np.mean([d.mean() for d in diffs]))
    boots = np.zeros(n_boot)
    for d in diffs:
        n = len(d)
        boots += np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    boots /= len(diffs)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_one = float(np.mean(boots <= 0)) if delta > 0 else float(np.mean(boots >= 0))
    return delta, (float(lo), float(hi)), min(1.0, 2 * p_one)


def mcnemar(hit_a, hit_b):
    """McNemar test on paired 0/1 outcomes (continuity-corrected chi-square, 1 dof).
    Returns (only_a, only_b, p)."""
    a, b = np.asarray(hit_a, dtype=bool), np.asarray(hit_b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    stat = (abs(only_a - only_b) - 1) ** 2 / n
    return only_a, only_b, math.erfc(math.sqrt(stat / 2))
