"""Metrics, model comparisons and ranking examples from the per-query dumps (CPU only).

Usage: PYTHONPATH=src python -m eval_metrics.report --data_dir DIR --out DIR \
         --model released=runs/rank_eval/released --model mt_align=runs/rank_eval/photo_mt_align_s42 ... \
         [--compare mt_align:single ...]

Each model directory holds <dataset>.npz from eval_metrics.ranking.save_dump. Writes:
  metrics.json   per model and dataset: both protocols, leak rates, rank histogram, views groups
  compare.json   per pair: deltas with paired-bootstrap CI and p, McNemar, per-query rank changes
  examples.json  per model: the queries whose target sits deepest; per pair and dataset: queries
                 with the largest rank gains and losses; all with top-5 lists
  tables.md      the same, as Markdown tables
Before anything else it checks that Recall@1/5/10 from the stored ranks equals the Recall the
evaluation run printed (stored in each dump's meta) and stops if not.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .ranking import load_dump

DATASETS = ("deepfashion", "f200k", "fashiongen_val")
DISPLAY = {"deepfashion": "DeepFashion", "f200k": "Fashion200K", "fashiongen_val": "FashionGen"}
MAIN = ("R@1", "R@5", "R@10", "R@50", "MRR", "MRR@10", "NDCG@10", "MedR")
COMPARED = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")
NO_MEAN = ("MedR", "MeanR")


def load_triplets(data_dir):
    out = {}
    with open(Path(data_dir) / "val_triplets.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            out[(t["dataset"], str(t["source_id"]), str(t["target_id"]))] = t
    return out


def load_model(path):
    dumps = {ds: load_dump(Path(path) / f"{ds}.npz") for ds in DATASETS if (Path(path) / f"{ds}.npz").exists()}
    if not dumps:
        raise FileNotFoundError(f"no <dataset>.npz in {path}")
    return dumps


def check_recall(name, dumps):
    """Recall from stored ranks must equal the Recall the evaluation printed."""
    for ds, d in dumps.items():
        printed = d["meta"].get("recall", {})
        for k in (1, 5, 10):
            mine = round(100 * float((d["rank_keep"] <= k).mean()), 2)
            if f"R@{k}" in printed and abs(mine - printed[f"R@{k}"]) > 0.005:
                raise ValueError(f"{name}/{ds}: R@{k} from ranks {mine} != printed {printed[f'R@{k}']}")


def per_query(rank):
    """Per-query vectors (0-100 scale) of the compared metrics."""
    out = {f"R@{k}": 100 * M.hits(rank, k) for k in (1, 5, 10)}
    out["MRR"] = 100 * M.reciprocal_rank(rank)
    out["NDCG@10"] = 100 * M.ndcg(rank, 10)
    return out


def model_metrics(dumps, triplets):
    out = {}
    for ds, d in dumps.items():
        views = [M.canonical_views(triplets[(ds, s, t)]["views_involved"])
                 for s, t in zip(d["source_ids"].tolist(), d["target_ids"].tolist())]
        out[ds] = {"n_queries": int(len(d["rank_keep"])), "n_gallery": int(len(d["gallery_keys"])),
                   "keep": M.summary(d["rank_keep"]), "excl": M.summary(d["rank_excl"]),
                   "leak": M.leak_rates(d["rank_src"]), "hist_top10": M.rank_histogram(d["rank_keep"]),
                   "buckets": M.rank_buckets(d["rank_keep"]),
                   "depth": M.depth(d["rank_keep"], len(d["gallery_keys"])),
                   "by_views": M.by_group(d["rank_keep"], views)}
    mean = {}
    for proto in ("keep", "excl", "leak"):
        keys = [k for k in next(iter(out.values()))[proto] if k not in NO_MEAN]
        mean[proto] = {k: round(float(np.mean([out[ds][proto][k] for ds in out])), 2) for k in keys}
    out["mean"] = mean
    return out


def _aligned(da, db):
    """Index arrays that put b's queries in a's order (queries matched by source and target)."""
    key_b = {(s, t): i for i, (s, t) in enumerate(zip(db["source_ids"].tolist(), db["target_ids"].tolist()))}
    keys_a = list(zip(da["source_ids"].tolist(), da["target_ids"].tolist()))
    if len(keys_a) != len(key_b) or any(k not in key_b for k in keys_a):
        raise ValueError("the two models were scored on different query sets")
    return np.array([key_b[k] for k in keys_a])


def compare(a, b, n_boot=1000):
    """a, b: dumps per dataset of two models. Deltas are a - b."""
    out = {"per_dataset": {}, "mean": {}}
    vecs = {proto: {m: ([], []) for m in COMPARED} for proto in ("keep", "excl")}
    for ds in [x for x in DATASETS if x in a and x in b]:
        da, db = a[ds], b[ds]
        ib = _aligned(da, db)
        row = {}
        for proto in ("keep", "excl"):
            ra, rb = da[f"rank_{proto}"], db[f"rank_{proto}"][ib]
            qa, qb = per_query(ra), per_query(rb)
            row[proto] = {}
            for m in COMPARED:
                vecs[proto][m][0].append(qa[m])
                vecs[proto][m][1].append(qb[m])
                delta, ci, p = M.paired_bootstrap([qa[m]], [qb[m]], n_boot=n_boot)
                row[proto][m] = {"delta": round(delta, 2), "ci95": [round(ci[0], 2), round(ci[1], 2)],
                                 "p": round(p, 4)}
            for k in (5, 10):
                only_a, only_b, p = M.mcnemar(ra <= k, rb <= k)
                row[proto][f"mcnemar_R@{k}"] = {"only_a": only_a, "only_b": only_b, "p": p}
            row[proto]["rank_change"] = {"a_better": int((ra < rb).sum()), "b_better": int((ra > rb).sum()),
                                         "equal": int((ra == rb).sum())}
        out["per_dataset"][ds] = row
    for proto in ("keep", "excl"):
        out["mean"][proto] = {}
        for m in COMPARED:
            delta, ci, p = M.paired_bootstrap(*vecs[proto][m], n_boot=n_boot)
            out["mean"][proto][m] = {"delta": round(delta, 2), "ci95": [round(ci[0], 2), round(ci[1], 2)],
                                     "p": round(p, 4)}
    return out


def examples(a, b, triplets, n=10, top=5):
    """Queries with the largest rank gains (a over b) and losses, source kept, with top lists."""
    out = {}
    for ds in [x for x in DATASETS if x in a and x in b]:
        da, db = a[ds], b[ds]
        ib = _aligned(da, db)
        ra, rb = da["rank_keep"], db["rank_keep"][ib]
        gain = np.log(rb) - np.log(ra)  # a moving the target from 40 to 4 counts like 10 to 1
        order = np.argsort(-gain, kind="stable")

        def item(i):
            s, t = str(da["source_ids"][i]), str(da["target_ids"][i])
            trip = triplets[(ds, s, t)]
            return {"source_id": s, "target_id": t, "text": trip["modification_text_short"],
                    "views_involved": trip["views_involved"],
                    "rank_a": int(ra[i]), "rank_b": int(rb[i]),
                    "top_a": da["gallery_keys"][da["top_idx"][i, :top]].tolist(),
                    "top_b": db["gallery_keys"][db["top_idx"][ib[i], :top]].tolist()}

        out[ds] = {"a_better": [item(i) for i in order[:n] if gain[i] > 0],
                   "b_better": [item(i) for i in order[::-1][:n] if gain[i] < 0]}
    return out


def worst(dumps, triplets, n=10, top=5):
    """Per dataset, the queries whose target this model ranks deepest (source kept), with top lists."""
    out = {}
    for ds, d in dumps.items():
        rank = d["rank_keep"]
        out[ds] = []
        for i in np.argsort(-rank, kind="stable")[:n]:
            s, t = str(d["source_ids"][i]), str(d["target_ids"][i])
            trip = triplets[(ds, s, t)]
            out[ds].append({"source_id": s, "target_id": t, "text": trip["modification_text_short"],
                            "views_involved": trip["views_involved"], "rank": int(rank[i]),
                            "n_gallery": int(len(d["gallery_keys"])), "source_rank": int(d["rank_src"][i]),
                            "top": d["gallery_keys"][d["top_idx"][i, :top]].tolist()})
    return out


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def tables(all_metrics, comparisons):
    lines = []
    for proto, title in (("keep", "source kept in the gallery (as the paper)"),
                         ("excl", "source removed from each query's ranking")):
        lines += [f"## Metrics, {title}", "", "| Model | Dataset | " + " | ".join(MAIN) + " | RLR@1 |",
                  "|---|---|" + "---|" * (len(MAIN) + 1)]
        for name, mm in all_metrics.items():
            for ds in [x for x in DATASETS if x in mm] + ["mean"]:
                row = mm[ds][proto]
                cells = [_fmt(row.get(k, "")) if k in row else "" for k in MAIN]
                leak = mm[ds]["leak"]["RLR@1"] if proto == "keep" else ""
                lines.append(f"| {name} | {DISPLAY.get(ds, 'Mean')} | " + " | ".join(cells) + f" | {_fmt(leak)} |")
        lines.append("")
    lines += ["## Where the target lands (source kept): % of queries at each rank", "",
              "| Model | Dataset | " + " | ".join([str(r) for r in range(1, 11)] + [">10"]) + " |",
              "|---|---|" + "---|" * 11]
    for name, mm in all_metrics.items():
        for ds in [x for x in DATASETS if x in mm]:
            h = mm[ds]["hist_top10"]
            lines.append(f"| {name} | {DISPLAY[ds]} | " + " | ".join(_fmt(v) for v in h.values()) + " |")
    buckets = list(next(iter(next(iter(all_metrics.values())).values()))["buckets"])
    lines += ["", "## How deep the target sits (source kept): % of queries per rank band, and depth in the gallery",
              "", "| Model | Dataset | " + " | ".join(buckets)
              + " | % of gallery above target, mean / median | bottom half | bottom 10% | worst rank |",
              "|---|---|" + "---|" * (len(buckets) + 4)]
    for name, mm in all_metrics.items():
        for ds in [x for x in DATASETS if x in mm]:
            b, dp = mm[ds]["buckets"], mm[ds]["depth"]
            lines.append(f"| {name} | {DISPLAY[ds]} | " + " | ".join(_fmt(v) for v in b.values())
                         + f" | {dp['mean_pct_above']:.2f} / {dp['median_pct_above']:.2f} | {dp['bottom_half']:.2f}"
                         f" | {dp['bottom_10pct']:.2f} | {dp['worst_rank']} of {mm[ds]['n_gallery']} |")
    lines += ["", "## Recall@5 by the views the modification text involves (source kept)", ""]
    for ds in DATASETS:
        groups = []
        for mm in all_metrics.values():
            for gname in mm.get(ds, {}).get("by_views", {}):
                if gname not in groups:
                    groups.append(gname)
        if not groups:
            continue
        lines += [f"**{DISPLAY[ds]}**", "", "| Views | n | " + " | ".join(all_metrics) + " |",
                  "|---|---|" + "---|" * len(all_metrics)]
        for gname in groups:
            n = next(mm[ds]["by_views"][gname]["n"] for mm in all_metrics.values() if gname in mm[ds]["by_views"])
            cells = [_fmt(mm[ds]["by_views"].get(gname, {}).get("R@5", "")) for mm in all_metrics.values()]
            lines.append(f"| {gname} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    for pair, c in comparisons.items():
        lines += [f"## {pair}: difference (first minus second), 95% CI, p", ""]
        for proto in ("keep", "excl"):
            lines += [f"Protocol: {proto}", "", "| Dataset | " + " | ".join(COMPARED) + " |",
                      "|---|" + "---|" * len(COMPARED)]
            rows = [(DISPLAY[ds], c["per_dataset"][ds][proto]) for ds in c["per_dataset"]] + [("Mean", c["mean"][proto])]
            for label, row in rows:
                cells = [f"{row[m]['delta']:+.2f} [{row[m]['ci95'][0]:+.2f}, {row[m]['ci95'][1]:+.2f}] p={row[m]['p']:.3f}"
                         for m in COMPARED]
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--compare", action="append", default=[], help="A:B (names given with --model)")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args(argv)

    triplets = load_triplets(args.data_dir)
    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[name] = load_model(path)
        check_recall(name, models[name])
    all_metrics = {name: model_metrics(d, triplets) for name, d in models.items()}
    comparisons = {}
    exs = {"deepest_targets": {name: worst(d, triplets) for name, d in models.items()}}
    for spec in args.compare:
        a, b = spec.split(":")
        comparisons[f"{a} vs {b}"] = compare(models[a], models[b], args.n_boot)
        exs[f"{a} vs {b}"] = examples(models[a], models[b], triplets)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (out / "compare.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (out / "examples.json").write_text(json.dumps(exs, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "tables.md").write_text(tables(all_metrics, comparisons), encoding="utf-8")
    print((out / "tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
