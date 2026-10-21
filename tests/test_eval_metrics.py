import json
import math

import numpy as np
import pytest
import torch

import procir_train.retrieval_eval as re_
from eval_metrics import metrics as M
from eval_metrics.ranking import load_dump, rank_queries, save_dump

# ---- ranking ----


def _toy():
    """Gallery of 7 products on 7 axes; 3 queries whose similarities are set by hand.
    Query 0: target 1 first.  Query 1: source 2 first, then 0, then target 3.
    Query 2: source 4 first, target 6 last (rank 7 kept, 6 with the source removed)."""
    g = np.eye(7, dtype=np.float32)
    q = np.zeros((3, 7), dtype=np.float32)
    q[0] = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.05]
    q[1] = [0.5, 0.0, 0.9, 0.3, 0.0, 0.0, 0.0]
    q[2] = [0.7, 0.6, 0.5, 0.4, 0.8, 0.3, 0.2]
    gt, src = [1, 3, 6], [0, 2, 4]
    return q, g, gt, src


def test_ranks_by_hand():
    q, g, gt, src = _toy()
    r = rank_queries(q, g, gt, src)
    assert r["rank_keep"].tolist() == [1, 3, 7]
    assert r["rank_excl"].tolist() == [1, 2, 6]
    assert r["rank_src"].tolist() == [2, 1, 1]
    assert r["top_idx"][1, :3].tolist() == [2, 0, 3]


def test_ties_break_by_lower_index():
    g = np.eye(3, dtype=np.float32)
    q = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    r = rank_queries(q, g, [2], [0])
    assert r["rank_keep"].tolist() == [3]
    assert r["rank_excl"].tolist() == [2]
    assert r["top_idx"][0].tolist() == [0, 1, 2]


def test_ranks_match_recall_of_eval_tools_and_chunking():
    rng = np.random.default_rng(0)
    q, g = rng.normal(size=(300, 16)), rng.normal(size=(120, 16))
    gt, src = rng.integers(0, 120, 300), rng.integers(0, 120, 300)
    r = rank_queries(q, g, gt, src, chunk=37)
    tq, tg = torch.tensor(q), torch.tensor(g)
    ref = re_.recall_at_k(tq, tg, gt.tolist(), ks=(1, 5, 10))
    ref_x = re_.recall_at_k(tq, tg, gt.tolist(), ks=(1, 5, 10), exclude=src.tolist())
    for k in (1, 5, 10):
        assert round(100 * M.hits(r["rank_keep"], k).mean(), 2) == ref[f"R@{k}"]
        assert round(100 * M.hits(r["rank_excl"], k).mean(), 2) == ref_x[f"R@{k}"]
    full = rank_queries(q, g, gt, src)
    assert all(np.array_equal(full[k], r[k]) for k in ("rank_keep", "rank_excl", "rank_src", "top_idx"))


def test_top_list_without_source_is_inside_stored_list():
    rng = np.random.default_rng(1)
    q, g = rng.normal(size=(50, 8)), rng.normal(size=(80, 8))
    gt, src = rng.integers(0, 80, 50), rng.integers(0, 80, 50)
    r = rank_queries(q, g, gt, src)
    for i in range(50):
        no_src = [j for j in r["top_idx"][i] if j != src[i]]
        assert len(no_src) >= 50
        hit = gt[i] in no_src[:10]
        assert hit == (r["rank_excl"][i] <= 10)


def test_save_and_load_dump(tmp_path):
    q, g, gt, src = _toy()
    keys = [f"p{i}" for i in range(7)]
    save_dump(tmp_path / "ds.npz", q, g, keys, ["p0", "p2", "p4"], ["p1", "p3", "p6"], gt, src,
              meta={"tool": "test"})
    d = load_dump(tmp_path / "ds.npz")
    assert d["rank_keep"].tolist() == [1, 3, 7]
    assert d["gallery_keys"].tolist() == keys
    assert d["q"].dtype == np.float16
    assert d["meta"]["tool"] == "test" and d["meta"]["n_gallery"] == 7


# ---- metrics ----


def test_metric_values_by_hand():
    rank = np.array([1, 3, 6])
    s = M.summary(rank, ks=(1, 5))
    assert s["R@1"] == pytest.approx(33.33)
    assert s["R@5"] == pytest.approx(66.67)
    assert s["MRR"] == pytest.approx(round(100 * (1 + 1 / 3 + 1 / 6) / 3, 2))
    assert s["MRR@10"] == s["MRR"]
    assert s["NDCG@10"] == pytest.approx(round(100 * (1 + 1 / math.log2(4) + 1 / math.log2(7)) / 3, 2))
    assert s["NDCG@5"] == pytest.approx(round(100 * (1 + 0.5) / 3, 2))
    assert s["MedR"] == 3 and s["MeanR"] == pytest.approx(10 / 3, abs=0.01)


def test_map_equals_mrr_with_one_target():
    rank = np.array([1, 2, 7, 15, 40])
    k = 10
    ap = np.where(rank <= k, 1.0 / rank, 0.0)  # AP@K with a single relevant item
    assert np.allclose(ap, M.reciprocal_rank(rank, k))


def test_leak_rates_and_histogram():
    assert M.leak_rates([1, 2, 1, 11], ks=(1, 10)) == {"RLR@1": 50.0, "RLR@10": 75.0}
    h = M.rank_histogram([1, 1, 3, 12], k=3)
    assert h == {"1": 50.0, "2": 0.0, "3": 25.0, ">3": 25.0}


@pytest.mark.parametrize("views,key", [
    (["front", "back"], "back+front"),
    (["back", "front"], "back+front"),
    (["front", "side", "back"], "back+front+side"),
    (["side/back"], "back+side"),
    (["detail"], "other"),
    (["front", "detail"], "front+other"),
    ([], "none"),
])
def test_canonical_views(views, key):
    assert M.canonical_views(views) == key


def test_by_group():
    out = M.by_group([1, 4, 20, 2], ["a", "a", "b", "a"], ks=(1, 5))
    assert out["a"] == {"n": 3, "R@1": 33.33, "R@5": 100.0, "MRR": round(100 * (1 + 0.25 + 0.5) / 3, 2)}
    assert list(out) == ["a", "b"]


def test_paired_bootstrap_detects_a_real_gap_and_not_a_null_one():
    rng = np.random.default_rng(0)
    b = [rng.random(2000) < 0.5, rng.random(3000) < 0.6]
    a = [x | (rng.random(len(x)) < 0.1) for x in b]  # a strictly better
    delta, (lo, hi), p = M.paired_bootstrap(a, b, n_boot=300)
    assert delta > 0 and lo > 0 and p < 0.01
    same = [rng.random(2000) < 0.5]
    delta0, (lo0, hi0), p0 = M.paired_bootstrap(same, same, n_boot=100)
    assert delta0 == 0 and lo0 == hi0 == 0 and p0 == 1.0


def test_mcnemar():
    a = np.array([1] * 30 + [0] * 10 + [1] * 50, dtype=bool)
    b = np.array([0] * 30 + [1] * 10 + [1] * 50, dtype=bool)
    only_a, only_b, p = M.mcnemar(a, b)
    assert (only_a, only_b) == (30, 10)
    assert p == pytest.approx(math.erfc(math.sqrt(((20 - 1) ** 2 / 40) / 2)))
    assert M.mcnemar(a, a)[2] == 1.0


# ---- upstream protocol with dumps ----


def test_upstream_score_matches_evaluate_py_and_dumps(tmp_path):
    from collections import OrderedDict

    from eval_metrics import upstream_dump as U

    up = U.load_upstream()
    torch.manual_seed(0)
    doc_emb, doc_dset = OrderedDict(), {}
    for ds, n in (("f200k", 40), ("deepfashion", 30)):
        for i in range(n):
            doc_emb[f"{ds}-{i}"], doc_dset[f"{ds}-{i}"] = torch.randn(8), ds
    tids, sids, q_dsets = [], [], []
    for ds, n in (("f200k", 40), ("deepfashion", 30)):
        for i in range(25):
            sids.append(f"{ds}-{i}")
            tids.append(f"{ds}-{(i * 7 + 3) % n}")
            q_dsets.append(ds)
    tids[0] = "missing"  # target without images: skipped, as in evaluate.py
    all_q = torch.randn(len(tids), 8)

    res = U.score(up, doc_emb, doc_dset, all_q, tids, sids, q_dsets, tmp_path, meta={"model_path": "m"})

    # evaluate.py phase 3, written out
    expected = {}
    for ds in ("deepfashion", "f200k"):
        pids = [p for p, d in doc_dset.items() if d == ds]
        idx = {p: i for i, p in enumerate(pids)}
        valid = [i for i, d in enumerate(q_dsets) if d == ds and tids[i] in idx]
        r = up.compute_recall(all_q[valid], torch.stack([doc_emb[p] for p in pids]), [idx[tids[i]] for i in valid])
        expected.update({f"{up.DS_DISPLAY[ds]}_{k}": v for k, v in r.items()})
    assert all(res[k] == v for k, v in expected.items())
    assert res["Average_R@5"] == round((res["DeepFashion_R@5"] + res["Fashion200K_R@5"]) / 2, 2)

    d = load_dump(tmp_path / "f200k.npz")
    assert len(d["rank_keep"]) == 24 and d["source_ids"][0] == "f200k-1"
    for k in (1, 5, 10):
        assert round(100 * float((d["rank_keep"] <= k).mean()), 2) == res[f"Fashion200K_R@{k}"]
    assert d["gallery_keys"][d["src"]].tolist() == d["source_ids"].tolist()
    assert d["gallery_keys"][d["gt"]].tolist() == d["target_ids"].tolist()
    assert d["meta"]["model_path"] == "m"


# ---- report ----


def _fake_model(root, name, rng, quality, triplet_rows):
    from eval_metrics.ranking import save_dump

    for ds, n_gal in (("deepfashion", 60), ("f200k", 90)):
        rows = [t for t in triplet_rows if t["dataset"] == ds]
        keys = [f"{ds}-{i}" for i in range(n_gal)]
        idx = {k: i for i, k in enumerate(keys)}
        g = rng.normal(size=(n_gal, 16))
        gt = np.array([idx[t["target_id"]] for t in rows])
        src = np.array([idx[t["source_id"]] for t in rows])
        q = rng.normal(size=(len(rows), 16)) + quality * g[gt]
        order = rng.permutation(len(rows))  # the two tools may list queries in different orders
        r = save_dump(root / name / f"{ds}.npz", q[order], g, keys,
                      [rows[i]["source_id"] for i in order], [rows[i]["target_id"] for i in order],
                      gt[order], src[order])
        meta_path = root / name / f"{ds}.json"
        meta = json.loads(meta_path.read_text())
        meta["recall"] = {f"R@{k}": round(100 * float((r["rank_keep"] <= k).mean()), 2) for k in (1, 5, 10)}
        meta_path.write_text(json.dumps(meta))


def test_report_end_to_end(tmp_path):
    import json as _json

    from eval_metrics import report as R

    rng = np.random.default_rng(0)
    rows = []
    for ds, n_gal in (("deepfashion", 60), ("f200k", 90)):
        for i in range(n_gal - 10):
            rows.append({"source_id": f"{ds}-{i}", "target_id": f"{ds}-{(i + 5) % n_gal}", "dataset": ds,
                         "views_involved": ["front", "back"] if i % 3 else ["side"],
                         "modification_text_short": f"text {i}"})
    data = tmp_path / "data"
    data.mkdir()
    (data / "val_triplets.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    _fake_model(tmp_path, "good", rng, 3.0, rows)
    _fake_model(tmp_path, "weak", rng, 0.3, rows)

    R.main(["--data_dir", str(data), "--out", str(tmp_path / "out"), "--n_boot", "200",
            "--model", f"good={tmp_path / 'good'}", "--model", f"weak={tmp_path / 'weak'}",
            "--compare", "good:weak"])
    mm = _json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert mm["good"]["mean"]["keep"]["R@5"] > mm["weak"]["mean"]["keep"]["R@5"]
    assert mm["good"]["f200k"]["n_queries"] == 80
    assert set(mm["good"]["f200k"]["by_views"]) == {"back+front", "side"}
    c = _json.loads((tmp_path / "out" / "compare.json").read_text())["good vs weak"]
    assert c["mean"]["keep"]["R@5"]["delta"] > 0 and c["mean"]["keep"]["R@5"]["ci95"][0] > 0
    rc = c["per_dataset"]["f200k"]["keep"]["rank_change"]
    assert rc["a_better"] + rc["b_better"] + rc["equal"] == 80
    ex = _json.loads((tmp_path / "out" / "examples.json").read_text())["good vs weak"]["f200k"]
    first = ex["a_better"][0]
    assert first["rank_a"] < first["rank_b"] and len(first["top_a"]) == 5 and first["text"].startswith("text")
    assert "## good vs weak" in (tmp_path / "out" / "tables.md").read_text()


def test_report_refuses_ranks_that_disagree_with_printed_recall(tmp_path):
    from eval_metrics import report as R
    from eval_metrics.ranking import save_dump

    q, g, gt, src = _toy()
    save_dump(tmp_path / "m" / "f200k.npz", q, g, [f"p{i}" for i in range(7)], ["a"] * 3, ["b"] * 3, gt, src,
              meta={"recall": {"R@1": 99.0}})
    with pytest.raises(ValueError, match="R@1 from ranks"):
        R.check_recall("m", R.load_model(tmp_path / "m"))


def test_rank_buckets_and_depth():
    rank = np.array([1, 3, 8, 30, 70, 300, 900, 1000])
    b = M.rank_buckets(rank)
    assert b == {"1": 12.5, "2-5": 12.5, "6-10": 12.5, "11-50": 12.5, "51-100": 12.5, "101-500": 12.5, ">500": 25.0}
    assert sum(b.values()) == 100.0
    d = M.depth([1, 501, 1000], 1001)  # 0%, 50%, 99.9% of the gallery above the target
    assert d["mean_pct_above"] == round((0 + 50 + 99.9) / 3, 2)
    assert d["median_pct_above"] == 50.0
    assert d["bottom_half"] == round(200 / 3, 2) and d["bottom_10pct"] == round(100 / 3, 2)
    assert d["worst_rank"] == 1000


def test_report_shows_deepest_targets(tmp_path):
    import json as _json

    from eval_metrics import report as R

    rng = np.random.default_rng(3)
    rows = [{"source_id": f"{ds}-{i}", "target_id": f"{ds}-{(i + 5) % n}", "dataset": ds,
             "views_involved": ["front"], "modification_text_short": f"t{i}"}
            for ds, n in (("deepfashion", 60), ("f200k", 90)) for i in range(n - 10)]
    data = tmp_path / "data"
    data.mkdir()
    (data / "val_triplets.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    _fake_model(tmp_path, "m", rng, 0.5, rows)
    R.main(["--data_dir", str(data), "--out", str(tmp_path / "out"), "--model", f"m={tmp_path / 'm'}"])
    deep = _json.loads((tmp_path / "out" / "examples.json").read_text())["deepest_targets"]["m"]["f200k"]
    ranks = [x["rank"] for x in deep]
    assert ranks == sorted(ranks, reverse=True) and deep[0]["n_gallery"] == 90
    mm = _json.loads((tmp_path / "out" / "metrics.json").read_text())["m"]["f200k"]
    assert abs(sum(mm["buckets"].values()) - 100) < 0.05
    assert "How deep the target sits" in (tmp_path / "out" / "tables.md").read_text()
