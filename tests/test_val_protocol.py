import json

import pytest

from conftest import ROOT


def _norm(paths):
    return [p.replace("\\", "/") for p in paths]


def _compare(data_dir, image_root, datasets, monkeypatch):
    import procir.datasets as up
    from procir.datasets import CIRValDataset, ProductValDataset
    from procir_train.retrieval_eval import val_gallery_and_queries

    # upstream opens the jsonl without an encoding (fine on Linux, cp1252 on Windows): force utf-8 for this test
    monkeypatch.setattr(up, "open", lambda f, *a, **k: open(f, *a, encoding="utf-8", **k), raising=False)
    up_products = ProductValDataset(str(data_dir), str(image_root)).samples
    up_queries = CIRValDataset(str(data_dir), str(image_root)).samples
    for ds in datasets:
        products, queries = val_gallery_and_queries(data_dir, image_root, ds)
        theirs_p = {f"{s['dataset']}/{s['product_id']}": _norm(s["image_paths"])
                    for s in up_products if s["dataset"] == ds}
        ours_p = {k: _norm(v) for k, v in products.items()}
        assert ours_p == theirs_p, ds
        assert ours_p, ds  # an empty comparison proves nothing
        theirs_q = [(s["source_id"], s["target_id"], s["modification_text_short"],
                     _norm(s["source_image_paths"]), _norm(s["target_image_paths"]))
                    for s in up_queries if s["dataset"] == ds]
        ours_q = [(r["source_id"], r["target_id"], r["mod_short"],
                   _norm(r["source_images"]), _norm(r["target_images"])) for r in queries]
        assert ours_q == theirs_q, ds
        assert ours_q, ds


def test_gallery_and_queries_match_upstream_datasets_on_synthetic_corpus(tmp_path, make_images, monkeypatch):
    root = tmp_path
    layout = {  # (dataset, product) -> number of images
        ("deepfashion", "A/id_1/01"): 3,
        ("deepfashion", "A/id_1/02"): 7,   # more than 5 views
        ("deepfashion", "A/id_2/01"): 1,
        ("deepfashion", "A/id_3/01"): 5,
        ("f200k", "p10"): 2,
        ("f200k", "p11"): 6,
        ("f200k", "p12"): 4,
    }
    for (ds, pid), n in layout.items():
        make_images(f"images/{ds}/{pid}", n)
    rows = [
        ("deepfashion", "A/id_1/01", "A/id_1/02", "t1"),
        ("deepfashion", "A/id_1/02", "A/id_2/01", "t2"),   # id_1/02 is a target and then a source
        ("deepfashion", "A/id_2/01", "A/id_3/01", "t3"),
        ("deepfashion", "A/id_3/01", "A/missing/01", "t4"),  # target without images
        ("deepfashion", "A/gone/01", "A/id_1/01", "t5"),     # source without images
        ("f200k", "p10", "p11", "t6"),
        ("f200k", "p11", "p12", "t7"),
        ("f200k", "p12", "p99", "t8"),
    ]
    data = tmp_path / "data"
    data.mkdir()
    with open(data / "val_triplets.jsonl", "w", encoding="utf-8") as f:
        for ds, s, t, text in rows:  # two datasets interleaved in one file
            f.write(json.dumps({"source_id": s, "target_id": t, "dataset": ds,
                                "modification_text_short": text}) + "\n")
    _compare(data, root / "images", ["deepfashion", "f200k"], monkeypatch)


@pytest.mark.parametrize("ds", ["deepfashion", "f200k"])
def test_gallery_and_queries_match_upstream_datasets_on_real_val(ds, monkeypatch):
    data_dir = ROOT / "data" / "FashionMV_hf" / "data" / "data"
    image_root = ROOT / "data" / "images"
    if not (data_dir / "val_triplets.jsonl").exists() or not (image_root / ds).exists():
        pytest.skip(f"real val annotations or {ds} images missing")
    _compare(data_dir, image_root, [ds], monkeypatch)
