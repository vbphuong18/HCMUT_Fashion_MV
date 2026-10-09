import importlib.util

import torch
import torch.nn.functional as F

from conftest import ROOT


def _load():
    spec = importlib.util.spec_from_file_location("clip_val_eval", ROOT / "src" / "baselines" / "clip_val_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit(*v):
    return F.normalize(torch.tensor(v, dtype=torch.float), dim=0)


def test_gallery_holds_sources_and_targets_like_upstream():
    m = _load()
    trips = [{"source_id": "a", "target_id": "b"}, {"source_id": "c", "target_id": "b"}]
    assert m.gallery_ids(trips) == ["a", "b", "c"]


def test_source_excluded_recall_skips_the_query_product():
    m = _load()
    trips = [{"source_id": "s", "target_id": "t"}]
    prod = {"s": _unit(1, 0, 0), "t": _unit(0.6, 0.8, 0), "x": _unit(0, 0, 1)}
    txt = _unit(0, 1, 0).unsqueeze(0)
    out = m.score(trips, prod, txt, alphas=(1.0,))  # image only: the source itself ranks first
    r = out["alpha"]["1.0"]
    assert out["n_gallery"] == 2 and out["n_queries"] == 1  # "x" is not named by any triplet
    assert r["with_source"]["R@1"] == 0.0 and r["source_excluded"]["R@1"] == 100.0


def test_triplets_without_images_are_dropped():
    m = _load()
    trips = [{"source_id": "s", "target_id": "t"}, {"source_id": "s", "target_id": "missing"}]
    prod = {"s": _unit(1, 0), "t": _unit(0, 1)}
    out = m.score(trips, prod, torch.stack([_unit(0, 1), _unit(0, 1)]), alphas=(0.0,))
    assert out["n_queries"] == 1 and out["alpha"]["0.0"]["with_source"]["R@1"] == 100.0


def test_list_images_keeps_the_first_five_sorted(tmp_path):
    m = _load()
    for name in ["b.jpg", "a.jpg", "f.png", "e.jpeg", "d.webp", "c.jpg", "note.txt"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in m.list_images(str(tmp_path))] == \
        ["a.jpg", "b.jpg", "c.jpg", "d.webp", "e.jpeg"]
