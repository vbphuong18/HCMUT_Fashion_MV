import importlib.util
import json
import types

import pytest
import torch
import torch.nn as nn

from conftest import ROOT
import procir_train.retrieval_eval as re_
from procir_train.retrieval_eval import recall_at_k, val_gallery_and_queries


def _upstream_compute_recall():
    spec = importlib.util.spec_from_file_location("upstream_evaluate",
                                                  ROOT / "external" / "FashionMV" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_recall


def test_recall_matches_upstream():
    torch.manual_seed(0)
    q, g = torch.randn(30, 16), torch.randn(50, 16)
    gt = torch.randint(0, 50, (30,)).tolist()
    assert recall_at_k(q, g, gt) == _upstream_compute_recall()(q, g, gt)


def test_source_exclusion():
    g = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0, 0.0]])
    assert recall_at_k(q, g, [1])["R@1"] == 0.0
    assert recall_at_k(q, g, [1], exclude=[0])["R@1"] == 100.0


def test_val_gallery_keeps_products_whose_partner_has_no_images(tmp_path, make_images):
    ann = tmp_path / "ann"
    ann.mkdir()
    rows = [{"source_id": "a", "target_id": "b", "dataset": "f200k", "views_involved": [],
             "modification_text_short": "x"},
            {"source_id": "c", "target_id": "d", "dataset": "f200k", "views_involved": [],
             "modification_text_short": "y"}]
    (ann / "val_triplets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for pid in ("a", "b", "c"):  # "d" has no images, like a product missing from the mirror
        make_images(f"images/f200k/{pid}", 6, size=(8, 8))
    products, queries = val_gallery_and_queries(ann, tmp_path / "images", "f200k")
    assert set(products) == {"f200k/a", "f200k/b", "f200k/c"}
    assert all(len(v) == 5 for v in products.values())
    assert [q["source_key"] for q in queries] == ["f200k/a"]
    assert queries[0]["mod_long"] is None


# ---- scoring path with a fake encoder (CPU, no processor) ----


class FakeEnc(nn.Module):
    def __init__(self, emb, multi_turn=None):
        super().__init__()
        self.emb = emb
        self.multi_turn = multi_turn  # None: no restriction
        self.lin = nn.Linear(1, 1)

    def encode_single(self, inputs):
        # gallery docs always use this; queries only when single-turn
        if self.multi_turn and any(x["text"] != "doc" for x in inputs):
            raise AssertionError("single-turn encode on a multi-turn query")
        return torch.stack([self.emb[x["key"]] for x in inputs])

    def encode_multiturn(self, inputs):
        if self.multi_turn is False:
            raise AssertionError("multi-turn encode in single-turn mode")
        q = torch.stack([self.emb[x["key"]] for x in inputs])
        return torch.zeros_like(q), q  # index [0] would give zeros -> wrong


@pytest.fixture
def fake_setup(monkeypatch):
    eye = torch.eye(4)
    emb = {k: eye[i] for i, k in enumerate(["d/a", "d/b", "d/c", "d/e"])}
    emb["q:a"] = emb["d/b"].clone()                    # correct
    emb["q:c"] = emb["d/c"] + 0.5 * emb["d/e"]         # top hit = own source, then target
    monkeypatch.setattr(re_, "load_images", lambda paths: paths)
    monkeypatch.setattr(re_, "doc_text", lambda p, n: "doc")
    monkeypatch.setattr(re_, "multiturn_query_text", lambda p, n, m: "q-mt")
    monkeypatch.setattr(re_, "singleturn_query_text", lambda p, n, m: "q-st")
    monkeypatch.setattr(re_, "process_visual", lambda proc, text, imgs, lo, hi: {
        "text": text, "key": imgs[0] if text == "doc" else "q:" + imgs[0].split("/")[1]})
    recs = [dict(dataset="d", source_key="d/a", target_key="d/b", source_images=["d/a"],
                 target_images=["d/b"], mod_short="x"),
            dict(dataset="d", source_key="d/c", target_key="d/e", source_images=["d/c"],
                 target_images=["d/e"], mod_short="y")]
    return emb, recs


def _cfg(multi_turn):
    return types.SimpleNamespace(image_min_pixels=1, image_max_pixels=2, multi_turn=multi_turn)


@pytest.mark.parametrize("multi_turn", [True, False])
def test_evaluate_records_scores(fake_setup, multi_turn):
    emb, recs = fake_setup
    enc = FakeEnc(emb, multi_turn)
    enc.train()
    seen = []
    orig = re_.process_visual
    re_.process_visual = lambda *a: (seen.append(a[1]), orig(*a))[1]
    try:
        res = re_.evaluate_records(enc, None, recs, _cfg(multi_turn), batch_size=1)["d"]
    finally:
        re_.process_visual = orig
    assert {t for t in seen if t != "doc"} == {"q-mt" if multi_turn else "q-st"}
    assert res["n_queries"] == 2 and res["n_gallery"] == 4
    assert res["R@1"] == 50.0 and res["R@5"] == 100.0 and res["R@10"] == 100.0
    assert res["R@1_src_excluded"] == 100.0
    assert enc.training  # restored


def test_evaluate_records_max_queries_and_mode_restore(fake_setup):
    emb, recs = fake_setup
    enc = FakeEnc(emb)
    enc.eval()
    res = re_.evaluate_records(enc, None, recs, _cfg(True), max_queries=1, batch_size=3)["d"]
    assert res["n_queries"] == 1 and res["n_gallery"] == 2
    assert res["R@1"] == 100.0
    assert not enc.training


def test_evaluate_val_empty_dataset_raises(tmp_path):
    (tmp_path / "val_triplets.jsonl").write_text("")
    with pytest.raises(ValueError, match="no queries/products for dataset f200k"):
        re_.evaluate_val(FakeEnc({}), None, tmp_path, tmp_path / "images", ["f200k"], True)


def _val_fixture(tmp_path, make_images):
    ann = tmp_path / "ann"
    ann.mkdir()
    row = {"source_id": "a", "target_id": "b", "dataset": "f200k", "modification_text_short": "x"}
    (ann / "val_triplets.jsonl").write_text(json.dumps(row) + "\n")
    for pid in ("a", "b"):
        make_images(f"images/f200k/{pid}", 6, size=(8, 8))
    return ann, tmp_path / "images"


def _recording_val(monkeypatch, ann, images, **kwargs):
    seen = []

    def fake_visual(proc, text, imgs, lo, hi):
        seen.append((len(imgs), lo, hi))
        return {"text": text, "key": "q" if text != "doc" else imgs[0]}

    monkeypatch.setattr(re_, "load_images", lambda paths: list(paths))
    monkeypatch.setattr(re_, "doc_text", lambda p, n: "doc")
    monkeypatch.setattr(re_, "multiturn_query_text", lambda p, n, m: "q-mt")
    monkeypatch.setattr(re_, "process_visual", fake_visual)

    class Enc(FakeEnc):
        def encode_single(self, inputs):
            return torch.ones(len(inputs), 2)

        def encode_multiturn(self, inputs):
            return torch.ones(len(inputs), 2), torch.ones(len(inputs), 2)

    re_.evaluate_val(Enc({}), None, ann, images, ["f200k"], True, **kwargs)
    return seen


def test_evaluate_val_defaults_to_upstream_protocol(tmp_path, make_images, monkeypatch):
    ann, images = _val_fixture(tmp_path, make_images)
    seen = _recording_val(monkeypatch, ann, images)
    assert seen and set(seen) == {(5, 128 * 128, 512 * 512)}


def test_evaluate_val_with_training_views_and_pixels(tmp_path, make_images, monkeypatch):
    ann, images = _val_fixture(tmp_path, make_images)
    seen = _recording_val(monkeypatch, ann, images, max_views=3, pixels=(128 * 128, 336 * 336))
    assert seen and set(seen) == {(3, 128 * 128, 336 * 336)}


def test_protocol_settings():
    cfg = types.SimpleNamespace(max_views=3, image_min_pixels=128 * 128, image_max_pixels=336 * 336)
    assert re_.protocol_settings("upstream", cfg) == (5, (128 * 128, 512 * 512))
    assert re_.protocol_settings("train", cfg) == (3, (128 * 128, 336 * 336))
    with pytest.raises(ValueError, match="protocol"):
        re_.protocol_settings("other", cfg)
