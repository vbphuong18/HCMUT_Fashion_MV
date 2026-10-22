import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import baselines.clip4cir_eval as ce
import baselines.qwen3vl_emb_eval as qe
from eval_metrics import report
from eval_metrics.ranking import load_dump


def test_combiner_loads_authors_state_dict_and_matches_formula():
    torch.manual_seed(0)
    c = ce.Combiner(6, 4, 5).eval()
    assert set(c.state_dict()) == {
        "text_projection_layer.weight", "text_projection_layer.bias", "image_projection_layer.weight",
        "image_projection_layer.bias", "combiner_layer.weight", "combiner_layer.bias", "output_layer.weight",
        "output_layer.bias", "dynamic_scalar.0.weight", "dynamic_scalar.0.bias", "dynamic_scalar.3.weight",
        "dynamic_scalar.3.bias"}
    img, txt = torch.randn(3, 6), torch.randn(3, 6)
    raw = torch.cat((F.relu(c.text_projection_layer(txt)), F.relu(c.image_projection_layer(img))), -1)
    a = c.dynamic_scalar(raw)
    ref = F.normalize(c.output_layer(F.relu(c.combiner_layer(raw))) + a * txt + (1 - a) * img, dim=-1)
    assert torch.allclose(c.combine_features(img, txt), ref, atol=1e-6)


def test_target_pad_only_above_ratio():
    from PIL import Image

    assert ce.target_pad(Image.new("RGB", (100, 110))).size == (100, 110)
    assert ce.target_pad(Image.new("RGB", (100, 200))).size == (160, 200)  # 200 / 1.25 = 160 wide
    assert ce.make_preprocess(32)(Image.new("RGB", (40, 90))).shape == (3, 32, 32)


def test_caption_as_authors_fiq_format():
    assert ce.caption("make it red.") == "Make it red"
    assert ce.caption("Is Shorter?") == "Is shorter"


@pytest.fixture
def val_data(tmp_path, make_images):
    """f200k val: a -> b ("to b"), c -> e ("to e"); products a, b, c, e with 2-3 views."""
    ann = tmp_path / "ann"
    ann.mkdir()
    rows = [{"source_id": "a", "target_id": "b", "dataset": "f200k", "views_involved": ["front"],
             "modification_text_short": "to b"},
            {"source_id": "c", "target_id": "e", "dataset": "f200k", "views_involved": ["front"],
             "modification_text_short": "to e"}]
    (ann / "val_triplets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for pid, n in (("a", 2), ("b", 3), ("c", 2), ("e", 3)):
        make_images(f"images/f200k/{pid}", n, size=(8, 8))
    return ann, tmp_path / "images"


AXES = {"a": 0, "b": 1, "c": 2, "e": 3}


def fake_encoders(image_calls):
    """Images on their product's axis (views differ slightly); "To b" pulls to b, "To e" weakly to e,
    so query a->b lands on b and query c->e on its source c, then e."""
    def run_images(paths):
        image_calls.append(len(paths))
        out = torch.zeros(len(paths), 8)
        for i, ps in enumerate(paths):
            p = Path(ps[0])
            out[i, AXES[p.parent.name]] = 1.0
            out[i, 6 + int(p.stem) % 2] = 0.05
        return out

    def encode_texts(texts):
        out = torch.zeros(len(texts), 8)
        for i, t in enumerate(texts):
            out[i, AXES[t[-1]]] = {"To b": 2.0, "To e": 0.6}[t]
        return out

    return (lambda _, imgs: imgs, run_images), encode_texts, lambda img, txt: F.normalize(img + txt, dim=-1)


def test_run_dataset_writes_report_readable_dumps(val_data, tmp_path, monkeypatch):
    from procir_train.retrieval_eval import val_gallery_and_queries

    monkeypatch.setattr(qe, "load_images", lambda paths: list(paths))
    products, queries = val_gallery_and_queries(*val_data, "f200k")
    out, calls = tmp_path / "out", []
    images, texts, combine = fake_encoders(calls)
    res = ce.run_dataset(out, "f200k", products, queries, images, texts, combine, 4, {"tool": "test"})
    assert set(res) == set(ce.STRATEGIES) and sum(calls) == 10  # every view encoded once
    for strategy in ce.STRATEGIES:
        d = load_dump(out / strategy / "f200k.npz")
        assert d["gallery_keys"].tolist() == ["f200k/a", "f200k/b", "f200k/c", "f200k/e"]
        assert d["rank_keep"].tolist() == [1, 2] and d["rank_excl"].tolist() == [1, 1]
        assert d["rank_src"].tolist() == [2, 1]
        assert d["meta"]["n_images"] == 10 and d["meta"]["n_query_views"] == 4
        report.check_recall(strategy, report.load_model(out / strategy))

    # second run: image features come from the cache
    calls.clear()
    ce.run_dataset(tmp_path / "out", "f200k", products, queries, images, texts, combine, 4, {"tool": "test"})
    assert calls == []
    assert ce.dataset_done(out, "f200k")
