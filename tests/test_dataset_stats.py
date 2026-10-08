import importlib.util
import json

from conftest import ROOT


def _stats():
    spec = importlib.util.spec_from_file_location("dataset_stats", ROOT / "scripts" / "dataset_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_stats_counts_products_views_and_text(tmp_path, make_images):
    s = _stats()
    rows = [{"dataset": "f200k", "source_id": "1", "target_id": "2",
             "modification_text_short": "make it red", "modification_text_long": "make it a long red dress"},
            {"dataset": "f200k", "source_id": "1", "target_id": "3",
             "modification_text_short": "blue", "modification_text_long": "make it blue"}]
    make_images("img/f200k/1", 2, size=(8, 6))
    make_images("img/f200k/2", 4, size=(8, 6))
    make_images("img/f200k/3", 6, size=(8, 6))
    out = s.split_stats(rows, tmp_path / "img", "f200k", sample_images=10)
    assert out["triplets"] == 2 and out["products"] == 3
    assert out["views_per_product"] == {"2": 1, "4": 1, "5+": 1}
    assert out["images_used_max5"] == 2 + 4 + 5
    assert out["targets_per_source_mean"] == 2.0
    assert out["mod_short_words"]["mean"] == 2.0  # "make it red" (3) and "blue" (1)
    assert out["image_size_median"] == [8, 6]


def test_main_writes_json_and_markdown(tmp_path, make_images, monkeypatch):
    s = _stats()
    ann = tmp_path / "ann"
    ann.mkdir()
    for split, ds in (("train", "deepfashion"), ("val", "deepfashion")):
        (ann / f"{split}_triplets.jsonl").write_text(json.dumps(
            {"dataset": ds, "source_id": "A/x/id_1/01", "target_id": "A/x/id_1/02",
             "modification_text_short": "x y"}) + "\n", encoding="utf-8")
        (ann / f"{split}_captions.jsonl").write_text(json.dumps(
            {"dataset": ds, "product_id": "A/x/id_1/01", "short_caption": "a b c", "long_caption": "a b c d"})
            + "\n", encoding="utf-8")
    make_images("img/deepfashion/A/x/id_1/01", 3)
    make_images("img/deepfashion/A/x/id_1/02", 3)
    out = tmp_path / "out"
    s.main(["--ann", str(ann), "--image-root", str(tmp_path / "img"), "--out", str(out),
            "--sources", "deepfashion", "--no-config-split"])
    data = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    assert data["deepfashion"]["train"]["triplets"] == 1
    assert data["deepfashion"]["captions"]["train"]["short_words"]["mean"] == 3.0
    assert "| DeepFashion |" in (out / "dataset_stats.md").read_text(encoding="utf-8")
