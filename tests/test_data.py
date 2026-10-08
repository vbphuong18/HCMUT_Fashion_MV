import json

import pytest

from procir_train.config import TrainConfig
from procir_train.data import (attach_images, build_records, load_captions, load_triplets,
                               split_dev, stratified_subsample)


@pytest.fixture
def corpus(tmp_path, make_images):
    rows = []
    for ds in ("deepfashion", "f200k"):
        for i in range(40):
            for j in range(2):
                rows.append({"source_id": f"s{i}", "target_id": f"t{i}_{j}", "dataset": ds,
                             "views_involved": ["front", "back"],
                             "modification_text_short": f"short {i}{j}",
                             "modification_text_long": f"long {i}{j}"})
    for r in rows:
        for k in ("source_id", "target_id"):
            if r[k] != "t0_0":  # one product without images in each dataset
                make_images(f"images/{r['dataset']}/{r[k]}", 2, size=(8, 8))
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / "train_triplets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    caps = [{"product_id": r[k], "dataset": r["dataset"], "long_caption": "L", "short_caption": "S"}
            for r in rows for k in ("source_id", "target_id")]
    (ann / "train_captions.jsonl").write_text("\n".join(json.dumps(c) for c in caps) + "\n")
    return ann, tmp_path / "images"


def test_load_triplets_filters_datasets(corpus):
    ann, _ = corpus
    assert len(load_triplets(ann / "train_triplets.jsonl", ["f200k"])) == 80


def test_attach_images_drops_triplets_without_images_and_caps_views(corpus):
    ann, images = corpus
    trip = load_triplets(ann / "train_triplets.jsonl", ["deepfashion", "f200k"])
    recs = attach_images(trip, str(images), max_views=1)
    assert len(recs) == 158  # 2 x (80 - 1)
    assert all(len(r["source_images"]) == 1 and len(r["target_images"]) == 1 for r in recs)
    assert recs[0]["source_key"] == "deepfashion/s0"


def test_split_dev_has_no_product_leakage(corpus):
    ann, images = corpus
    recs = attach_images(load_triplets(ann / "train_triplets.jsonl", ["deepfashion", "f200k"]),
                         str(images), 3)
    train, dev = split_dev(recs, 0.05, seed=42)
    train_p = {r[k] for r in train for k in ("source_key", "target_key")}
    dev_p = {r[k] for r in dev for k in ("source_key", "target_key")}
    assert dev and not (train_p & dev_p)
    assert {r["dataset"] for r in dev} == {"deepfashion", "f200k"}
    assert split_dev(recs, 0.05, seed=42) == (train, dev)


def test_stratified_subsample_takes_fraction_per_dataset(corpus):
    ann, images = corpus
    recs = attach_images(load_triplets(ann / "train_triplets.jsonl", ["deepfashion", "f200k"]),
                         str(images), 3)
    sub = stratified_subsample(recs, 0.5, seed=1)
    assert sum(r["dataset"] == "deepfashion" for r in sub) == round(0.5 * 79)
    assert sum(r["dataset"] == "f200k" for r in sub) == round(0.5 * 79)
    assert stratified_subsample(recs, 0.5, seed=1) == sub


def test_split_dev_drops_bridging_triplets_exactly_once(corpus):
    recs = []
    for i in range(10):
        # shared targets (t_{i+1} reached from s_i and s_{i+1}) and chains (t_i is also a source)
        for tgt in (f"t{i}", f"t{(i + 1) % 10}"):
            recs.append({"dataset": "x", "source_id": f"s{i}", "target_id": tgt,
                         "source_key": f"x/s{i}", "target_key": f"x/{tgt}",
                         "source_images": ["a"], "target_images": ["b"],
                         "mod_short": "m", "mod_long": "l"})
        recs.append({"dataset": "x", "source_id": f"t{i}", "target_id": f"s{(i + 3) % 10}",
                     "source_key": f"x/t{i}", "target_key": f"x/s{(i + 3) % 10}",
                     "source_images": ["a"], "target_images": ["b"],
                     "mod_short": "m", "mod_long": "l"})
    train, dev = split_dev(recs, 0.2, seed=7)
    train_p = {r[k] for r in train for k in ("source_key", "target_key")}
    dev_p = {r[k] for r in dev for k in ("source_key", "target_key")}
    assert dev and train and not (train_p & dev_p)
    in_train, in_dev = {id(r) for r in train}, {id(r) for r in dev}
    assert not (in_train & in_dev)
    dropped = [r for r in recs if id(r) not in in_train | in_dev]
    assert dropped
    assert len(train) + len(dev) + len(dropped) == len(recs)


def test_split_and_subsample_order_not_clustered(corpus):
    ann, images = corpus
    recs = attach_images(load_triplets(ann / "train_triplets.jsonl", ["deepfashion", "f200k"]),
                         str(images), 3)
    sub = stratified_subsample(recs, 1.0, seed=3)
    assert len({r["dataset"] for r in sub[:20]}) == 2
    _, dev = split_dev(recs, 0.3, seed=3)
    df = [r["source_key"] for r in dev if r["dataset"] == "deepfashion"]
    assert len(df) > 2 and df != sorted(df)


def test_build_records_and_captions(corpus):
    ann, images = corpus
    cfg = TrainConfig(data_dir=str(ann), image_root=str(images), train_fraction=1.0)
    train, dev, stats = build_records(cfg)
    assert stats["deepfashion"]["triplets"] == 80 and stats["deepfashion"]["usable"] == 79
    for ds in ("deepfashion", "f200k"):
        st = stats[ds]
        assert st["train"] == st["train_pool"]
        assert st["dropped"] >= 0
        assert st["train_pool"] + st["dev"] + st["dropped"] == st["usable"]
    assert len(train) == sum(stats[d]["train"] for d in stats)
    assert len(dev) == sum(stats[d]["dev"] for d in stats)
    caps = load_captions(ann / "train_captions.jsonl")
    assert caps["f200k/t3_1"] == {"long": "L", "short": "S"}


def test_build_records_raises_when_dataset_has_no_images(corpus, tmp_path):
    ann, _ = corpus
    empty = tmp_path / "empty_images"
    empty.mkdir()
    cfg = TrainConfig(data_dir=str(ann), image_root=str(empty), train_fraction=1.0)
    with pytest.raises(ValueError, match=r"deepfashion: 0 of 80 train triplets.*empty_images"):
        build_records(cfg)
