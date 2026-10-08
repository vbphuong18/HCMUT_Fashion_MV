import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from conftest import ROOT

sys.path.insert(0, str(ROOT / "src" / "data"))


def _parquet(path, item_ids):
    rows = [{"bytes": b"\xff\xd8fake", "path": None} for _ in item_ids]
    pq.write_table(pa.table({"item_ID": item_ids, "image": rows}), path)


def _triplets(path, dataset, pairs):
    with open(path, "w") as f:
        for s, t in pairs:
            f.write(json.dumps({"source_id": s, "target_id": t, "dataset": dataset}) + "\n")


def test_f200k_extracts_only_wanted_products(tmp_path):
    import extract_f200k

    src = tmp_path / "src"
    src.mkdir()
    _parquet(src / "a.parquet", ["111_0", "111_1", "222_0", "999_0"])
    trip = tmp_path / "train_triplets.jsonl"
    _triplets(trip, "f200k", [("111", "222")])
    out = tmp_path / "out"
    extract_f200k.main(["--src", str(src), "--out", str(out), "--triplets", str(trip)])
    assert sorted(p.name for p in (out / "111").iterdir()) == ["111_0.jpg", "111_1.jpg"]
    assert (out / "222" / "222_0.jpg").exists()
    assert not (out / "999").exists()


def test_deepfashion_maps_item_ids_to_nested_product_dirs(tmp_path):
    import extract_deepfashion

    src = tmp_path / "src"
    src.mkdir()
    _parquet(src / "a.parquet", ["WOMEN_Dresses_id_00000001_02_1_front",
                                 "WOMEN_Dresses_id_00000001_02_3_back",
                                 "MEN_Tees_id_00000009_01_1_front"])
    trip = tmp_path / "train_triplets.jsonl"
    _triplets(trip, "deepfashion", [("WOMEN/Dresses/id_00000001/02", "WOMEN/Dresses/id_00000001/02")])
    out = tmp_path / "out"
    extract_deepfashion.main(["--src", str(src), "--out", str(out), "--triplets", str(trip)])
    pdir = out / "WOMEN" / "Dresses" / "id_00000001" / "02"
    assert len(list(pdir.iterdir())) == 2
    assert not (out / "MEN").exists()


def test_extract_raises_without_shards(tmp_path):
    import pytest
    import parquet_extract

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        parquet_extract.extract(str(empty), str(tmp_path / "out"), lambda i: i)
