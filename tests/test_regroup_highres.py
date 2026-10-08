import json
import os
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "src" / "data"))


def test_regroup_splits_colorways_into_product_dirs(tmp_path):
    import regroup_deepfashion_highres as rg

    item = tmp_path / "img_highres" / "WOMEN" / "Dresses" / "id_00000001"
    item.mkdir(parents=True)
    for name in ("02_1_front.jpg", "02_3_back.jpg", "03_1_front.jpg"):
        (item / name).write_bytes(b"x")
    trip = tmp_path / "t.jsonl"
    trip.write_text(json.dumps({"source_id": "WOMEN/Dresses/id_00000001/02",
                                "target_id": "WOMEN/Dresses/id_00000001/03",
                                "dataset": "deepfashion"}) + "\n")
    out = tmp_path / "out"
    rg.main(["--src", str(tmp_path / "img_highres"), "--out", str(out), "--triplets", str(trip)])
    assert sorted(p.name for p in (out / "WOMEN/Dresses/id_00000001/02").iterdir()) == \
        ["02_1_front.jpg", "02_3_back.jpg"]
    assert [p.name for p in (out / "WOMEN/Dresses/id_00000001/03").iterdir()] == ["03_1_front.jpg"]


def test_regroup_reports_missing(tmp_path):
    import regroup_deepfashion_highres as rg

    (tmp_path / "src").mkdir()
    n, found = rg.regroup(str(tmp_path / "src"), str(tmp_path / "out"), {"MEN/Tees/id_9/01"})
    assert n == 0 and found == set()


def _make_item(tmp_path, names=("02_1_front.jpg", "03_1_front.jpg")):
    item = tmp_path / "src" / "WOMEN" / "Dresses" / "id_00000001"
    item.mkdir(parents=True)
    for name in names:
        (item / name).write_bytes(b"x")
    return item


def test_regroup_ignores_non_images_and_is_idempotent(tmp_path):
    import regroup_deepfashion_highres as rg

    item = _make_item(tmp_path)
    (item / "02_Thumbs.db").write_bytes(b"x")
    (item / "02_x").mkdir()
    (item / "02_x" / "inner.jpg").write_bytes(b"x")
    pid = "WOMEN/Dresses/id_00000001/02"
    src, out = str(tmp_path / "src"), str(tmp_path / "out")
    n, found = rg.regroup(src, out, {pid})
    assert n == 1 and found == {pid}
    assert os.listdir(os.path.join(out, pid)) == ["02_1_front.jpg"]
    n2, found2 = rg.regroup(src, out, {pid})
    assert n2 == 0 and found2 == {pid}


def test_regroup_leaves_no_tmp_files(tmp_path):
    import regroup_deepfashion_highres as rg

    _make_item(tmp_path)
    pid = "WOMEN/Dresses/id_00000001/03"
    rg.regroup(str(tmp_path / "src"), str(tmp_path / "out"), {pid})
    assert not [f for f in os.listdir(tmp_path / "out" / pid) if f.endswith(".tmp")]


def test_regroup_symlink_mode(tmp_path):
    import regroup_deepfashion_highres as rg

    _make_item(tmp_path)
    probe = tmp_path / "probe"
    try:
        os.symlink(str(tmp_path / "src"), str(probe))
    except (OSError, NotImplementedError):
        pytest.skip("os.symlink not permitted on this machine")
    pid = "WOMEN/Dresses/id_00000001/02"
    src, out = str(tmp_path / "src"), str(tmp_path / "out")
    n, _ = rg.regroup(src, out, {pid}, symlink=True)
    dst = os.path.join(out, pid, "02_1_front.jpg")
    assert n == 1 and os.path.islink(dst)
    assert rg.regroup(src, out, {pid}, symlink=True)[0] == 0


def test_regroup_symlink_not_permitted_exits(tmp_path, monkeypatch):
    import regroup_deepfashion_highres as rg

    _make_item(tmp_path)

    def deny(*a, **k):
        raise OSError(1314, "privilege not held")

    monkeypatch.setattr(rg.os, "symlink", deny)
    with pytest.raises(SystemExit, match="symlinks not permitted"):
        rg.regroup(str(tmp_path / "src"), str(tmp_path / "out"),
                   {"WOMEN/Dresses/id_00000001/02"}, symlink=True)
