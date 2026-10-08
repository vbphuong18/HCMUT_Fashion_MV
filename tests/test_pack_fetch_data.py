import importlib.util
import json
import zipfile

from conftest import ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _repo(tmp_path):
    """A tiny repo: annotations + images, with one product folder that no triplet references."""
    data = tmp_path / "data"
    ann = data / "FashionMV_hf" / "data" / "data"
    _jsonl(ann / "train_triplets.jsonl", [
        {"dataset": "f200k", "source_id": "1", "target_id": "2"},
        {"dataset": "fashiongen_train", "source_id": "10", "target_id": "11"}])
    _jsonl(ann / "val_triplets.jsonl", [{"dataset": "fashiongen_val", "source_id": "20", "target_id": "21"}])
    _jsonl(ann / "train_captions.jsonl", [{"dataset": "f200k", "product_id": "1", "short_caption": "x"}])
    img = data / "images_official_hr"
    for ds, pid in [("f200k", "1"), ("f200k", "2"), ("f200k", "999"), ("fashiongen_train", "10"),
                    ("fashiongen_train", "11"), ("fashiongen_train", "77"), ("fashiongen_val", "20"),
                    ("fashiongen_val", "21")]:
        (img / ds / pid).mkdir(parents=True)
        (img / ds / pid / f"{pid}_0.jpg").write_bytes(b"x" * 100)
    return data


def test_pack_keeps_only_referenced_products_and_annotations(tmp_path):
    pack = _load("pack_data")
    data = _repo(tmp_path)
    out = tmp_path / "dist"
    manifest = pack.pack(data, out, parts=["annotations", "f200k", "fashiongen"], max_bytes=10**9)
    names = {p.name for p in out.glob("*.zip")}
    assert names == {"procir_annotations.zip", "procir_f200k.zip", "procir_fashiongen.zip"}
    with zipfile.ZipFile(out / "procir_f200k.zip") as z:
        members = set(z.namelist())
    assert "data/images_official_hr/f200k/1/1_0.jpg" in members
    assert not any("/999/" in m for m in members)
    with zipfile.ZipFile(out / "procir_fashiongen.zip") as z:
        fg = z.namelist()
    assert any("fashiongen_val/21/" in m for m in fg) and not any("/77/" in m for m in fg)
    with zipfile.ZipFile(out / "procir_annotations.zip") as z:
        assert "data/FashionMV_hf/data/data/train_captions.jsonl" in z.namelist()
    assert manifest["procir_f200k.zip"]["files"] == 2


def test_pack_splits_large_parts(tmp_path):
    pack = _load("pack_data")
    data = _repo(tmp_path)
    pack.pack(data, tmp_path / "dist", parts=["fashiongen"], max_bytes=250)
    zips = sorted(p.name for p in (tmp_path / "dist").glob("*.zip"))
    assert zips == ["procir_fashiongen_1.zip", "procir_fashiongen_2.zip"]  # 4 files x 100 B, <= 250 B each


def test_fetch_extracts_into_repo_and_skips_done_zips(tmp_path):
    pack, fetch = _load("pack_data"), _load("fetch_data")
    data = _repo(tmp_path)
    pack.pack(data, tmp_path / "dist", parts=["annotations", "f200k"], max_bytes=10**9)
    dest = tmp_path / "server"
    done = fetch.extract_all(tmp_path / "dist", dest)
    assert sorted(done) == ["procir_annotations.zip", "procir_f200k.zip"]
    assert (dest / "data/images_official_hr/f200k/2/2_0.jpg").read_bytes() == b"x" * 100
    assert (dest / "data/FashionMV_hf/data/data/val_triplets.jsonl").exists()
    assert fetch.extract_all(tmp_path / "dist", dest) == []  # markers make re-runs a no-op


def test_fetch_downloads_with_gdown_when_given_a_folder(tmp_path):
    fetch = _load("fetch_data")
    calls = []
    fetch.download_folder("https://drive.google.com/drive/folders/abc", tmp_path / "dl", run=calls.append)
    cmd = calls[0]
    assert "gdown" in " ".join(map(str, cmd)) and "--folder" in cmd and "https://drive.google.com/drive/folders/abc" in cmd
