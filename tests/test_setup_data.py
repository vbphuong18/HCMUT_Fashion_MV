import importlib.util
import json
import zipfile

import pytest

from conftest import ROOT


def _setup():
    spec = importlib.util.spec_from_file_location("setup_data", ROOT / "scripts" / "setup_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _ann(data_root):
    ann = data_root / "FashionMV_hf" / "data" / "data"
    _write_jsonl(ann / "train_triplets.jsonl", [
        {"dataset": "deepfashion", "source_id": "MEN/Denim/id_1/01", "target_id": "MEN/Denim/id_1/02",
         "modification_text_short": "x", "modification_text_long": "xx"}])
    _write_jsonl(ann / "val_triplets.jsonl", [
        {"dataset": "deepfashion", "source_id": "WOMEN/Tees/id_2/03", "target_id": "MEN/Denim/id_1/01",
         "modification_text_short": "y"}])
    return ann


def _inshop_tree(root, jpeg):
    files = ["MEN/Denim/id_1/01_1_front.jpg", "MEN/Denim/id_1/01_2_side.jpg", "MEN/Denim/id_1/02_1_front.jpg",
             "WOMEN/Tees/id_2/03_1_front.jpg", "WOMEN/Tees/id_2/04_1_front.jpg"]
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes(jpeg)
    return files


@pytest.fixture
def jpeg():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "JPEG")
    return buf.getvalue()


def test_find_inshop_root_accepts_both_layouts(tmp_path):
    s = _setup()
    (tmp_path / "a" / "img_highres" / "MEN").mkdir(parents=True)
    (tmp_path / "a" / "img_highres" / "WOMEN").mkdir()
    assert s.find_inshop_root(tmp_path / "a") == tmp_path / "a" / "img_highres"
    assert s.find_inshop_root(tmp_path / "a" / "img_highres") == tmp_path / "a" / "img_highres"
    with pytest.raises(FileNotFoundError):
        s.find_inshop_root(tmp_path / "missing")


def test_extract_inshop_python_fallback_only_takes_men_and_women(tmp_path, jpeg):
    s = _setup()
    z = tmp_path / "img_highres.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("img_highres/MEN/Denim/id_1/01_1_front.jpg", jpeg)
        zf.writestr("img_highres/WOMEN/Tees/id_2/03_1_front.jpg", jpeg)
        zf.writestr("img_highres/CLOTHING/other.jpg", jpeg)  # Category benchmark images: not needed
    root = s.extract_inshop(z, tmp_path / "out", password=None, use_unzip=False)
    assert (root / "MEN" / "Denim" / "id_1" / "01_1_front.jpg").exists()
    assert not (tmp_path / "out" / "img_highres" / "CLOTHING").exists()


def test_extract_inshop_uses_unzip_with_password(tmp_path):
    s = _setup()
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        for g in ("MEN", "WOMEN"):
            (tmp_path / "out" / "img_highres" / g).mkdir(parents=True, exist_ok=True)

    root = s.extract_inshop(tmp_path / "x.zip", tmp_path / "out", password="pw", use_unzip=True, run=fake_run)
    cmd = calls[0]
    assert cmd[0] == "unzip" and cmd[cmd.index("-P") + 1] == "pw"
    assert "img_highres/MEN/*" in cmd and "img_highres/WOMEN/*" in cmd
    assert root == tmp_path / "out" / "img_highres"


def test_stage_deepfashion_from_extracted_folder(tmp_path, jpeg):
    s = _setup()
    data = tmp_path / "data"
    ann = _ann(data)
    _inshop_tree(tmp_path / "inshop" / "img_highres", jpeg)
    s.stage_deepfashion(tmp_path / "inshop", data, ann)
    out = data / "images_official_hr" / "deepfashion"
    assert sorted(p.name for p in (out / "MEN/Denim/id_1/01").iterdir()) == ["01_1_front.jpg", "01_2_side.jpg"]
    assert (out / "WOMEN/Tees/id_2/03/03_1_front.jpg").exists()
    assert not (out / "WOMEN/Tees/id_2/04").exists()  # not referenced by any triplet


def test_stage_fashiongen_downloads_then_extracts(tmp_path):
    s = _setup()
    calls = []
    s.stage_fashiongen(tmp_path / "data", run=calls.append, unzip=lambda *a: None)
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("kaggle datasets download bothin/fashiongen-validation -f fashiongen_256_256_train.h5" in j
               for j in joined)
    prep = [j for j in joined if "prepare_fashiongen.py" in j]
    assert len(prep) == 2
    assert any("fashiongen_256_256_validation.h5" in j and j.endswith("fashiongen_val") for j in prep)


def test_stage_fashiongen_skips_existing_h5(tmp_path):
    s = _setup()
    raw = tmp_path / "data" / "raw" / "fashiongen"
    raw.mkdir(parents=True)
    for f in s.FASHIONGEN_FILES.values():
        (raw / f).write_bytes(b"h5")
    calls = []
    s.stage_fashiongen(tmp_path / "data", run=calls.append, unzip=lambda *a: None)
    assert not any("kaggle" in " ".join(map(str, c)) for c in calls)


def test_verify_reports_coverage_and_status_file(tmp_path, jpeg):
    s = _setup()
    data = tmp_path / "data"
    ann = _ann(data)
    _inshop_tree(tmp_path / "inshop", jpeg)
    s.stage_deepfashion(tmp_path / "inshop", data, ann)
    ok = s.verify(data, ann, datasets=["deepfashion"])
    assert ok
    status = json.loads((data / "data_status.json").read_text(encoding="utf-8"))
    assert status["train"]["deepfashion"] == {"usable": 1, "total": 1}
    assert status["val"]["deepfashion"] == {"usable": 1, "total": 1}

    import shutil
    shutil.rmtree(data / "images_official_hr" / "deepfashion" / "WOMEN")
    assert not s.verify(data, ann, datasets=["deepfashion"])


def test_stage_f200k_downloads_full_images_then_crops_like_the_release(tmp_path, monkeypatch):
    s = _setup()
    import crop_f200k
    import download_f200k_urls

    calls = []
    monkeypatch.setattr(download_f200k_urls, "main", lambda argv: calls.append(("download", argv)))
    monkeypatch.setattr(crop_f200k, "main", lambda argv: calls.append(("crop", argv)))
    data, ann = tmp_path / "data", tmp_path / "ann"
    urls_zip = tmp_path / "fashion-200k.zip"
    with zipfile.ZipFile(urls_zip, "w") as z:
        z.writestr("fashion-200k/image_urls.txt", "")
    s.stage_f200k(urls_zip, data, ann)
    (kind1, dl), (kind2, crop) = calls
    assert kind1 == "download" and dl[dl.index("--out") + 1] == str(data / "images_official_hr" / "f200k_source")
    assert kind2 == "crop"
    assert crop[crop.index("--src") + 1] == str(data / "images_official_hr" / "f200k_source")
    assert crop[crop.index("--dst") + 1] == str(data / "images_official_hr" / "f200k")
    assert crop[crop.index("--zip") + 1] == str(urls_zip)


def test_stage_f200k_needs_the_zip_for_the_crop_boxes(tmp_path):
    s = _setup()
    txt = tmp_path / "image_urls.txt"
    txt.write_text("")
    with pytest.raises(SystemExit, match="detection"):
        s.stage_f200k(txt, tmp_path / "data", tmp_path / "ann")
