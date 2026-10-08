import importlib.util
import json

from conftest import ROOT


def _pack():
    spec = importlib.util.spec_from_file_location("pack_kaggle", ROOT / "scripts" / "pack_kaggle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_code_bundle_has_what_the_notebook_runs(tmp_path):
    out = tmp_path / "code"
    _pack().main(["code", "--out", str(out), "--owner", "someone"])
    for rel in ("src/procir_train/pipeline.py", "src/procir_train/train.py", "external/FashionMV/evaluate.py",
                "external/FashionMV/procir/model.py", "configs/kaggle_t4.yaml", "configs/procir_mt_align.yaml",
                "scripts/check_env.py", "requirements-train.txt"):
        assert (out / rel).is_file(), rel
    assert not list(out.rglob("__pycache__"))
    assert not (out / "external" / "FashionMV" / ".git").exists()
    meta = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/procir-code"


def test_images_metadata_is_written_in_place(tmp_path):
    images = tmp_path / "images"
    (images / "deepfashion").mkdir(parents=True)
    _pack().main(["images", "--src", str(images), "--owner", "someone"])
    meta = json.loads((images / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/procir-images"
