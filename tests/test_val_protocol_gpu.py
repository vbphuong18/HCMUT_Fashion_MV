import json
import re
import subprocess
import sys

import pytest
import torch

from conftest import ROOT

pytestmark = pytest.mark.gpu
VAL = ROOT / "data" / "FashionMV_hf" / "data" / "data" / "val_triplets.jsonl"


def test_evaluate_val_matches_upstream_evaluate(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not VAL.exists() or not (ROOT / "data" / "images" / "deepfashion").exists():
        pytest.skip("FashionMV val annotations or DeepFashion images missing")
    from gpu_helpers import make_perturbed_run
    from procir_train.encoder import load_encoder
    from procir_train.export import export_merged
    from procir_train.retrieval_eval import evaluate_val
    from procir_train.train import load_checkpoint

    run = tmp_path / "run"
    cfg = make_perturbed_run(run)
    export_merged(run / "config.yaml", run / "ckpt" / "latest.pt", tmp_path / "export")
    sub = tmp_path / "val"
    sub.mkdir()
    (sub / "val_triplets.jsonl").write_text("\n".join(VAL.read_text(encoding="utf-8").splitlines()[:200]) + "\n",
                                            encoding="utf-8")

    proc_out = subprocess.run([sys.executable, str(ROOT / "external" / "FashionMV" / "evaluate.py"),
                    "--model_path", str(tmp_path / "export"), "--image_root", str(ROOT / "data" / "images"),
                    "--data_dir", str(sub), "--datasets", "deepfashion",
                    "--output_dir", str(tmp_path / "upstream"), "--batch_size", "8"],
                              capture_output=True, text=True, encoding="utf-8")
    if proc_out.returncode != 0:
        print(proc_out.stdout, proc_out.stderr)
    proc_out.check_returncode()
    theirs = json.loads((tmp_path / "upstream" / "eval_results.json").read_text(encoding="utf-8"))
    m = re.search(r"DeepFashion\s.*\(queries=(\d+), gallery=(\d+)\)", proc_out.stdout)
    assert m, proc_out.stdout

    enc, proc = load_encoder(cfg, torch.device("cuda"))
    load_checkpoint(run / "ckpt" / "latest.pt", enc)
    ours = evaluate_val(enc, proc, sub, ROOT / "data" / "images", ["deepfashion"],
                        multi_turn=True, batch_size=8)["deepfashion"]
    assert (ours["n_queries"], ours["n_gallery"]) == (int(m.group(1)), int(m.group(2)))
    for k in ("R@1", "R@5", "R@10"):
        assert abs(ours[k] - theirs[f"DeepFashion_{k}"]) <= 1.0, (k, ours[k], theirs[f"DeepFashion_{k}"])
