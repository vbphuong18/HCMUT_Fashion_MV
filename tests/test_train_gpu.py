import json
import shutil

import pytest
import torch

from conftest import ROOT

pytestmark = pytest.mark.gpu


def _losses(run_dir):
    lines = [json.loads(l) for l in (run_dir / "train_log.jsonl").read_text().splitlines() if l.strip()]
    return {l["step"]: l["loss"] for l in lines if "loss" in l}, [l for l in lines if "dev" in l]


def test_smoke_train_then_deterministic_resume(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not (ROOT / "data/FashionMV_hf/data/data/train_triplets.jsonl").exists():
        pytest.skip("FashionMV training annotations are not downloaded")
    if not (ROOT / "data/images/deepfashion").exists():
        pytest.skip("DeepFashion training images are not extracted")
    from procir_train.train import main

    monkeypatch.chdir(ROOT)
    run1 = tmp_path / "run1"
    main(["--config", "configs/smoke.yaml", "--set", f"output_dir={run1}"])
    losses1, devs = _losses(run1)
    assert sorted(losses1) == list(range(1, 21))
    first, last = [losses1[s] for s in range(1, 6)], [losses1[s] for s in range(16, 21)]
    assert sum(last) / 5 < sum(first) / 5
    assert devs and "R@5" in next(iter(devs[-1]["dev"].values()))
    assert (run1 / "ckpt" / "latest.pt").exists()

    run2 = tmp_path / "run2"
    (run2 / "ckpt").mkdir(parents=True)
    shutil.copyfile(run1 / "ckpt" / "step_000010.pt", run2 / "ckpt" / "latest.pt")
    main(["--config", "configs/smoke.yaml", "--set", f"output_dir={run2}", "eval_every=0", "--resume"])
    losses2, _ = _losses(run2)
    assert sorted(losses2) == list(range(11, 21))
    for s in range(11, 21):
        assert losses2[s] == pytest.approx(losses1[s], rel=1e-2), s
