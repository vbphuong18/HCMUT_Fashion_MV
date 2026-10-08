import sys

import pytest
import torch
import yaml

from conftest import ROOT
from procir_train import pipeline


def _config(tmp_path, **extra):
    cfg = {"output_dir": str(tmp_path / "run"), "data_dir": "DD", "image_root": "IR",
           "datasets": ["deepfashion"], **extra}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _ckpt(run_dir, step, total):
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "meta": {"total": total}}, run_dir / "ckpt" / "latest.pt")


def test_run_state(tmp_path):
    run = tmp_path / "run"
    assert pipeline.run_state(run) == ("fresh", 0, None)
    _ckpt(run, 40, 220)
    assert pipeline.run_state(run) == ("partial", 40, 220)
    _ckpt(run, 220, 220)
    assert pipeline.run_state(run) == ("done", 220, 220)


def test_find_data_dir(tmp_path):
    deep = tmp_path / "hf" / "data"
    deep.mkdir(parents=True)
    (deep / "train_triplets.jsonl").write_text("")
    assert pipeline.find_data_dir(tmp_path / "hf") == deep
    with pytest.raises(FileNotFoundError):
        pipeline.find_data_dir(tmp_path / "missing")


class Recorder:
    """Stands in for subprocess: records commands, fakes the files each stage would write."""

    def __init__(self, run_dir, train_reaches):
        self.cmds, self.run_dir, self.train_reaches = [], run_dir, train_reaches

    def __call__(self, cmd):
        self.cmds.append(cmd)
        if "procir_train.train" in cmd:
            step, total = self.train_reaches
            _ckpt(self.run_dir, step, total)
            (self.run_dir / "config.yaml").write_text("x")


def _stage_names(cmds):
    names = []
    for c in cmds:
        joined = " ".join(map(str, c))
        for key, name in (("check_env.py", "env"), ("procir_train.train", "train"),
                          ("procir_train.export", "export"), ("evaluate.py", "eval_upstream"),
                          ("procir_train.retrieval_eval", "eval_ours")):
            if key in joined:
                names.append(name)
    return names


def test_full_pipeline_multi_turn(tmp_path):
    rec = Recorder(tmp_path / "run", (220, 220))
    pipeline.main(["--config", str(_config(tmp_path)), "--eval-protocols", "upstream", "train"], run=rec)
    assert _stage_names(rec.cmds) == ["env", "train", "export", "eval_upstream", "eval_ours"]
    train_cmd = rec.cmds[1]
    assert train_cmd[:3] == [sys.executable, "-m", "procir_train.train"] and "--resume" not in train_cmd
    up = rec.cmds[3]
    assert up[1] == str(ROOT / "external" / "FashionMV" / "evaluate.py")
    assert up[up.index("--model_path") + 1] == str(tmp_path / "run" / "export")
    assert up[up.index("--datasets") + 1:up.index("--datasets") + 2] == ["deepfashion"]
    ours = rec.cmds[4]
    assert ours[ours.index("--protocol") + 1] == "train"


def test_partial_training_skips_export_and_eval(tmp_path):
    rec = Recorder(tmp_path / "run", (40, 220))
    code = pipeline.main(["--config", str(_config(tmp_path))], run=rec)
    assert _stage_names(rec.cmds) == ["env", "train"]
    assert code == 0
    assert "40/220" in (tmp_path / "run" / "STATUS").read_text(encoding="utf-8")


def test_resume_flag_when_checkpoint_exists(tmp_path):
    _ckpt(tmp_path / "run", 40, 220)
    rec = Recorder(tmp_path / "run", (220, 220))
    pipeline.main(["--config", str(_config(tmp_path)), "--stages", "train", "--skip-env-check"], run=rec)
    assert _stage_names(rec.cmds) == ["train"]
    assert "--resume" in rec.cmds[0]


def test_single_turn_never_uses_upstream_evaluate(tmp_path):
    _ckpt(tmp_path / "run", 220, 220)
    (tmp_path / "run" / "config.yaml").write_text("x")
    rec = Recorder(tmp_path / "run", (220, 220))
    cfg = _config(tmp_path, multi_turn=False)
    pipeline.main(["--config", str(cfg), "--stages", "eval", "--skip-env-check"], run=rec)
    assert _stage_names(rec.cmds) == ["eval_ours"]
    assert rec.cmds[0][rec.cmds[0].index("--protocol") + 1] == "upstream"


def test_overrides_and_eval_paths_follow_set(tmp_path):
    rec = Recorder(tmp_path / "other", (5, 5))
    pipeline.main(["--config", str(_config(tmp_path)), "--set", f"output_dir={tmp_path / 'other'}",
                   "image_root=/imgs", "--skip-env-check", "--allow-slow"], run=rec)
    assert f"output_dir={tmp_path / 'other'}" in rec.cmds[0]
    up = rec.cmds[2]
    assert up[up.index("--image_root") + 1] == "/imgs"


def test_allow_slow_is_forwarded_to_env_check(tmp_path):
    rec = Recorder(tmp_path / "run", (5, 5))
    pipeline.main(["--config", str(_config(tmp_path)), "--stages", "train", "--allow-slow"], run=rec)
    assert "--allow-slow" in rec.cmds[0]


def test_resume_flag_when_only_a_log_exists(tmp_path):
    """A session killed before the first checkpoint leaves train_log.jsonl; train.py needs --resume then."""
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "train_log.jsonl").write_text('{"step": 1}\n')
    rec = Recorder(tmp_path / "run", (5, 5))
    pipeline.main(["--config", str(_config(tmp_path)), "--stages", "train", "--skip-env-check"], run=rec)
    assert "--resume" in rec.cmds[0]


def test_export_checkpoint_best_uses_best_pt(tmp_path):
    _ckpt(tmp_path / "run", 220, 220)
    (tmp_path / "run" / "config.yaml").write_text("x")
    torch.save({"step": 100}, tmp_path / "run" / "ckpt" / "best.pt")
    rec = Recorder(tmp_path / "run", (220, 220))
    pipeline.main(["--config", str(_config(tmp_path)), "--stages", "export", "eval", "--skip-env-check",
                   "--eval-protocols", "upstream", "train", "--export-checkpoint", "best"], run=rec)
    export, ours = rec.cmds[0], rec.cmds[2]
    assert export[export.index("--checkpoint") + 1] == str(tmp_path / "run" / "ckpt" / "best.pt")
    assert ours[ours.index("--checkpoint") + 1] == str(tmp_path / "run" / "ckpt" / "best.pt")


def test_export_checkpoint_best_falls_back_to_latest(tmp_path):
    _ckpt(tmp_path / "run", 220, 220)
    (tmp_path / "run" / "config.yaml").write_text("x")
    rec = Recorder(tmp_path / "run", (220, 220))
    pipeline.main(["--config", str(_config(tmp_path)), "--stages", "export", "--skip-env-check",
                   "--export-checkpoint", "best"], run=rec)
    assert rec.cmds[0][rec.cmds[0].index("--checkpoint") + 1] == str(tmp_path / "run" / "ckpt" / "latest.pt")
