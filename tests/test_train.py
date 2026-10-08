import pytest
import torch
import torch.nn as nn

from procir_train.config import TrainConfig
from procir_train.train import check_resume_config, cosine_lambda, load_checkpoint, save_checkpoint


def test_cosine_without_warmup():
    f = cosine_lambda(100)
    assert f(0) == pytest.approx(1.0)
    assert f(50) == pytest.approx(0.5)
    assert f(100) == pytest.approx(0.0)


def test_checkpoint_roundtrip_saves_only_trainable(tmp_path):
    m = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    m[0].requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(10))
    m(torch.ones(1, 3)).sum().backward()
    opt.step()
    sched.step()
    path = tmp_path / "ck.pt"
    save_checkpoint(path, m, opt, sched, step=7, cfg=TrainConfig())
    saved = torch.load(path, weights_only=False)
    assert set(saved["trainable"]) == {"1.weight", "1.bias"}
    expected = m[1].weight.detach().clone()
    with torch.no_grad():
        m[1].weight.zero_()
    opt2 = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, cosine_lambda(10))
    assert load_checkpoint(path, m, opt2, sched2) == 7
    assert torch.equal(m[1].weight, expected)
    assert sched2.last_epoch == 1


def test_checkpoint_keeps_fp32_master_of_bf16_params(tmp_path):
    from procir_train.precision import MasterWeights

    m = nn.Sequential(nn.Linear(3, 3)).bfloat16()
    mw = MasterWeights(m.named_parameters())
    opt = torch.optim.AdamW(mw.masters, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(10))
    with torch.no_grad():
        mw.masters[0].add_(1e-6)  # below bf16 resolution: only the master sees it
    path = tmp_path / "ck.pt"
    save_checkpoint(path, m, opt, sched, step=3, cfg=TrainConfig(), master=mw)
    saved = torch.load(path, weights_only=False)["trainable"]
    assert saved["0.weight"].dtype == torch.float32
    assert torch.equal(saved["0.weight"], mw.masters[0].detach())

    m2 = nn.Sequential(nn.Linear(3, 3)).bfloat16()
    mw2 = MasterWeights(m2.named_parameters())
    opt2 = torch.optim.AdamW(mw2.masters, lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, cosine_lambda(10))
    assert load_checkpoint(path, m2, opt2, sched2, master=mw2) == 3
    assert torch.equal(mw2.masters[0], mw.masters[0])
    assert torch.equal(m2[0].weight, m[0].weight)
    # loaders without a master (export, val eval) just get the value cast to the model dtype
    m3 = nn.Sequential(nn.Linear(3, 3)).bfloat16()
    load_checkpoint(path, m3)
    assert torch.equal(m3[0].weight, m[0].weight)


def test_resume_refuses_changed_schedule_fields():
    saved = dict(vars(TrainConfig()))
    check_resume_config(saved, TrainConfig(eval_every=0, chunk_size=4))  # harmless changes
    with pytest.raises(ValueError, match="batch_size"):
        check_resume_config(saved, TrainConfig(batch_size=32))


def test_resume_refuses_missing_saved_key():
    saved = dict(vars(TrainConfig()))
    del saved["lr"]
    with pytest.raises(ValueError, match="lr"):
        check_resume_config(saved, TrainConfig())


def test_resume_data_check():
    from procir_train.train import check_resume_data

    check_resume_data({"total": 5, "n_train": 10}, 5, 10)
    check_resume_data({}, 5, 10)  # old checkpoint without meta: skipped
    check_resume_data(None, 5, 10)
    with pytest.raises(ValueError, match="n_train"):
        check_resume_data({"total": 5, "n_train": 10}, 5, 11)
    with pytest.raises(ValueError, match="total"):
        check_resume_data({"total": 5, "n_train": 10}, 6, 10)


def _saved(tmp_path, model):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(10))
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, opt, sched, step=1, cfg=TrainConfig(), meta={"total": 3, "n_train": 9})
    assert torch.load(path, weights_only=False)["meta"] == {"total": 3, "n_train": 9}
    return path


def test_load_checkpoint_rejects_extra_checkpoint_param(tmp_path):
    path = _saved(tmp_path, nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3)))
    small = nn.Sequential(nn.Linear(3, 3))
    with pytest.raises(KeyError, match="not in model"):
        load_checkpoint(path, small)


def test_load_checkpoint_rejects_missing_trainable_param(tmp_path):
    frozen = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    frozen[0].requires_grad_(False)
    path = _saved(tmp_path, frozen)
    all_trainable = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    with pytest.raises(ValueError, match="0.weight"):
        load_checkpoint(path, all_trainable)


def test_load_checkpoint_rejects_shape_mismatch(tmp_path):
    path = _saved(tmp_path, nn.Sequential(nn.Linear(3, 3)))
    with pytest.raises(ValueError, match="shape mismatch for 0.weight"):
        load_checkpoint(path, nn.Sequential(nn.Linear(3, 4)))


def test_check_captions_reports_missing():
    from procir_train.train import check_captions

    recs = [{"source_key": "a", "target_key": "b"}, {"source_key": "c", "target_key": "d"}]
    check_captions(recs, {"a": 1, "b": 1, "c": 1, "d": 1})
    with pytest.raises(ValueError, match=r"2 of 4.*\['b', 'd'\]"):
        check_captions(recs, {"a": 1, "c": 1})


def test_should_stop_by_steps_and_wall_clock():
    from procir_train.train import should_stop

    cfg = TrainConfig(stop_after_steps=3, stop_after_hours=0.5)
    assert not should_stop(cfg, steps_done=2, hours=0.1)
    assert should_stop(cfg, steps_done=3, hours=0.1)
    assert should_stop(cfg, steps_done=1, hours=0.5)
    never = TrainConfig()
    assert not should_stop(never, steps_done=10**6, hours=10**3)


def test_stop_after_hours_is_validated_and_not_a_resume_key():
    from procir_train.train import RESUME_KEYS

    with pytest.raises(ValueError, match="stop_after_hours"):
        TrainConfig(stop_after_hours=-1.0)
    assert "stop_after_hours" not in RESUME_KEYS  # each Kaggle session sets its own budget
