import pytest

from procir_train.config import TrainConfig, load_config, save_config


def test_defaults_follow_proposal_scale():
    cfg = TrainConfig()
    assert cfg.base_model == "Qwen/Qwen3.5-0.8B"
    assert (cfg.batch_size, cfg.max_views, cfg.image_max_pixels) == (64, 3, 336 * 336)
    assert (cfg.tau, cfg.lambda_doc, cfg.lambda_src) == (0.07, 0.25, 0.25)
    assert cfg.lora and cfg.lora_dropout == 0.0


def test_yaml_roundtrip_and_float_coercion(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lr: 1e-4\nbatch_size: 32\nchunk_size: 8\ndatasets: [deepfashion]\n")
    cfg = load_config(p)
    assert cfg.lr == pytest.approx(1e-4) and isinstance(cfg.lr, float)
    assert cfg.datasets == ["deepfashion"]
    save_config(cfg, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml") == cfg


def test_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("cot: false\n")
    cfg = load_config(p, ["cot=true", "lr=2e-4", "max_steps=20", "output_dir=/tmp/x"])
    assert cfg.cot is True and cfg.lr == pytest.approx(2e-4)
    assert cfg.max_steps == 20 and cfg.output_dir == "/tmp/x"


def test_rejects_unknown_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("learning_rate: 1e-4\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


def test_rejects_lora_dropout_because_gradcache_replays_forward():
    with pytest.raises(ValueError, match="lora_dropout"):
        TrainConfig(lora_dropout=0.1)


def test_batch_must_be_multiple_of_chunk():
    with pytest.raises(ValueError, match="chunk_size"):
        TrainConfig(batch_size=64, chunk_size=7)


def test_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown datasets"):
        TrainConfig(datasets=["fashiongen_val"])


def test_rejects_negative_step_limits():
    with pytest.raises(ValueError, match="stop_after_steps"):
        TrainConfig(stop_after_steps=-1)


def test_roundtrip_default_config(tmp_path):
    save_config(TrainConfig(), tmp_path / "d.yaml")
    assert load_config(tmp_path / "d.yaml") == TrainConfig()


def test_utf8_yaml_with_non_ascii_comment(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_bytes("# cấu hình huấn luyện\nepochs: 2\n".encode("utf-8"))
    assert load_config(p).epochs == 2


def _load(tmp_path, overrides):
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 1\n")
    return load_config(p, overrides)


def test_list_override_scalar_and_comma(tmp_path):
    assert _load(tmp_path, ["datasets=deepfashion"]).datasets == ["deepfashion"]
    assert _load(tmp_path, ["datasets=deepfashion,f200k"]).datasets == ["deepfashion", "f200k"]
    assert _load(tmp_path, ["datasets=deepfashion, f200k"]).datasets == ["deepfashion", "f200k"]
    assert _load(tmp_path, ["datasets=[f200k]"]).datasets == ["f200k"]


def test_list_override_rejects_other_types(tmp_path):
    with pytest.raises(ValueError, match="datasets"):
        _load(tmp_path, ["datasets=5"])


def test_int_coercion(tmp_path):
    assert _load(tmp_path, ["max_steps=1e3"]).max_steps == 1000
    assert _load(tmp_path, ["max_steps=20.0"]).max_steps == 20
    for bad in ["max_steps=1.5", "max_steps=abc", "batch_size=true"]:
        with pytest.raises(ValueError, match=bad.split("=")[0]):
            _load(tmp_path, [bad])


def test_float_coercion_rejects_bool_and_text(tmp_path):
    for bad in ["lr=true", "lr=abc"]:
        with pytest.raises(ValueError, match="lr"):
            _load(tmp_path, [bad])


def test_none_and_wrong_str_rejected(tmp_path):
    for bad in ["output_dir=", "lr=", "datasets=", "cot=", "output_dir=123"]:
        with pytest.raises(ValueError, match=bad.split("=")[0]):
            _load(tmp_path, [bad])


def test_none_in_yaml_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lr:\n")
    with pytest.raises(ValueError, match="lr"):
        load_config(p)


@pytest.mark.parametrize("kw,name", [
    ({"batch_size": 0}, "batch_size"),
    ({"chunk_size": 0}, "chunk_size"),
    ({"tau": 0.0}, "tau"),
    ({"lr": 0.0}, "lr"),
    ({"max_views": 0}, "max_views"),
    ({"epochs": 0}, "epochs"),
    ({"lora_r": 0}, "lora_r"),
    ({"datasets": []}, "datasets"),
])
def test_rejects_non_positive_values(kw, name):
    with pytest.raises(ValueError, match=name):
        TrainConfig(**kw)


@pytest.mark.parametrize("kw,name", [
    ({"log_every": 0}, "log_every"),
    ({"save_every": 0}, "save_every"),
    ({"eval_batch_size": 0}, "eval_batch_size"),
    ({"grad_clip": 0.0}, "grad_clip"),
    ({"lambda_doc": -0.1}, "lambda_doc"),
    ({"lambda_src": -0.1}, "lambda_src"),
    ({"weight_decay": -1.0}, "weight_decay"),
    ({"lr": float("inf")}, "lr"),
    ({"tau": float("nan")}, "tau"),
    ({"lambda_doc": float("inf")}, "lambda_doc"),
    ({"lambda_src": float("nan")}, "lambda_src"),
    ({"weight_decay": float("inf")}, "weight_decay"),
    ({"grad_clip": float("inf")}, "grad_clip"),
])
def test_rejects_bad_loop_and_float_values(kw, name):
    with pytest.raises(ValueError, match=name):
        TrainConfig(**kw)


@pytest.mark.parametrize("kw,name", [
    ({"max_caption_tokens": 0}, "max_caption_tokens"),
    ({"num_workers": -1}, "num_workers"),
    ({"image_min_pixels": 0}, "image_min_pixels"),
    ({"image_min_pixels": 500000, "image_max_pixels": 400}, "image_min_pixels"),
    ({"dev_eval_max_queries": -1}, "dev_eval_max_queries"),
    ({"lora_alpha": 0}, "lora_alpha"),
])
def test_rejects_bad_token_pixel_worker_and_lora_values(kw, name):
    with pytest.raises(ValueError, match=name):
        TrainConfig(**kw)


def test_accepts_boundary_values():
    TrainConfig(num_workers=0, dev_eval_max_queries=0, image_min_pixels=1,
                image_max_pixels=1, max_caption_tokens=1, lora_alpha=1)
