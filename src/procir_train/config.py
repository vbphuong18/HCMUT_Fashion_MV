"""Training configuration for the ProCIR reproduction."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_DATASETS = ("deepfashion", "f200k", "fashiongen_train")


@dataclass
class TrainConfig:
    # paths (relative to the repository root)
    base_model: str = "Qwen/Qwen3.5-0.8B"
    data_dir: str = "data/FashionMV_hf/data/data"
    image_root: str = "data/images"
    output_dir: str = "runs/debug"
    # data
    datasets: list = field(default_factory=lambda: ["deepfashion", "f200k"])
    train_fraction: float = 0.25
    dev_product_fraction: float = 0.05
    max_train_triplets: int = 0  # 0 = no cap (smoke runs only)
    max_views: int = 3
    image_min_pixels: int = 128 * 128
    image_max_pixels: int = 336 * 336
    max_caption_tokens: int = 512
    # ProCIR variant switches (paper Table 3)
    multi_turn: bool = True
    align: bool = True
    cot: bool = False
    # loss
    tau: float = 0.07
    lambda_doc: float = 0.25
    lambda_src: float = 0.25
    # optimisation
    batch_size: int = 64
    chunk_size: int = 8
    epochs: int = 1
    max_steps: int = 0  # 0 = run the full schedule; >0 also shortens the LR and CoT schedules
    stop_after_steps: int = 0  # 0 = never; stops the loop early but keeps full-run schedules (pilots)
    stop_after_hours: float = 0.0  # 0 = never; checkpoint and stop after this much training time (Kaggle sessions)
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    # parameter-efficient tuning
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = True
    # loop
    num_workers: int = 4
    log_every: int = 10
    eval_every: int = 200  # 0 = no dev evaluation
    save_every: int = 200
    save_every_minutes: float = 0.0  # 0 = off; also checkpoint when this much time passed since the last save
    dev_eval_max_queries: int = 1000  # 0 = use all dev queries
    eval_batch_size: int = 16

    def __post_init__(self):
        unknown = set(self.datasets) - set(VALID_DATASETS)
        if unknown:
            raise ValueError(f"unknown datasets: {sorted(unknown)}")
        if self.lora_dropout != 0.0:
            raise ValueError("lora_dropout must be 0.0: GradCache re-runs the forward pass "
                             "and needs identical activations")
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        for name in ("batch_size", "chunk_size", "max_views", "epochs", "lora_r",
                     "log_every", "save_every", "eval_batch_size", "max_caption_tokens",
                     "image_min_pixels", "lora_alpha"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.image_min_pixels > self.image_max_pixels:
            raise ValueError("image_min_pixels must be <= image_max_pixels")
        for name in ("num_workers", "dev_eval_max_queries"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("lr", "tau", "lambda_doc", "lambda_src", "weight_decay", "grad_clip", "stop_after_hours",
                     "save_every_minutes"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("lr", "tau", "grad_clip"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("lambda_doc", "lambda_src", "weight_decay", "stop_after_hours", "save_every_minutes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.batch_size % self.chunk_size != 0:
            raise ValueError("batch_size must be a multiple of chunk_size")
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be in (0, 1]")
        if not 0.0 < self.dev_product_fraction < 1.0:
            raise ValueError("dev_product_fraction must be in (0, 1)")
        if min(self.eval_every, self.max_steps, self.stop_after_steps, self.max_train_triplets) < 0:
            raise ValueError("eval_every, max_steps, stop_after_steps and max_train_triplets must be >= 0")


def _coerce(name, value, default):
    if value is None:
        raise ValueError(f"{name} must not be empty/null")
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, got {value!r}")
        return value
    if isinstance(default, int):
        err = ValueError(f"{name} must be an integer, got {value!r}")
        if isinstance(value, bool):
            raise err
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise err from None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise err
    if isinstance(default, float):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {value!r}") from None
    if isinstance(default, list):
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ValueError(f"{name} must be a list or comma-separated string, got {value!r}")
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got {value!r}")
        return value
    return value


def load_config(path, overrides=None):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        data[key.strip()] = yaml.safe_load(raw)
    defaults = dataclasses.asdict(TrainConfig())
    unknown = set(data) - set(defaults)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**{k: _coerce(k, v, defaults[k]) for k, v in data.items()})


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False), encoding="utf-8")
