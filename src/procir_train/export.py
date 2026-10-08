"""Merge LoRA and the trained <emb_all> vector into a plain checkpoint for external/FashionMV/evaluate.py.

Usage: PYTHONPATH=src:external/FashionMV python -m procir_train.export \
         --config RUN/config.yaml --checkpoint RUN/ckpt/latest.pt --out RUN/export
"""
import argparse

import torch

from .config import load_config
from .encoder import load_encoder
from .train import load_checkpoint


ARCH_KEYS = ("base_model", "lora", "lora_r", "lora_alpha")


def export_merged(config_path, checkpoint_path, out_dir):
    cfg = load_config(config_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("config", {})
    diff = {k: (saved.get(k), getattr(cfg, k)) for k in ARCH_KEYS if saved.get(k) != getattr(cfg, k)}
    if diff:
        raise ValueError("config.yaml does not match the checkpoint's saved config "
                         f"(key: (checkpoint, config.yaml)): {diff}")
    encoder, processor = load_encoder(cfg, torch.device("cpu"))
    load_checkpoint(checkpoint_path, encoder)
    vlm = encoder.vlm
    weight = vlm.get_input_embeddings().weight
    with torch.no_grad():
        weight[encoder.emb_token_id] = encoder.emb_vector.to(weight.dtype)
    encoder._hook.remove()
    model = encoder.peft_model.merge_and_unload() if encoder.peft_model is not None else vlm
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True  # training turned it off; restore the default for downstream users
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None:
        text_cfg.use_cache = True
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    export_merged(args.config, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
