import torch

from procir_train.config import TrainConfig, save_config
from procir_train.encoder import load_encoder
from procir_train.train import cosine_lambda, save_checkpoint


def make_perturbed_run(run_dir):
    """Checkpoint with large LoRA and <emb_all> deltas: a skipped merge or embedding write cannot pass unnoticed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(output_dir=str(run_dir))
    save_config(cfg, run_dir / "config.yaml")
    enc, _ = load_encoder(cfg, torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in enc.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
        enc.emb_vector.add_(torch.randn(enc.emb_vector.shape, generator=g) * 0.05)
    params = [p for p in enc.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda(10))
    (run_dir / "ckpt").mkdir(exist_ok=True)
    save_checkpoint(run_dir / "ckpt" / "latest.pt", enc, optimizer, scheduler, step=0, cfg=cfg)
    return cfg
