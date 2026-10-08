"""Train ProCIR at single-GPU scale.

Usage (from the repository root):
  PYTHONPATH=src:external/FashionMV python -m procir_train.train --config configs/procir_mt_align.yaml
  ... --set cot=true seed=1 output_dir=runs/x      # override config keys
  ... --resume                                      # continue from <output_dir>/ckpt/latest.pt
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .data import build_records, load_captions
from .dataset import StepBatchSampler, TripletTrainDataset, collate, split_chunks
from .encoder import load_encoder
from .gradcache import gradcache_step
from .losses import procir_loss
from .precision import MasterWeights
from .retrieval_eval import evaluate_records


def cosine_lambda(total_steps):
    def f(step):
        return 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))
    return f


# Fields that change the data order, the schedules or the model; resuming with any of them changed
# would silently train a different run. chunk_size may change (GradCache is exact), e.g. after an OOM.
RESUME_KEYS = ("base_model", "data_dir", "image_root", "datasets", "train_fraction", "dev_product_fraction",
               "max_train_triplets", "max_views", "image_min_pixels", "image_max_pixels", "max_caption_tokens",
               "multi_turn", "align", "cot", "tau", "lambda_doc", "lambda_src", "batch_size", "epochs",
               "max_steps", "lr", "weight_decay", "grad_clip", "seed", "lora", "lora_r", "lora_alpha")


_MISSING = object()


def dev_score(res):
    """Model-selection score on the dev split: mean R@5 over datasets (the paper's headline metric)."""
    return sum(r["R@5"] for r in res.values()) / len(res)


def best_from_meta(meta):
    meta = meta or {}
    return meta.get("best_dev", float("-inf")), meta.get("best_step")


def should_save(cfg, step, total, stopping, minutes_since_save, new_best):
    timer = bool(cfg.save_every_minutes) and minutes_since_save >= cfg.save_every_minutes
    return step % cfg.save_every == 0 or step == total or stopping or new_best or timer


def should_stop(cfg, steps_done, hours):
    """Early stop of this process (pilot step count or wall-clock budget); the schedules stay those of the full run."""
    if cfg.stop_after_steps and steps_done >= cfg.stop_after_steps:
        return True
    return bool(cfg.stop_after_hours) and hours >= cfg.stop_after_hours


def check_resume_config(saved, cfg):
    diffs = {k: (saved.get(k, "<missing>"), getattr(cfg, k)) for k in RESUME_KEYS
             if saved.get(k, _MISSING) != getattr(cfg, k)}
    if diffs:
        raise ValueError(f"config differs from the checkpoint (saved, current); refusing to resume: {diffs}")


def check_resume_data(meta, total, n_train):
    """Refuse to resume when the data or the schedule length changed (skipped for keys the checkpoint lacks)."""
    meta = meta or {}
    diffs = {k: (meta[k], v) for k, v in (("total", total), ("n_train", n_train))
             if k in meta and meta[k] != v}
    if diffs:
        raise ValueError(f"training data/schedule differs from the checkpoint (saved, current); "
                         f"refusing to resume: {diffs}")


def check_captions(records, captions):
    """Fail fast: a missing caption would otherwise raise KeyError inside a DataLoader worker mid-run."""
    keys = {r[k] for r in records for k in ("source_key", "target_key")}
    missing = sorted(keys - set(captions))
    if missing:
        raise ValueError(f"{len(missing)} of {len(keys)} product keys have no caption "
                         f"(needed because align or cot is on); examples: {missing[:5]}")


def save_checkpoint(path, encoder, optimizer, scheduler, step, cfg, meta=None, master=None):
    """With `master`, trainable values are its fp32 copies, so bf16 parameters resume without rounding."""
    path = Path(path)
    if master is not None:
        trainable = master.state_dict()
    else:
        trainable = {n: p.detach().cpu().clone() for n, p in encoder.named_parameters() if p.requires_grad}
    state = {
        "step": step,
        "trainable": trainable,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": vars(cfg),
        "meta": dict(meta or {}),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, encoder, optimizer=None, scheduler=None, master=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(encoder.named_parameters())
    missing = sorted(set(ck["trainable"]) - set(params))
    if missing:
        raise KeyError(f"checkpoint parameters not in model: {missing[:5]}")
    absent = sorted(n for n, p in params.items() if p.requires_grad and n not in ck["trainable"])
    if absent:
        raise ValueError(f"trainable model parameters missing from checkpoint: {absent[:5]}")
    for name, tensor in ck["trainable"].items():
        if tuple(params[name].shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
                             f"vs model {tuple(params[name].shape)}")
    with torch.no_grad():
        for name, tensor in ck["trainable"].items():
            params[name].copy_(tensor)
    if master is not None:
        master.load_state_dict(ck["trainable"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ck["scheduler"])
    return ck["step"]


def write_env(path):
    import peft
    import transformers

    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines(),
    }
    Path(path).write_text(json.dumps(info, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config, args.set)
    out = Path(cfg.output_dir)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    latest = out / "ckpt" / "latest.pt"
    log_path = out / "train_log.jsonl"
    saved_meta = None
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        check_resume_config(saved["config"], cfg)
        saved_meta = saved.get("meta")
        del saved
    elif args.resume:
        print(f"no checkpoint at {latest}, starting from scratch", flush=True)
    elif log_path.exists() and log_path.stat().st_size > 0:
        raise ValueError(f"{log_path} already exists: pass --resume to continue that run "
                         f"or use a fresh output_dir")
    save_config(cfg, out / "config.yaml")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    write_env(out / "env.json")

    train, dev, stats = build_records(cfg)
    if cfg.max_train_triplets:
        train = train[: cfg.max_train_triplets]
    stats["train_used"] = len(train)
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    needed = {r[k] for r in train + dev for k in ("source_key", "target_key")}
    captions = {k: v for k, v in load_captions(Path(cfg.data_dir) / "train_captions.jsonl").items()
                if k in needed}

    if cfg.align or cfg.cot:
        check_captions(train, captions)

    total = (len(train) // cfg.batch_size) * cfg.epochs
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    if total <= 0:
        raise ValueError(f"no training steps: {len(train)} records for batch size {cfg.batch_size}")

    if args.resume and latest.exists():
        check_resume_data(saved_meta, total, len(train))

    encoder, processor = load_encoder(cfg, device)
    encoder.train()

    master = MasterWeights(encoder.named_parameters())  # fp32 copies of bf16 trainables (full fine-tune)
    params = master.masters
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = LambdaLR(optimizer, cosine_lambda(total))
    start = 0
    if args.resume and latest.exists():
        start = load_checkpoint(latest, encoder, optimizer, scheduler, master)
        print(f"resumed from step {start}", flush=True)
    best_dev, best_step = best_from_meta(saved_meta)
    best = out / "ckpt" / "best.pt"

    sampler = StepBatchSampler(len(train), cfg.batch_size, total, cfg.seed, start_step=start)
    dataset = TripletTrainDataset(train, captions, processor, cfg, total_steps=total)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.num_workers)

    step = start
    t_start = t_saved = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        t_last = time.time()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = gradcache_step(
                lambda chunk: encoder.encode(chunk, cfg),
                split_chunks(batch, cfg.chunk_size),
                lambda reps: procir_loss(reps, batch["source_key"], batch["target_key"], cfg),
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"non-finite grad_norm ({float(grad_norm)}) at step {step + 1}")
            lr_used = scheduler.get_last_lr()[0]  # the LR this step actually used
            optimizer.step()
            master.copy_to_model()
            scheduler.step()
            step += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_sec = time.time() - t_last  # this step only (data wait + GradCache), not a running mean

            if step % cfg.log_every == 0 or step == total:
                rec = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in parts.items()},
                       "lr": lr_used, "grad_norm": float(grad_norm),
                       "sec_per_step": step_sec,
                       "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                       "max_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            new_best = False
            if dev and cfg.eval_every and (step % cfg.eval_every == 0 or step == total):
                res = evaluate_records(encoder, processor, dev, cfg, max_queries=cfg.dev_eval_max_queries,
                                       batch_size=cfg.eval_batch_size)
                score = dev_score(res)
                new_best = score > best_dev
                if new_best:
                    best_dev, best_step = score, step
                rec = {"step": step, "dev": res, "dev_score": score, "best_step": best_step}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            stopping = step < total and should_stop(cfg, step - start, (time.time() - t_start) / 3600)
            if should_save(cfg, step, total, stopping, (time.time() - t_saved) / 60, new_best):
                ck = out / "ckpt" / f"step_{step:06d}.pt"
                save_checkpoint(ck, encoder, optimizer, scheduler, step, cfg, master=master,
                                meta={"total": total, "n_train": len(train), "best_dev": best_dev,
                                      "best_step": best_step})
                tmp_latest = latest.with_suffix(".tmp")
                shutil.copyfile(ck, tmp_latest)
                os.replace(tmp_latest, latest)
                if new_best:  # model selection on the dev split (the proposal: dev picks the checkpoint)
                    tmp_best = best.with_suffix(".tmp")
                    shutil.copyfile(ck, tmp_best)
                    os.replace(tmp_best, best)
                t_saved = time.time()
            if stopping:
                print(f"stopping early at step {step}/{total}; continue with --resume", flush=True)
                break
            t_last = time.time()  # eval/save time is not charged to the next step


if __name__ == "__main__":
    main()
