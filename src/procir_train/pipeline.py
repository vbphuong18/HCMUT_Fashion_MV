"""One entry point for every environment (Docker server, Kaggle): env check -> train -> export -> eval.

Each stage runs in its own process so GPU memory is released in between. Re-running the same command
continues an unfinished run (--resume is added when <output_dir>/ckpt/latest.pt exists); when training
stops early (stop_after_hours / stop_after_steps), export and eval are skipped and <output_dir>/STATUS
says how far it got.

Usage (from the repository root, PYTHONPATH=src:external/FashionMV):
  python -m procir_train.pipeline --config configs/procir_mt_align.yaml [--set key=value ...]
      [--stages train export eval] [--eval-protocols upstream train] [--eval-datasets deepfashion f200k]
      [--eval-batch-size 16] [--allow-slow] [--skip-env-check]

Eval: --eval-protocols upstream runs external/FashionMV/evaluate.py unmodified on the merged export
(multi_turn runs) or procir_train.retrieval_eval with the same protocol (single-turn runs);
'train' adds retrieval_eval --protocol train (the run's own views/pixels) as a diagnostic.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from .config import load_config

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("train", "export", "eval")


def run_state(output_dir):
    """('fresh' | 'partial' | 'done', step, total) from <output_dir>/ckpt/latest.pt."""
    latest = Path(output_dir) / "ckpt" / "latest.pt"
    if not latest.exists():
        return "fresh", 0, None
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    step, total = ck["step"], (ck.get("meta") or {}).get("total")
    return ("done" if total is not None and step >= total else "partial"), step, total


def find_data_dir(root):
    """The folder holding train_triplets.jsonl under `root` (HF snapshots nest it differently)."""
    root = Path(root)
    hits = sorted(root.rglob("train_triplets.jsonl")) if root.exists() else []
    if not hits:
        raise FileNotFoundError(f"no train_triplets.jsonl under {root}")
    return hits[0].parent


def _subprocess_run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = dict(os.environ)
    paths = [str(ROOT / "src"), str(ROOT / "external" / "FashionMV")]
    env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in [env.get("PYTHONPATH")] if p])
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main(argv=None, run=_subprocess_run):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--eval-protocols", nargs="+", choices=("upstream", "train"), default=["upstream"])
    ap.add_argument("--eval-datasets", nargs="+", default=None,
                    help="default: the run's datasets (fashiongen_train is scored as fashiongen_val)")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--export-checkpoint", choices=("latest", "best"), default="latest",
                    help="best = ckpt/best.pt (highest dev mean R@5), falling back to latest when absent")
    ap.add_argument("--allow-slow", action="store_true", help="accept GPUs below sm80 / missing kernels (Kaggle T4)")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)
    raw_run = run

    def run(cmd):
        raw_run([str(c) for c in cmd])

    cfg = load_config(args.config, args.set)
    config_path = Path(args.config).resolve()  # stages run with cwd = repository root
    out = Path(cfg.output_dir)
    if not out.is_absolute():  # relative paths in configs are relative to the repository root
        out = ROOT / out
    py = sys.executable

    if not args.skip_env_check:
        cmd = [py, str(ROOT / "scripts" / "check_env.py"), str(out / "env_check.json")]
        run(cmd + (["--allow-slow"] if args.allow_slow else []))

    if "train" in args.stages:
        cmd = [py, "-m", "procir_train.train", "--config", config_path]
        if args.set:
            cmd += ["--set", *args.set]
        if (out / "ckpt" / "latest.pt").exists() or (out / "train_log.jsonl").exists():
            cmd.append("--resume")  # also after a session killed before its first checkpoint
        run(cmd)

    state, step, total = run_state(out)
    if state != "done":
        msg = (f"training not finished: step {step}/{total}. Re-run the same command to resume; "
               "export and eval are skipped." if state == "partial" else "no checkpoint yet; nothing to export or evaluate.")
        out.mkdir(parents=True, exist_ok=True)
        (out / "STATUS").write_text(msg + "\n", encoding="utf-8")
        print(msg, flush=True)
        return 0
    (out / "STATUS").write_text(f"training finished: step {step}/{total}\n", encoding="utf-8")

    run_cfg, latest, export_dir = out / "config.yaml", out / "ckpt" / "latest.pt", out / "export"
    if args.export_checkpoint == "best":
        if (out / "ckpt" / "best.pt").exists():
            latest = out / "ckpt" / "best.pt"
        else:
            print("no ckpt/best.pt (dev eval off?): exporting latest.pt", flush=True)
    if "export" in args.stages:
        run([py, "-m", "procir_train.export", "--config", run_cfg, "--checkpoint", latest, "--out", export_dir])

    if "eval" in args.stages:
        datasets = args.eval_datasets or [("fashiongen_val" if d == "fashiongen_train" else d) for d in cfg.datasets]
        for protocol in args.eval_protocols:
            if protocol == "upstream" and cfg.multi_turn:
                run([py, ROOT / "external" / "FashionMV" / "evaluate.py", "--model_path", export_dir,
                     "--image_root", cfg.image_root, "--data_dir", cfg.data_dir, "--datasets", *datasets,
                     "--output_dir", out / "eval_upstream", "--batch_size", args.eval_batch_size])
            else:
                run([py, "-m", "procir_train.retrieval_eval", "--config", run_cfg, "--checkpoint", latest,
                     "--data_dir", cfg.data_dir, "--image_root", cfg.image_root, "--datasets", *datasets,
                     "--out", out / f"eval_{protocol}.json", "--protocol", protocol,
                     "--batch_size", args.eval_batch_size])
    return 0


if __name__ == "__main__":
    sys.exit(main())
