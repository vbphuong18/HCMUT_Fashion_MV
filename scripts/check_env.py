"""Fail fast if this machine cannot run ProCIR training (bf16 + linear-attention kernels).

Usage: python scripts/check_env.py [--allow-slow] [out.json]

--allow-slow (Kaggle T4 and similar): a GPU below sm80 (bf16 emulated) or missing fla /
causal_conv1d kernels (transformers falls back to its PyTorch reference) become warnings in the
JSON instead of errors. Training still works, many times slower. No CUDA at all is always an error.
"""
import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch


def _importable(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def main(out_path=None, allow_slow=False):
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available")
    major, minor = torch.cuda.get_device_capability(0)
    kernels = {"fla": _importable("fla"), "causal_conv1d": _importable("causal_conv1d")}
    problems = []
    if major < 8:
        problems.append(f"GPU sm{major}{minor} has no native bf16; rent Ampere or newer (sm80+)")
    missing = [k for k, ok in kernels.items() if not ok]
    if missing:
        problems.append(f"linear-attention kernels missing ({', '.join(missing)}); "
                        "pip install flash-linear-attention causal-conv1d --no-build-isolation")
    if problems and not allow_slow:
        sys.exit("; ".join(problems))
    import peft
    import transformers

    info = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": f"sm{major}{minor}",
        "kernels": kernels,
        "warnings": problems,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines(),
    }
    text = json.dumps(info, indent=2)
    print(text)
    for w in problems:
        print(f"WARNING (slow mode): {w}", flush=True)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--allow-slow", action="store_true")
    args = ap.parse_args()
    main(args.out, allow_slow=args.allow_slow)
