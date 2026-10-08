"""Summarise a training run (speed, memory, final loss and dev recall) into one JSON.

Usage: python scripts/summarize_run.py RUN_DIR [out.json]
"""
import json
import statistics
import sys
from pathlib import Path

WARMUP_STEPS = 5  # Triton autotuning and cache warm-up inflate the first steps


def summarize(run_dir):
    log_path = Path(run_dir) / "train_log.jsonl"
    if not log_path.exists():
        return {"run": str(run_dir), "error": "no train_log.jsonl (crashed before the first logged step, e.g. OOM)"}
    lines = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue  # process killed mid-write
    # --resume appends to the log and rewinds to the last checkpoint, so step numbers can repeat:
    # keep the last occurrence of each step.
    by_step = {}
    for l in lines:
        if "loss" in l:
            by_step[l["step"]] = l
    steps = [by_step[k] for k in sorted(by_step)]
    devs = [l for l in lines if "dev" in l]
    timed = [l["sec_per_step"] for l in steps if l["step"] > WARMUP_STEPS] or [l["sec_per_step"] for l in steps]
    alloc = [l["max_mem_gb"] for l in steps if "max_mem_gb" in l]
    reserved = [l["max_mem_reserved_gb"] for l in steps if "max_mem_reserved_gb" in l]
    return {
        "run": str(run_dir),
        "steps": steps[-1]["step"] if steps else 0,
        "sec_per_step_median": statistics.median(timed) if timed else None,
        "max_mem_allocated_gb": max(alloc) if alloc else None,
        "max_mem_reserved_gb": max(reserved) if reserved else None,
        "final_loss": steps[-1]["loss"] if steps else None,
        "final_dev": devs[-1]["dev"] if devs else None,
    }


if __name__ == "__main__":
    result = summarize(sys.argv[1])
    text = json.dumps(result, indent=2)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[2]).write_text(text, encoding="utf-8")
