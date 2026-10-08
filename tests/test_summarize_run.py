import json
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))


def test_summarize_skips_warmup_and_reports_reserved_memory(tmp_path):
    from summarize_run import summarize

    secs = [30.0, 30.0, 30.0, 30.0, 30.0, 8.0, 9.0]
    lines = [{"step": i + 1, "loss": 4.0 - 0.1 * i, "l_cir": 1.0, "lr": 1e-4, "grad_norm": 1.0,
              "sec_per_step": s, "max_mem_gb": 18.0, "max_mem_reserved_gb": 20.0 + i}
             for i, s in enumerate(secs)]
    lines.append({"step": 7, "dev": {"deepfashion": {"R@5": 40.0}}})
    (tmp_path / "train_log.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    s = summarize(tmp_path)
    assert s["steps"] == 7 and s["sec_per_step_median"] == 8.5
    assert s["max_mem_allocated_gb"] == 18.0 and s["max_mem_reserved_gb"] == 26.0
    assert s["final_loss"] == 3.4 and s["final_dev"] == {"deepfashion": {"R@5": 40.0}}


def test_summarize_without_log_reports_error(tmp_path):
    from summarize_run import summarize

    s = summarize(tmp_path)
    assert s["run"] == str(tmp_path) and "error" in s


def _step(step, loss, sec, alloc=18.0, reserved=20.0):
    return {"step": step, "loss": loss, "sec_per_step": sec, "max_mem_gb": alloc, "max_mem_reserved_gb": reserved}


def test_summarize_keeps_last_occurrence_of_resumed_steps(tmp_path):
    from summarize_run import summarize

    first = [_step(i, 5.0 - 0.1 * i, 30.0 if i <= 5 else 100.0, reserved=30.0) for i in range(1, 9)]
    resumed = [_step(i, 1.0 + 0.1 * i, 8.0 + i - 6, reserved=21.0) for i in range(6, 11)]
    (tmp_path / "train_log.jsonl").write_text("\n".join(json.dumps(l) for l in first + resumed) + "\n")
    s = summarize(tmp_path)
    assert s["steps"] == 10
    assert s["final_loss"] == 2.0
    # steps 6..10 only (all after warmup) with resumed sec values 8,9,10,11,12 -> median 10
    assert s["sec_per_step_median"] == 10.0
    assert s["max_mem_reserved_gb"] == 30.0  # steps 1..5 survive from the first attempt


def test_summarize_skips_truncated_line(tmp_path):
    from summarize_run import summarize

    good = [_step(i, 4.0 - 0.1 * i, 8.0) for i in range(1, 4)]
    text = "\n".join(json.dumps(l) for l in good) + '\n{"step": 4, "loss": 3.'
    (tmp_path / "train_log.jsonl").write_text(text)
    s = summarize(tmp_path)
    assert s["steps"] == 3


def test_summarize_missing_memory_keys_report_none(tmp_path):
    from summarize_run import summarize

    (tmp_path / "train_log.jsonl").write_text(json.dumps({"step": 1, "loss": 1.0, "sec_per_step": 5.0}) + "\n")
    s = summarize(tmp_path)
    assert s["max_mem_allocated_gb"] is None and s["max_mem_reserved_gb"] is None
