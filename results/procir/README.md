# ProCIR reproduction runs (Kaggle)

All runs use the authors' checkpoint (`yuandaxia/ProCIR`) and unmodified `evaluate.py`, on one NVIDIA Tesla T4.
`causal-conv1d` failed to build in every run and `flash-linear-attention` was not installed, so transformers fell back to the reference PyTorch implementation of Qwen3.5's linear-attention layers (correct but slow).

| Kernel (`src/kaggle/`) | Kaggle slug | Triplets | Batch | Gallery | Outcome | Wall time |
|---|---|---|---|---|---|---|
| `deepfashion` | `vtbphuongsdh242/procir-eval-deepfashion` | 5,188 (all) | 32 | 2,791 | **Complete**: R@1/5/10 = 33.58 / 74.42 / 85.25 | 10h54m (gallery 3h48m, queries 7h04m) |
| `f200k` | `vtbphuongsdh242/procir-eval-f200k` | 8,920 (all usable) | 32 | 6,908 | Stopped at the 12 h session limit with 76% of the gallery encoded (165/216 batches) | 12h00m |
| `f200k_v3` | `vtbphuongsdh242/procir-eval-f200k-v3-subsample` | 2,000 (seed 42) | 48 | 2,968 | Gallery finished (7h10m), then the run errored as query encoding started; no traceback in the log | 7h13m |
| `f200k_v4` | `vbichphuong/procir-eval-f200k-v4-safe-batch` | 1,500 (seed 42) | 8 | 2,367 | **Complete**: R@1/5/10 = 41.27 / 87.93 / 94.53 | 8h43m (gallery 5h19m, queries 3h20m) |

`deepfashion_v3` was never pushed to Kaggle.

Files:
- `deepfashion_full.json` and `f200k_sample1500.json` are the `eval_results.json` outputs of the two completed runs.
- `logs/*.log` are the raw Kaggle kernel logs (a JSON list of `{stream_name, time, data}` entries, where `time` is in seconds since the kernel started).

Kaggle CLI on Windows writes empty log files unless it is run with `PYTHONUTF8=1`.
