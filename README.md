# HCMUT FashionMV — Internship 1 (Thực tập 1)

Summary and reproduction of the paper **FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data** (Yuan et al., 2026, [arXiv:2604.10297](https://arxiv.org/abs/2604.10297)), with zero-shot CLIP / FashionCLIP baselines.

- Student: Võ Thị Bích Phượng — 2470570, HCMUT
- Advisor: Assoc. Prof. Dr. Võ Thị Ngọc Châu

## Repository layout

```
.
├── report/                     # Internship 1 report (LaTeX, book format)
│   ├── TT1_report.pdf          # compiled report
│   ├── main.tex
│   ├── chapter/ frontmatter/ appendix/ references/
│   └── figures/
├── src/
│   ├── data/                   # extract substitute images (HF parquet -> data/images/)
│   ├── baselines/              # CLIP / FashionCLIP late-fusion baselines
│   └── kaggle/<version>/       # Kaggle kernels running ProCIR's evaluate.py on GPU
├── results/
│   ├── procir/                 # ProCIR reproduction (Kaggle outputs)
│   └── baselines/              # baseline JSON results + logs/
└── docs/
    ├── proposal/               # research proposal (.tex + .pdf)
    └── VoThiBichPhuong2470570_literature_overview.pdf
```

Local-only folders (not in the repo): `data/` (annotations, checkpoint, images, raw zips), `FashionMV/` (clone of the official code), `personal/`, `archive/`.

## Main results (Recall@K, %)

| Model | DeepFashion R@5 / R@10 | Fashion200K* R@5 / R@10 |
|-------|------------------------|-------------------------|
| CLIP ViT-B/32, late fusion (α = 0.5) | 22.11 / 33.85 | 22.87 / 33.60 |
| FashionCLIP, late fusion (α = 0.25) | 55.17 / 67.27 | 51.00 / 62.80 |
| ProCIR (reproduced) | 74.42 / 85.25 | 87.93 / 94.53 |
| ProCIR (paper, full gallery) | 89.2 / 94.9 | 77.6 / 86.6 |

\* Subsample of 1,500 triplets (seed 42), gallery of 2,367 products versus 10,720 in the paper, so these numbers are not directly comparable with the paper's. See Chapter 6 of the report.

## Reproducing

1. Get the data into `data/` (not redistributed here):
   - Annotations: <https://huggingface.co/datasets/yuandaxia/FashionMV> → `data/FashionMV_hf/data/`
   - Substitute images: `Marqo/deepfashion-inshop`, `Marqo/fashion200k` parquet → `data/raw/<name>_parquet/`, then run `python src/data/extract_deepfashion.py` and `python src/data/extract_f200k.py`
2. Baselines (CPU): `python src/baselines/baseline_ext.py --model patrickjohncyh/fashion-clip`
3. ProCIR (Kaggle GPU): `kaggle kernels push -p src/kaggle/deepfashion` (the kernel downloads the code, checkpoint and images by itself)

Official code: <https://github.com/yuandaxia2001/FashionMV> · Checkpoint: <https://huggingface.co/yuandaxia/ProCIR>

## Training ProCIR (TT2)

The official repository has no training code yet; `src/procir_train/` reimplements it on top of the
pinned upstream eval package (`external/FashionMV`, commit 1c2f05c). Choices made where the paper is
silent are marked `Assumption N` in the code.

```bash
git submodule update --init
pip install -r requirements-train.txt
pip install flash-linear-attention causal-conv1d --no-build-isolation   # GPU box, CUDA devel image
python scripts/check_env.py runs/env_check.json                          # must pass before any run
python src/data/extract_f200k.py --triplets data/FashionMV_hf/data/data/train_triplets.jsonl
python src/data/extract_f200k.py --triplets data/FashionMV_hf/data/data/val_triplets.jsonl
python src/data/extract_deepfashion.py --triplets data/FashionMV_hf/data/data/train_triplets.jsonl   # low-res images (smoke test)
python src/data/extract_deepfashion.py                                                              # val triplets (script default)
# data/raw/img_highres must exist first: download and unzip img_highres.zip from the MMLab DeepFashion In-shop page
python src/data/regroup_deepfashion_highres.py --src data/raw/img_highres --symlink \
  --triplets data/FashionMV_hf/data/data/train_triplets.jsonl \
  --triplets data/FashionMV_hf/data/data/val_triplets.jsonl
ln -s ../images/f200k data/images_official_hr/f200k

pytest                       # CPU unit tests (no network, no GPU)
pytest -m hf                 # prompt/dataset tests (downloads the Qwen3.5 processor)
PYTHONPATH=src:external/FashionMV pytest -m gpu   # GPU tests

PYTHONPATH=src:external/FashionMV python -m procir_train.train --config configs/smoke.yaml   # 20 steps on data/images; needs a fresh output_dir
PYTHONPATH=src:external/FashionMV python -m procir_train.train --config configs/procir_mt_align.yaml
PYTHONPATH=src:external/FashionMV python -m procir_train.export \
  --config runs/procir_mt_align_seed42/config.yaml \
  --checkpoint runs/procir_mt_align_seed42/ckpt/latest.pt --out runs/procir_mt_align_seed42/export
python external/FashionMV/evaluate.py --model_path runs/procir_mt_align_seed42/export \
  --image_root data/images_official_hr --data_dir data/FashionMV_hf/data/data --datasets deepfashion f200k
```

Every run needs a fresh `output_dir` (pilots and smoke runs included): `train.py` refuses to start without `--resume` when `<output_dir>/train_log.jsonl` already exists, so delete `runs/smoke` or pass `--set output_dir=...`. `--resume` continues from `<output_dir>/ckpt/latest.pt` and appends to the log.

If DataLoader workers fail with "Too many open files" on Linux, run `ulimit -n 65535` (or lower `num_workers`).
`retrieval_eval --datasets` accepts `deepfashion f200k fashiongen_val`.

The eight ProCIR variants (paper Table 3, pretrained init) are `--set multi_turn=… align=… cot=…`.
Upstream `evaluate.py` always builds the two-turn query, so score `multi_turn=false` runs with the
same protocol through `PYTHONPATH=src:external/FashionMV python -m procir_train.retrieval_eval
--config RUN/config.yaml --checkpoint RUN/ckpt/latest.pt --data_dir data/FashionMV_hf/data/data
--image_root data/images_official_hr --datasets deepfashion f200k --out RUN/eval_val.json`.
Add `--protocol train` to score the same val gallery with the run's own `max_views` and pixel range
(3 views, 336² for `procir_mt_align.yaml`) instead of the upstream 5 views, 512²; the gap between the
two measures the train–eval mismatch. Reported numbers use the default `--protocol upstream`.

`lora=false` (full fine-tune) keeps the model in bf16 but gives every bf16 trainable an fp32 master
copy (`procir_train/precision.py`): plain AdamW on bf16 weights drops most updates at lr ≈ 1e-5.
Its checkpoints store the fp32 masters, so they are about twice the size of the bf16 weights.

### One command per run: `procir_train.pipeline`

`python -m procir_train.pipeline --config CONFIG [--set key=value ...]` runs env check → train → export
→ eval, each stage in its own process. Re-running the same command resumes the run. With
`stop_after_hours` (or `stop_after_steps`) training checkpoints and stops early; export and eval are then
skipped and `<output_dir>/STATUS` says how far it got. Options: `--stages train export eval`,
`--eval-protocols upstream train`, `--eval-datasets ...`, `--allow-slow` (GPU below sm80 or missing
kernels, e.g. Kaggle T4).

### Data on a new machine: `scripts/setup_data.py`

Builds `data/FashionMV_hf/data/data/*.jsonl` and `data/images_official_hr/{deepfashion,f200k,
fashiongen_train,fashiongen_val}/` (the `image_root` of the configs). Two inputs cannot be downloaded
by script and must be copied over first: the MMLab DeepFashion In-shop `img_highres` zip (the part
with `MEN/` and `WOMEN/`; password from the MMLab e-mail) and the Fashion200K zip from the
xthan/fashion-200k Google Drive (it holds `fashion-200k/image_urls.txt`). FashionGen comes from
Kaggle, so Kaggle API credentials (`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`) are needed.

```bash
export DEEPFASHION_PASSWORD='...'          # never commit it
python scripts/setup_data.py all \
  --deepfashion /path/img_highres.zip --f200k-urls /path/fashion-200k.zip
python scripts/setup_data.py verify        # coverage table + data/data_status.json; exit 1 if incomplete
```

Every stage resumes when re-run (`annotations`, `deepfashion`, `f200k`, `fashiongen`, `verify` can be
named individually). The Fashion200K stage downloads ~200k full images from the URLs (~10 images/s,
~5 h) into `f200k_source/`, then crops each to the garment box the official release used
(`src/data/crop_f200k.py`: the detection whose score `labels/*_detect_all.txt` records), so `f200k/`
matches the released, cropped Fashion200K images (checked against the Marqo/fashion200k mirror: 60/60
crops within 3 px). Train and val then come from one source. Disk: about 15 GB of
images, plus ~30 GB of FashionGen h5/zip in `data/raw/fashiongen/` that can be deleted afterwards.
With Docker: `docker compose -f docker/compose.yaml run --rm -e DEEPFASHION_PASSWORD procir python scripts/setup_data.py all ...`.

### Docker server (proposal-scale runs; GPU sm80+, 24 GB)

The image holds dependencies only (`docker/Dockerfile`, CUDA devel base, builds `causal-conv1d`); the
repository is bind-mounted at `/workspace`, so either build `data/` there with `scripts/setup_data.py`
or copy `data/FashionMV_hf/data/data/` and `data/images_official_hr/` from another machine.

```bash
git submodule update --init
docker/run.sh build           # once; the causal-conv1d compile takes a while
docker/run.sh check           # must report the GPU, fla and causal_conv1d
docker/run.sh test            # CPU + hf + gpu test suites
docker/run.sh smoke           # 20-step train + export on DeepFashion
docker/run.sh train configs/procir_mt_align.yaml --set datasets=[deepfashion] output_dir=runs/df_mt_align_s42
docker/run.sh logs <name printed by train>
```

### Kaggle (pilots on a T4)

`src/kaggle/train/procir_train.ipynb` runs the same pipeline with `configs/kaggle_t4.yaml`
(DeepFashion only, chunk 4). Setup and the multi-session resume procedure are in the notebook's first cell:
`scripts/pack_kaggle.py` builds the private `procir-code` dataset and writes the metadata for
`procir-images` (= `data/images_official_hr`); annotations and the base model download inside the notebook.
On a T4 bf16 is emulated and `causal-conv1d` does not build, so expect minutes per step and ~10 h per
dataset for eval; each session trains up to `SESSION_HOURS` and the next one resumes.

## Building the report

```bash
docker run --rm -v "$PWD/report:/work" -w /work texlive/texlive:latest-full \
  latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
cp report/build/main.pdf report/TT1_report.pdf
```
