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

## Building the report

```bash
docker run --rm -v "$PWD/report:/work" -w /work texlive/texlive:latest-full \
  latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
cp report/build/main.pdf report/TT1_report.pdf
```
