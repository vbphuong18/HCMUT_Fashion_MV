# HCMUT FashionMV — Internship 1 (Thực tập 1)

Summary and reproduction of the paper **FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data** (Yuan et al., 2026, [arXiv:2604.10297](https://arxiv.org/abs/2604.10297)), with zero-shot CLIP / FashionCLIP baselines.

- Student: Võ Thị Bích Phượng — 2470570, HCMUT
- Advisor: Assoc. Prof. Dr. Võ Thị Ngọc Châu

## Contents

| Path | Description |
|------|-------------|
| `TT1_Report/` | LaTeX source of the report (book format, 7 chapters); the compiled PDF is `TT1_Report/out/main.pdf` |
| `TT1_Report/scripts/` | Image extraction, Kaggle evaluation kernels, and baseline scripts |
| `kaggle_kernel_*/` | Kaggle kernel versions (`script.py` + `kernel-metadata.json`) used to run ProCIR on GPU |
| `out/` | Raw results: baseline JSON files and run logs |
| `decuong_MultiViewCIR_v3.tex` | Research proposal (LaTeX source) |
| `VoThiBichPhuong2470570_MultiViewCIR_v2.pdf` | Research proposal (PDF) |
| `VoThiBichPhuong2470570_literature_overview.pdf` | Literature review |

## Main results (Recall@K, %)

| Model | DeepFashion R@5 / R@10 | Fashion200K* R@5 / R@10 |
|-------|------------------------|-------------------------|
| CLIP ViT-B/32, late fusion (α = 0.5) | 22.11 / 33.85 | 22.87 / 33.60 |
| FashionCLIP, late fusion (α = 0.25) | 55.17 / 67.27 | 51.00 / 62.80 |
| ProCIR (reproduced) | 74.42 / 85.25 | 87.93 / 94.53 |
| ProCIR (paper, full gallery) | 89.2 / 94.9 | 77.6 / 86.6 |

\* Subsample of 1,500 triplets (seed 42), gallery of 2,367 products versus 10,720 in the paper, so these numbers are not directly comparable with the paper's. See Chapter 6 of the report.

## Not included

Datasets, images and checkpoints are not redistributed. To get them:

- Evaluation code: <https://github.com/yuandaxia2001/FashionMV>
- Annotations: <https://huggingface.co/datasets/yuandaxia/FashionMV>
- Checkpoint: <https://huggingface.co/yuandaxia/ProCIR>
- Substitute images: `Marqo/deepfashion-inshop` and `Marqo/fashion200k` on HuggingFace

## Build the report

```bash
docker run --rm -v "$PWD/TT1_Report:/work" -w /work texlive/texlive:latest-full \
  latexmk -pdf -interaction=nonstopmode -outdir=out main.tex
```
