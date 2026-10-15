#!/usr/bin/env bash
cd /home/jovyan/HCMUT_Fashion_MV
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/bin/python
D=data/FashionMV_hf/data/data
HR=data/images_official_hr
run() { name=$1; shift; echo "=== $name start $(date)"; "$@" >> runs/$name.log 2>&1; echo "=== $name exit $? $(date)"; }

$PY -c "import fla, peft, transformers, triton; from packaging.version import Version as V; assert V(triton.__version__) >= V('3.7.1')" \
  || { echo "packages missing: Run All in the notebook first"; exit 1; }
test -f ~/procir_f200k_fix_val.zip || { echo "upload procir_f200k_fix_val.zip to the home folder first"; exit 1; }
test -f runs/f200k_fix_val.done || { $PY -m zipfile -e ~/procir_f200k_fix_val.zip . && touch runs/f200k_fix_val.done; }
echo "fixed val crops applied $(date)"

# the decisive test: authors' checkpoint, unmodified evaluate.py, Fashion200K val with the fixed crops
run f200kfix_released $PY external/FashionMV/evaluate.py --model_path data/FashionMV_hf/model --image_root $HR --data_dir $D --datasets f200k --output_dir runs/f200kfix_released/eval_upstream --batch_size 8
# supporting evidence: every other model on the same fixed images
run f200kfix_fashionclip $PY src/baselines/clip_val_eval.py --model patrickjohncyh/fashion-clip --datasets f200k --out runs/f200kfix_fashionclip/eval.json
run f200kfix_clip $PY src/baselines/clip_val_eval.py --model openai/clip-vit-base-patch32 --datasets f200k --out runs/f200kfix_clip/eval.json
run f200kfix_mt_align $PY external/FashionMV/evaluate.py --model_path runs/fast_mt_align_s42/export --image_root $HR --data_dir $D --datasets f200k --output_dir runs/f200kfix_mt_align/eval_upstream --batch_size 8
