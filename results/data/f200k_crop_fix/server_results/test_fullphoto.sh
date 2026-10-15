#!/usr/bin/env bash
cd /home/jovyan/HCMUT_Fashion_MV
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/bin/python
FULL=data/images_f200k_fullphoto
run() { name=$1; shift; echo "=== $name start $(date)"; "$@" >> runs/$name.log 2>&1; echo "=== $name exit $? $(date)"; }
test -d $FULL/f200k || $PY -m zipfile -e ~/procir_f200k_val_fullphoto.zip . || { echo "upload procir_f200k_val_fullphoto.zip first"; exit 1; }
echo "products: $(ls $FULL/f200k | wc -l)"
# the decisive test: authors' checkpoint, unmodified evaluate.py, every Fashion200K val image uncropped
run f200k_fullphoto_released $PY external/FashionMV/evaluate.py --model_path data/FashionMV_hf/model --image_root $FULL --data_dir data/FashionMV_hf/data/data --datasets f200k --output_dir runs/f200k_fullphoto_released/eval_upstream --batch_size 8
# supporting evidence from two unrelated models
run f200k_fullphoto_fashionclip $PY src/baselines/clip_val_eval.py --model patrickjohncyh/fashion-clip --image-root $FULL --datasets f200k --out runs/f200k_fullphoto_fashionclip/eval.json
run f200k_fullphoto_clip $PY src/baselines/clip_val_eval.py --model openai/clip-vit-base-patch32 --image-root $FULL --datasets f200k --out runs/f200k_fullphoto_clip/eval.json
