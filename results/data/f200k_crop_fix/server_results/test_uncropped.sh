#!/usr/bin/env bash
cd /home/jovyan/HCMUT_Fashion_MV
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/bin/python
UNC=data/images_f200k_uncropped
while pgrep -f 'runs/[t]est_crop_fix.sh' >/dev/null; do sleep 60; done   # one GPU job at a time
test -d $UNC/f200k || $PY -m zipfile -e ~/procir_f200k_val_uncropped.zip . || { echo "upload procir_f200k_val_uncropped.zip first"; exit 1; }
echo "products: $(ls $UNC/f200k | wc -l); start $(date)"
$PY external/FashionMV/evaluate.py --model_path data/FashionMV_hf/model --image_root $UNC --data_dir data/FashionMV_hf/data/data --datasets f200k --output_dir runs/f200k_uncropped_released/eval_upstream --batch_size 8 >> runs/f200k_uncropped_released.log 2>&1
echo "exit $? $(date)"
