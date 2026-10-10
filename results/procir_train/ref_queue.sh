#!/usr/bin/env bash
cd /home/jovyan/HCMUT_Fashion_MV
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# one GPU job at a time: wait for the training queue
while pgrep -f 'runs/[q]ueue.sh' >/dev/null; do sleep 300; done
echo "=== ref_procir_released start $(date)"
mkdir -p runs/ref_procir_released
test -f data/FashionMV_hf/model/config.json || /opt/conda/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('yuandaxia/ProCIR', local_dir='data/FashionMV_hf/model')" && /opt/conda/bin/python external/FashionMV/evaluate.py --model_path data/FashionMV_hf/model --image_root data/images_official_hr --data_dir data/FashionMV_hf/data/data --datasets deepfashion f200k fashiongen_val --output_dir runs/ref_procir_released/eval_upstream --batch_size 8 >> runs/ref_procir_released.log 2>&1
echo "=== ref_procir_released exit $? $(date)"
echo "=== ref_clip start $(date)"
mkdir -p runs/ref_clip
/opt/conda/bin/python src/baselines/clip_val_eval.py --model openai/clip-vit-base-patch32 --datasets deepfashion f200k fashiongen_val --out runs/ref_clip/eval.json >> runs/ref_clip.log 2>&1
echo "=== ref_clip exit $? $(date)"
echo "=== ref_fashionclip start $(date)"
mkdir -p runs/ref_fashionclip
/opt/conda/bin/python src/baselines/clip_val_eval.py --model patrickjohncyh/fashion-clip --datasets deepfashion f200k fashiongen_val --out runs/ref_fashionclip/eval.json >> runs/ref_fashionclip.log 2>&1
echo "=== ref_fashionclip exit $? $(date)"
