#!/usr/bin/env bash
cd /home/jovyan/HCMUT_Fashion_MV
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== fast_mt_align_s42 start $(date)"
/opt/conda/bin/python -m procir_train.pipeline --config configs/procir_fast.yaml --set output_dir=runs/fast_mt_align_s42 chunk_size=4 gradient_checkpointing=true eval_batch_size=8 train_fraction=0.25 --stages train export eval --eval-protocols upstream train --eval-batch-size 8 --export-checkpoint best --allow-slow >> runs/fast_mt_align_s42.log 2>&1
echo "=== fast_mt_align_s42 exit $? $(date)"
echo "=== fast_single_s42 start $(date)"
/opt/conda/bin/python -m procir_train.pipeline --config configs/procir_fast.yaml --set output_dir=runs/fast_single_s42 chunk_size=4 gradient_checkpointing=true eval_batch_size=8 train_fraction=0.25 multi_turn=false align=false --stages train export eval --eval-protocols upstream train --eval-batch-size 8 --export-checkpoint best --allow-slow >> runs/fast_single_s42.log 2>&1
echo "=== fast_single_s42 exit $? $(date)"
