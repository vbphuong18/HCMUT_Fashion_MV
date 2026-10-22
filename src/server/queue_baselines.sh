#!/usr/bin/env bash
# Zero-shot baseline on the lab server: Qwen3-VL-Embedding-2B (the paper's "Qwen3-V-2B", Table 2),
# used as released, no training on FashionMV. Same val triplets, galleries and images as ProCIR
# (data/images_official_hr, Fashion200K photos), source products kept in the gallery.
#
#   joint     all views of a product in one input (paper: Joint)        (~2-3.5 h)
#   perview   one input per view, scored as MeanPool and as MaxSim       (~2-3.5 h)
#
# Start it at any time: it downloads the model weights first (no GPU), then waits until no other
# training or evaluation is running, then scores. Every step is skipped when its output exists, so
# the same command resumes. At the end it packs ~/baselines.tgz for download.
#
#   cd ~/HCMUT_Fashion_MV && nohup bash src/server/queue_baselines.sh >> runs/baselines_queue.log 2>&1 &
set -u
REPO=${REPO:-$HOME/HCMUT_Fashion_MV}
PY=${PY:-/opt/conda/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-VL-Embedding-2B}
cd "$REPO" || exit 1
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/FashionMV_hf/data/data
HR=data/images_official_hr
OUT=runs/baselines/qwen3vl_emb_2b
mkdir -p "$OUT"
exec 9> "$OUT/.lock"
! command -v flock > /dev/null || flock -n 9 || { echo "queue_baselines.sh is running already: this copy exits"; exit 1; }

n=$(ls "$HR/f200k" | wc -l)
[ "$n" -eq 67680 ] || { echo "expected the 67680 Fashion200K photo folders in $HR/f200k, found $n"; exit 1; }
$PY -c "from transformers import Qwen3VLModel, Qwen3VLProcessor" \
  || { echo "this transformers has no Qwen3-VL (needs >= 4.57)"; exit 1; }

# ---- 1. weights, while the GPU may still be busy ----
echo "=== download $MODEL $(date)"
$PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))" \
  || { echo "could not download $MODEL"; exit 1; }

# ---- 2. wait for the GPU ----
busy='queue_rank_[e]val|queue_f200k_[v]ariants|procir_[t]rain|[e]valuate.py|retrieval_[e]val|upstream_[d]ump'
if pgrep -f "$busy" > /dev/null; then
  echo "=== waiting for the running jobs to finish $(date)"
  while pgrep -f "$busy" > /dev/null; do sleep 300; done
  echo "=== GPU free $(date)"
fi

# ---- 3. scoring ----
run() {  # run NAME DONE_FILE COMMAND...
  local name=$1 done_file=$2
  shift 2
  if [ -e "$done_file" ]; then echo "--- $name: already done"; return 0; fi
  echo "=== $name start $(date)"
  "$@" >> "$OUT/$name.log" 2>&1
  local rc=$?
  echo "=== $name exit $rc $(date)"
  return $rc
}
SETS="deepfashion f200k fashiongen_val"
run joint "$OUT/joint/eval.json" $PY -m baselines.qwen3vl_emb_eval --model "$MODEL" \
    --data_dir $D --image_root "$HR" --out_dir "$OUT" --datasets $SETS --modes joint --batch_size 8 || exit 1
run perview "$OUT/maxsim/eval.json" $PY -m baselines.qwen3vl_emb_eval --model "$MODEL" \
    --data_dir $D --image_root "$HR" --out_dir "$OUT" --datasets $SETS --modes perview --batch_size 8 || exit 1

$PY - "$OUT" << 'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print("R@5        DeepFashion  Fashion200K  FashionGen   (paper Table 2, Joint: 75.7 / 61.1 / 63.0)")
for s in ("joint", "meanpool", "maxsim"):
    p = out / s / "eval.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    print(f"{s:10s} " + "  ".join(f"{r[d]['R@5']:11.2f}" if d in r else "          -" for d in ("deepfashion", "f200k", "fashiongen_val")))
EOF

git rev-parse HEAD > "$OUT/git_commit.txt"
tar -czf "$HOME/baselines.tgz" --exclude='.lock' --exclude='cache' "$OUT" runs/baselines_queue.log
echo "=== queue finished $(date): download ~/baselines.tgz ($(du -h "$HOME/baselines.tgz" | cut -f1))"
