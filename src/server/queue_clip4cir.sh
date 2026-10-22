#!/usr/bin/env bash
# CLIP4Cir baseline on the lab server (paper Table 2, "CLIP4Cir", 0.25B): the authors' released
# FashionIQ RN50x4_fullft checkpoint (fine-tuned CLIP RN50x4 + Combiner), no training on FashionMV.
# Same val triplets, galleries and images as ProCIR; every view encoded alone, scored as MeanPool and
# MaxSim (~15-30 min).
#
# Start it at any time: it installs open_clip without touching torch, downloads the checkpoint
# (streams the 8.5 GB release, keeps only FashionIQ/RN50x4_fullft, ~0.95 GB), then waits until no
# other training or evaluation is running, then scores. The same command resumes. At the end it packs
# ~/clip4cir.tgz for download.
#
#   cd ~/HCMUT_Fashion_MV && nohup bash src/server/queue_clip4cir.sh >> runs/clip4cir_queue.log 2>&1 &
set -u
REPO=${REPO:-$HOME/HCMUT_Fashion_MV}
PY=${PY:-/opt/conda/bin/python}
cd "$REPO" || exit 1
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
D=data/FashionMV_hf/data/data
HR=data/images_official_hr
CK=data/clip4cir
CFG=FashionIQ/RN50x4_fullft
OUT=runs/baselines/clip4cir_rn50x4_fiq
mkdir -p "$OUT" "$CK"
exec 9> "$OUT/.lock"
! command -v flock > /dev/null || flock -n 9 || { echo "queue_clip4cir.sh is running already: this copy exits"; exit 1; }

n=$(ls "$HR/f200k" | wc -l)
[ "$n" -eq 67680 ] || { echo "expected the 67680 Fashion200K photo folders in $HR/f200k, found $n"; exit 1; }

# ---- 1. open_clip, without letting pip replace torch / torchvision ----
if ! $PY -c "import open_clip" 2> /dev/null; then
  echo "=== install open_clip $(date)"
  $PY -m pip install -q --no-deps open_clip_torch ftfy wcwidth timm
fi
$PY -c "import open_clip, torch, torchvision; print('open_clip', open_clip.__version__, 'torch', torch.__version__)" \
  || { echo "open_clip does not import (missing dependency above): tell Claude"; exit 1; }

# ---- 2. checkpoint ----
if [ ! -s "$CK/$CFG/fiq_comb_RN50x4_fullft.pt" ]; then
  echo "=== download CLIP4Cir weights $(date)"
  U=https://github.com/ABaldrati/CLIP4Cir/releases/download/weights/pretrained_models.tar.gz.part-
  (set -o pipefail; for i in 000 001 002 003 004; do curl -sSfL "$U$i" || exit 1; done \
     | tar -xz -C "$CK" --wildcards "*$CFG/*") \
    || [ -s "$CK/$CFG/fiq_comb_RN50x4_fullft.pt" ] || { echo "could not download the CLIP4Cir weights"; exit 1; }
fi
ls -la "$CK/$CFG"

# ---- 3. wait for the GPU ----
busy='queue_rank_[e]val|queue_f200k_[v]ariants|queue_[b]aselines|procir_[t]rain|[e]valuate.py|retrieval_[e]val|upstream_[d]ump|qwen3vl_emb_[e]val'
if pgrep -f "$busy" > /dev/null; then
  echo "=== waiting for the running jobs to finish $(date)"
  while pgrep -f "$busy" > /dev/null; do sleep 300; done
  echo "=== GPU free $(date)"
fi

# ---- 4. scoring ----
if [ -e "$OUT/maxsim/eval.json" ]; then
  echo "--- clip4cir: already done"
else
  echo "=== clip4cir start $(date)"
  $PY -m baselines.clip4cir_eval --data_dir $D --image_root "$HR" --out_dir "$OUT" \
      --clip_path "$CK/$CFG/fiq_clip_RN50x4_fullft.pt" --combiner_path "$CK/$CFG/fiq_comb_RN50x4_fullft.pt" \
      --batch_size 64 >> "$OUT/clip4cir.log" 2>&1
  rc=$?
  echo "=== clip4cir exit $rc $(date)"
  [ $rc -eq 0 ] || exit 1
fi

$PY - "$OUT" << 'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print("R@5        DeepFashion  Fashion200K  FashionGen   (paper Table 2, CLIP4Cir MeanPool: 28.0 / 13.3 / 16.2)")
for s in ("meanpool", "maxsim"):
    p = out / s / "eval.json"
    if p.exists():
        r = json.loads(p.read_text())
        print(f"{s:10s} " + "  ".join(f"{r[d]['R@5']:11.2f}" if d in r else "          -" for d in ("deepfashion", "f200k", "fashiongen_val")))
EOF

git rev-parse HEAD > "$OUT/git_commit.txt"
tar -czf "$HOME/clip4cir.tgz" --exclude='.lock' --exclude='cache' "$OUT" runs/clip4cir_queue.log
echo "=== queue finished $(date): download ~/clip4cir.tgz ($(du -h "$HOME/clip4cir.tgz" | cut -f1))"
