#!/usr/bin/env bash
# Re-score the models on the photos (the paper's images) and keep, for every query, the target's
# rank, the source's rank and the top-51 list (eval_metrics.ranking), so MRR, Median Rank, NDCG,
# leak rates, bootstrap tests and the ranking examples can be computed on CPU afterwards.
#
#   released          the authors' checkpoint, evaluate.py protocol      (~1.5 h)
#   mt_align          photo_mt_align_s42, our tool (multi-turn)           (~2-3.5 h)
#   single            photo_single_s42, our tool (single-turn)            (~2-3.5 h)
#   mt_align_upstream photo_mt_align_s42 export, evaluate.py protocol     (~1.5 h, cross-check only)
#
# Each step compares its Recall with the run's earlier evaluation (same images, same tool) and
# prints both. Every step is skipped when its output exists, so the same command resumes the queue.
# At the end it packs ~/rank_eval.tgz (dumps + logs, ~0.4 GB) for download.
#
#   cd ~/HCMUT_Fashion_MV && nohup bash src/server/queue_rank_eval.sh >> runs/rank_eval.log 2>&1 &
set -u
REPO=${REPO:-$HOME/HCMUT_Fashion_MV}
PY=${PY:-/opt/conda/bin/python}
cd "$REPO" || exit 1
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/FashionMV_hf/data/data
HR=data/images_official_hr
OUT=runs/rank_eval
SETS="deepfashion f200k fashiongen_val"
mkdir -p "$OUT"
exec 9> "$OUT/.lock"
! command -v flock > /dev/null || flock -n 9 || { echo "queue_rank_eval.sh is running already: this copy exits"; exit 1; }

if pgrep -f 'queue_f200k_[v]ariants|procir_[t]rain|[e]valuate.py|retrieval_[e]val|upstream_[d]ump' > /dev/null; then
  echo "another training or evaluation is using the GPU: wait for it to finish, then start this queue again"; exit 1
fi
$PY -c "import fla, peft, transformers, triton" \
  || { echo "packages missing (the notebook restarted?): in procir_server.ipynb run only the first four code cells" \
            "(parameters, helpers, repo, packages), not Run All, which can launch training; then start this queue again"; exit 1; }
n=$(ls "$HR/f200k" | wc -l)
[ "$n" -eq 67680 ] || { echo "expected the 67680 Fashion200K photo folders in $HR/f200k, found $n"; exit 1; }
[ -e data/images_official_hr/f200k_cropped ] && echo "note: $HR/f200k_cropped exists; scoring uses $HR/f200k (photos)"

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

same_recall() {  # same_recall NEW_DIR OLD_JSON: Recall from the dumps against the earlier evaluation
  $PY - "$1" "$2" << 'EOF'
import json, sys
from pathlib import Path
new_dir, old_path = Path(sys.argv[1]), Path(sys.argv[2])
if not old_path.exists():
    print(f"  (no earlier evaluation at {old_path} to compare with)"); sys.exit(0)
old = json.loads(old_path.read_text())
names = {"deepfashion": "DeepFashion", "f200k": "Fashion200K", "fashiongen_val": "FashionGen-val"}
worst = 0.0
for ds, name in names.items():
    meta = new_dir / f"{ds}.json"
    if not meta.exists():
        continue
    new = json.loads(meta.read_text())["recall"]
    for k in (1, 5, 10):
        before = old[ds][f"R@{k}"] if ds in old else old.get(f"{name}_R@{k}")
        if before is None:
            continue
        worst = max(worst, abs(new[f"R@{k}"] - before))
        print(f"  {ds:15s} R@{k:<2d} now {new[f'R@{k}']:6.2f}  before {before:6.2f}")
print(f"  largest difference {worst:.2f} points" + ("" if worst <= 0.1 else "  <-- more than 0.1: tell Claude"))
EOF
}

run released "$OUT/released/f200k.npz" $PY -m eval_metrics.upstream_dump \
    --model_path data/FashionMV_hf/model --image_root "$HR" --data_dir $D \
    --out_dir "$OUT/released" --batch_size 8 || exit 1
same_recall "$OUT/released" runs/check_photo/eval_upstream/eval_results.json

ours() {  # ours NAME RUN: our tool, upstream protocol, with dumps
  mkdir -p "$OUT/$1"
  run "$1" "$OUT/$1/eval.json" $PY -m procir_train.retrieval_eval \
      --config "runs/$2/config.yaml" --checkpoint "runs/$2/ckpt/latest.pt" \
      --data_dir $D --image_root "$HR" --datasets $SETS --protocol upstream --batch_size 8 \
      --dump_dir "$OUT/$1" --out "$OUT/$1/eval.json.tmp" || return 1
  [ ! -e "$OUT/$1/eval.json.tmp" ] || mv "$OUT/$1/eval.json.tmp" "$OUT/$1/eval.json"
}
ours mt_align photo_mt_align_s42 || exit 1
same_recall "$OUT/mt_align" runs/photo_mt_align_s42/eval_upstream_ourtool.json
ours single photo_single_s42 || exit 1
same_recall "$OUT/single" runs/photo_single_s42/eval_upstream.json

run mt_align_upstream "$OUT/mt_align_upstream/f200k.npz" $PY -m eval_metrics.upstream_dump \
    --model_path runs/photo_mt_align_s42/export --image_root "$HR" --data_dir $D \
    --out_dir "$OUT/mt_align_upstream" --batch_size 8 || exit 1
same_recall "$OUT/mt_align_upstream" runs/photo_mt_align_s42/eval_upstream/eval_results.json

git rev-parse HEAD > "$OUT/git_commit.txt"
tar -czf "$HOME/rank_eval.tgz" --exclude='.lock' --exclude='*.tmp' "$OUT" runs/rank_eval.log
echo "=== queue finished $(date): download ~/rank_eval.tgz ($(du -h "$HOME/rank_eval.tgz" | cut -f1))"
