#!/usr/bin/env bash
# Fashion200K image variants on the lab server: build them from the uncropped photos, check each with
# the authors' checkpoint, then train on them. DeepFashion and FashionGen are the same in every run.
#
#   photo     every image uncropped (what the paper uses; authors' checkpoint R@5 78.29)   -> main runs
#   mixed     images in the Fashion200K release cropped as released, the others uncropped (75.51)
#   crop      every image cropped as the release does (74.70)
#   top_crop  unlabelled images cropped to the top detection, our first data (67.71): the earlier
#             run runs/fast_mt_align_s42 was trained on it and is only re-scored here
#
# Needs in $HOME: procir_f200k_1.zip and procir_f200k_2.zip (the photos, from scripts/pack_data.py)
# and the Fashion200K release zip fashion-200k*.zip (its detection/ and labels/ give the crop boxes).
#
#   cd ~/HCMUT_Fashion_MV && nohup bash src/server/queue_f200k_variants.sh >> runs/variants_queue.log 2>&1 &
#
# Every step is skipped when its output exists and training resumes from its last checkpoint, so the
# same command continues the queue after an interruption.
set -u
REPO=${REPO:-$HOME/HCMUT_Fashion_MV}
PY=${PY:-/opt/conda/bin/python}
cd "$REPO" || exit 1
export PYTHONPATH=src:external/FashionMV TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/FashionMV_hf/data/data
HR=data/images_official_hr
VAR=data/images_variants
SETS="deepfashion f200k fashiongen_val"
COMMON="chunk_size=4 gradient_checkpointing=true eval_batch_size=8 train_fraction=0.25"
mkdir -p runs

run() {  # run NAME DONE_FILE COMMAND...
  local name=$1 done_file=$2
  shift 2
  if [ -e "$done_file" ]; then echo "--- $name: already done"; return 0; fi
  echo "=== $name start $(date)"
  "$@" >> "runs/$name.log" 2>&1
  local rc=$?
  echo "=== $name exit $rc $(date)"
  return $rc
}

if pgrep -f 'runs/[q]ueue.sh|procir_[t]rain|[e]valuate.py' > /dev/null; then
  echo "another training or evaluation is running: stop it or wait, then start this queue again"; exit 1
fi
$PY -c "import fla, peft, transformers, triton; from packaging.version import Version as V; assert V(triton.__version__) >= V('3.7.1')" \
  || { echo "packages missing (the notebook restarted?): Run All in the notebook, then start this queue again"; exit 1; }

# ---- 1. the photos replace the cropped Fashion200K images ----
if [ ! -e runs/f200k_photos.done ]; then
  for z in "$HOME/procir_f200k_1.zip" "$HOME/procir_f200k_2.zip"; do
    [ -f "$z" ] || { echo "upload $(basename "$z") to the home folder first"; exit 1; }
  done
  rm -rf "$HR/f200k" data/images_f200k_uncropped data/images_f200k_fullphoto  # crops and test copies
  free=$(df --output=avail -BG "$REPO" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 50 ] || { echo "only ${free}G free, 50G needed: remove ~/ckpt_download and uploaded test zips"; exit 1; }
  for z in "$HOME/procir_f200k_1.zip" "$HOME/procir_f200k_2.zip"; do
    $PY -m zipfile -e "$z" . || { echo "could not extract $z"; exit 1; }
  done
  n=$(ls "$HR/f200k" | wc -l)
  [ "$n" -eq 67680 ] || { echo "expected 67680 Fashion200K products, found $n"; exit 1; }
  touch runs/f200k_photos.done
  echo "photos in place: $n products ($(date))"
fi

# ---- 2. the cropped variants, built from the photos ----
REL=$(ls "$HOME"/fashion-200k*.zip 2> /dev/null | head -1)
build() {  # build NAME UNLABELLED_POLICY
  [ -e "runs/f200k_variant_$1.done" ] && return 0
  [ -n "$REL" ] || { echo "upload the Fashion200K release zip (fashion-200k*.zip) to the home folder first"; exit 1; }
  mkdir -p "$VAR/$1"
  for d in deepfashion fashiongen_train fashiongen_val; do ln -sfn "$REPO/$HR/$d" "$VAR/$1/$d"; done
  echo "building variant $1 ($(date))"
  $PY src/data/crop_f200k.py --src "$HR/f200k" --dst "$VAR/$1/f200k" --zip "$REL" --mirror none \
      --unlabelled "$2" --workers 4 >> "runs/f200k_variant_$1.log" 2>&1 || { echo "building $1 failed"; exit 1; }
  touch "runs/f200k_variant_$1.done"
  tail -n 1 "runs/f200k_variant_$1.log"
}
build mixed keep
build crop category

# ---- 3. each image set against the authors' checkpoint, before any training on it ----
check() {  # check NAME IMAGE_ROOT
  run "check_$1" "runs/check_$1/eval_upstream/eval_results.json" $PY external/FashionMV/evaluate.py \
      --model_path data/FashionMV_hf/model --image_root "$2" --data_dir $D --datasets f200k \
      --output_dir "runs/check_$1/eval_upstream" --batch_size 8
}
check photo "$HR"
$PY - << 'EOF' || { echo "the photos do not reproduce the paper on Fashion200K: not training"; exit 1; }
import json
r5 = json.load(open("runs/check_photo/eval_upstream/eval_results.json"))["Fashion200K_R@5"]
print(f"authors' checkpoint on the photos: R@5 = {r5} (paper 77.6, earlier test 78.29)")
raise SystemExit(0 if r5 >= 77.0 else 1)
EOF

# ---- 4. training; both configurations are scored at the last step ----
train() {  # train NAME IMAGE_ROOT "PROTOCOLS" [KEY=VALUE...]
  local name=$1 root=$2 protocols=$3
  shift 3
  run "$name" "runs/$name/queue_done" $PY -m procir_train.pipeline --config configs/procir_fast.yaml \
      --set "output_dir=runs/$name" "image_root=$root" $COMMON "$@" \
      --stages train export eval --eval-protocols $protocols --eval-batch-size 8 \
      --export-checkpoint latest --allow-slow && mkdir -p "runs/$name" && touch "runs/$name/queue_done"
}
on_photos() {  # a model trained on another variant, scored on the images the paper uses
  run "$1_on_photos" "runs/$1/eval_photos/eval_results.json" $PY external/FashionMV/evaluate.py \
      --model_path "runs/$1/export" --image_root "$HR" --data_dir $D --datasets f200k \
      --output_dir "runs/$1/eval_photos" --batch_size 8
}

train photo_mt_align_s42 "$HR" "upstream train"
train photo_single_s42 "$HR" "upstream train" multi_turn=false align=false
# MT+Align with the tool that scores the single-turn run (upstream protocol, source-excluded recalls)
run photo_mt_align_s42_ourtool runs/photo_mt_align_s42/eval_upstream_ourtool.json $PY -m procir_train.retrieval_eval \
    --config runs/photo_mt_align_s42/config.yaml --checkpoint runs/photo_mt_align_s42/ckpt/latest.pt \
    --data_dir $D --image_root "$HR" --datasets $SETS \
    --out runs/photo_mt_align_s42/eval_upstream_ourtool.json --protocol upstream --batch_size 8

check mixed "$VAR/mixed"
train mixed_mt_align_s42 "$VAR/mixed" "upstream"
on_photos mixed_mt_align_s42

check crop "$VAR/crop"
train crop_mt_align_s42 "$VAR/crop" "upstream"
on_photos crop_mt_align_s42

on_photos fast_mt_align_s42  # the top_crop run of 9 October (its export is the best dev step, 400)

echo "=== queue finished $(date)"
$PY - << 'EOF'
import json
from pathlib import Path


def r5(path, key):
    p = Path(path)
    if not p.exists():
        return "    -"
    d = json.loads(p.read_text())
    return f"{(d[key] if key in d else d['f200k']['R@5']):5.2f}"


print("Fashion200K R@5 (upstream evaluate.py)      own images   photos")
print(f"authors' checkpoint / photo                 {r5('runs/check_photo/eval_upstream/eval_results.json', 'Fashion200K_R@5')}")
print(f"authors' checkpoint / mixed                 {r5('runs/check_mixed/eval_upstream/eval_results.json', 'Fashion200K_R@5')}")
print(f"authors' checkpoint / crop                  {r5('runs/check_crop/eval_upstream/eval_results.json', 'Fashion200K_R@5')}")
for name in ("photo_mt_align_s42", "mixed_mt_align_s42", "crop_mt_align_s42", "fast_mt_align_s42"):
    own = r5(f"runs/{name}/eval_upstream/eval_results.json", "Fashion200K_R@5")
    photos = own if name.startswith("photo") else r5(f"runs/{name}/eval_photos/eval_results.json", "Fashion200K_R@5")
    print(f"{name:42s}  {own}        {photos}")
print(f"photo_single_s42 (our tool)                 {r5('runs/photo_single_s42/eval_upstream.json', '-')}")
EOF
