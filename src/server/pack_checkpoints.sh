#!/usr/bin/env bash
# Hand the checkpoints of the variant runs over for download, one run at a time: the home volume
# cannot hold all of them packed at once. For each run it waits until ~/ckpt_download holds no
# .tar.part files (the previous run was downloaded and its parts deleted), packs runs/<run>/ckpt and
# runs/<run>/export into 2 GB parts with a <run>.SHA256SUMS file, writes READY.txt, and waits again.
#
#   cd ~/HCMUT_Fashion_MV && nohup bash src/server/pack_checkpoints.sh >> runs/pack_checkpoints.log 2>&1 &
#
# For each run: download every file in ~/ckpt_download, then delete its .tar.part files; the next
# run is packed within a minute. A run is packed once (runs/.packed/<run>); delete that marker to
# pack it again. With WAIT=0 it packs the runs one after another without waiting, as long as the
# free space lasts, and stops with a message when it does not.
set -u
REPO=${REPO:-$HOME/HCMUT_Fashion_MV}
OUT=${OUT:-$HOME/ckpt_download}
POLL=${POLL:-60}
WAIT=${WAIT:-1}  # WAIT=0: pack every run without waiting for downloads, while the free space lasts
RUNS=${RUNS:-"photo_mt_align_s42 photo_single_s42 mixed_mt_align_s42 crop_mt_align_s42 mixed_other_mt_align_s42"}
cd "$REPO" || exit 1
mkdir -p "$OUT" runs/.packed
exec 9> runs/.packed/.lock  # one copy at a time: two would pack the same run into the same files
! command -v flock > /dev/null || flock -n 9 || { echo "pack_checkpoints.sh is running already ($(date)): this copy exits"; exit 1; }

for run in $RUNS; do
  if [ -e "runs/.packed/$run" ]; then echo "--- $run: packed already"; continue; fi
  [ -d "runs/$run/ckpt" ] && [ -d "runs/$run/export" ] || { echo "runs/$run has no ckpt/ or export/"; exit 1; }
  if [ "$WAIT" = 1 ]; then
    while ls "$OUT"/*.tar.part* > /dev/null 2>&1; do sleep "$POLL"; done  # previous run not taken yet
  fi
  need=$(du -sBG "runs/$run/ckpt" "runs/$run/export" | awk '{s += $1} END {print s + 1}')
  free=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$need" ] || { echo "$run needs ${need}G, only ${free}G free"; exit 1; }
  rm -f "$OUT/READY.txt"
  echo "=== packing $run ($(date))"
  tar -cf - "runs/$run/ckpt" "runs/$run/export" | split -b 2000M -d - "$OUT/$run.tar.part" \
    || { echo "packing $run failed"; exit 1; }
  (cd "$OUT" && sha256sum "$run".tar.part* > "$run.SHA256SUMS")
  echo "$run is ready: download every file in this folder, then delete the $run.tar.part files" > "$OUT/READY.txt"
  touch "runs/.packed/$run"
  echo "=== $run ready ($(date)): $(ls "$OUT"/"$run".tar.part* | wc -l) parts"
done
rm -f "$OUT/READY.txt"
echo "=== all runs packed ($(date))"
