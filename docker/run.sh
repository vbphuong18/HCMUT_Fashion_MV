#!/usr/bin/env bash
# Docker server helper. Run from the repository root on the GPU server.
#   docker/run.sh build                      # build the image (first time ~10-20 min: causal-conv1d compiles)
#   docker/run.sh check                      # GPU / kernel check, must pass before any run
#   docker/run.sh test                       # CPU + GPU test suites inside the container
#   docker/run.sh smoke                      # 20-step end-to-end run (train + export) on DeepFashion
#   docker/run.sh train CONFIG [ARGS...]     # full pipeline in the background, e.g.
#       docker/run.sh train configs/procir_mt_align.yaml --set cot=true seed=1 output_dir=runs/mt_align_cot_s1
#   docker/run.sh logs NAME                  # follow a background run (NAME printed by `train`)
# Re-running the same `train` command resumes the run from <output_dir>/ckpt/latest.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker/compose.yaml)
RUN=("${COMPOSE[@]}" run --rm procir)

cmd=${1:-help}; shift || true
case "$cmd" in
  build) "${COMPOSE[@]}" build ;;
  check) "${RUN[@]}" python scripts/check_env.py runs/env_check.json ;;
  test)  "${RUN[@]}" bash -c "pytest -q && pytest -q -m 'hf or gpu'" ;;
  smoke)
    rm -rf runs/smoke
    "${RUN[@]}" python -m procir_train.pipeline --config configs/smoke.yaml \
      --set image_root=data/images_official_hr --stages train export ;;
  train)
    config=${1:?usage: docker/run.sh train CONFIG [--set key=value ...]}; shift
    name="procir-$(basename "$config" .yaml)-$(date +%m%d-%H%M)"
    "${COMPOSE[@]}" run -d --name "$name" procir \
      python -m procir_train.pipeline --config "$config" "$@"
    echo "started $name; follow with: docker/run.sh logs $name" ;;
  logs)  docker logs -f "${1:?usage: docker/run.sh logs NAME}" ;;
  *) sed -n '2,11p' "$0" ;;
esac
