#!/usr/bin/env bash
# Run an AXLE sweep pinned to specific GPU(s), logging to a file.
#
#   scripts/run_gpu.sh <gpu_ids> <run_tag> <hydra args...>
#
# Examples:
#   # Germany loss ablation across shift protocols on GPU 2:
#   scripts/run_gpu.sh 2 germany_ablation -m loss=mse,invvar,hetero,axle \
#       data=germany model=lstm protocol=cv10,loyo,loro train.epochs=30 train.batch_size=4096
#
#   # Argentina soybean go/no-go on GPU 3:
#   scripts/run_gpu.sh 3 arg_soy -m loss=mse,hetero,axle data=argentina crop=soybean \
#       model=lstm protocol=cv10,loyo,loro train.epochs=30 train.batch_size=4096
#
# Runs in the foreground; wrap in tmux or append '&' to background it. Hydra
# multirun (-m) runs the sweep sequentially on the pinned GPU.
set -euo pipefail

GPUS="${1:?gpu id(s), e.g. 2 or 2,3}"; shift
TAG="${1:?a short run tag}"; shift
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/${TAG}_${STAMP}.log"
mkdir -p logs

echo "GPU=$GPUS  tag=$TAG  log=$LOG"
echo "args: $*"
CUDA_VISIBLE_DEVICES="$GPUS" python -m axle.train "$@" \
    hydra.sweep.dir="multirun/${TAG}_${STAMP}" \
    hydra.run.dir="outputs/${TAG}_${STAMP}" \
    2>&1 | tee "$LOG"
echo "done -> $LOG"
