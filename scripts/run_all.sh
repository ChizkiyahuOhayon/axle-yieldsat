#!/usr/bin/env bash
# Reproduce the AXLE benchmark grid end-to-end.
#   Usage: bash scripts/run_all.sh /path/to/Both.zip /path/to/Preprocessed [SEEDS]
# Builds the per-country caches (with reliability join) then sweeps the
# loss ablation over backbones and shift protocols, 3 seeds by default.
set -euo pipefail

BOTH="${1:?path to Both.zip}"
PREPROC="${2:?path to Preprocessed root (holds <Country>/merge_*.nc)}"
SEEDS="${3:-0,1,2}"

echo "== 1/3  building caches =="
python scripts/prepare.py --root "$PREPROC" --both "$BOTH" --out data/cache

echo "== 2/3  estimating harvester directions (d_f, needed by AXLE-M2) =="
python scripts/estimate_directions.py data/cache/*/

echo "== 3/3  training grid =="
# In-distribution + shift protocols; full loss ablation; three backbones.
python -m axle.train -m \
    data=germany,argentina,brazil,uruguay \
    model=lstm,tempcnn,transformer \
    loss=mse,invvar,hetero,axle,axle_spatial \
    protocol=cv10,loyo,loro \
    seed="$SEEDS"

echo "done. per-run metrics.json + predictions.parquet under outputs/"
