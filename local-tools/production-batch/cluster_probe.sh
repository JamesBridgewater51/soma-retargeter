#!/usr/bin/env bash
set -euo pipefail
export STORAGE_ROOT=/data/liuminghao
REPO="$STORAGE_ROOT/agibot-x2/soma-retargeter-production"
PYTHON="$STORAGE_ROOT/agibot-x2/soma-retargeter/.venv/bin/python"
LOG_DIR="$STORAGE_ROOT/logs/soma-retargeter/x2-production/cluster-probe"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/probe.log" 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO"
cd "$REPO"
printf 'CLUSTER_PROBE_START %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" -I -c 'import sys; sys.path.insert(0,"/data/liuminghao/agibot-x2/soma-retargeter-production"); import soma_retargeter, newton, warp; print("SOURCE",soma_retargeter.__file__); print("NEWTON",newton.__version__,"WARP",warp.__version__)'
"$PYTHON" -I "$REPO/local-tools/temporal-ablation/run.py" \
  --motion fire_in_the_hole_180_R_002__A473 --variant cluster-control \
  --weight-set optimized \
  --temporal-config "$REPO/local-tools/production-batch/probe-disabled.json" \
  --output-dir "$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/cluster-probe/control"
printf 'CLUSTER_PROBE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
