#!/usr/bin/env bash
set -euo pipefail

PACKET=${1:?launch packet path required}
SHARD=${2:?shard index required}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=/data/liuminghao/agibot-x2/soma-retargeter/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

LOG_DIR=$("$PYTHON" -I -c 'import json,sys; from pathlib import Path; packet=json.load(open(sys.argv[1])); print(Path(packet["paths"]["logs"])/("worker-"+sys.argv[2]))' "$PACKET" "$SHARD")
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/worker.log") 2>&1
printf 'CLUSTER_WORKER_START shard=%s time=%s host=%s\n' "$SHARD" "$(date -u +%FT%TZ)" "$HOSTNAME"
exec "$PYTHON" -I "$SCRIPT_DIR/launch_full.py" worker --packet "$PACKET" --shard-index "$SHARD"
