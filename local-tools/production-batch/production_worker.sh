#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -lt 5 ]]; then
  printf 'Usage: production_worker.sh REPO OPTIMIZER_JSON TEMPORAL_JSON SHARD_INDEX SHARD_COUNT [SEARCH_BATCH]\n' >&2
  exit 2
fi
REPO="$1"
OPTIMIZER_CONFIG="$2"
TEMPORAL_CONFIG="$3"
SHARD_INDEX="$4"
SHARD_COUNT="$5"
SEARCH_BATCH="${6:-1024}"
export STORAGE_ROOT="${STORAGE_ROOT:-/data/liuminghao}"
PYTHON="$STORAGE_ROOT/agibot-x2/soma-retargeter/.venv/bin/python"
MANIFEST="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/full-input-manifest.jsonl"
LOG_ROOT="$STORAGE_ROOT/logs/soma-retargeter/x2-production/full"
mkdir -p "$LOG_ROOT"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
exec >> "$LOG_ROOT/shard-$SHARD_INDEX-$ATTEMPT.log" 2>&1
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
printf 'PRODUCTION_SHARD_START shard=%s count=%s batch=%s utc=%s\n' "$SHARD_INDEX" "$SHARD_COUNT" "$SEARCH_BATCH" "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec "$PYTHON" -I "$REPO/app/bvh_to_csv_converter.py" --production-batch batched \
  --storage-root "$STORAGE_ROOT" --manifest "$MANIFEST" \
  --optimizer-config "$OPTIMIZER_CONFIG" --conversion-overrides "$TEMPORAL_CONFIG" \
  --expected-count 142220 --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
  --gpu 0 --batch-size "$SEARCH_BATCH" --final-batch-size 128 --fast-bvh --reuse-geometry --retries 1
