#!/usr/bin/env bash
set -euo pipefail
export STORAGE_ROOT=/data/liuminghao
REPO="$STORAGE_ROOT/agibot-x2/soma-retargeter-world-fix"
PYTHON="$STORAGE_ROOT/agibot-x2/soma-retargeter/.venv/bin/python"
TOOLS="$REPO/local-tools/production-batch"
ARTIFACTS="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/world-isolation/gpu"
LOGS="$STORAGE_ROOT/logs/soma-retargeter/x2-production/world-isolation/gpu"
INPUTS="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/throughput/cluster-batch-large-s025/inputs"
mkdir -p "$ARTIFACTS" "$LOGS"
exec > "$LOGS/job.log" 2>&1
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
printf 'WORLD_MEMORY_PROFILE_START %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for SIZE in 512 1024; do
  WEIGHTS="$STORAGE_ROOT/optimizers/soma-retargeter/agibot-x2ultra/throughput/cluster-batch-s025/b512"
  if [[ "$SIZE" -eq 1024 ]]; then
    WEIGHTS="$STORAGE_ROOT/optimizers/soma-retargeter/agibot-x2ultra/throughput/cluster-batch-large-s025/b1024-actual1024"
  fi
  "$PYTHON" -I "$TOOLS/profile_batch.py" --profile-path "$ARTIFACTS/b$SIZE-memory.json" \
    --entrypoint "$TOOLS/world_fix_profile.py" -- \
    --manifest "$INPUTS/manifest-b$SIZE.jsonl" --weights "$WEIGHTS" \
    --optimizer-config "$TOOLS/optimizer-native50.json" \
    --conversion-overrides "$INPUTS/temporal-provisional-s025.json" \
    --output-root "$ARTIFACTS/b$SIZE" > "$LOGS/b$SIZE.log" 2>&1
  if [[ "$SIZE" -eq 512 ]]; then
    "$PYTHON" -I "$TOOLS/compare_optimized_outputs.py" --manifest "$INPUTS/manifest-b512.jsonl" \
      --baseline-weights "$WEIGHTS" --weights "$WEIGHTS" \
      --baseline-csv "$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/throughput/cluster-batch-s025/b512/csv" \
      --csv "$ARTIFACTS/b512" --output "$ARTIFACTS/b512-native-equivalence.json"
  fi
  printf 'WORLD_MEMORY_CASE_COMPLETE b%s %s\n' "$SIZE" "$(date -u +%FT%TZ)"
done
printf 'WORLD_MEMORY_PROFILE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
