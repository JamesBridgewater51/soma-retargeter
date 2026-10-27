#!/usr/bin/env bash
set -euo pipefail
export STORAGE_ROOT=/data/liuminghao
REPO="$STORAGE_ROOT/agibot-x2/soma-retargeter-batch-benchmark"
PYTHON="$STORAGE_ROOT/agibot-x2/soma-retargeter/.venv/bin/python"
TOOLS="$REPO/local-tools/production-batch"
RUN=cluster-batch-s025
ARTIFACTS="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/throughput/$RUN"
LOGS="$STORAGE_ROOT/logs/soma-retargeter/x2-production/throughput/$RUN"
WEIGHTS="$STORAGE_ROOT/optimizers/soma-retargeter/agibot-x2ultra/throughput/$RUN"
TEMPORAL="$STORAGE_ROOT/artifacts/soma-retargeter/agibot-x2ultra/temporal-final-family-comparison/plan-optimized/smooth-0.25.json"
mkdir -p "$LOGS" "$ARTIFACTS"
exec > "$LOGS/job.log" 2>&1
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
printf 'TRUEBATCH_PROBE_START %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" -I "$TOOLS/prepare_gpu_benchmark.py" --artifact-root "$ARTIFACTS/inputs" --temporal-config "$TEMPORAL"
for SCENARIO in native force-accept; do
  "$PYTHON" -I "$TOOLS/smoke_batched_parity.py" \
    --motion "$STORAGE_ROOT/datasets/bones-seed/raw/soma-uniform/validation/broad_jump_002__A360.bvh" \
    --optimizer-config "$TOOLS/optimizer-native50.json" \
    --conversion-overrides "$ARTIFACTS/inputs/temporal-provisional-s025.json" \
    --device cuda:0 --scenario "$SCENARIO" --fullpost > "$LOGS/parity-$SCENARIO.log" 2>&1
  printf 'GPU_PARITY_COMPLETE %s %s\n' "$SCENARIO" "$(date -u +%FT%TZ)"
done
for SIZE in 8 32 128 512; do
  CASE="b$SIZE"
  printf 'TRUEBATCH_CASE_START %s %s\n' "$CASE" "$(date -u +%FT%TZ)"
  "$PYTHON" -I "$TOOLS/run_batched.py" \
    --optimizer-config "$ARTIFACTS/inputs/optimizer-budget2.json" \
    --conversion-overrides "$ARTIFACTS/inputs/temporal-provisional-s025.json" \
    --manifest "$ARTIFACTS/inputs/manifest-$CASE.jsonl" --expected-count "$SIZE" \
    --gpu 0 --batch-size "$SIZE" --length-bucket-frames 256 --retries 0 \
    --output-root "$ARTIFACTS/$CASE/csv" --weights-root "$WEIGHTS/$CASE" \
    --log-root "$LOGS/$CASE" --artifact-root "$ARTIFACTS/$CASE" > "$LOGS/$CASE.log" 2>&1
  "$PYTHON" -I "$TOOLS/summarize_gpu_benchmark.py" --artifact-root "$ARTIFACTS/$CASE"
  printf 'TRUEBATCH_CASE_COMPLETE %s %s\n' "$CASE" "$(date -u +%FT%TZ)"
done
printf 'TRUEBATCH_PROBE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
