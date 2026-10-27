#!/usr/bin/env bash
set -euo pipefail
export STORAGE_ROOT=/data/liuminghao
REPO="$STORAGE_ROOT/agibot-x2/soma-retargeter-hard-budget-pilot"
PYTHON="$STORAGE_ROOT/agibot-x2/soma-retargeter/.venv/bin/python"
TOOLS="$REPO/local-tools/production-batch"
ARTIFACTS="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/pilots/hard-budget8"
LOGS="$STORAGE_ROOT/logs/soma-retargeter/x2-production/pilots/hard-budget8"
WEIGHTS="$STORAGE_ROOT/optimizers/soma-retargeter/agibot-x2ultra/pilots/hard-budget8"
mkdir -p "$ARTIFACTS" "$LOGS"
exec > "$LOGS/job.log" 2>&1
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
printf 'HARD_BUDGET_PILOT_START %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" -I "$TOOLS/prepare_quality_pilot.py" --output-root "$ARTIFACTS/inputs" --selected-config "$TOOLS/temporal-selected.json"
"$PYTHON" -I "$TOOLS/smoke_batched_parity.py" \
  --motion "$STORAGE_ROOT/datasets/bones-seed/raw/soma-uniform/validation/broad_jump_002__A360.bvh" \
  --optimizer-config "$ARTIFACTS/inputs/optimizer-budget8.json" \
  --conversion-overrides "$ARTIFACTS/inputs/temporal-selected.json" \
  --device cuda:0 --scenario force-accept --reuse-geometry --fullpost > "$LOGS/hard-b1-b2-compat.log" 2>&1
printf 'HARD_B1_B2_COMPAT_COMPLETE %s\n' "$(date -u +%FT%TZ)"
"$PYTHON" -I "$TOOLS/run_batched.py" \
  --optimizer-config "$ARTIFACTS/inputs/optimizer-budget8.json" \
  --conversion-overrides "$ARTIFACTS/inputs/temporal-selected.json" \
  --manifest "$ARTIFACTS/inputs/selected11.jsonl" --expected-count 11 \
  --gpu 0 --batch-size 16 --length-bucket-frames 100000 --final-batch-size 128 \
  --snapshot-iterations 4,8 --fast-bvh --reuse-geometry --retries 0 \
  --output-root "$ARTIFACTS/budget8-csv" --weights-root "$WEIGHTS" \
  --log-root "$LOGS/fit" --artifact-root "$ARTIFACTS/fit" > "$LOGS/fit.log" 2>&1
SETTINGS=$("$PYTHON" -I -c 'from pathlib import Path;import sys;p=list(Path(sys.argv[1]).glob("*.settings.json"));assert len(p)==1;print(p[0])' "$ARTIFACTS/fit")
"$PYTHON" -I "$TOOLS/pilot_quality.py" --settings "$SETTINGS" --manifest "$ARTIFACTS/inputs/selected11.jsonl" \
  --source-root "$STORAGE_ROOT/datasets/bones-seed/raw/soma-uniform/full" --output-root "$ARTIFACTS/quality" > "$LOGS/quality.log" 2>&1
printf 'HARD_BUDGET_PILOT_COMPLETE %s\n' "$(date -u +%FT%TZ)"
