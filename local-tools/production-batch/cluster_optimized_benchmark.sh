#!/usr/bin/env bash
set -euo pipefail
export STORAGE_ROOT=/data/liuminghao
REPO="$STORAGE_ROOT/agibot-x2/soma-retargeter-batch-benchmark-optimized"
PYTHON="$STORAGE_ROOT/agibot-x2/soma-retargeter/.venv/bin/python"
TOOLS="$REPO/local-tools/production-batch"
RUN=cluster-batch-optimized-s025
ARTIFACTS="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/throughput/$RUN"
INPUTS="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/throughput/cluster-batch-large-s025/inputs"
LOGS="$STORAGE_ROOT/logs/soma-retargeter/x2-production/throughput/$RUN"
WEIGHTS="$STORAGE_ROOT/optimizers/soma-retargeter/agibot-x2ultra/throughput/$RUN"
mkdir -p "$LOGS" "$ARTIFACTS"
exec > "$LOGS/job.log" 2>&1
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
printf 'OPTIMIZED_BATCH_START %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
PREVIOUS_FPS=0
for REQUESTED in 512 1024 2048 4096; do
  ACTUAL="$REQUESTED"
  SUCCESS=0
  for ATTEMPT in 1 2 3; do
    CASE="b$REQUESTED-actual$ACTUAL"
    mkdir -p "$ARTIFACTS/$CASE"
    printf 'OPTIMIZED_CASE_START requested=%s actual=%s attempt=%s %s\n' "$REQUESTED" "$ACTUAL" "$ATTEMPT" "$(date -u +%FT%TZ)"
    set +e
    timeout 1800 "$PYTHON" -I "$TOOLS/profile_batch.py" --profile-path "$ARTIFACTS/$CASE/memory-profile.json" -- \
      --optimizer-config "$INPUTS/optimizer-budget2.json" \
      --conversion-overrides "$INPUTS/temporal-provisional-s025.json" \
      --manifest "$INPUTS/manifest-b$ACTUAL.jsonl" --expected-count "$ACTUAL" \
      --gpu 0 --batch-size "$ACTUAL" --length-bucket-frames 256 --retries 0 --fast-bvh --reuse-geometry \
      --output-root "$ARTIFACTS/$CASE/csv" --weights-root "$WEIGHTS/$CASE" \
      --log-root "$LOGS/$CASE" --artifact-root "$ARTIFACTS/$CASE" > "$LOGS/$CASE.log" 2>&1
    STATUS=$?
    set -e
    if [[ "$STATUS" -eq 0 ]]; then
      "$PYTHON" -I "$TOOLS/summarize_gpu_benchmark.py" --artifact-root "$ARTIFACTS/$CASE"
      SUCCESS=1
      break
    fi
    if [[ "$STATUS" -ne 88 && "$STATUS" -ne 137 && "$STATUS" -ne 247 ]]; then exit "$STATUS"; fi
    if [[ "$ACTUAL" -le 512 ]]; then break; fi
    ACTUAL=$((ACTUAL / 2))
    printf 'MEMORY_HALVE requested=%s next_actual=%s\n' "$REQUESTED" "$ACTUAL"
  done
  if [[ "$SUCCESS" -ne 1 ]]; then exit 88; fi
  if [[ "$REQUESTED" -eq 512 && "$ACTUAL" -eq 512 ]]; then
    "$PYTHON" -I "$TOOLS/compare_optimized_outputs.py" \
      --manifest "$INPUTS/manifest-b512.jsonl" \
      --baseline-weights "$STORAGE_ROOT/optimizers/soma-retargeter/agibot-x2ultra/throughput/cluster-batch-s025/b512" \
      --weights "$WEIGHTS/$CASE" \
      --baseline-csv "$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/throughput/cluster-batch-s025/b512/csv" \
      --csv "$ARTIFACTS/$CASE/csv" --output "$ARTIFACTS/$CASE/native-equivalence.json"
  fi
  printf 'OPTIMIZED_CASE_COMPLETE requested=%s actual=%s %s\n' "$REQUESTED" "$ACTUAL" "$(date -u +%FT%TZ)"
  CURRENT_FPS=$("$PYTHON" -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["frames_per_second_including_search_scoring_final_csv"])' "$ARTIFACTS/$CASE/measured-summary.json")
  if [[ "$REQUESTED" -ge 1024 ]]; then
    if ! "$PYTHON" -I -c 'import sys;raise SystemExit(0 if float(sys.argv[1]) > 1.05*float(sys.argv[2]) else 1)' "$CURRENT_FPS" "$PREVIOUS_FPS"; then
      printf 'STOP_SCALING_PLATEAU remaining_cases_skipped\n'
      break
    fi
  fi
  PREVIOUS_FPS="$CURRENT_FPS"
  if [[ -f "$ARTIFACTS/stop-after-current" ]]; then
    printf 'STOP_AFTER_CURRENT remaining_cases_skipped\n'
    break
  fi
done
printf 'OPTIMIZED_BATCH_EXIT %s\n' "$(date -u +%FT%TZ)"
