#!/usr/bin/env bash
set -euo pipefail
export STORAGE_ROOT=/data/liuminghao
REPO="$STORAGE_ROOT/agibot-x2/soma-retargeter-batch-benchmark"
PYTHON="$STORAGE_ROOT/agibot-x2/soma-retargeter/.venv/bin/python"
TOOLS="$REPO/local-tools/production-batch"
RUN=cluster-batch-large-s025
ARTIFACTS="$STORAGE_ROOT/artifacts/soma-retargeter/x2-production/throughput/$RUN"
LOGS="$STORAGE_ROOT/logs/soma-retargeter/x2-production/throughput/$RUN"
WEIGHTS="$STORAGE_ROOT/optimizers/soma-retargeter/agibot-x2ultra/throughput/$RUN"
mkdir -p "$LOGS"
exec > "$LOGS/job.log" 2>&1
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
printf 'LARGE_BATCH_PROBE_START %s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
PREVIOUS_FPS=531.7247147435769
for REQUESTED in 32 1024 2048 4096; do
  ACTUAL="$REQUESTED"
  SUCCESS=0
  for ATTEMPT in 1 2 3; do
    CASE="b$REQUESTED-actual$ACTUAL"
    mkdir -p "$ARTIFACTS/$CASE"
    printf 'PROFILED_CASE_START requested=%s actual=%s attempt=%s %s\n' "$REQUESTED" "$ACTUAL" "$ATTEMPT" "$(date -u +%FT%TZ)"
    set +e
    "$PYTHON" -I "$TOOLS/profile_batch.py" --profile-path "$ARTIFACTS/$CASE/memory-profile.json" -- \
      --optimizer-config "$ARTIFACTS/inputs/optimizer-budget2.json" \
      --conversion-overrides "$ARTIFACTS/inputs/temporal-provisional-s025.json" \
      --manifest "$ARTIFACTS/inputs/manifest-b$ACTUAL.jsonl" --expected-count "$ACTUAL" \
      --gpu 0 --batch-size "$ACTUAL" --length-bucket-frames 256 --retries 0 \
      --output-root "$ARTIFACTS/$CASE/csv" --weights-root "$WEIGHTS/$CASE" \
      --log-root "$LOGS/$CASE" --artifact-root "$ARTIFACTS/$CASE" > "$LOGS/$CASE.log" 2>&1
    STATUS=$?
    set -e
    if [[ "$STATUS" -eq 0 ]]; then
      "$PYTHON" -I "$TOOLS/summarize_gpu_benchmark.py" --artifact-root "$ARTIFACTS/$CASE"
      SUCCESS=1
      break
    fi
    if [[ "$STATUS" -ne 88 && "$STATUS" -ne 137 && "$STATUS" -ne 247 ]]; then
      printf 'CASE_FAILED_NONMEMORY status=%s\n' "$STATUS"
      exit "$STATUS"
    fi
    if [[ "$ACTUAL" -le 512 ]]; then break; fi
    ACTUAL=$((ACTUAL / 2))
    printf 'MEMORY_HALVE requested=%s next_actual=%s\n' "$REQUESTED" "$ACTUAL"
  done
  if [[ "$SUCCESS" -ne 1 ]]; then exit 88; fi
  printf 'PROFILED_CASE_COMPLETE requested=%s actual=%s %s\n' "$REQUESTED" "$ACTUAL" "$(date -u +%FT%TZ)"
  if [[ "$REQUESTED" -ge 1024 ]]; then
    CURRENT_FPS=$("$PYTHON" -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["frames_per_second_including_search_scoring_final_csv"])' "$ARTIFACTS/$CASE/measured-summary.json")
    if ! "$PYTHON" -I -c 'import sys;raise SystemExit(0 if float(sys.argv[1]) > 1.05*float(sys.argv[2]) else 1)' "$CURRENT_FPS" "$PREVIOUS_FPS"; then
      printf 'STOP_SCALING_PLATEAU current_fps=%s previous_fps=%s remaining_cases_skipped\n' "$CURRENT_FPS" "$PREVIOUS_FPS"
      break
    fi
    PREVIOUS_FPS="$CURRENT_FPS"
  fi
  if [[ -f "$ARTIFACTS/stop-after-current" ]]; then
    printf 'STOP_AFTER_CURRENT remaining_cases_skipped\n'
    break
  fi
done
printf 'LARGE_BATCH_PROBE_EXIT %s\n' "$(date -u +%FT%TZ)"
