#!/usr/bin/env python3
"""Report measured complete two-iteration batch throughput and explicit projections."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-frames", type=int, default=124554605)
    args = parser.parse_args()
    progress_paths = sorted(args.artifact_root.glob("*.progress.json"))
    if len(progress_paths) != 1:
        raise ValueError(f"Expected one fresh benchmark progress record, found {len(progress_paths)}")
    progress = batch.load_json(progress_paths[0])
    if progress["failed"] or progress["resumed"] or progress["completed"] != progress["total"]:
        raise ValueError("Benchmark incomplete, failed or mixed with resumed clips")
    measured = progress["batch"]
    evaluations = measured["evaluations"]
    visited = sum(row["frames"] for row in evaluations)
    search_seconds = sum(row["seconds"] for row in evaluations)
    frames = measured["frames"]
    two_iteration_hours = args.dataset_frames * progress["wall_seconds"] / frames / 3600
    final_hours = args.dataset_frames * measured["conversion_seconds"] / frames / 3600
    maximum_probe_hours = args.dataset_frames * 301 * search_seconds / visited / 3600 + final_hours
    summary = {"clips": measured["clips"], "source_frames": frames,
               "wall_seconds": progress["wall_seconds"],
               "frames_per_second_including_search_scoring_final_csv": frames / progress["wall_seconds"],
               "search_forward_batches": len(evaluations), "search_frame_visits": visited,
               "effective_full_frame_search_passes": visited / frames,
               "final_conversion_seconds": measured["conversion_seconds"],
               "measured_budget": 2,
               "full_dataset_budget2_one_gpu_hours_linear_projection": two_iteration_hours,
               "full_dataset_budget2_ten_gpu_hours_ideal_projection": two_iteration_hours / 10,
               "full_dataset_budget50_maximum_301_search_passes_one_gpu_hours_linear_projection": maximum_probe_hours,
               "caveat": "Single 769–1024-frame bucket, warmed after separate GPU parity/compilation; budget2 not quality-approved production budget. "
                          "Ten GPUs are hypothetical linear scaling; 301-pass estimate is not a rigorous runtime bound."}
    batch.atomic_json(args.artifact_root / "measured-summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
