#!/usr/bin/env python3
"""Read-only whole-cohort planning using the exact frozen production dispatch code."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offload_cohorts import (FROZEN_GROUPING, eligible_whole_cohort, file_sha256,
                             load_frozen_runtime, read_verified_manifest, reconstruct_groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Immutable production source used by the main workers")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=4096,
                        help="Every member of an eligible cohort must meet this threshold")
    args = parser.parse_args()
    rows = read_verified_manifest(args.manifest)
    if len(rows) != 142220 or sum(row["frames"] for row in rows) != 124554605:
        raise ValueError("Unexpected full manifest coverage")
    runtime = load_frozen_runtime(args.repo)
    cohorts, ranks = reconstruct_groups(runtime, rows)
    done = {path.stem for path in (args.artifacts / "done").glob("*.json")}
    candidates, groups, totals = [], [], defaultdict(lambda: {"clips": 0, "frames": 0})
    for index, cohort in enumerate(cohorts):
        rank = ranks[index]
        totals[rank]["clips"] += len(cohort)
        totals[rank]["frames"] += sum(row["frames"] for row in cohort)
        if not eligible_whole_cohort(cohort, done, args.weights, args.minimum_frames):
            continue
        candidates.extend({"stem": row["stem"], "frames": row["frames"], "source_sha256": row["sha256"],
                           "original_rank": rank, "global_group": index} for row in cohort)
        groups.append({"original_rank": rank, "global_group": index, "cohort_clips": len(cohort),
                       "unclaimed_long_clips": len(cohort),
                       "unclaimed_long_frames": sum(row["frames"] for row in cohort),
                       "maximum_frames": max(row["frames"] for row in cohort)})
    candidates.sort(key=lambda row: (row["frames"], row["original_rank"]), reverse=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "allocated": False,
              "mirror": {"decoded_topology": "one certified topology", **vars(FROZEN_GROUPING),
                         "runtime_sha256": file_sha256(runtime.__file__)},
              "rank_input_totals": dict(totals), "receipt_filenames_observed": len(done),
              "candidate_count": len(candidates), "candidate_frames": sum(row["frames"] for row in candidates),
              "groups": sorted(groups, key=lambda row: row["unclaimed_long_frames"], reverse=True),
              "motions": candidates,
              "safety": "Only complete eligible original cohorts are emitted. No lock was acquired. "
                        "Recheck actual free GPU and every cohort member before real producer lock acquisition. "
                        "Receipt presence is only an operational exclusion, not a completion audit."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"plan": str(args.output), "candidates": len(candidates),
                      "frames": report["candidate_frames"], "allocated": False}), flush=True)


if __name__ == "__main__":
    main()
