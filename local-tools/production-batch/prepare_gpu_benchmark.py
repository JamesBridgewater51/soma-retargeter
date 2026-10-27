#!/usr/bin/env python3
"""Prepare unique real-motion length-matched manifests for a bounded GPU probe."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch
from run_batched import inspect_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, default=Path(os.environ.get("STORAGE_ROOT", "/data/liuminghao")))
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--sizes", default="8,32,128,512")
    args = parser.parse_args()
    sizes = [int(token) for token in args.sizes.split(",")]
    root = args.storage_root
    selected = []
    hierarchy = None
    source_root = root / "datasets/bones-seed/raw/soma-uniform/full"
    manifest = root / "datasets/bones-seed/metadata/soma-uniform-full/extracted-files.jsonl"
    for row in batch.source_rows(manifest, source_root):
        motion = inspect_motion(row)
        if not 769 <= motion["frames"] <= 1024:
            continue
        if hierarchy is None:
            hierarchy = motion["hierarchy_sha256"]
        if motion["hierarchy_sha256"] != hierarchy:
            continue
        selected.append(motion)
        if len(selected) == max(sizes):
            break
    if len(selected) != max(sizes):
        raise ValueError(f"Only {len(selected)} unique same-hierarchy clips in requested 769–1024-frame bucket")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        path = args.artifact_root / f"manifest-b{size}.jsonl"
        if path.exists():
            raise FileExistsError(path)
        with path.open("x") as stream:
            for motion in selected[:size]:
                stream.write(json.dumps(motion) + "\n")
    config = copy.deepcopy(batch.load_json(Path(__file__).with_name("optimizer-native50.json")))
    config["outer_iterations"] = 2
    batch.atomic_json(args.artifact_root / "optimizer-budget2.json", config)
    batch.atomic_json(args.artifact_root / "temporal-provisional-s025.json", batch.load_json(args.temporal_config))
    batch.atomic_json(args.artifact_root / "selection.json", {
        "sizes": sizes, "unique_motions": len(selected), "hierarchy_sha256": hierarchy,
        "minimum_frames": min(row["frames"] for row in selected),
        "maximum_frames": max(row["frames"] for row in selected),
        "total_frames": sum(row["frames"] for row in selected),
        "source_manifest_sha256": batch.digest(manifest),
        "outer_iterations": 2, "production_default_iterations": 50,
        "note": "Real unique clips, one explicit length bucket; provisional temporal variant, not production selection."})
    print(json.dumps({"event": "BENCHMARK_INPUTS_READY", "clips": len(selected), "sizes": sizes}), flush=True)


if __name__ == "__main__":
    main()
