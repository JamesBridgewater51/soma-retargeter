#!/usr/bin/env python3
"""Profile full-post world isolation using existing independent native weight tables."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch
from run_batched import inspect_motion, load_animations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.storage_root = "/data/liuminghao"
    args.provenance_file = []
    os.environ["STORAGE_ROOT"] = args.storage_root
    settings = batch.prepare_settings(args)
    settings["fast_bvh"] = True
    batch.initialize_worker("0", settings["optimizer"]["extra_robot_paths"])
    import newton
    import warp as wp
    from batched_optimizer import PerSequencePipeline
    from soma_retargeter.io.csv import save_csv
    from soma_retargeter.utils.newton_utils import get_filtered_joint_names
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    jobs = [inspect_motion(row) for row in batch.source_rows(args.manifest,
        Path(args.storage_root) / "datasets/bones-seed/raw/soma-uniform/full")]
    configs = []
    for row in jobs:
        directory = args.weights / row["stem"]
        receipt = batch.load_json(directory / "optimized.done.json")
        path = directory / "retargeter.json"
        if receipt["source_sha256"] != row["sha256"] or receipt["sha256"] != batch.digest(path):
            raise ValueError("Existing independent weight provenance mismatch")
        config = batch.load_json(path)
        batch.verify_weights_only(config, settings["base_retargeter"])
        configs.append(config)
    started = time.monotonic()
    skeleton, animations = load_animations(jobs, settings)
    loaded = time.monotonic()
    full_config = copy.deepcopy(settings["base_retargeter"])
    full_config.update(copy.deepcopy(settings["conversion_overrides"]))
    pipeline = PerSequencePipeline(skeleton, full_config, configs)
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    pipeline.add_input_motions(animations, [transform] * len(jobs), scale_animation=True)
    prepared = time.monotonic()
    buffers = pipeline.execute()
    executed = time.monotonic()
    headers = batch.ROOT_HEADER + get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    args.output_root.mkdir(parents=True, exist_ok=True)
    for row, buffer in zip(jobs, buffers, strict=True):
        path = args.output_root / f"{row['stem']}.csv"
        save_csv(str(path), headers[len(batch.ROOT_HEADER):], buffer)
        validation = batch.validate_csv(path, headers, row["frames"], batch.velocity_plan(settings, row["fps"]))
        batch.atomic_json(path.with_suffix(".validation.json"), validation)
    report = {"benchmark_kind": "full-post-forward-only; existing independent fitted weights, not search ETA",
              "clips": len(jobs), "frames": sum(row["frames"] for row in jobs),
              "load_seconds": loaded - started, "prepare_seconds": prepared - loaded,
              "execute_seconds": executed - prepared, "publish_seconds": time.monotonic() - executed,
              "wall_seconds": time.monotonic() - started,
              "limb_contact_pair_count": int(pipeline.limb_stabilizer.model.shape_contact_pair_count),
              "expected_isolated_pair_count": 497 * len(jobs)}
    batch.atomic_json(args.output_root / "world-profile.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
