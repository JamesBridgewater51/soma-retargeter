#!/usr/bin/env python3
"""Compare exported full-post quality of genuine per-motion native search prefixes."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch
from run_batched import load_animations, inspect_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    settings = batch.load_json(args.settings)
    import os
    os.environ["STORAGE_ROOT"] = "/data/liuminghao"
    batch.verify_local_imports()
    import numpy as np
    import newton
    import warp as wp
    from app.tools import ik_weight_optimizer as native
    from batched_optimizer import IndependentBatchOptimizer, PerSequencePipeline, finite_metrics
    from soma_retargeter.io import csv as csv_io
    from soma_retargeter.utils.newton_utils import get_filtered_joint_names
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    wp.init()
    wp.set_device("cuda:0")
    jobs = [inspect_motion(row) for row in batch.source_rows(args.manifest, args.source_root)]
    skeleton, animations = load_animations(jobs, settings)
    workspace = native.OptimizerConfigWorkspace("soma", "agibot_x2ultra", Path(settings["weights"]))
    scorer = IndependentBatchOptimizer(skeleton, animations, [row["stem"] for row in jobs], settings, workspace)
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    report = {"inner_config": settings["conversion_overrides"], "frames": sum(row["frames"] for row in jobs),
              "metric_source": "exported CSV reconstructed at recorded native FPS; native tracking/smoothness functions", "budgets": {}}
    for budget in (4, 8):
        started = time.monotonic()
        directory = args.output_root / f"budget-{budget}"
        directory.mkdir(parents=True, exist_ok=True)
        if budget == 8:
            paths = [Path(settings["output"]) / f"{row['stem']}.csv" for row in jobs]
        else:
            configs = [batch.load_json(Path(settings["weights"]) / row["stem"] / "snapshots" / f"budget-{budget}.json")["retargeter_config"] for row in jobs]
            full_config = copy.deepcopy(settings["base_retargeter"])
            full_config.update(copy.deepcopy(settings["conversion_overrides"]))
            pipeline = PerSequencePipeline(skeleton, full_config, configs)
            pipeline.add_input_motions(animations, [transform] * len(jobs), scale_animation=True)
            buffers = pipeline.execute()
            headers = batch.ROOT_HEADER + get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
            paths = []
            for row, buffer in zip(jobs, buffers, strict=True):
                path = directory / f"{row['stem']}.csv"
                csv_io.save_csv(str(path), headers[len(batch.ROOT_HEADER):], buffer)
                attestation = batch.validate_csv(path, headers, row["frames"], batch.velocity_plan(settings, row["fps"]))
                batch.atomic_json(path.with_suffix(".validation.json"), attestation)
                paths.append(path)
        q = [np.asarray(csv_io.load_csv(str(path), fps=row["fps"]).data, dtype=np.float32)
             for row, path in zip(jobs, paths, strict=True)]
        poses = scorer.evaluator.evaluate(q)
        metrics = [finite_metrics(scorer.score(index, pose, coordinates).aggregate)
                   for index, (pose, coordinates) in enumerate(zip(poses, q, strict=True))]
        total = sum(row["frames"] for row in jobs)
        pooled_position = math.sqrt(sum(row["frames"] * metric["ik_position_rmse_m"] ** 2
                                       for row, metric in zip(jobs, metrics, strict=True)) / total)
        pooled_smoothness = sum(row["frames"] * metric["ik_smoothness_loss"]
                                for row, metric in zip(jobs, metrics, strict=True)) / total
        report["budgets"][budget] = {"pooled_position_rmse_m": pooled_position,
                                     "frame_weighted_native_smoothness_loss": pooled_smoothness,
                                     "quality_seconds": time.monotonic() - started,
                                     "per_motion": [{"stem": row["stem"], "frames": row["frames"], **metric}
                                                    for row, metric in zip(jobs, metrics, strict=True)]}
        batch.atomic_json(args.output_root / "quality.json", report)
        print(json.dumps({"event": "PILOT_BUDGET_QUALITY", "budget": budget,
                          "position_rmse_m": pooled_position, "native_smoothness_loss": pooled_smoothness}), flush=True)


if __name__ == "__main__":
    main()
