#!/usr/bin/env python3
"""Numerically compare two short native singleton searches against batched states."""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


def trimmed_bvh(source, destination, start, frames):
    lines = source.read_text().splitlines(keepends=True)
    header_index = next(index for index, line in enumerate(lines) if line.strip().startswith("Frames:"))
    timing_index = next(index for index, line in enumerate(lines) if line.strip().startswith("Frame Time:"))
    motion = [line for line in lines[timing_index + 1:] if line.strip()]
    if len(motion) < start + frames:
        raise ValueError("Source too short for distinct parity snippets")
    lines[header_index] = f"Frames: {frames}\n"
    destination.write_text("".join(lines[:timing_index + 1] + motion[start:start + frames]))


def assert_equivalent(candidate, state, reference):
    import numpy as np
    expected = batch.load_json(reference["optimized_retargeter_config"])
    expected_accepts = sum(row["parameter_update"].get("line_search_accepted", 0)
                           for row in reference["history"][1:])
    if state.iterations != len(reference["history"]) - 1 or state.accepts != expected_accepts:
        raise AssertionError("Batched optimizer changed independent native acceptance decisions")
    for field in ("position_weight_scales", "rotation_weight_scales"):
        np.testing.assert_allclose(getattr(state.params, field), reference["final_parameters"][field], rtol=1e-5, atol=1e-6)
    for effector in expected["ik_match_table"]:
        for field in ("t_weight", "r_weight"):
            np.testing.assert_allclose(candidate["ik_match_table"][effector][field],
                                       expected["ik_match_table"][effector][field], rtol=1e-4, atol=1e-3)
    for name in ("tracking_loss", "ik_position_rmse_m", "ik_orientation_rmse_rad",
                 "ik_smoothness_loss", "ik_frame_loss_p95", "ik_worst_frame_loss"):
        np.testing.assert_allclose(state.initial_metrics[name], reference["history"][0]["aggregate_metrics"][name],
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(state.anchor.aggregate[name], reference["history"][-1]["aggregate_metrics"][name],
                                   rtol=1e-4, atol=1e-6)
    return {"motion": reference["motions"][0], "iterations": state.iterations, "probes": state.probes,
            "accepts": state.accepts, "initial_tracking_loss": state.initial_metrics["tracking_loss"],
            "final_tracking_loss": state.anchor.aggregate["tracking_loss"]}


def fullpost_parity(skeleton, animations, configs, settings):
    import newton
    import numpy as np
    import warp as wp
    from batched_optimizer import PerSequencePipeline
    from soma_retargeter.io.csv import save_csv
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils.newton_utils import get_filtered_joint_names
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    full_config = copy.deepcopy(settings["base_retargeter"])
    full_config.update(copy.deepcopy(settings["conversion_overrides"]))
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    combined = PerSequencePipeline(skeleton, full_config, configs)
    combined.add_input_motions(animations, [transform] * len(animations), scale_animation=True)
    combined_buffers = combined.execute()
    headers = batch.ROOT_HEADER + get_filtered_joint_names(combined.robot_builder, [newton.JointType.REVOLUTE])
    if len(headers) != 38 or not combined.post_processing_enabled or not combined.contact_processing_enabled:
        raise AssertionError("Full-post batch omitted native processing or X2 columns")
    with tempfile.TemporaryDirectory(prefix="batch-fullpost-") as directory:
        for index, (animation, config, buffer) in enumerate(zip(animations, configs, combined_buffers, strict=True)):
            scalar_config = copy.deepcopy(config)
            scalar_config.update(copy.deepcopy(settings["conversion_overrides"]))
            scalar = SomaRetargetingPipeline(skeleton, "soma", "agibot_x2ultra", retarget_config=scalar_config)
            scalar.add_input_motions([animation], [transform], scale_animation=True)
            scalar_buffer = scalar.execute()[0]
            np.testing.assert_allclose(buffer.data, scalar_buffer.data, rtol=1e-4, atol=1e-5)
            if buffer.num_frames != animation.num_frames or abs(buffer.sample_rate - animation.sample_rate) > 1e-6:
                raise AssertionError("Full-post export changed frames or FPS")
            path = Path(directory) / f"clip-{index}.csv"
            save_csv(str(path), headers[len(batch.ROOT_HEADER):], buffer)
            batch.validate_csv(path, headers, animation.num_frames)
    return {"fullpost_scalar_parity": "PASS", "csv_columns": len(headers),
            "clips": len(combined_buffers), "frames": [buffer.num_frames for buffer in combined_buffers]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--storage-root", default="/data/liuminghao")
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenario", choices=("native", "force-accept"), default="native")
    parser.add_argument("--fullpost", action="store_true", help="Also compare native full-post trajectories and validate CSV exports")
    args = parser.parse_args()
    args.provenance_file = []
    os.environ["STORAGE_ROOT"] = args.storage_root
    settings = batch.prepare_settings(args)
    import warp as wp
    from app.tools import ik_weight_optimizer as native
    from soma_retargeter.io.bvh import load_bvh
    from batched_optimizer import IndependentBatchOptimizer
    wp.init()
    wp.set_device(args.device)
    config = copy.deepcopy(settings["optimizer"])
    config.update(outer_iterations=2, convergence_patience=1)
    config["ik_weight_line_search"] = {"enabled": True, "max_backtracks": 1, "shrink": 0.5, "min_improvement": 0.00005}
    if args.scenario == "force-accept":
        config["ik_weight_line_search"]["enabled"] = False
        config.setdefault("clip_guard", {})["mode"] = "off"
    settings["optimizer"] = config
    evidence = []
    fullpost = None
    with tempfile.TemporaryDirectory(prefix="batched-parity-") as directory:
        root = Path(directory)
        paths = [root / "clip-a.bvh", root / "clip-b.bvh"]
        for index, path in enumerate(paths):
            trimmed_bvh(args.motion, path, index * 8, 8)
        references = []
        with tempfile.TemporaryFile(mode="w+") as transcript:
            try:
                with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                    for path in paths:
                        singleton = copy.deepcopy(config)
                        singleton.update(input_folder=str(path), output_folder=str(root / path.stem))
                        references.append(native.run_optimizer(singleton, batch.REPO))
                    skeleton, first = load_bvh(str(paths[0]))
                    animations = [first, load_bvh(str(paths[1]), skeleton)[1]]
                    workspace = native.OptimizerConfigWorkspace("soma", "agibot_x2ultra", root / "batch")
                    optimizer = IndependentBatchOptimizer(skeleton, animations, [path.name for path in paths], settings, workspace)
                    candidates = optimizer.run()
                    for candidate, state, reference in zip(candidates, optimizer.states, references, strict=True):
                        evidence.append({"batch_size": 2, **assert_equivalent(candidate, state, reference)})
                    one = IndependentBatchOptimizer(skeleton, animations[:1], [paths[0].name], settings, workspace)
                    one_candidate = one.run()[0]
                    evidence.append({"batch_size": 1, **assert_equivalent(one_candidate, one.states[0], references[0])})
                    if args.scenario == "force-accept" and any(row["accepts"] != 2 for row in evidence):
                        raise AssertionError("Forced acceptance smoke did not exercise two accepted updates")
                    if args.fullpost:
                        fullpost = fullpost_parity(skeleton, animations, candidates, settings)
            except Exception:
                transcript.seek(0)
                print(transcript.read()[-32768:], flush=True)
                raise
    print(json.dumps({"parity": "PASS", "device": args.device, "frames_per_clip": 8,
                      "outer_iterations": 2, "max_backtracks": 1, "scenario": args.scenario,
                      "clips": evidence, "fullpost": fullpost}), flush=True)


if __name__ == "__main__":
    main()
