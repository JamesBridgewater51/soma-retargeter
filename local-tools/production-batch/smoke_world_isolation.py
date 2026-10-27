#!/usr/bin/env python3
"""Measure limb collision-pair scaling and verify isolated-world kinematic parity."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch
from smoke_batched_parity import trimmed_bvh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.storage_root = "/data/liuminghao"
    args.provenance_file = []
    os.environ["STORAGE_ROOT"] = args.storage_root
    settings = batch.prepare_settings(args)
    import numpy as np
    import newton
    import warp as wp
    from batched_optimizer import PerSequencePipeline
    from soma_retargeter.io import bvh, csv as csv_io
    from soma_retargeter.utils.newton_utils import get_filtered_joint_names
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    wp.init()
    wp.set_device("cpu")
    full_config = copy.deepcopy(settings["base_retargeter"])
    full_config.update(copy.deepcopy(settings["conversion_overrides"]))
    with tempfile.TemporaryDirectory(prefix="world-isolation-") as directory:
        root = Path(directory)
        paths = [root / "a.bvh", root / "b.bvh"]
        for index, path in enumerate(paths):
            trimmed_bvh(args.motion, path, index * 8, 8)
        skeleton, first = bvh.load_bvh(str(paths[0]))
        animations = [first, bvh.load_bvh(str(paths[1]), skeleton)[1]]
        template = PerSequencePipeline(skeleton, full_config, [full_config])
        robot = template.limb_stabilizer.robot_builder

        def build(count, isolated):
            started = time.monotonic()
            builder = newton.ModelBuilder()
            for _ in range(count):
                if isolated:
                    builder.add_world(robot, xform=wp.transform_identity())
                else:
                    builder.add_builder(robot, xform=wp.transform_identity())
            builder.add_ground_plane()
            model = builder.finalize()
            return model, {"environments": count, "isolated": isolated,
                           "seconds": time.monotonic() - started,
                           "collision_pairs": int(model.shape_contact_pair_count)}

        report = {"builds": [], "fk_parity": [], "fullpost_csv_parity": []}
        for count in (1, 2, 16, 32, 64):
            reference, old_stats = build(count, False)
            isolated, new_stats = build(count, True)
            report["builds"].extend([old_stats, new_stats])
            if count <= 2:
                for field in ("joint_q_start", "joint_parent", "joint_child", "joint_type"):
                    np.testing.assert_array_equal(getattr(reference, field).numpy(), getattr(isolated, field).numpy())
                for field in ("body_label", "joint_label"):
                    leaf = lambda label: label.rsplit("/", 1)[-1]
                    if list(map(leaf, getattr(reference, field))) != list(map(leaf, getattr(isolated, field))):
                        raise AssertionError("Body/joint ordering changed")
                q = reference.joint_q.numpy().copy().reshape(count, -1)
                for index in range(count):
                    q[index, 0] += index * 0.2
                    q[index, 7:] += (index + 1) * 0.01
                reference.joint_q.assign(q.reshape(-1))
                isolated.joint_q.assign(q.reshape(-1))
                states = [model.state() for model in (reference, isolated)]
                for model, state in zip((reference, isolated), states, strict=True):
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                np.testing.assert_array_equal(states[0].body_q.numpy(), states[1].body_q.numpy())
                report["fk_parity"].append({"environments": count, "maximum_difference": 0.0})
            print(json.dumps({"event": "WORLD_BUILD_PROFILE", "old": old_stats, "new": new_stats}), flush=True)
        transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
        for count in (1, 2):
            outputs = []
            for isolated in (False, True):
                pipeline = PerSequencePipeline(skeleton, full_config, [full_config] * count)
                if not isolated:
                    def old_build(self, num_envs):
                        return build(num_envs, False)[0]
                    pipeline.limb_stabilizer._build_model = types.MethodType(old_build, pipeline.limb_stabilizer)
                    pipeline.limb_stabilizer.ik_model = build(1, False)[0]
                pipeline.add_input_motions(animations[:count], [transform] * count, scale_animation=True)
                outputs.append(pipeline.execute())
            headers = batch.ROOT_HEADER + get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
            for index in range(count):
                np.testing.assert_allclose(outputs[0][index].data, outputs[1][index].data, rtol=1e-6, atol=1e-7)
                for version in (0, 1):
                    path = root / f"b{count}-clip{index}-version{version}.csv"
                    csv_io.save_csv(str(path), headers[len(batch.ROOT_HEADER):], outputs[version][index])
                    batch.validate_csv(path, headers, 8, batch.velocity_plan(settings, 120.0))
            report["fullpost_csv_parity"].append({"environments": count, "result": "PASS"})
    report["result"] = "PASS"
    batch.atomic_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
