import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("csv_velocity", Path(__file__).with_name("csv_velocity.py"))
VELOCITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VELOCITY)


def independent_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class CSVVelocityTests(unittest.TestCase):
    def urdf(self, directory, limits):
        path = Path(directory) / "robot.urdf"
        joints = "".join(f'<joint name="{name}" type="revolute"><parent link="root"/>'
                         f'<child link="{name}_link"/><limit velocity="{limit}"/></joint>'
                         for name, limit in limits.items())
        links = '<link name="root"/>' + "".join(f'<link name="{name}_link"/>' for name in limits)
        path.write_text(f'<robot name="test">{links}{joints}</robot>')
        return path

    def test_real_csv_header_order_degrees_fps_scale_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf = self.urdf(directory, {"hip": 2, "knee": 4})
            plan = {**VELOCITY.load_velocity_contract(urdf, velocity_scale=0.5), "native_fps": 120}
            path = Path(directory) / "motion.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Frame", "knee", "hip"])
                for frame in range(3):
                    writer.writerow([frame, frame * math.degrees(1 / 120), frame * math.degrees(0.25 / 120)])
            with path.open(newline="") as stream:
                reader = csv.reader(stream)
                accumulator = VELOCITY.VelocityAccumulator(next(reader)[1:], plan)
                for row in reader:
                    accumulator.observe_angles([float(angle) for angle in row[1:]])
            report = accumulator.finish()
            self.assertEqual((report["frames"], report["checked_pairs"], report["violation_count"]), (3, 4, 0))
            self.assertAlmostEqual(report["max_ratio"], 0.5)
            self.assertEqual(report["worst_joint"], "knee")
            self.assertEqual(report["worst_frame_interval"], [0, 1])
            self.assertEqual((report["native_fps"], report["velocity_scale"]), (120.0, 0.5))
            self.assertEqual((report["angle_units"], report["limit_units"]), ("deg", "rad/s"))
            self.assertEqual(report["urdf_sha256"], hashlib.sha256(urdf.read_bytes()).hexdigest())
            self.assertEqual(report["ordered_joint_names_sha256"], independent_fingerprint(["knee", "hip"]))
            self.assertEqual(report["velocity_limits_sha256"], independent_fingerprint({"hip": 2.0, "knee": 4.0}))
            self.assertEqual(report["effective_limits_sha256"], independent_fingerprint({"hip": 1.0, "knee": 2.0}))
            json.dumps(report, allow_nan=False)
            path.unlink()
            VELOCITY.validate_attestation(report, ["knee", "hip"], 3, plan)

    def test_zero_one_two_three_frame_boundaries_and_per_clip_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 1}))
            for frames, fps in [(0, 120), (1, 120), (2, 60), (3, 120)]:
                with self.subTest(frames=frames, fps=fps):
                    accumulator = VELOCITY.VelocityAccumulator(["hip"], {**contract, "native_fps": fps})
                    for _ in range(frames):
                        accumulator.observe_angles([17.25])
                    if not frames:
                        with self.assertRaisesRegex(ValueError, "empty CSV"):
                            accumulator.finish()
                        continue
                    report = accumulator.finish()
                    self.assertEqual(report["frames"], frames)
                    self.assertEqual(report["checked_pairs"], frames - 1)
                    self.assertEqual(report["max_ratio"], 0)
                    self.assertEqual(report["worst_frame_interval"], [0, 1] if frames > 1 else None)
                    self.assertEqual(report["native_fps"], fps)
                    VELOCITY.validate_attestation(report, ["hip"], frames, {**contract, "native_fps": fps})
                    if frames == 1:
                        for field in ["worst_joint", "worst_frame_interval"]:
                            corrupted = {key: value for key, value in report.items() if key != field}
                            with self.assertRaises(ValueError):
                                VELOCITY.validate_attestation(corrupted, ["hip"], frames, {**contract, "native_fps": fps})

    def test_relative_rounding_tolerance_accepts_small_excess_but_not_real_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 1}))
            for ratio, fails in [(1.00005, False), (1.0002, True)]:
                with self.subTest(ratio=ratio):
                    accumulator = VELOCITY.VelocityAccumulator(["hip"], {**contract, "native_fps": 120})
                    accumulator.observe_angles([0])
                    accumulator.observe_angles([math.degrees(ratio / 120)])
                    if fails:
                        with self.assertRaises(VELOCITY.VelocityValidationError) as caught:
                            accumulator.finish()
                        self.assertEqual(caught.exception.attestation["status"], "failed")
                    else:
                        report = accumulator.finish()
                        self.assertEqual(report["status"], "passed")
                        self.assertGreater(report["max_ratio"], 1.0)
                        self.assertEqual(report["relative_tolerance"], 1e-4)

    def test_violations_count_samples_and_intervals_and_preserve_worst_location(self):
        with tempfile.TemporaryDirectory() as directory:
            unit = math.radians(1) * 120
            contract = VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": unit, "knee": 2 * unit}))
            accumulator = VELOCITY.VelocityAccumulator(["hip", "knee"], {**contract, "native_fps": 120})
            for angles in [[0, 0], [2, 4], [2, 8], [2, 8]]:
                accumulator.observe_angles(angles)
            with self.assertRaises(VELOCITY.VelocityValidationError) as caught:
                accumulator.finish()
            report = caught.exception.attestation
            self.assertEqual((report["frames"], report["checked_pairs"]), (4, 6))
            self.assertEqual((report["violation_count"], report["exceedance_frames"]), (3, 2))
            self.assertEqual(report["max_ratio"], 2)
            self.assertEqual((report["worst_joint"], report["worst_frame_interval"]), ("hip", [0, 1]))

    def test_malformed_urdf_and_ambiguous_or_invalid_angular_limits_are_rejected(self):
        cases = ["<robot>", "<other/>", '<robot name="empty"/>',
                 '<robot><joint type="revolute"><limit velocity="1"/></joint></robot>',
                 '<robot><joint name="a" type="revolute"><limit velocity="1"/></joint><joint name="a" type="continuous"><limit velocity="2"/></joint></robot>',
                 '<robot><joint name="a" type="unknown"/></robot>',
                 '<robot><joint name="a" type="revolute"/></robot>',
                 '<robot><joint name="a" type="revolute"><limit velocity="1"/><limit velocity="2"/></joint></robot>',
                 '<robot><joint name="a" type="prismatic"><limit velocity="1"/></joint></robot>']
        cases += [f'<robot><joint name="a" type="continuous"><limit velocity="{value}"/></joint></robot>'
                  for value in ["0", "-1", "nan", "inf", "not-a-number"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.urdf"
            for payload in cases:
                with self.subTest(payload=payload):
                    path.write_text(payload)
                    with self.assertRaises(ValueError):
                        VELOCITY.load_velocity_contract(path)

    def test_malformed_contract_header_units_fps_scale_and_tolerance_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = {**VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 1})), "native_fps": 120}
            cases = [(None, {}), ([], {}), (["hip", "hip"], {}), ("hip", {}), ([None], {}),
                     (["missing"], {}), (["hip"], {"schema": True}), (["hip"], {"schema": 2}),
                     (["hip"], {"urdf_sha256": "invalid"}), (["hip"], {"limit_units": "deg/s"}),
                     (["hip"], {"limits_rad_s": {"hip": True}}), (["hip"], {"limits_rad_s": {"hip": 0}})]
            cases += [(["hip"], {field: value}) for field in ["native_fps", "velocity_scale"]
                      for value in [None, True, 0, -1, float("nan"), float("inf")]]
            cases.append((["hip"], {"relative_tolerance": 1e-3}))
            for names, overrides in cases:
                with self.subTest(names=names, overrides=overrides):
                    with self.assertRaises(ValueError):
                        VELOCITY.VelocityAccumulator(names, {**base, **overrides})

    def test_bad_rows_do_not_advance_history_and_overflow_is_not_attested(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {**VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 1})), "native_fps": 120}
            accumulator = VELOCITY.VelocityAccumulator(["hip"], plan)
            accumulator.observe_angles([0])
            for angles in [None, "0", [], [0, 1], [float("nan")], [float("inf")], ["invalid"], [1e308]]:
                with self.subTest(angles=angles):
                    with self.assertRaises(ValueError):
                        accumulator.observe_angles(angles)
                    self.assertEqual(accumulator.frames, 1)
            accumulator.observe_angles([math.degrees(0.5 / 120)])
            self.assertAlmostEqual(accumulator.finish()["max_ratio"], 0.5)

    def test_mutable_row_and_plan_are_snapshotted(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {**VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 1})), "native_fps": 120}
            names, row = ["hip"], [0]
            accumulator = VELOCITY.VelocityAccumulator(names, plan)
            accumulator.observe_angles(row)
            row[0] = math.degrees(0.5 / 120)
            accumulator.observe_angles(row)
            names[0] = "mutated"
            plan["limits_rad_s"]["hip"] = 100
            plan["velocity_scale"] = 100
            report = accumulator.finish()
            self.assertAlmostEqual(report["max_ratio"], 0.5)
            self.assertEqual(report["velocity_scale"], 1)
            self.assertEqual(report["velocity_limits_sha256"], independent_fingerprint({"hip": 1.0}))

    def test_attestation_corruptions_and_missing_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {**VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 1})), "native_fps": 120}
            accumulator = VELOCITY.VelocityAccumulator(["hip"], plan)
            accumulator.observe_angles([0])
            accumulator.observe_angles([math.degrees(0.5 / 120)])
            report = accumulator.finish()
            changes = [("schema", 2), ("schema", True), ("enabled", False), ("enabled", 1),
                       ("status", "failed"), ("frames", 3), ("frames", True), ("joint_count", 2),
                       ("checked_pairs", 0), ("violation_count", 1), ("exceedance_frames", 1),
                       ("native_fps", 60), ("native_fps", "120"), ("angle_units", "rad"),
                       ("limit_units", "deg/s"), ("limit_basis", "unverified"),
                       ("relative_tolerance", 1e-3), ("velocity_scale", 0.5), ("velocity_scale", True),
                       ("urdf_sha256", "0" * 64), ("ordered_joint_names_sha256", "0" * 64),
                       ("velocity_limits_sha256", "0" * 64), ("effective_limits_sha256", "0" * 64),
                       ("fingerprint_schema", "unknown"), ("worst_joint", "unknown")]
            changes += [("max_ratio", value) for value in [1.001, -0.1, float("nan"), float("inf"), True, "0"]]
            changes += [("worst_frame_interval", value) for value in
                        [[0, 0], [-1, 0], [0, 2], [1, 2], [0, True], (0, 1), None]]
            for field, value in changes:
                with self.subTest(field=field, value=value):
                    corrupted = copy.deepcopy(report)
                    corrupted[field] = value
                    with self.assertRaises(ValueError):
                        VELOCITY.validate_attestation(corrupted, ["hip"], 2, plan)
            for field in report:
                with self.subTest(missing=field):
                    corrupted = copy.deepcopy(report)
                    del corrupted[field]
                    with self.assertRaises(ValueError):
                        VELOCITY.validate_attestation(corrupted, ["hip"], 2, plan)

    def test_attestation_is_bound_to_authoritative_names_limits_scale_fps_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {**VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 1, "knee": 2})), "native_fps": 120}
            accumulator = VELOCITY.VelocityAccumulator(["hip", "knee"], plan)
            accumulator.observe_angles([0, 0])
            accumulator.observe_angles([0, 0])
            report = accumulator.finish()
            cases = [(["knee", "hip"], 2, plan), (["hip", "knee"], 3, plan),
                     (["hip", "knee"], True, plan), (["hip", "knee"], 0, plan)]
            cases += [(["hip", "knee"], 2, {**plan, **mutation}) for mutation in
                      [{"native_fps": 60}, {"velocity_scale": 0.5}, {"urdf_sha256": "0" * 64},
                       {"limits_rad_s": {"hip": 2, "knee": 1}}]]
            without_scale = {key: value for key, value in plan.items() if key != "velocity_scale"}
            cases.append((["hip", "knee"], 2, without_scale))
            for names, frames, authoritative in cases:
                with self.subTest(names=names, frames=frames, plan=authoritative):
                    with self.assertRaises(ValueError):
                        VELOCITY.validate_attestation(report, names, frames, authoritative)

    def test_angle_wraps_are_checked_as_actual_exported_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {**VELOCITY.load_velocity_contract(self.urdf(directory, {"hip": 10})), "native_fps": 120}
            wrap = VELOCITY.VelocityAccumulator(["hip"], plan)
            wrap.observe_angles([179])
            wrap.observe_angles([-179])
            with self.assertRaises(VELOCITY.VelocityValidationError):
                wrap.finish()


if __name__ == "__main__":
    unittest.main()
