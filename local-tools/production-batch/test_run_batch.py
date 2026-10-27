import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("production_batch", Path(__file__).with_name("run_batch.py"))
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


class BatchValidationTests(unittest.TestCase):
    def test_import_guard_rejects_modules_outside_the_isolated_repository(self):
        self.assertEqual(BATCH.verified_module_path(BATCH), str(Path(BATCH.__file__).resolve()))
        with self.assertRaisesRegex(RuntimeError, "outside isolated repository"):
            BATCH.verified_module_path(tempfile)

    def test_resume_requires_matching_content_and_unchanged_product(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product, receipt = root / "retargeter.json", root / "done.json"
            BATCH.atomic_json(product, {"weights": [1.2, 3.4]})
            BATCH.atomic_json(receipt, {"content_key": "clip-a", "sha256": BATCH.digest(product)})
            self.assertIsNotNone(BATCH.receipt_valid(receipt, "clip-a", product))
            self.assertIsNone(BATCH.receipt_valid(receipt, "clip-b", product))
            BATCH.atomic_json(product, {"weights": [9.0]})
            self.assertIsNone(BATCH.receipt_valid(receipt, "clip-a", product))
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["done.json", "retargeter.json"])

    def test_csv_rejects_nonfinite_wrong_frames_headers_width_and_indices(self):
        cases = [
            (["Frame", "joint"], [[0, 1.0]], 1, False),
            (["Frame", "joint"], [[0, "nan"]], 1, True),
            (["Frame", "joint"], [[0, "inf"]], 1, True),
            (["Frame", "joint"], [[1, 0.1]], 1, True),
            (["Frame", "wrong"], [[0, 0.1]], 1, True),
            (["Frame", "joint"], [[0]], 1, True),
            (["Frame", "joint"], [], 1, True),
            (["Frame", "joint"], [[0, 0.1]], 2, True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            for headers, rows, expected_frames, invalid in cases:
                with self.subTest(headers=headers, rows=rows, expected_frames=expected_frames):
                    with path.open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(headers)
                        writer.writerows(rows)
                    if invalid:
                        with self.assertRaises(ValueError):
                            BATCH.validate_csv(path, ["Frame", "joint"], expected_frames)
                    else:
                        BATCH.validate_csv(path, ["Frame", "joint"], expected_frames)

    def test_hard_velocity_uses_joint_names_native_fps_and_one_frame_boundary(self):
        names = [f"joint_{index}" for index in reversed(range(31))]
        headers = BATCH.ROOT_HEADER + names
        limits = {f"joint_{index}": 1.0 + index / 10.0 for index in range(31)}
        cases = [(1, 0.0, 120.0, False), (2, 1.0, 120.0, False),
                 (2, 1.00005, 60.0, False), (2, 1.0002, 120.0, True)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard.csv"
            for frames, ratio, fps, invalid in cases:
                with self.subTest(frames=frames, ratio=ratio, fps=fps):
                    first = [0.0] * len(headers)
                    second = first.copy()
                    second[0] = 1
                    second[headers.index("joint_7")] = math.degrees(limits["joint_7"] * ratio / fps)
                    with path.open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(headers)
                        writer.writerows([first] if frames == 1 else [first, second])
                    plan = {"limits_rad_s": limits, "native_fps": fps,
                            "relative_tolerance": 1e-4, "urdf_sha256": "a" * 64, "velocity_scale": 1.0}
                    if invalid:
                        with self.assertRaisesRegex(ValueError, "violates URDF speed"):
                            BATCH.validate_csv(path, headers, frames, plan)
                    else:
                        report = BATCH.validate_csv(path, headers, frames, plan)["velocity"]
                        self.assertEqual(report["violation_count"], 0)
                        self.assertEqual(report["checked_pairs"], 31 * (frames - 1))
                        self.assertAlmostEqual(report["max_ratio"], ratio)

    def test_missing_urdf_joint_limit_never_silently_disables_speed_validation(self):
        headers = BATCH.ROOT_HEADER + [f"joint_{index}" for index in range(31)]
        plan = {"limits_rad_s": {}, "native_fps": 120.0,
                "relative_tolerance": 1e-4, "urdf_sha256": "a" * 64, "velocity_scale": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            path.write_text(",".join(headers) + "\n" + ",".join(["0"] * len(headers)) + "\n")
            with self.assertRaisesRegex(ValueError, "missing a matching URDF angular velocity limit"):
                BATCH.validate_csv(path, headers, 1, plan)

    def test_optimization_cannot_change_scaler_or_nonweight_configuration(self):
        baseline = {"human_robot_scaler_config": "fixed-scaler.json", "ik_iterations": 24,
                    "ik_match_table": {"LeftFoot": {"t_weight": 1, "r_weight": 2, "body": "foot"}}}
        for mutation in ("weights", "scaler", "iterations", "mapping", "negative", "nonfinite"):
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(baseline)
                if mutation == "weights":
                    candidate["ik_match_table"]["LeftFoot"]["t_weight"] = 4.5
                elif mutation == "scaler":
                    candidate["human_robot_scaler_config"] = "different.json"
                elif mutation == "iterations":
                    candidate["ik_iterations"] = 1
                elif mutation == "mapping":
                    candidate["ik_match_table"]["RightFoot"] = candidate["ik_match_table"].pop("LeftFoot")
                else:
                    candidate["ik_match_table"]["LeftFoot"]["r_weight"] = -1 if mutation == "negative" else float("nan")
                if mutation == "weights":
                    BATCH.verify_weights_only(candidate, baseline)
                    self.assertEqual(candidate["ik_match_table"]["LeftFoot"]["t_weight"], 4.5)
                else:
                    with self.assertRaises(ValueError):
                        BATCH.verify_weights_only(candidate, baseline)

    def test_selected_temporal_solver_applies_during_postoff_search_without_shared_mutation(self):
        baseline = {"retargeter_overrides": {"enable_post_processing": False, "enable_contact_processing": False}}
        selected = {"temporal_ik": {"enabled": True, "weight": 0.3, "hard_velocity": True}}
        configured = BATCH.optimizer_with_selected_temporal(baseline, selected)
        self.assertEqual(configured["retargeter_overrides"]["temporal_ik"], selected["temporal_ik"])
        self.assertFalse(configured["retargeter_overrides"]["enable_post_processing"])
        self.assertFalse(configured["retargeter_overrides"]["enable_contact_processing"])
        configured["retargeter_overrides"]["temporal_ik"]["weight"] = 99
        self.assertEqual(selected["temporal_ik"]["weight"], 0.3)
        self.assertNotIn("temporal_ik", baseline["retargeter_overrides"])

    def test_native_summary_keeps_metrics_and_guard_phases_without_full_history(self):
        history = [{"aggregate_metrics": {"tracking_loss": 1.0, "missing_metric": float("nan")},
                    "parameter_update": {"clip_guard_active_mode": "gate"}},
                   {"aggregate_metrics": {"tracking_loss": 0.8},
                    "parameter_update": {"clip_guard_active_mode": "penalty"}}]
        summary = BATCH.native_summary({"history": history, "motions": ["one.bvh"]}, {"outer_iterations": 50})
        self.assertEqual(summary["initial_metrics"], {"tracking_loss": 1.0})
        self.assertEqual(summary["final_metrics"], {"tracking_loss": 0.8})
        self.assertEqual(summary["active_guard_modes"], ["gate", "penalty"])
        self.assertNotIn("history", summary)
        json.dumps(summary, allow_nan=False)

    def test_manifest_rejects_duplicate_stems_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            for filenames in (["walk.bvh", "walk.bvh"], ["../walk.bvh"], ["/walk.bvh"], ["walk.csv"]):
                with self.subTest(filenames=filenames):
                    manifest.write_text("\n".join(json.dumps({"output_filename": name}) for name in filenames))
                    with self.assertRaises(ValueError):
                        list(BATCH.source_rows(manifest, directory))

    def test_bvh_metadata_preserves_source_sample_rate_and_rejects_invalid_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "motion.bvh"
            for interval in ("0.008333333333333333", "0.008333", "0.033333", "0", "nan"):
                with self.subTest(interval=interval):
                    source.write_text(f"HIERARCHY\nMOTION\n  Frames: 12\n\tFrame Time: {interval}\n")
                    if interval not in ("0", "nan"):
                        metadata = BATCH.bvh_metadata(source)
                        self.assertEqual(metadata["frames"], 12)
                        self.assertEqual(metadata["fps"], round(1.0 / float(interval), 2))
                        self.assertEqual(metadata["source_frame_time"], float(interval))
                        self.assertEqual(metadata["source_fps"], 1.0 / float(interval))
                    else:
                        with self.assertRaises(ValueError):
                            BATCH.bvh_metadata(source)


if __name__ == "__main__":
    unittest.main()
