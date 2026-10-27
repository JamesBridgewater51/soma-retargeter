#!/usr/bin/env python3
"""Compare optimized-I/O results with previously validated fresh native outputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline-weights", "weights", "baseline-csv", "csv", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    maximum_translation = maximum_angle = 0.0
    for row in rows:
        stem = Path(row["output_filename"]).stem
        expected = batch.load_json(args.baseline_weights / stem / "retargeter.json")
        actual = batch.load_json(args.weights / stem / "retargeter.json")
        if expected != actual:
            raise AssertionError(f"Independent optimized configuration changed: {stem}")
        baseline_path, actual_path = args.baseline_csv / f"{stem}.csv", args.csv / f"{stem}.csv"
        with baseline_path.open() as stream:
            baseline_header = stream.readline()
        with actual_path.open() as stream:
            actual_header = stream.readline()
        if baseline_header != actual_header:
            raise AssertionError(f"CSV schema changed: {stem}")
        expected_csv = np.loadtxt(baseline_path, delimiter=",", skiprows=1, ndmin=2)
        actual_csv = np.loadtxt(actual_path, delimiter=",", skiprows=1, ndmin=2)
        if expected_csv.shape != actual_csv.shape or not np.array_equal(expected_csv[:, 0], actual_csv[:, 0]):
            raise AssertionError(f"CSV frame layout changed: {stem}")
        translation = np.max(np.abs(expected_csv[:, 1:4] - actual_csv[:, 1:4]))
        angle = np.max(np.abs((expected_csv[:, 4:] - actual_csv[:, 4:] + 180) % 360 - 180))
        maximum_translation = max(maximum_translation, float(translation))
        maximum_angle = max(maximum_angle, float(angle))
        if translation > 1e-4 or angle > 1e-3:
            raise AssertionError(f"CSV pose changed beyond 1e-4cm/1e-3degree: {stem}")
    report = {"native_equivalence": "PASS", "clips": len(rows), "weight_configurations": "exact",
              "maximum_translation_difference_cm": maximum_translation,
              "maximum_circular_angle_difference_degrees": maximum_angle}
    batch.atomic_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
