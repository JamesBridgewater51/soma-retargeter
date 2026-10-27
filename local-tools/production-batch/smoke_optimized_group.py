#!/usr/bin/env python3
"""Exercise fast decoding, retained inputs, immutable geometry and validated stage resume."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch
from run_batched import execute_group, inspect_motion, product_paths
from smoke_batched_parity import trimmed_bvh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--storage-root", default="/data/liuminghao")
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True)
    args = parser.parse_args()
    args.provenance_file = []
    os.environ["STORAGE_ROOT"] = args.storage_root
    settings = batch.prepare_settings(args)
    settings.update(engine="native-independent-batched", reuse_geometry=True, fast_bvh=True)
    settings["optimizer"]["outer_iterations"] = 1
    settings["optimizer"]["ik_weight_line_search"] = {"enabled": False}
    settings["optimizer"].setdefault("clip_guard", {})["mode"] = "off"
    import warp as wp
    wp.init()
    wp.set_device("cpu")
    with tempfile.TemporaryDirectory(prefix="optimized-group-") as directory:
        root = Path(directory)
        for category in ("weights", "output", "logs", "artifacts"):
            settings[category] = str(root / category)
        jobs = []
        for index, frames in enumerate((1, 3, 8)):
            source = root / f"clip-{frames}.bvh"
            trimmed_bvh(args.motion, source, index * 8, frames)
            jobs.append(inspect_motion({"stem": source.stem, "source": str(source),
                "bytes": source.stat().st_size, "sha256": batch.digest(source), "archive_member": source.name}))
        with tempfile.TemporaryFile(mode="w+") as transcript:
            try:
                with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                    fresh = execute_group(jobs, settings)
                    hashes = [batch.digest(product_paths(job, settings)["csv"]) for job in jobs]
                    resumed = execute_group(jobs, settings)
                    if fresh["resumed"] != 0 or resumed["resumed"] != len(jobs):
                        raise AssertionError("Optimized group failed stage resume validation")
                    if hashes != [batch.digest(product_paths(job, settings)["csv"]) for job in jobs]:
                        raise AssertionError("Validated resume rewrote completed CSVs")
            except Exception:
                transcript.seek(0)
                print(transcript.read()[-32768:], flush=True)
                raise
    print(json.dumps({"optimized_group": "PASS", "fast_bvh": True, "reuse_geometry": True,
                      "source_lengths": [1, 3, 8], "fresh_resumed": fresh["resumed"],
                      "second_run_resumed": resumed["resumed"], "device": "cpu",
                      "outer_iterations": 1, "scenario": "force-accept"}), flush=True)


if __name__ == "__main__":
    main()
