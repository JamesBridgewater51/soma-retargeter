#!/usr/bin/env python3
"""Measure fixed-weight native batching/context throughput, not optimization quality."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import copy
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


def forward_case(request, settings):
    """Return validated forward-pass timings; errors propagate and stop the benchmark."""
    import numpy as np
    import warp as wp
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    batch.verified_module_path(sys.modules[SomaRetargetingPipeline.__module__])
    with tempfile.TemporaryFile(mode="w+") as transcript:
        try:
            with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                started = time.monotonic()
                skeleton, animation = load_bvh(request["motion"])
                animations = [animation] + [load_bvh(request["motion"], skeleton)[1]
                                            for _ in range(request["batch_size"] - 1)]
                config = copy.deepcopy(settings["base_retargeter"])
                config.update(copy.deepcopy(settings["conversion_overrides"]))
                pipeline = SomaRetargetingPipeline(skeleton, "soma", "agibot_x2ultra", retarget_config=config)
                transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
                pipeline.add_input_motions(animations, [transform] * len(animations), scale_animation=True)
                wp.synchronize()
                ready = time.monotonic()
                buffers = pipeline.execute()
                wp.synchronize()
                solved = time.monotonic()
                metadata = batch.bvh_metadata(request["motion"])
                if len(buffers) != request["batch_size"]:
                    raise ValueError("Wrong batch output count")
                for buffer in buffers:
                    if (buffer.num_frames != metadata["frames"]
                            or abs(float(buffer.sample_rate) - metadata["fps"]) > 1e-3
                            or not np.isfinite(buffer.data).all()):
                        raise ValueError("Invalid fixed-pass benchmark output")
                batch.verify_protected_files(settings)
                return {"batch_size": len(buffers), "source_frames": metadata["frames"],
                        "output_frames": metadata["frames"] * len(buffers),
                        "setup_seconds": ready - started, "execute_seconds": solved - ready,
                        "validated_total_seconds": time.monotonic() - started}
        except Exception:
            transcript.seek(0)
            print(transcript.read()[-32768:], flush=True)
            raise


def measure_case(args, settings, concurrency, batch_size):
    repeats = max(args.repeats, concurrency)
    request = {"motion": str(args.motion.resolve()), "batch_size": batch_size}
    started = time.monotonic()
    print(json.dumps({"event": "BENCHMARK_START", "mode": args.mode,
                      "workers": concurrency, "batch_size": batch_size, "repeats": repeats}), flush=True)
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=multiprocessing.get_context("spawn"),
                             initializer=batch.initialize_worker,
                             initargs=(args.gpu, settings["optimizer"]["extra_robot_paths"])) as pool:
        futures = [pool.submit(forward_case, request, settings) for _ in range(repeats)]
        timings = [future.result() for future in futures]
        measured_seconds = time.monotonic() - started
    frames = sum(timing["output_frames"] for timing in timings)
    return {"mode": args.mode, "workers": concurrency, "batch_size": batch_size,
            "repeats": repeats, "frames": frames, "seconds_including_worker_startup": measured_seconds,
            "frames_per_second_including_startup": frames / measured_seconds,
            "full_dataset_fixed_pass_gpu_hours_at_observed_rate": args.dataset_frames / (frames / measured_seconds) / 3600,
            "timings": timings,
            "warning": "Fixed shared weights only: not independently optimized clips; not an optimizer ETA."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--storage-root", default="/data/liuminghao")
    parser.add_argument("--mode", choices=("batch", "contexts"), required=True)
    parser.add_argument("--sizes", default="1,8,32", help="Batch sizes or concurrent process counts")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dataset-frames", type=int, default=124554605)
    args = parser.parse_args()
    sizes = [int(token) for token in args.sizes.split(",")]
    if not sizes or min(sizes) < 1 or args.repeats < 1:
        raise ValueError("Benchmark sizes/repeats must be positive")
    import os
    os.environ["STORAGE_ROOT"] = str(Path(args.storage_root).resolve())
    args.provenance_file = []
    settings = batch.prepare_settings(args)
    evidence = Path(settings["artifacts"]) / "throughput"
    evidence.mkdir(parents=True, exist_ok=True)
    run_id = f"native-{args.mode}-{uuid.uuid4().hex[:12]}"
    batch.atomic_json(evidence / f"{run_id}.settings.json", settings)
    for size in sizes:
        measured = measure_case(args, settings, 1 if args.mode == "batch" else size,
                                size if args.mode == "batch" else 1)
        batch.atomic_json(evidence / f"{run_id}-{size}.json", measured)
        print(json.dumps(measured), flush=True)


if __name__ == "__main__":
    main()
