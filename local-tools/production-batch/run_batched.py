#!/usr/bin/env python3
"""Length-bucketed independent singleton optimization using batched native IK."""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


def inspect_motion(job):
    source = Path(job["source"])
    if source.stat().st_size != job["bytes"]:
        raise ValueError(f"Source size differs from manifest: {source}")
    hierarchy = hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            if line.strip() == b"MOTION":
                break
            hierarchy.update(line)
    return {**job, **batch.bvh_metadata(source), "hierarchy_sha256": hierarchy.hexdigest()}


def motion_groups(selected, args):
    buckets = defaultdict(list)
    for job in selected:
        inspected = inspect_motion(job)
        hierarchy = "certified-soma-x2-decoded" if getattr(args, "decoded_bvh", False) else inspected["hierarchy_sha256"]
        key = (hierarchy, (inspected["frames"] - 1) // args.length_bucket_frames)
        pending = buckets[key]
        maximum_frames = max([inspected["frames"], *(row["frames"] for row in pending)])
        if pending and (len(pending) + 1) * maximum_frames > args.max_batch_frames:
            yield buckets.pop(key)
            pending = buckets[key]
        pending.append(inspected)
        if len(pending) == args.batch_size:
            yield buckets.pop(key)
    for key in sorted(buckets):
        yield buckets[key]


def product_paths(job, settings):
    weights = Path(settings["weights"]) / job["stem"]
    return {"directory": weights, "config": weights / "retargeter.json",
            "optimized": weights / "optimized.done.json",
            "csv": Path(settings["output"]) / f"{job['stem']}.csv",
            "done": Path(settings["artifacts"]) / "done" / f"{job['stem']}.json"}


def optimizer_receipt(job, settings):
    paths = product_paths(job, settings)
    return batch.receipt_valid(paths["optimized"], batch.optimization_key(job, settings), paths["config"])


def completed_receipt(job, settings):
    optimized = optimizer_receipt(job, settings)
    if not optimized:
        return None
    paths = product_paths(job, settings)
    done = batch.receipt_valid(paths["done"], batch.conversion_key_for(optimized, settings), paths["csv"])
    if done:
        try:
            checked = batch.validate_csv(paths["csv"], done["headers"], job["frames"], batch.velocity_plan(settings, job["fps"]))
            if done.get("validation") != checked:
                return None
            if done["frames"] != job["frames"] or not math.isclose(done["fps"], job["fps"], rel_tol=1e-8):
                return None
            return done
        except (ValueError, KeyError, OSError):
            return None
    return None


def load_animations(jobs, settings, prepared=None):
    if settings.get("fast_bvh", False):
        from fast_bvh import load_bvh
    else:
        from soma_retargeter.io.bvh import load_bvh
    if prepared is None:
        skeleton, first = load_bvh(jobs[0]["source"])
        cached = {jobs[0]["stem"]: first}
    else:
        skeleton, cached = prepared
    certification = None
    if settings.get("decoded_bvh", False):
        from decoded_bvh import certify_soma_x2_topology, load_decoded_bvh
        certification = certify_soma_x2_topology(skeleton)
    for job in jobs:
        if job["stem"] not in cached:
            if certification is None:
                cached[job["stem"]] = load_bvh(job["source"], skeleton)[1]
            else:
                cached[job["stem"]] = load_decoded_bvh(
                    job["source"], skeleton,
                    decoder="fast" if settings.get("fast_bvh", False) else "native",
                    certification=certification)[1]
    animations = [cached[job["stem"]] for job in jobs]
    for job, animation in zip(jobs, animations, strict=True):
        if not math.isclose(float(animation.sample_rate), job["fps"], rel_tol=1e-8):
            raise ValueError("Native BVH sample-rate rounding contract changed")
    return skeleton, animations


def fit_missing(jobs, settings):
    from app.tools.ik_weight_optimizer import OptimizerConfigWorkspace
    from batched_optimizer import IndependentBatchOptimizer
    gate_started = time.monotonic()
    missing = [job for job in jobs if not optimizer_receipt(job, settings)]
    if not missing:
        return {"evaluations": [], "timings": {"resume_scan_seconds": time.monotonic() - gate_started}, "prepared": None}
    started = time.monotonic()
    skeleton, animations = load_animations(missing, settings)
    print(json.dumps({"event": "BVH_LOAD_PROFILE", "clips": len(missing),
                      "seconds": time.monotonic() - started}), flush=True)
    workspace = OptimizerConfigWorkspace("soma", "agibot_x2ultra", Path(settings["weights"]))
    optimizer = IndependentBatchOptimizer(skeleton, animations, [job["stem"] for job in missing], settings, workspace)
    search_started = time.monotonic()
    configs = optimizer.run()
    searched = time.monotonic()
    elapsed = searched - started
    batch.verify_protected_files(settings)
    for job, config, summary in zip(missing, configs, optimizer.summaries(), strict=True):
        paths = product_paths(job, settings)
        batch.verify_weights_only(config, settings["base_retargeter"])
        batch.atomic_json(paths["config"], config)
        batch.atomic_json(paths["directory"] / "native-summary.json", summary)
        batch.atomic_json(paths["optimized"], {
            "content_key": batch.optimization_key(job, settings), "sha256": batch.digest(paths["config"]),
            "source_sha256": job["sha256"], "source": job["source"], "batch_wall_seconds": elapsed,
            "batch_clips": len(missing), "outer_iterations": settings["optimizer"]["outer_iterations"],
            "engine": settings["engine"]})
    for budget, prefix_configs in optimizer.parameter_snapshots.items():
        for index, (job, prefix_config) in enumerate(zip(missing, prefix_configs, strict=True)):
            batch.atomic_json(product_paths(job, settings)["directory"] / "snapshots" / f"budget-{budget}.json",
                              {"budget": budget, "source_sha256": job["sha256"], "retargeter_config": prefix_config,
                               **optimizer.snapshot_statistics[budget][index]})
    timings = {"resume_scan_seconds": started - gate_started,
               "bvh_and_optimizer_setup_seconds": search_started - started,
               "search_seconds": searched - search_started,
               "weights_publish_seconds": time.monotonic() - searched,
               "optimizer_setup": optimizer.setup_timings}
    print(json.dumps({"event": "FIT_PROFILE", **timings}), flush=True)
    return {"evaluations": optimizer.evaluation_timings, "timings": timings,
            "prepared": (skeleton, {job["stem"]: animation for job, animation in zip(missing, animations, strict=True)})}


def export_buffer(job, settings, buffer, headers):
    from soma_retargeter.io.csv import save_csv
    paths = product_paths(job, settings)
    if buffer.num_frames != job["frames"] or not math.isclose(float(buffer.sample_rate), job["fps"], rel_tol=1e-6):
        raise ValueError(f"Final frame/FPS mismatch: {job['stem']}")
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["csv"].with_name(f".{paths['csv'].name}.{uuid.uuid4().hex}.tmp")
    try:
        save_csv(str(temporary), headers[len(batch.ROOT_HEADER):], buffer)
        checked = batch.validate_csv(temporary, headers, job["frames"], batch.velocity_plan(settings, job["fps"]))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, paths["csv"])
    finally:
        temporary.unlink(missing_ok=True)
    optimized = optimizer_receipt(job, settings)
    if not optimized:
        raise ValueError("No valid independent optimization receipt at export")
    batch.atomic_json(paths["done"], {
        "content_key": batch.conversion_key_for(optimized, settings), "sha256": batch.digest(paths["csv"]),
        "optimization_key": optimized["content_key"], "weights_sha256": optimized["sha256"],
        "source_sha256": job["sha256"], "archive_member": job["archive_member"],
        "headers": headers, "frames": job["frames"], "fps": job["fps"],
        "source_fps": job["source_fps"], "source_frame_time": job["source_frame_time"],
        "engine": settings["engine"], "final_batch_size": settings.get("final_batch_size", 128), "validation": checked})


def convert_missing(jobs, settings, prepared=None):
    cap = settings.get("final_batch_size", 128)
    if len(jobs) > cap:
        started = time.monotonic()
        for offset in range(0, len(jobs), cap):
            convert_missing(jobs[offset:offset + cap], settings, prepared)
            gc.collect()
        return time.monotonic() - started
    import newton
    import warp as wp
    from batched_optimizer import PerSequencePipeline
    from soma_retargeter.utils.newton_utils import get_filtered_joint_names
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    gate_started = time.monotonic()
    missing = [job for job in jobs if not completed_receipt(job, settings)]
    if not missing:
        return 0.0
    configs = [batch.load_json(product_paths(job, settings)["config"]) for job in missing]
    for config in configs:
        batch.verify_weights_only(config, settings["base_retargeter"])
    full_config = copy.deepcopy(settings["base_retargeter"])
    full_config.update(copy.deepcopy(settings["conversion_overrides"]))
    for flag in ("enable_post_processing", "enable_contact_processing"):
        if full_config.get(flag, True) is not True:
            raise ValueError(f"Full conversion requires {flag}=true")
    load_started = time.monotonic()
    skeleton, animations = load_animations(missing, settings, prepared)
    loaded = time.monotonic()
    pipeline = PerSequencePipeline(skeleton, full_config, configs)
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    started = time.monotonic()
    pipeline.add_input_motions(animations, [transform] * len(animations), scale_animation=True)
    prepared = time.monotonic()
    buffers = pipeline.execute()
    executed = time.monotonic()
    headers = batch.ROOT_HEADER + get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    if len(headers) != 38 or len(set(headers)) != 38:
        raise ValueError("X2 Ultra export requires 38 unique CSV columns")
    if len(buffers) != len(missing):
        raise ValueError("Batched full conversion output count mismatch")
    batch.verify_protected_files(settings)
    for job, buffer in zip(missing, buffers, strict=True):
        export_buffer(job, settings, buffer, headers)
    finished = time.monotonic()
    print(json.dumps({"event": "CONVERSION_PROFILE", "receipt_config_seconds": load_started - gate_started,
                      "bvh_load_seconds": loaded - load_started, "pipeline_prepare_seconds": prepared - loaded,
                      "execute_seconds": executed - prepared, "publish_seconds": finished - executed}), flush=True)
    return finished - started


def execute_group(jobs, settings):
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        for job in sorted(jobs, key=lambda row: row["stem"]):
            directory = product_paths(job, settings)["directory"]
            directory.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((directory / ".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch.verify_protected_files(settings)
        for job in jobs:
            if batch.digest(job["source"]) != job["sha256"]:
                raise ValueError(f"Source content differs from manifest: {job['source']}")
        resumed = sum(bool(completed_receipt(job, settings)) for job in jobs)
        print(json.dumps({"event": "BATCH_START", "clips": len(jobs), "resumed": resumed,
                          "frames": sum(job["frames"] for job in jobs)}), flush=True)
        gated = time.monotonic()
        fitted = fit_missing(jobs, settings)
        conversion_seconds = convert_missing(jobs, settings, fitted["prepared"])
    return {"clips": len(jobs), "frames": sum(job["frames"] for job in jobs), "resumed": resumed,
            "wall_seconds": time.monotonic() - started, "conversion_seconds": conversion_seconds,
            "gate_seconds": gated - started, "fit_profile": fitted["timings"],
            "evaluations": fitted["evaluations"]}


def shard_batches(groups, shard_index, shard_count):
    return [group for index, group in enumerate(groups) if index % shard_count == shard_index]


def run(args):
    os.environ["STORAGE_ROOT"] = str(Path(args.storage_root).resolve())
    root = Path(args.storage_root)
    args.provenance_file = []
    settings = batch.prepare_settings(args)
    settings["engine"] = "native-independent-batched"
    settings["reuse_geometry"] = args.reuse_geometry
    settings["fast_bvh"] = args.fast_bvh
    settings["decoded_bvh"] = args.decoded_bvh
    settings["final_batch_size"] = args.final_batch_size
    settings["max_batch_frames"] = args.max_batch_frames
    settings["snapshot_iterations"] = [int(token) for token in args.snapshot_iterations.split(",") if token]
    if any(budget < 1 or budget > settings["optimizer"]["outer_iterations"] for budget in settings["snapshot_iterations"]):
        raise ValueError("Snapshot prefixes must be within the explicitly configured optimizer budget")
    for key, override in (("output", args.output_root), ("weights", args.weights_root),
                          ("logs", args.log_root), ("artifacts", args.artifact_root)):
        if override:
            settings[key] = str(Path(override).resolve())
    rows = list(batch.source_rows(args.manifest or root / "datasets/bones-seed/metadata/soma-uniform-full/extracted-files.jsonl",
                                  args.source_root or root / "datasets/bones-seed/raw/soma-uniform/full"))
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} source entries, got {len(rows)}")
    selected = rows if args.shard_mode == "batch" else [
        job for job in rows if job["index"] % args.shard_count == args.shard_index
    ]
    if args.motion:
        selected = [job for job in selected if job["stem"] in set(args.motion)]
        if {job["stem"] for job in selected} != set(args.motion):
            raise ValueError("Not every explicitly selected motion is present in this shard")
    if args.limit:
        selected = selected[:args.limit]
    groups = motion_groups(selected, args)
    if args.shard_mode == "batch":
        groups = shard_batches(groups, args.shard_index, args.shard_count)
        selected = [job for group in groups for job in group]
    settings["shard_mode"] = args.shard_mode
    artifact_root = Path(settings["artifacts"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_id = f"batched-shard-{args.shard_index}-of-{args.shard_count}-{uuid.uuid4().hex[:12]}"
    batch.atomic_json(artifact_root / f"{run_id}.settings.json", settings)
    batch.initialize_worker(args.gpu, settings["optimizer"]["extra_robot_paths"])
    started = time.monotonic()
    completed = resumed = failures = frames = 0
    log_path = Path(settings["logs"]) / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        for group_index, group in enumerate(groups):
            outcome = None
            for attempt in range(args.retries + 1):
                try:
                    outcome = execute_group(group, settings)
                    break
                except Exception:
                    log.write(json.dumps({"group": group_index, "attempt": attempt + 1,
                                          "error": traceback.format_exc()}) + "\n")
            if outcome is None:
                failures += len(group)
                continue
            completed += outcome["clips"]
            resumed += outcome["resumed"]
            frames += outcome["frames"]
            elapsed = time.monotonic() - started
            progress = {"run_id": run_id, "group": group_index, "completed": completed, "total": len(selected),
                        "failed": failures, "resumed": resumed, "frames": frames, "wall_seconds": elapsed,
                        "clips_per_second": completed / elapsed,
                        "eta_seconds": (len(selected) - completed - failures) * elapsed / completed,
                        "batch": outcome}
            log.write(json.dumps(progress, allow_nan=False) + "\n")
            batch.atomic_json(artifact_root / f"{run_id}.progress.json", progress)
            print(json.dumps(progress), flush=True)
    return int(failures > 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default=os.environ.get("STORAGE_ROOT", "/data/liuminghao"))
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--manifest")
    parser.add_argument("--source-root")
    parser.add_argument("--expected-count", type=int, default=142220)
    parser.add_argument("--motion", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-frames", type=int, default=4000000,
                        help="Bound padded search frames while allowing large same-length cohorts")
    parser.add_argument("--final-batch-size", type=int, default=128,
                        help="Maximum independent full-post conversion environments; search batch is unchanged")
    parser.add_argument("--fast-bvh", action="store_true", help="Use bit-exact vectorized native BVH decoding")
    parser.add_argument("--decoded-bvh", action="store_true",
                        help="Own-decode and certify exact native SOMA topology conformance before batching")
    parser.add_argument("--reuse-geometry", action="store_true",
                        help="Reuse immutable post-OFF model geometry; keep all per-probe solver/history state fresh")
    parser.add_argument("--length-bucket-frames", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-mode", choices=("sequence", "batch"), default="sequence",
                        help="Assign whole global cohorts to workers instead of scattering each cohort")
    parser.add_argument("--snapshot-iterations", default="", help="Explicit native-search prefix budgets to retain for pilot quality comparisons")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.final_batch_size < 1 or args.max_batch_frames < 1 or args.length_bucket_frames < 1 or args.limit < 0 or args.retries < 0
            or not 0 <= args.shard_index < args.shard_count):
        parser.error("Invalid batch, bucket, limit, retry or shard arguments")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
