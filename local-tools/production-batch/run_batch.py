#!/usr/bin/env python3
"""Independent native IK optimization and validated CSV export for every BVH."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import contextlib
import copy
import csv
import fcntl
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT_HEADER = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
               "root_rotateX", "root_rotateY", "root_rotateZ"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key_for(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def optimization_key(job, settings):
    return key_for({"schema": 1, "source_sha256": job["sha256"],
                    "optimizer": settings["optimizer"], "provenance": settings["provenance"],
                    "engine": settings.get("engine", "native-singleton"),
                    "reuse_geometry": settings.get("reuse_geometry", False),
                    "fast_bvh": settings.get("fast_bvh", False)})


def conversion_key_for(optimized, settings):
    return key_for({"optimized": optimized["sha256"], "optimization_key": optimized["content_key"],
                    "conversion": settings["conversion_overrides"], "provenance": settings["provenance"],
                    "final_batch_size": settings.get("final_batch_size", 128)})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def bvh_metadata(path):
    frames = fps = None
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("Frames:"):
                frames = int(line.split(":", 1)[1])
            elif line.startswith("Frame Time:"):
                dt = float(line.split(":", 1)[1])
                if not math.isfinite(dt) or dt <= 0:
                    raise ValueError("Invalid BVH sample interval")
                fps = 1.0 / dt
                break
    if frames is None or frames < 1 or fps is None:
        raise ValueError(f"Missing BVH frames/FPS: {path}")
    # Native BVH loading quantizes sample rates to two decimals; retain the header timing too.
    return {"frames": frames, "fps": round(fps, 2), "source_fps": fps, "source_frame_time": dt}


def velocity_plan(settings, fps):
    configured = settings.get("velocity_validation")
    return {**configured, "native_fps": fps} if configured else None


def validate_csv(path, headers, frames, velocity=None):
    accumulator = None
    if velocity:
        from csv_velocity import VelocityAccumulator
        names = headers[len(ROOT_HEADER):]
        if len(names) != 31 or len(set(names)) != 31:
            raise ValueError("Hard-velocity CSV must contain 31 uniquely named actuated joints")
        accumulator = VelocityAccumulator(names, velocity)
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != headers:
            raise ValueError(f"Unexpected CSV headers: {path}")
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != len(headers):
                raise ValueError(f"Invalid CSV width at row {count}")
            values = [float(item) for item in row]
            if not all(math.isfinite(item) for item in values) or values[0] != count - 1:
                raise ValueError(f"Nonfinite data or invalid frame index at row {count}")
            if accumulator is not None:
                accumulator.observe_angles(values[len(ROOT_HEADER):])
    if count != frames:
        raise ValueError(f"CSV has {count} frames, expected {frames}")
    report = accumulator.finish() if accumulator is not None else {
        "enabled": False, "violation_count": 0, "max_ratio": None, "checked_pairs": 0}
    return {"schema": 2, "frames": count, "columns": len(headers), "velocity": report}


def receipt_valid(path, expected_key, product):
    try:
        receipt = load_json(path)
        return receipt if (receipt["content_key"] == expected_key
                           and receipt["sha256"] == digest(product)) else None
    except (OSError, ValueError, KeyError):
        return None


def source_rows(manifest, source_root):
    seen = set()
    with Path(manifest).open() as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            filename = item["output_filename"]
            if Path(filename).name != filename or Path(filename).suffix.lower() != ".bvh":
                raise ValueError(f"Unsafe/non-BVH manifest filename: {filename!r}")
            stem = Path(filename).stem
            if stem in seen:
                raise ValueError(f"Duplicate flattened output stem: {stem}")
            seen.add(stem)
            yield {**item, "index": index, "stem": stem,
                   "source": str(Path(source_root) / filename)}


def verified_module_path(module):
    filename = Path(module.__file__).resolve()
    if not filename.is_relative_to(REPO.resolve()):
        raise RuntimeError(f"Refusing imported code outside isolated repository: {module.__name__} -> {filename}")
    return str(filename)


def verify_local_imports():
    sys.path.insert(0, str(REPO))
    return {name: verified_module_path(importlib.import_module(name)) for name in (
        "soma_retargeter", "soma_retargeter.robotics.robot_registry")}


def initialize_worker(gpu, extra_robot_paths):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    origins = verify_local_imports()
    import warp as wp
    from soma_retargeter.robotics import robot_registry
    wp.init()
    wp.set_device("cuda:0")
    for path in extra_robot_paths:
        robot_registry.register_robots_path(path)
    print(json.dumps({"event": "WORKER_READY", "pid": os.getpid(), "gpu": gpu,
                      "repo": str(REPO), "module_origins": origins,
                      "python": sys.executable, "storage_root": os.environ["STORAGE_ROOT"]}), flush=True)


def optimize_and_convert(job, settings):
    """One lock, one BVH, one native optimizer, one private conversion pipeline."""
    stem = job["stem"]
    weights_dir = Path(settings["weights"]) / stem
    weights_dir.mkdir(parents=True, exist_ok=True)
    with (weights_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({"event": "SEQUENCE_START", "pid": os.getpid(),
                          "stem": stem, "source": job["source"], "time": time.time()}), flush=True)
        return process_locked(job, settings, weights_dir)


def process_locked(job, settings, weights_dir):
    started = time.monotonic()
    log_path = Path(settings["logs"]) / f"{job['stem']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_protected_files(settings)
            source = Path(job["source"])
            if source.stat().st_size != job["bytes"] or digest(source) != job["sha256"]:
                raise ValueError("Source content differs from extraction manifest")
            metadata = bvh_metadata(source)
            optimize_key = optimization_key(job, settings)
            config_path = weights_dir / "retargeter.json"
            optimize_receipt = weights_dir / "optimized.done.json"
            optimized = receipt_valid(optimize_receipt, optimize_key, config_path)
            optimize_seconds = 0.0
            if not optimized:
                tick = time.monotonic()
                optimized_config, summary = run_native_singleton(source, weights_dir, settings)
                verify_weights_only(optimized_config, settings["base_retargeter"])
                atomic_json(config_path, optimized_config)
                atomic_json(weights_dir / "native-summary.json", summary)
                optimize_seconds = time.monotonic() - tick
                optimized = {"content_key": optimize_key, "sha256": digest(config_path),
                             "source_sha256": job["sha256"], "source": str(source),
                             "seconds": optimize_seconds, "outer_iterations": settings["optimizer"]["outer_iterations"]}
                atomic_json(optimize_receipt, optimized)
            own_config = load_json(config_path)
            verify_weights_only(own_config, settings["base_retargeter"])
            own_config.update(copy.deepcopy(settings["conversion_overrides"]))
            for flag in ("enable_post_processing", "enable_contact_processing"):
                if own_config.get(flag, True) is not True:
                    raise ValueError(f"Full conversion requires {flag}=true")
            conversion_key = conversion_key_for(optimized, settings)
            output = Path(settings["output"]) / f"{job['stem']}.csv"
            done_path = Path(settings["artifacts"]) / "done" / f"{job['stem']}.json"
            done = receipt_valid(done_path, conversion_key, output)
            convert_seconds = 0.0
            resumed = bool(done)
            if done:
                try:
                    checked = validate_csv(output, done["headers"], metadata["frames"], velocity_plan(settings, metadata["fps"]))
                    if done["frames"] != metadata["frames"] or not math.isclose(done["fps"], metadata["fps"], rel_tol=1e-8):
                        raise ValueError("Resume FPS/frame metadata mismatch")
                    if done.get("validation") != checked:
                        done["validation"] = checked
                        atomic_json(done_path, done)
                except (OSError, KeyError, ValueError):
                    done = None
                    resumed = False
            if not done:
                tick = time.monotonic()
                headers, checked = convert(source, output, own_config,
                                           {**metadata, "velocity_plan": velocity_plan(settings, metadata["fps"])})
                convert_seconds = time.monotonic() - tick
                done = {"content_key": conversion_key, "sha256": digest(output),
                        "optimization_key": optimize_key, "weights_sha256": optimized["sha256"],
                        "source_sha256": job["sha256"], "archive_member": job["archive_member"],
                        "headers": headers, **metadata, "seconds": convert_seconds, "validation": checked}
                atomic_json(done_path, done)
            verify_protected_files(settings)
            return {"stem": job["stem"], "ok": True, "resumed": resumed,
                    "seconds": time.monotonic() - started, "optimize_seconds": optimize_seconds,
                    "convert_seconds": convert_seconds, **metadata}
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            return {"stem": job["stem"], "ok": False, "error": error,
                    "seconds": time.monotonic() - started}


def finite_metrics(metrics):
    return {name: float(number) for name, number in metrics.items()
            if isinstance(number, (int, float)) and math.isfinite(number)}


def native_summary(result, config):
    history = result["history"]
    return {"motions": result["motions"], "outer_iteration_budget": config["outer_iterations"],
            "recorded_iterations": max(0, len(history) - 1),
            "initial_metrics": finite_metrics(history[0]["aggregate_metrics"]),
            "final_metrics": finite_metrics(history[-1]["aggregate_metrics"]),
            "active_guard_modes": sorted({row["parameter_update"].get("clip_guard_active_mode", "unknown")
                                          for row in history}),
            "temporal_ik": config.get("retargeter_overrides", {}).get("temporal_ik"),
            "note": "Compare tracking metrics directly; objectives can change scale across guard modes."}


def run_native_singleton(source, weights_dir, settings):
    from app.tools.ik_weight_optimizer import run_optimizer
    verified_module_path(sys.modules[run_optimizer.__module__])
    config = copy.deepcopy(settings["optimizer"])
    with tempfile.TemporaryDirectory(prefix=".native-", dir=weights_dir) as directory:
        config.update(input_folder=str(source), output_folder=directory)
        with tempfile.TemporaryFile(mode="w+") as transcript:
            try:
                with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                    result = run_optimizer(config, REPO)
            except Exception:
                transcript.seek(0)
                print(transcript.read()[-32768:], flush=True)
                raise
        if result.get("motions") != [str(source.resolve())]:
            raise ValueError("Native optimizer did not optimize exactly this BVH")
        return load_json(result["optimized_retargeter_config"]), native_summary(result, config)


def verify_protected_files(settings):
    for filename, expected_hash in settings["protected_files"].items():
        if digest(filename) != expected_hash:
            raise ValueError(f"Native retargeter/scaler changed during execution: {filename}")


def verify_weights_only(config, baseline):
    candidate = copy.deepcopy(config)
    reference = copy.deepcopy(baseline)
    if candidate.get("human_robot_scaler_config") != reference.get("human_robot_scaler_config"):
        raise ValueError("Optimizer changed the human-to-robot scaler reference")
    if candidate.get("ik_match_table", {}).keys() != reference["ik_match_table"].keys():
        raise ValueError("Optimizer changed the effector mapping")
    for name, row in candidate["ik_match_table"].items():
        for field in ("t_weight", "r_weight"):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid optimized weight {name}.{field}")
            row[field] = reference["ik_match_table"][name][field]
    if candidate != reference:
        raise ValueError("Optimizer modified something other than IK weights")


def convert(source, output, config, metadata):
    import newton
    import warp as wp
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.io.csv import save_csv
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils.newton_utils import get_filtered_joint_names
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    verified_module_path(sys.modules[SomaRetargetingPipeline.__module__])
    skeleton, animation = load_bvh(str(source))
    if not math.isclose(float(animation.sample_rate), metadata["fps"], rel_tol=1e-8):
        raise ValueError("Native BVH sample-rate rounding contract changed")
    pipeline = SomaRetargetingPipeline(skeleton, "soma", "agibot_x2ultra", retarget_config=config)
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    pipeline.add_input_motions([animation], [transform], scale_animation=True)
    buffers = pipeline.execute()
    if len(buffers) != 1 or buffers[0].num_frames != metadata["frames"]:
        raise ValueError("Conversion returned incorrect buffer/frame count")
    if not math.isclose(float(buffers[0].sample_rate), metadata["fps"], rel_tol=1e-6):
        raise ValueError("Conversion changed source FPS")
    names = get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    if len(names) != 31 or len(set(names)) != 31:
        raise ValueError("X2 Ultra export requires exactly 31 unique actuated joint names")
    headers = ROOT_HEADER + names
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_csv(str(temporary), names, buffers[0])
        checked = validate_csv(temporary, headers, metadata["frames"], metadata.get("velocity_plan"))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return headers, checked


def urdf_velocity_validation(entry, asset_root, temporal_config):
    if not temporal_config.get("hard_velocity", False):
        return None
    if "urdf_path" not in entry.desc or "newton_asset" in entry.desc:
        raise ValueError("Independent hard-velocity validation requires an explicit URDF asset")
    from csv_velocity import load_velocity_contract
    urdf = asset_root / entry.desc["urdf_path"]
    return load_velocity_contract(urdf, velocity_scale=temporal_config.get("velocity_scale", 1.0))


def optimizer_with_selected_temporal(optimizer, conversion_overrides):
    configured = copy.deepcopy(optimizer)
    if "temporal_ik" in conversion_overrides:
        configured.setdefault("retargeter_overrides", {})["temporal_ik"] = copy.deepcopy(conversion_overrides["temporal_ik"])
    return configured


def prepare_settings(args):
    origins = verify_local_imports()
    from soma_retargeter.robotics import robot_registry
    optimizer = load_json(args.optimizer_config)
    if optimizer.get("source_type", "soma") != "soma" or optimizer.get("target_type") != "agibot_x2ultra":
        raise ValueError("Expected SOMA to agibot_x2ultra optimizer configuration")
    optimizer.setdefault("outer_iterations", 50)
    if not isinstance(optimizer["outer_iterations"], int) or optimizer["outer_iterations"] < 1:
        raise ValueError("outer_iterations must be a positive explicit search budget")
    optimizer["plot"] = {"enabled": False, "live": False}
    optimizer["save_probe_diagnostics"] = False
    for key in ("input_folder", "output_folder"):
        optimizer.pop(key, None)
    extra_paths = optimizer.get("extra_robot_paths", [])
    if isinstance(extra_paths, str):
        extra_paths = [extra_paths]
    optimizer["extra_robot_paths"] = extra_paths
    for extra in extra_paths:
        if not Path(extra).is_absolute():
            raise ValueError("extra_robot_paths must be absolute")
        robot_registry.register_robots_path(extra)
    entry = robot_registry.registry.get(optimizer["target_type"])
    asset_root = Path(entry.manifest_dir)
    baseline_path = asset_root / entry.retarget_configs["soma"]
    baseline = load_json(baseline_path)
    provenance = {}
    files = set(asset_root.rglob("*"))
    files.update((REPO / "soma_retargeter").rglob("*.py"))
    files.update((REPO / "app/tools").rglob("*.py"))
    files.update(Path(__file__).resolve().parent.glob("*.py"))
    files.add(REPO / "uv.lock")
    files.add(REPO / "app/bvh_to_csv_converter.py")
    files.update(Path(p) for p in args.provenance_file)
    for path in sorted(files):
        if path.is_file() and "__pycache__" not in path.parts:
            provenance[str(path)] = digest(path)
    root = Path(args.storage_root)
    relative = {
        "output": "datasets/bones-seed/converted/agibot-x2ultra/per-sequence-optimized",
        "weights": "optimizers/soma-retargeter/agibot-x2ultra/full-per-sequence",
        "logs": "logs/soma-retargeter/x2-production",
        "artifacts": "artifacts/soma-retargeter/x2-production",
    }
    overrides = load_json(args.conversion_overrides)
    optimizer = optimizer_with_selected_temporal(optimizer, overrides)
    forbidden = {"ik_match_table", "human_robot_scaler_config", "smooth_joint_filter_weight",
                 "smooth_joint_filter_objective_body_masks"}.intersection(overrides)
    if forbidden:
        raise ValueError(f"Conversion overrides may not replace weights or scaler: {forbidden}")
    return {**{key: str(root / value) for key, value in relative.items()},
            "optimizer": optimizer, "base_retargeter": baseline, "module_origins": origins,
            "conversion_overrides": overrides, "provenance": provenance,
            "velocity_validation": urdf_velocity_validation(
                entry, asset_root, overrides.get("temporal_ik", baseline.get("temporal_ik", {}))),
            "protected_files": {str(path): digest(path) for path in (
                baseline_path, asset_root / baseline["human_robot_scaler_config"])}}


def run(args):
    if (not 0 <= args.shard_index < args.shard_count or args.workers_per_gpu < 1
            or args.retries < 0 or args.limit < 0 or args.expected_count < 1):
        raise ValueError("Invalid shard/worker/retry/count arguments")
    os.environ["STORAGE_ROOT"] = str(Path(args.storage_root).resolve())
    root = Path(args.storage_root)
    args.manifest = args.manifest or str(root / "datasets/bones-seed/metadata/soma-uniform-full/extracted-files.jsonl")
    args.source_root = args.source_root or str(root / "datasets/bones-seed/raw/soma-uniform/full")
    settings = prepare_settings(args)
    rows = list(source_rows(args.manifest, args.source_root))
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} manifest entries, found {len(rows)}")
    jobs = [row for row in rows if row["index"] % args.shard_count == args.shard_index]
    if args.limit:
        jobs = jobs[:args.limit]
    run_id = f"shard-{args.shard_index}-of-{args.shard_count}-{uuid.uuid4().hex[:12]}"
    artifact_root = Path(settings["artifacts"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_root / f"{run_id}.settings.json", settings)
    gpus = args.gpus.split(",")
    if not all(gpu.strip() for gpu in gpus):
        raise ValueError("Empty GPU identifier")
    ctx = multiprocessing.get_context("spawn")
    executors = [ProcessPoolExecutor(max_workers=args.workers_per_gpu, mp_context=ctx,
                                    initializer=initialize_worker,
                                    initargs=(gpu, settings["optimizer"]["extra_robot_paths"])) for gpu in gpus]
    pending = {}
    iterator = iter(jobs)
    completed = failures = resumed = frames = 0
    durations = []
    started = time.monotonic()
    events_path = Path(settings["logs"]) / f"{run_id}.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for executor in executors:
            for _ in range(args.workers_per_gpu):
                job = next(iterator, None)
                if job is not None:
                    pending[executor.submit(optimize_and_convert, job, settings)] = (executor, job, 0)
        with events_path.open("a", buffering=1) as events:
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    executor, job, attempt = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        result = {"stem": job["stem"], "ok": False, "error": traceback.format_exc()}
                    result.update(attempt=attempt + 1, timestamp=time.time())
                    events.write(json.dumps(result, allow_nan=False) + "\n")
                    if not result["ok"] and attempt < args.retries:
                        pending[executor.submit(optimize_and_convert, job, settings)] = (executor, job, attempt + 1)
                        continue
                    completed += 1
                    failures += not result["ok"]
                    resumed += result.get("resumed", False)
                    frames += result.get("frames", 0)
                    if result["ok"] and not result.get("resumed"):
                        durations.append(result["seconds"])
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed
                    progress = {"run_id": run_id, "completed": completed, "total": len(jobs),
                                "failed": failures, "resumed": resumed, "elapsed_seconds": elapsed,
                                "clips_per_second_including_resume": rate,
                                "eta_seconds_including_resume": (len(jobs) - completed) / rate,
                                "frames": frames, "workers": len(gpus) * args.workers_per_gpu,
                                "fresh_clip_seconds_median": statistics.median(durations) if durations else None}
                    atomic_json(artifact_root / f"{run_id}.progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    next_job = next(iterator, None)
                    if next_job is not None:
                        pending[executor.submit(optimize_and_convert, next_job, settings)] = (executor, next_job, 0)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
    elapsed = time.monotonic() - started
    fresh = completed - failures - resumed
    clean_benchmark = fresh > 0 and resumed == 0 and failures == 0
    summary = {"run_id": run_id, "fresh_clips": fresh, "resumed_clips": resumed,
               "failed_clips": failures, "wall_seconds_including_startup": elapsed,
               "workers": len(gpus) * args.workers_per_gpu,
               "fresh_clip_seconds_median": statistics.median(durations) if durations else None,
               "benchmark_unmixed": clean_benchmark,
               "full_dataset_hours_at_measured_rate": len(rows) * elapsed / fresh / 3600 if clean_benchmark else None,
               "required_speedup_for_nine_hours": len(rows) * elapsed / fresh / 32400 if clean_benchmark else None,
               "caveat": "Observed shard/prefix sample, not representative proof or guaranteed GPU scaling."}
    atomic_json(artifact_root / f"{run_id}.summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 1 if failures else 0


def parse_args():
    root = Path(os.environ.get("STORAGE_ROOT", "/data/liuminghao"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default=str(root))
    parser.add_argument("--manifest")
    parser.add_argument("--source-root")
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True,
                        help="Explicit selected temporal variant JSON; use {} only for a deliberately native conversion")
    parser.add_argument("--provenance-file", action="append", default=[], help="Additional external asset or dependency lock file")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical CUDA device IDs or UUIDs")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--expected-count", type=int, default=142220)
    parser.add_argument("--limit", type=int, default=0, help="Smoke/benchmark prefix within this shard; zero means all")
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
