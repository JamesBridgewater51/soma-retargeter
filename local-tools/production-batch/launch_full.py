#!/usr/bin/env python3
"""Prepare, dispatch, and inspect frozen full-corpus X2 worker packets."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

MANIFEST_SHA256 = "e552d0fc5fb81848dadb0e4265926e179a01df915da491c5a0bbd0c5b22d7055"
QUEUE = "queue-t-reserved-d"
IMAGE = "registry.pjlab.org.cn/ccr-ailab-image/cuda-13.2-python-3.12-pytorch-2.12-nccl-2.29-ubuntu22.04:latest"
PLACEMENTS = (("ecp-tcluster-a844", "host-10-140-62-36", 4),
              ("ecp-tcluster-ab45", "host-10-140-108-112", 6))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def platform_query(expression):
    program = ("import json; from clusterx.launcher.ssp.ssp import SSPCluster; "
               "client=SSPCluster().client; " + expression)
    queried = subprocess.run(["/usr/bin/python3.12", "-I", "-c", program],
                             check=True, capture_output=True, text=True, timeout=65)
    return json.loads(queried.stdout)


def available_fragments():
    nodes = platform_query("print(json.dumps(client.list_queue_nodes('cluster-t', "
                           "'queue-t-reserved-d',100).get('nodes',[])))")
    allowed = {node for node, _, _ in PLACEMENTS}
    return {node["name"]: {resource["resource_type"]: int(float(resource["unallocated"]))
                           for resource in node["summary_data"]}
            for node in nodes if node["name"] in allowed}


def worker_plan(capacity):
    workers = []
    for node, hostname, cpus in PLACEMENTS:
        resources = capacity[node]
        count = min(resources["DEVICE"], resources["CPU"] // cpus,
                    resources["MEMORY"] // 64)
        workers.extend({"node": node, "hostname": hostname, "cpus": cpus,
                        "memory_gib": 64} for _ in range(count))
    if not workers:
        raise ValueError("No authorized fragmented GPU fits the worker resources")
    return workers


def protected_code(repo):
    paths = set((repo / "soma_retargeter").rglob("*.py"))
    paths.update((repo / "app").rglob("*.py"))
    paths.update((repo / "local-tools/production-batch").glob("*.py"))
    paths.add(repo / "local-tools/production-batch/cluster_worker.sh")
    paths.add(repo / "uv.lock")
    return {str(path): digest(path) for path in sorted(paths)}


def assert_packet_inputs(packet):
    expected = {packet["manifest"]: MANIFEST_SHA256, **packet["protected_sha256"]}
    for filename, checksum in expected.items():
        if digest(filename) != checksum:
            raise ValueError(f"Frozen launch input changed: {filename}")
    if len(packet["workers"]) != packet["shard_count"]:
        raise ValueError("Packet shard count differs from worker count")


def runner_arguments(packet, index):
    paths = packet["paths"]
    command = [packet["python"], "-I", packet["runner"], "--storage-root", packet["storage_root"],
               "--manifest", packet["manifest"], "--source-root", packet["source_root"],
               "--expected-count", "142220", "--optimizer-config", packet["optimizer_config"],
               "--conversion-overrides", packet["conversion_overrides"],
               "--batch-size", str(packet["batch_size"]), "--final-batch-size", str(packet["final_batch_size"]),
               "--length-bucket-frames", "128", "--shard-mode", "batch",
               "--shard-count", str(packet["shard_count"]), "--shard-index", str(index),
               "--gpu", "0", "--retries", "0"]
    for option, key in (("--output-root", "output"), ("--weights-root", "weights"),
                        ("--log-root", "logs"), ("--artifact-root", "artifacts")):
        command.extend([option, paths[key]])
    command.extend(packet["runner_flags"])
    return command


def prepare_packet(args):
    repo, root = args.repo.resolve(), args.storage_root.resolve()
    manifest = root / "artifacts/soma-retargeter/x2-production/full-input-manifest.jsonl"
    optimizer, conversion = args.optimizer_config.resolve(), args.conversion_overrides.resolve()
    if load_json(optimizer).get("outer_iterations") != 4:
        raise ValueError("Full production requires the explicitly selected native four-step budget")
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("Full input manifest differs from the verified corpus")
    workers = worker_plan(available_fragments())
    packet = packet_description(args, repo, root, workers)
    packet.update(manifest=str(manifest), optimizer_config=str(optimizer), conversion_overrides=str(conversion))
    packet["protected_sha256"] = protected_code(repo)
    packet["protected_sha256"].update({str(path): digest(path) for path in (optimizer, conversion)})
    if args.packet.exists():
        raise FileExistsError(args.packet)
    atomic_json(args.packet, packet)
    print(json.dumps({"event": "PACKET_READY", "packet": str(args.packet), "workers": workers}), flush=True)


def packet_description(args, repo, root, workers):
    token = uuid.uuid4().hex[:8]
    for index, worker in enumerate(workers):
        worker.update(index=index, job_name=f"x2-full4-{token}-{index:02d}")
    return {"schema": 1, "created_at": timestamp(), "repo": str(repo), "storage_root": str(root),
            "manifest_sha256": MANIFEST_SHA256, "expected_sequences": 142220,
            "expected_frames": 124554605, "source_root": str(root / "datasets/bones-seed/raw/soma-uniform/full"),
            "python": str(root / "agibot-x2/soma-retargeter/.venv/bin/python"),
            "runner": str(repo / "local-tools/production-batch/run_batched.py"),
            "worker_script": str(repo / "local-tools/production-batch/cluster_worker.sh"),
            "shard_count": len(workers), "workers": workers, "batch_size": args.batch_size,
            "final_batch_size": args.final_batch_size, "runner_flags": args.runner_flag,
            "paths": {"output": str(args.output_root.resolve()), "weights": str(args.weights_root.resolve()),
                      "logs": str(args.log_root.resolve()), "artifacts": str(args.artifact_root.resolve())}}


def submission_arguments(packet, packet_path, worker):
    command = ["clusterx", "run", "--job-name", worker["job_name"], "--num-nodes", "1",
               "--gpus-per-task", "1", "--cpus-per-task", str(worker["cpus"]),
               "--memory-per-task", str(worker["memory_gib"]), "--shm-size-gib", "8",
               "--include", worker["hostname"], "--queue", QUEUE, "--cluster-name", "cluster-t",
               "--priority", "1", "--retry", "0", "--no-env", "--image", IMAGE]
    for key, value in (("STORAGE_ROOT", packet["storage_root"]), ("OMP_NUM_THREADS", "1"),
                       ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("PYTHONUNBUFFERED", "1")):
        command.extend(["--environment", f"{key}={value}"])
    command.extend(["bash", packet["worker_script"], str(packet_path), str(worker["index"])])
    return command


def assert_capacity(workers):
    available = available_fragments()
    required = Counter(worker["node"] for worker in workers)
    for node, count in required.items():
        cpus = sum(worker["cpus"] for worker in workers if worker["node"] == node)
        if (available[node]["DEVICE"] < count or available[node]["CPU"] < cpus
                or available[node]["MEMORY"] < count * 64):
            raise ValueError(f"Prepared worker plan no longer fits {node}: {available[node]}")


def dispatch(args, packet):
    if not args.freeze_ack:
        raise ValueError("Submission requires the producer/root freeze acknowledgement")
    assert_packet_inputs(packet)
    receipt_path = args.packet.with_suffix(".dispatch.json")
    receipt = load_json(receipt_path) if receipt_path.exists() else {"freeze_ack": args.freeze_ack, "jobs": []}
    submitted = {job["job_name"] for job in receipt["jobs"] if job["accepted"]}
    pending = [worker for worker in packet["workers"] if worker["job_name"] not in submitted]
    assert_capacity(pending)
    for worker in pending:
        dispatch_one(packet, args.packet, worker, receipt)
        atomic_json(receipt_path, receipt)
        if not receipt["jobs"][-1]["accepted"]:
            raise RuntimeError(f"Submission failed; see {receipt_path}")
    print(json.dumps({"event": "DISPATCH_COMPLETE", **receipt}), flush=True)


def dispatch_one(packet, packet_path, worker, receipt):
    command = submission_arguments(packet, packet_path, worker)
    submitted = subprocess.run(command, capture_output=True, text=True, timeout=90)
    directory = Path(packet["paths"]["logs"]) / "submissions"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{worker['job_name']}.log"
    log_path.write_text(submitted.stdout + submitted.stderr)
    receipt["jobs"].append({**worker, "job_id": worker["job_name"], "submitted_at": timestamp(),
                            "accepted": submitted.returncode == 0, "returncode": submitted.returncode,
                            "submission_log": str(log_path)})
    print(json.dumps({"event": "JOB_SUBMISSION", **receipt["jobs"][-1]}), flush=True)


def run_worker(args, packet):
    assert_packet_inputs(packet)
    worker = packet["workers"][args.shard_index]
    if worker["index"] != args.shard_index:
        raise ValueError("Worker index differs from packet ordering")
    environment = dict(os.environ, STORAGE_ROOT=packet["storage_root"], PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    status_path = Path(packet["paths"]["artifacts"]) / f"worker-{args.shard_index}.status.json"
    started = time.monotonic()
    atomic_json(status_path, {"state": "started", "worker": worker, "started_at": timestamp()})
    print(json.dumps({"event": "REAL_WORKER_START", "worker": worker, "time": timestamp()}), flush=True)
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True)
        returncode = subprocess.run(runner_arguments(packet, args.shard_index),
                                    cwd=packet["repo"], env=environment).returncode
        failure = None
    except (OSError, subprocess.CalledProcessError) as error:
        returncode, failure = 1, str(error)
    status = {"state": "completed" if returncode == 0 else "failed", "worker": worker,
              "returncode": returncode, "error": failure, "finished_at": timestamp(),
              "seconds": time.monotonic() - started}
    atomic_json(status_path, status)
    print(json.dumps({"event": "REAL_WORKER_FINISH", **status}), flush=True)
    return returncode


def show_status(packet):
    root = Path(packet["paths"]["artifacts"])
    statuses = [load_json(path) for path in sorted(root.glob("worker-*.status.json"))]
    progress = [load_json(path) for path in sorted(root.glob("*.progress.json"), key=lambda path: path.stat().st_mtime)]
    latest = {}
    for row in progress:
        shard = row["run_id"].split("-of-", 1)[0].removeprefix("batched-shard-")
        latest[shard] = row
    print(json.dumps({"event": "FULL_PROGRESS", "time": timestamp(), "expected_sequences": 142220,
                      "workers": statuses, "latest_shards": latest,
                      "completed": sum(row["completed"] for row in latest.values()),
                      "failed": sum(row["failed"] for row in latest.values()),
                      "reported_eta_seconds": max((row["eta_seconds"] for row in latest.values()), default=None)}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "worker", "status"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--storage-root", type=Path, default=Path("/data/liuminghao"))
    parser.add_argument("--optimizer-config", type=Path)
    parser.add_argument("--conversion-overrides", type=Path)
    for name in ("output", "weights", "log", "artifact"):
        parser.add_argument(f"--{name}-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--final-batch-size", type=int, default=128)
    parser.add_argument("--runner-flag", action="append", default=[])
    parser.add_argument("--freeze-ack")
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        required = (args.repo, args.optimizer_config, args.conversion_overrides,
                    args.output_root, args.weights_root, args.log_root, args.artifact_root)
        if not all(required) or min(args.batch_size, args.final_batch_size) < 1:
            parser.error("Prepare requires repository, both configs, all output roots, and positive batch sizes")
    return args


def main():
    args = parse_args()
    args.packet = args.packet.resolve()
    if args.action == "prepare":
        prepare_packet(args)
        return 0
    packet = load_json(args.packet)
    if args.action == "worker":
        return run_worker(args, packet)
    if args.action == "status":
        show_status(packet)
        return 0
    with args.packet.with_suffix(".dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch(args, packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
