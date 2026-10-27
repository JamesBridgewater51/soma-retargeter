#!/usr/bin/env python3
"""Write launch commands without allocating GPUs or choosing the production budget."""
import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch

IMAGE = "registry.pjlab.org.cn/ccr-ailab-image/cuda-13.2-python-3.12-pytorch-2.12-nccl-2.29-ubuntu22.04:latest"
ALLOWED = {"host-10-140-62-36", "host-10-140-108-112"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--placement", action="append", required=True, help="Authorized hostname:CPUs:GPU-worker-count")
    parser.add_argument("--search-batch", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    optimizer = batch.load_json(args.optimizer_config)
    temporal = batch.load_json(args.temporal_config)
    if int(optimizer.get("outer_iterations", 0)) < 4:
        raise ValueError("Full production requires an explicitly validated budget >=4; budget2 is benchmark-only")
    placements = []
    for specification in args.placement:
        hostname, cpus, count = specification.split(":")
        cpus, count = int(cpus), int(count)
        if hostname not in ALLOWED or cpus not in (4, 6) or not 1 <= count <= 5:
            raise ValueError("Invalid authorized fragmented-node placement")
        placements.extend([(hostname, cpus)] * count)
    commands = []
    for index, (hostname, cpus) in enumerate(placements):
        command = ["clusterx", "run", "--job-name", f"x2-full-s{index}", "--num-nodes", "1",
                   "--gpus-per-task", "1", "--cpus-per-task", str(cpus), "--memory-per-task", "64",
                   "--shm-size-gib", "8", "--include", hostname, "--image", IMAGE, "--no-env", "--retry", "0",
                   "bash", str(args.repo / "local-tools/production-batch/production_worker.sh"),
                   str(args.repo), str(args.optimizer_config), str(args.temporal_config),
                   str(index), str(len(placements)), str(args.search_batch)]
        commands.append({"shard_index": index, "shard_count": len(placements), "hostname": hostname,
                         "cpus": cpus, "gpus": 1, "memory_gib": 64, "command": shlex.join(command)})
    packet = {"submitted": False, "optimizer": optimizer, "temporal": temporal,
              "optimizer_sha256": batch.digest(args.optimizer_config), "temporal_sha256": batch.digest(args.temporal_config),
              "expected_sequences": 142220, "expected_frames": 124554605, "final_batch_size": 128,
              "manifest": "/data/liuminghao/artifacts/soma-retargeter/x2-production/full-input-manifest.jsonl",
              "manifest_sha256": "e552d0fc5fb81848dadb0e4265926e179a01df915da491c5a0bbd0c5b22d7055",
              "commands": commands,
              "note": "Recheck free GPU/CPU capacity immediately before executing; this file allocates nothing."}
    batch.atomic_json(args.output, packet)
    print(json.dumps({"launch_packet": str(args.output), "workers": len(commands), "submitted": False}), flush=True)


if __name__ == "__main__":
    main()
