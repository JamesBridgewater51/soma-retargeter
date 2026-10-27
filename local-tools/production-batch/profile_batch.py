#!/usr/bin/env python3
"""Run one batch case with sampled RSS/GPU high-water marks and bounded termination."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


def rss_gib(pid):
    try:
        fields = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return 0.0
    peaks = [float(line.split()[1]) / 1024 ** 2 for line in fields if line.startswith(("VmHWM:", "VmRSS:"))]
    return max(peaks, default=0.0)


def gpu_gib():
    measured = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5, check=True)
    rows = measured.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("Memory guard requires exactly one visible allocated GPU")
    return float(rows[0]) / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, default=Path(__file__).with_name("run_batched.py"))
    parser.add_argument("--rss-limit-gib", type=float, default=56.0)
    parser.add_argument("--gpu-limit-gib", type=float, default=72.0)
    args, runner_args = parser.parse_known_args()
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    entrypoint = args.entrypoint.resolve()
    if entrypoint.parent != Path(__file__).resolve().parent:
        raise ValueError("Profiling entrypoint must be a trusted sibling tool")
    command = [sys.executable, "-I", str(entrypoint), *runner_args]
    started = time.monotonic()
    peak_rss = peak_gpu = 0.0
    stopped_for_memory = False
    with subprocess.Popen(command) as process:
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, rss_gib(process.pid))
                peak_gpu = max(peak_gpu, gpu_gib())
                if peak_rss >= args.rss_limit_gib or peak_gpu >= args.gpu_limit_gib:
                    stopped_for_memory = True
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(1)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        returncode = process.wait()
    peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2)
    stopped_for_memory = stopped_for_memory or peak_rss >= args.rss_limit_gib or peak_gpu >= args.gpu_limit_gib
    report = {"returncode": returncode, "stopped_for_memory": stopped_for_memory,
              "peak_rss_gib": peak_rss, "sampled_peak_gpu_gib": peak_gpu,
              "rss_limit_gib": args.rss_limit_gib, "gpu_limit_gib": args.gpu_limit_gib,
              "wall_seconds_including_startup": time.monotonic() - started,
              "sampling_interval_seconds": 1,
              "note": "RSS includes kernel process high-water mark; GPU is sampled whole allocated device, not an exact peak."}
    batch.atomic_json(args.profile_path, report)
    print(json.dumps({"event": "MEMORY_PROFILE", **report}), flush=True)
    return 88 if stopped_for_memory else returncode


if __name__ == "__main__":
    raise SystemExit(main())
