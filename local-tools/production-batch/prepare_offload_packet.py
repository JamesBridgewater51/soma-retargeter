#!/usr/bin/env python3
"""Prepare a no-allocation straggler helper packet against immutable production settings."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offload_cohorts import (FROZEN_GROUPING, file_sha256, load_frozen_runtime,
                             read_verified_manifest, reconstruct_groups, validate_whole_proposal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--helper-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.helper_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in args.helper_id):
        raise ValueError("Helper ID must be a simple unique identifier")
    os.environ["STORAGE_ROOT"] = "/data/liuminghao"
    tools = args.repo.resolve() / "local-tools/production-batch"
    specification = importlib.util.spec_from_file_location("frozen_offload_batch", tools / "run_batch.py")
    frozen = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(frozen)
    proposal = frozen.load_json(args.proposal)
    artifact_root = Path("/data/liuminghao/artifacts/soma-retargeter/x2-production")
    manifest = artifact_root / "full-input-manifest.jsonl"
    source_rows = read_verified_manifest(manifest)
    runtime = load_frozen_runtime(args.repo)
    cohorts, cohort_ranks = reconstruct_groups(runtime, source_rows)
    cohort_proof = validate_whole_proposal(proposal, cohorts, cohort_ranks)
    all_rows = {Path(row["output_filename"]).stem: row for row in source_rows}
    chosen = [all_rows[row["stem"]] for row in proposal["motions"]]
    if len(chosen) != proposal["clips"] or len({row["output_filename"] for row in chosen}) != len(chosen):
        raise ValueError("Proposal contains duplicate or missing source motions")
    for expected, row in zip(proposal["motions"], chosen, strict=True):
        if row["sha256"] != expected["source_sha256"] or row["frames"] != expected["frames"]:
            raise ValueError("Proposal differs from the authoritative full manifest")
    optimizer = tools / "optimizer-selected.json"
    temporal = tools / "temporal-selected.json"
    helper = frozen.prepare_settings(SimpleNamespace(optimizer_config=optimizer,
        conversion_overrides=temporal, storage_root="/data/liuminghao", provenance_file=[]))
    helper.update(engine="native-independent-batched", fast_bvh=True, decoded_bvh=True,
                  reuse_geometry=True, final_batch_size=128, max_batch_frames=4000000, snapshot_iterations=[])
    main_paths = sorted(artifact_root.glob("*-of-10-*.settings.json"))
    if len(main_paths) != 10:
        raise ValueError(f"Expected ten actual main settings files, found {len(main_paths)}")
    main_settings = [frozen.load_json(path) for path in main_paths]
    runtime_digest = file_sha256(runtime.__file__)
    for captured in main_settings:
        if captured["provenance"].get(str(Path(runtime.__file__).resolve())) != runtime_digest:
            raise ValueError("Grouping implementation differs from actual production provenance")
        if not captured.get("decoded_bvh") or captured.get("max_batch_frames") != FROZEN_GROUPING.max_batch_frames:
            raise ValueError("Captured worker grouping differs from the supported frozen production run")
    reference = main_settings[0]
    for key in ("output", "weights", "artifacts"):
        helper[key] = reference[key]
    helper["logs"] = str(Path(reference["logs"]) / "offload-helpers" / args.helper_id)
    parity_checks = 0
    for row in chosen:
        expected_key = frozen.optimization_key(row, helper)
        symbolic = {"content_key": expected_key, "sha256": "symbolic-same-uncomputed-optimized-config"}
        for main_settings_row in main_settings:
            if expected_key != frozen.optimization_key(row, main_settings_row):
                raise ValueError("Helper optimization key differs from an actual main worker")
            if frozen.conversion_key_for(symbolic, helper) != frozen.conversion_key_for(symbolic, main_settings_row):
                raise ValueError("Changing helper logs would change the conversion key")
            parity_checks += 1
    locks = [str(Path(helper["weights"]) / row["stem"] / ".lock") for row in proposal["motions"]]
    claimed = [path for path in locks if Path(path).exists()]
    python = "/data/liuminghao/agibot-x2/soma-retargeter/.venv/bin/python"
    argv = [python, "-I", str(tools / "run_batched.py"), "--storage-root", "/data/liuminghao",
            "--manifest", str(manifest), "--source-root", "/data/liuminghao/datasets/bones-seed/raw/soma-uniform/full",
            "--expected-count", "142220", "--optimizer-config", str(optimizer), "--conversion-overrides", str(temporal),
            "--shard-mode", "batch", "--shard-index", "0", "--shard-count", "1",
            "--batch-size", str(len(chosen)), "--length-bucket-frames", "32768", "--max-batch-frames", "4000000",
            "--final-batch-size", "128", "--gpu", "0", "--retries", "1",
            "--fast-bvh", "--decoded-bvh", "--reuse-geometry",
            "--output-root", helper["output"], "--weights-root", helper["weights"],
            "--artifact-root", helper["artifacts"], "--log-root", helper["logs"]]
    for row in proposal["motions"]:
        argv.extend(["--motion", row["stem"]])
    report = {"schema": 1, "kind": "unsubmitted-direct-frozen-helper", "allocated": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "helper_id": args.helper_id,
              "repo": str(args.repo.resolve()), "proposal": str(args.proposal),
              "whole_original_groups": proposal["whole_global_groups"], "original_rank": proposal["original_rank"],
              "clips": len(chosen), "frames": sum(row["frames"] for row in chosen),
              "cohort_membership": {**cohort_proof, "runtime_sha256": runtime_digest,
                                    "grouping": vars(FROZEN_GROUPING)},
              "optimization_and_conversion_settings_checks": parity_checks,
              "conversion_parity_method": "Same symbolic optimized hash supplied to both key functions; not a weight receipt",
              "claim_files_present_at_prepare": claimed, "ready_if_gpu_free_and_claims_unchanged": not claimed,
              "main_settings_files": [str(path) for path in main_paths], "helper_scientific_settings": helper,
              "argv": argv, "shell_command": shlex.join(argv),
              "stdout": str(Path(helper["logs"]) / "worker.stdout.log"),
              "exit_receipt": str(artifact_root / "operations" / args.helper_id / "exit.json"),
              "fresh_launch_gate": ["Verify an actually free authorized GPU and sufficient CPU/RAM",
                  "Revalidate every whole-cohort member remains without done receipt or existing lock",
                  "Recheck frozen settings keys before actual producer acquires locks",
                  "Use a detached supervisor; never main cluster_worker worker-0 status paths",
                  "Exclude helper 0-of-1 progress from main 0..9-of-10 ETA; count physical receipts once"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"packet": str(args.output), "allocated": False, "key_checks": parity_checks,
                      "clips": len(chosen), "existing_claims": len(claimed)}), flush=True)


if __name__ == "__main__":
    main()
