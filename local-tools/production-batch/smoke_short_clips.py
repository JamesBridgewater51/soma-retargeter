#!/usr/bin/env python3
"""Exercise mixed one-, two-, three-frame optimization and full-post CSV export."""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch
from smoke_batched_parity import assert_equivalent, fullpost_parity, trimmed_bvh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--storage-root", default="/data/liuminghao")
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--conversion-overrides", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.provenance_file = []
    os.environ["STORAGE_ROOT"] = args.storage_root
    settings = batch.prepare_settings(args)
    import warp as wp
    from app.tools import ik_weight_optimizer as native
    from batched_optimizer import IndependentBatchOptimizer
    from soma_retargeter.io.bvh import load_bvh
    wp.init()
    wp.set_device(args.device)
    config = copy.deepcopy(settings["optimizer"])
    config["outer_iterations"] = 1
    config["ik_weight_line_search"] = {"enabled": False}
    config.setdefault("clip_guard", {})["mode"] = "off"
    settings["optimizer"] = config
    with tempfile.TemporaryDirectory(prefix="short-clip-parity-") as directory:
        root = Path(directory)
        paths = [root / f"clip-{frames}.bvh" for frames in (1, 2, 3)]
        for index, path in enumerate(paths):
            trimmed_bvh(args.motion, path, index * 8, index + 1)
        with tempfile.TemporaryFile(mode="w+") as transcript:
            try:
                with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                    references = []
                    for path in paths:
                        singleton = copy.deepcopy(config)
                        singleton.update(input_folder=str(path), output_folder=str(root / path.stem))
                        references.append(native.run_optimizer(singleton, batch.REPO))
                    skeleton, first = load_bvh(str(paths[0]))
                    animations = [first] + [load_bvh(str(path), skeleton)[1] for path in paths[1:]]
                    workspace = native.OptimizerConfigWorkspace("soma", "agibot_x2ultra", root / "batch")
                    optimizer = IndependentBatchOptimizer(skeleton, animations, [path.name for path in paths], settings, workspace)
                    configs = optimizer.run()
                    evidence = [assert_equivalent(candidate, state, reference)
                                for candidate, state, reference in zip(configs, optimizer.states, references, strict=True)]
                    exported = fullpost_parity(skeleton, animations, configs, settings)
            except Exception:
                transcript.seek(0)
                print(transcript.read()[-32768:], flush=True)
                raise
    print(json.dumps({"boundary_parity": "PASS", "source_lengths": [1, 2, 3], "device": args.device,
                      "outer_iterations": 1, "scenario": "force-accept", "clips": evidence,
                      "fullpost": exported}), flush=True)


if __name__ == "__main__":
    main()
