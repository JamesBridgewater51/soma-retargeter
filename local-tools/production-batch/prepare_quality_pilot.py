#!/usr/bin/env python3
"""Prepare the fixed eleven-motion 4/8-prefix pilot from the audited full manifest."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/data/liuminghao")
    names = {row["name"] for row in batch.load_json(root / "datasets/bones-seed/metadata/x2-validation-selection.json")}
    rows = [json.loads(line) for line in (root / "artifacts/soma-retargeter/x2-production/full-input-manifest.jsonl").open()]
    selected = [row for row in rows if Path(row["output_filename"]).stem in names]
    if len(selected) != 11 or sum(row["frames"] for row in selected) != 7255:
        raise ValueError("Selected pilot must contain exactly eleven unique clips and 7,255 source frames")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "selected11.jsonl"
    with manifest.open("x") as stream:
        for row in selected:
            stream.write(json.dumps(row) + "\n")
    config = batch.load_json(Path(__file__).with_name("optimizer-native50.json"))
    config["outer_iterations"] = 8
    batch.atomic_json(args.output_root / "optimizer-budget8.json", config)
    batch.atomic_json(args.output_root / "temporal-selected.json", batch.load_json(args.selected_config))
    print(json.dumps({"event": "QUALITY_PILOT_INPUTS", "clips": len(selected), "frames": 7255,
                      "outer_budget": 8, "prefixes": [4, 8]}), flush=True)


if __name__ == "__main__":
    main()
