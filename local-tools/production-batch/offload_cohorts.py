"""Archival ops gates for exact immutable production cohort membership."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace


MANIFEST_SHA256 = "e552d0fc5fb81848dadb0e4265926e179a01df915da491c5a0bbd0c5b22d7055"


@dataclass(frozen=True)
class Grouping:
    batch_size: int = 2048
    max_batch_frames: int = 4000000
    length_bucket_frames: int = 128
    shard_count: int = 10


FROZEN_GROUPING = Grouping()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_verified_manifest(path, expected_sha256=MANIFEST_SHA256):
    if file_sha256(path) != expected_sha256:
        raise ValueError("Offload manifest differs from the verified production input")
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream]
    names = [row["output_filename"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Production input contains duplicate source filenames")
    return rows


def load_frozen_runtime(repo):
    tools = Path(repo).resolve() / "local-tools/production-batch"
    sys.path.insert(0, str(tools))
    specification = importlib.util.spec_from_file_location("offload_frozen_runtime", tools / "run_batched.py")
    runtime = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runtime)
    return runtime


def _manifest_metadata(row):
    if type(row.get("frames")) is not int or row["frames"] < 1:
        raise ValueError("Verified source metadata needs a positive integer frame count")
    return row


def reconstruct_groups(runtime, rows, grouping=FROZEN_GROUPING):
    # Reuse the exact frozen grouping bytecode without re-reading all BVH headers.
    # Only its metadata reader is replaced in a private globals copy, never in the live module.
    namespace = dict(runtime.motion_groups.__globals__, inspect_motion=_manifest_metadata)
    original = runtime.motion_groups
    metadata_groups = FunctionType(original.__code__, namespace, original.__name__,
                                   original.__defaults__, original.__closure__)
    args = SimpleNamespace(batch_size=grouping.batch_size, max_batch_frames=grouping.max_batch_frames,
                           length_bucket_frames=grouping.length_bucket_frames, decoded_bvh=True)
    prepared = [{**row, "stem": Path(row["output_filename"]).stem} for row in rows]
    groups = list(metadata_groups(prepared, args))
    identifiers = {tuple(row["stem"] for row in group): index for index, group in enumerate(groups)}
    ranks = {}
    for rank in range(grouping.shard_count):
        for group in runtime.shard_batches(groups, rank, grouping.shard_count):
            index = identifiers[tuple(row["stem"] for row in group)]
            if index in ranks:
                raise ValueError("Frozen sharding assigned a cohort more than once")
            ranks[index] = rank
    if len(ranks) != len(groups):
        raise ValueError("Frozen sharding omitted a source cohort")
    return groups, ranks


def _claimed_ranks(claim):
    if type(claim) is int:
        return {claim}
    if isinstance(claim, str) and claim.startswith("multi-"):
        try:
            return {int(rank) for rank in claim[6:].split("-")}
        except ValueError as error:
            raise ValueError("Invalid multi-rank claim") from error
    raise ValueError("Proposal must explicitly identify original rank(s)")


def validate_whole_proposal(proposal, groups, ranks):
    claimed = proposal.get("whole_global_groups")
    if not isinstance(claimed, list) or not claimed or any(type(index) is not int for index in claimed):
        raise ValueError("Proposal needs explicit integer whole-cohort IDs")
    if len(set(claimed)) != len(claimed) or any(index < 0 or index >= len(groups) for index in claimed):
        raise ValueError("Duplicate or unknown whole-cohort ID")
    expected = {row["stem"]: (index, row) for index in claimed for row in groups[index]}
    motions = proposal.get("motions", [])
    selected = [motion.get("stem") for motion in motions]
    if len(set(selected)) != len(selected) or set(selected) != set(expected):
        raise ValueError("Selected motions must equal the exact union of claimed whole cohorts")
    actual_ranks = {ranks[index] for index in claimed}
    if _claimed_ranks(proposal.get("original_rank")) != actual_ranks:
        raise ValueError("Claimed original rank differs from frozen cohort sharding")
    for motion in motions:
        index, source = expected[motion["stem"]]
        if (type(motion.get("global_group")) is not int or type(motion.get("original_rank")) is not int
                or motion["global_group"] != index or motion["original_rank"] != ranks[index]):
            raise ValueError("Motion cohort/rank claim differs from frozen production membership")
        if motion.get("source_sha256") != source["sha256"] or motion.get("frames") != source["frames"]:
            raise ValueError("Motion differs from verified production source metadata")
    if proposal.get("clips") != len(motions) or proposal.get("frames") != sum(row["frames"] for _, row in expected.values()):
        raise ValueError("Proposal coverage totals differ from its complete source cohorts")
    return {"whole_cohort_membership_verified": True, "global_groups": sorted(claimed),
            "original_ranks": sorted(actual_ranks), "clips": len(motions)}


def eligible_whole_cohort(group, completed, weights, minimum_frames):
    if not group or any(row["frames"] < minimum_frames for row in group):
        return False
    for row in group:
        stem = row["stem"]
        if stem in completed or (Path(weights) / stem / ".lock").exists():
            return False
    return True
