import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batched as runtime
from offload_cohorts import (Grouping, eligible_whole_cohort, file_sha256, read_verified_manifest,
                             reconstruct_groups, validate_whole_proposal)


def source(name, frames=10):
    return {"output_filename": f"{name}.bvh", "frames": frames, "sha256": "a" * 64}


def proposal(groups, ranks, identifiers):
    motions = [{"stem": row["stem"], "frames": row["frames"], "source_sha256": row["sha256"],
                "global_group": index, "original_rank": ranks[index]}
               for index in identifiers for row in groups[index]]
    owners = sorted({ranks[index] for index in identifiers})
    claim = owners[0] if len(owners) == 1 else "multi-" + "-".join(map(str, owners))
    return {"whole_global_groups": identifiers, "original_rank": claim, "motions": motions,
            "clips": len(motions), "frames": sum(row["frames"] for row in motions)}


class WholeCohortTests(unittest.TestCase):
    def setUp(self):
        self.groups, self.ranks = reconstruct_groups(
            runtime, [source(name) for name in ("a", "b", "c", "d", "e")],
            Grouping(batch_size=2, max_batch_frames=1000, length_bucket_frames=128, shard_count=2))

    def test_real_runtime_whole_union_and_shard_rank_are_accepted(self):
        valid = proposal(self.groups, self.ranks, [0, 2])
        report = validate_whole_proposal(valid, self.groups, self.ranks)
        self.assertEqual(report["clips"], 3)
        self.assertEqual(report["original_ranks"], [0])

    def test_partial_cohort_with_correct_source_fields_and_adjusted_totals_is_rejected(self):
        invalid = proposal(self.groups, self.ranks, [0])
        invalid["motions"] = invalid["motions"][:1]
        invalid.update(clips=1, frames=10)
        with self.assertRaisesRegex(ValueError, "exact union.*whole cohorts"):
            validate_whole_proposal(invalid, self.groups, self.ranks)

    def test_forged_cohort_and_rank_claims_are_rejected(self):
        for mutation in ("top-rank", "motion-rank", "motion-group", "claimed-group", "duplicate-group"):
            with self.subTest(mutation=mutation):
                invalid = proposal(self.groups, self.ranks, [0])
                if mutation == "top-rank":
                    invalid["original_rank"] = 1
                elif mutation == "motion-rank":
                    invalid["motions"][0]["original_rank"] = 1
                elif mutation == "motion-group":
                    invalid["motions"][0]["global_group"] = 1
                elif mutation == "claimed-group":
                    invalid["whole_global_groups"] = [1]
                else:
                    invalid["whole_global_groups"] = [0, 0]
                with self.assertRaises(ValueError):
                    validate_whole_proposal(invalid, self.groups, self.ranks)

    def test_extra_motion_from_unclaimed_cohort_is_rejected(self):
        invalid = proposal(self.groups, self.ranks, [0])
        invalid["motions"].append(proposal(self.groups, self.ranks, [1])["motions"][0])
        invalid.update(clips=3, frames=30)
        with self.assertRaisesRegex(ValueError, "exact union"):
            validate_whole_proposal(invalid, self.groups, self.ranks)

    def test_actual_runtime_uses_padded_frame_cap_and_leftover_dispatch(self):
        groups, ranks = reconstruct_groups(runtime, [source("a", 10), source("b", 11), source("c", 10)],
            Grouping(batch_size=3, max_batch_frames=32, length_bucket_frames=128, shard_count=2))
        self.assertEqual([[row["stem"] for row in group] for group in groups], [["a", "b"], ["c"]])
        self.assertEqual(ranks, {0: 0, 1: 1})
        invalid = proposal(groups, ranks, [0])
        invalid["motions"].pop()
        invalid.update(clips=1, frames=10)
        with self.assertRaisesRegex(ValueError, "whole cohorts"):
            validate_whole_proposal(invalid, groups, ranks)

    def test_any_existing_lock_receipt_or_short_member_excludes_entire_cohort(self):
        for exclusion in ("lock", "done", "short"):
            with self.subTest(exclusion=exclusion), tempfile.TemporaryDirectory() as directory:
                group = copy.deepcopy(self.groups[0])
                completed = set()
                if exclusion == "lock":
                    root = Path(directory) / "a"
                    root.mkdir()
                    (root / ".lock").write_text("")
                elif exclusion == "done":
                    completed.add("a")
                else:
                    group[0]["frames"] = 1
                self.assertFalse(eligible_whole_cohort(group, completed, directory, 10))

    def test_header_metadata_cannot_be_replaced_without_matching_verified_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps(source("a", 10)) + "\n")
            expected = file_sha256(path)
            self.assertEqual(read_verified_manifest(path, expected)[0]["frames"], 10)
            path.write_text(json.dumps(source("a", 11)) + "\n")
            with self.assertRaisesRegex(ValueError, "verified production input"):
                read_verified_manifest(path, expected)

    def test_untouched_whole_cohort_is_eligible_without_creating_claim_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(eligible_whole_cohort(self.groups[0], set(), directory, 10))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
