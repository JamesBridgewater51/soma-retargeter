"""Dispatch keeps complete IK cohorts together without losing source coverage."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_batched import shard_batches


class BatchShardingTests(unittest.TestCase):
    def test_workers_partition_whole_cohorts_without_overlap_or_missing_sources(self):
        cohorts = [[0, 1, 2], [3], [4, 5], [6, 7, 8, 9]]
        for workers in (1, 2, 3, 8):
            with self.subTest(workers=workers):
                assigned = [shard_batches(iter(cohorts), rank, workers) for rank in range(workers)]
                flattened = [source for groups in assigned for group in groups for source in group]
                self.assertEqual(sorted(flattened), list(range(10)))
                for rank, groups in enumerate(assigned):
                    self.assertEqual(groups, cohorts[rank::workers])

    def test_empty_global_cohorts_yield_empty_worker_assignments(self):
        self.assertEqual(shard_batches(iter(()), 2, 4), [])


if __name__ == '__main__':
    unittest.main()
