"""CPU-only native parity and focused malformed-payload regression tests.

Run with the existing project environment and source BVHs as positional arguments.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import warp as wp

import fast_bvh
from soma_retargeter.io.bvh import load_bvh as native_load_bvh


HIERARCHY = '''HIERARCHY
ROOT ns:Root
{
 OFFSET 1 2 3
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Child
 {
  OFFSET 10 20 30
  CHANNELS 3 Yrotation Zrotation Xrotation
  End Site
  {
   OFFSET 40 50 60
  }
 }
}
MOTION
'''


class FastBVHTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'motion.bvh'

    def write_motion(self, frames=3, payload=None):
        if payload is None:
            payload = '\n'.join(' '.join(str(x) for x in row) for row in
                                np.arange(frames * 9).reshape(frames, 9) - 8)
        self.path.write_text(HIERARCHY + f'Frames: {frames}\nFrame Time: 0.0333667\n' + payload + '\n')

    def test_native_exact_for_short_boundaries_and_channel_orders(self):
        for frames in (1, 2, 3):
            with self.subTest(frames=frames):
                self.write_motion(frames)
                expected_skeleton, expected = native_load_bvh(str(self.path))
                actual_skeleton, actual = fast_bvh.load_bvh(str(self.path))
                self.assertEqual(actual_skeleton.joint_names, ['Root', 'Child'])
                np.testing.assert_array_equal(actual_skeleton.parent_indices, expected_skeleton.parent_indices)
                np.testing.assert_array_equal(actual_skeleton.reference_local_transforms, expected_skeleton.reference_local_transforms)
                np.testing.assert_array_equal(actual.local_transforms, expected.local_transforms)
                self.assertEqual(actual.sample_rate, expected.sample_rate)
                self.assertEqual(actual.num_frames, frames)

    def test_existing_skeleton_conformance_and_header_reuse(self):
        self.write_motion()
        skeleton, expected = native_load_bvh(str(self.path))
        _, first = fast_bvh.load_bvh(str(self.path))
        other = self.path.with_name('other.bvh')
        other.write_text(self.path.read_text())
        reused, _ = fast_bvh.load_bvh(str(other))
        self.assertIsNot(reused, first.skeleton)
        np.testing.assert_array_equal(reused.reference_local_transforms, first.skeleton.reference_local_transforms)
        returned, actual = fast_bvh.load_bvh(str(other), skeleton)
        self.assertIs(returned, skeleton)
        self.assertIs(actual.skeleton, skeleton)
        np.testing.assert_array_equal(actual.local_transforms, expected.local_transforms)

    def test_returned_skeleton_mutations_do_not_contaminate_later_loads(self):
        self.write_motion()
        expected_skeleton, expected = native_load_bvh(str(self.path))
        first_skeleton, _ = fast_bvh.load_bvh(str(self.path))
        first_skeleton.joint_names[0] = 'mutated'
        first_skeleton.parent_indices[1] = -1
        first_skeleton._reference_local_transforms[:] = 0
        first_skeleton.up_axis = wp.vec3(1, 0, 0)
        actual_skeleton, actual = fast_bvh.load_bvh(str(self.path))
        self.assertEqual(actual_skeleton.joint_names, expected_skeleton.joint_names)
        np.testing.assert_array_equal(actual_skeleton.parent_indices, expected_skeleton.parent_indices)
        np.testing.assert_array_equal(actual_skeleton.reference_local_transforms, expected_skeleton.reference_local_transforms)
        np.testing.assert_array_equal(actual_skeleton.up_axis, expected_skeleton.up_axis)
        np.testing.assert_array_equal(actual.local_transforms, expected.local_transforms)

    def test_changed_offsets_do_not_reuse_previous_header(self):
        self.write_motion()
        first_skeleton, _ = fast_bvh.load_bvh(str(self.path))
        self.path.write_text(self.path.read_text().replace('OFFSET 10 20 30', 'OFFSET 40 50 60'))
        expected_skeleton, expected = native_load_bvh(str(self.path))
        actual_skeleton, actual = fast_bvh.load_bvh(str(self.path))
        self.assertIsNot(actual_skeleton, first_skeleton)
        np.testing.assert_array_equal(actual_skeleton.reference_local_transforms, expected_skeleton.reference_local_transforms)
        np.testing.assert_array_equal(actual.local_transforms, expected.local_transforms)

    def test_rejects_zero_frames(self):
        self.write_motion(0)
        with self.assertRaises(ValueError):
            fast_bvh.load_bvh(str(self.path))

    def test_rejects_bad_count_nonfinite_and_trailing_junk(self):
        for payload in ('0 ' * 8, '0 ' * 10, 'nan ' + '0 ' * 8,
                        'inf ' + '0 ' * 8, '1e100 ' + '0 ' * 8,
                        '0 ' * 9 + 'junk'):
            with self.subTest(payload=payload):
                self.write_motion(1, payload)
                with self.assertRaises(ValueError):
                    fast_bvh.load_bvh(str(self.path))


def compare_sources(paths, repeats):
    reports = []
    for path in paths:
        native_times, fast_times = [], []
        breakdown = []
        for _ in range(repeats):
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                native_skeleton, native = native_load_bvh(str(path))
            native_times.append(time.perf_counter() - started)
            timing = {}
            started = time.perf_counter()
            skeleton, actual = fast_bvh.load_bvh(str(path), timings=timing)
            fast_times.append(time.perf_counter() - started)
            breakdown.append(timing)
            np.testing.assert_array_equal(skeleton.parent_indices, native_skeleton.parent_indices)
            np.testing.assert_array_equal(skeleton.reference_local_transforms, native_skeleton.reference_local_transforms)
            assert skeleton.joint_names == native_skeleton.joint_names
            assert actual.sample_rate == native.sample_rate
            assert actual.num_frames == native.num_frames
            np.testing.assert_array_equal(actual.local_transforms, native.local_transforms)
            assert actual.local_transforms.tobytes() == native.local_transforms.tobytes()
        difference = np.abs(actual.local_transforms - native.local_transforms)
        report = dict(path=str(path), frames=actual.num_frames, joints=skeleton.num_joints,
                      sample_rate=actual.sample_rate, max_absolute_difference=float(difference.max()),
                      exact_quaternion_values=bool(np.array_equal(actual.local_transforms[..., 3:], native.local_transforms[..., 3:])),
                      bitwise_identical=actual.local_transforms.tobytes() == native.local_transforms.tobytes(),
                      native_seconds=native_times, fast_seconds=fast_times, fast_breakdown=breakdown)
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='*', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    wp.init()
    wp.set_device('cpu')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FastBVHTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    compare_sources(args.sources, args.repeats)
