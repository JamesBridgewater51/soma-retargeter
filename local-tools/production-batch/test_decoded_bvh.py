"""Real CPU native/fast decode tests; require the normal SOMA runtime assets."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import warp as wp

import decoded_bvh
from soma_retargeter.io.bvh import BVHImporter, load_bvh as native_load_bvh
from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.pipelines import utils as pipeline_utils


def _motion_fixture(skeleton, changed_rest=False):
    """Build a small real BVH using runtime names/topology, not stored source motion."""
    children = [[] for _ in skeleton.joint_names]
    for index, parent in enumerate(skeleton.parent_indices):
        if parent >= 0:
            children[int(parent)].append(index)
    lines = ['HIERARCHY']
    offsets = skeleton.reference_local_transforms[:, :3].copy() * 100.0
    if changed_rest:
        offsets[1, 0] += 12.5

    def joint(index, depth):
        indent = ' ' * depth
        lines.extend([f'{indent}{"ROOT" if index == 0 else "JOINT"} {skeleton.joint_names[index]}', indent + '{'])
        lines.append(indent + ' OFFSET ' + ' '.join(str(float(component)) for component in offsets[index]))
        channels = '6 Xposition Yposition Zposition Zrotation Xrotation Yrotation' if index == 0 else '3 Zrotation Xrotation Yrotation'
        lines.append(indent + ' CHANNELS ' + channels)
        for child in children[index]:
            joint(child, depth + 1)
        lines.append(indent + '}')

    joint(0, 0)
    channels = 6 + 3 * (skeleton.num_joints - 1)
    lines.extend(['MOTION', 'Frames: 3', 'Frame Time: 0.008333'])
    for frame in range(3):
        values = np.arange(channels, dtype=float) * (frame + 1) * 0.01
        values[:3] = [frame, 0, 100]
        lines.append(' '.join(str(component) for component in values))
    return '\n'.join(lines) + '\n'


class DecodedBVHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device('cpu')
        source = pipeline_utils.get_source_type_from_str('soma')
        cls.layout, _ = BVHImporter().create_skeleton(pipeline_utils.get_source_zero_pose_asset_path(source))

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.paths = [Path(directory.name) / 'canonical.bvh', Path(directory.name) / 'different-rest.bvh']
        for path, changed in zip(self.paths, (False, True)):
            path.write_text(_motion_fixture(self.layout, changed))

    def test_own_reference_decode_preserves_all_values_and_metadata_in_both_orders(self):
        expected = [native_load_bvh(str(path)) for path in self.paths]
        self.assertFalse(np.array_equal(expected[0][0].reference_local_transforms,
                                         expected[1][0].reference_local_transforms))
        self.assertFalse(np.array_equal(expected[0][1].local_transforms, expected[1][1].local_transforms))
        for decoder in ('native', 'fast'):
            for order in ((0, 1), (1, 0)):
                with self.subTest(decoder=decoder, order=order):
                    canonical, first = decoded_bvh.load_decoded_bvh(self.paths[order[0]], decoder=decoder)
                    certification = decoded_bvh.certify_soma_x2_topology(canonical)
                    returned, second = decoded_bvh.load_decoded_bvh(
                        self.paths[order[1]], canonical, decoder=decoder, certification=certification)
                    self.assertIs(returned, canonical)
                    self.assertIs(second.skeleton, canonical)
                    for actual, index in zip((first, second), order):
                        reference = expected[index][1]
                        self.assertEqual(actual.local_transforms.tobytes(), reference.local_transforms.tobytes())
                        self.assertEqual(actual.sample_rate, reference.sample_rate)
                        self.assertEqual(actual.num_frames, reference.num_frames)
                        self.assertEqual(actual.skeleton.joint_names, reference.skeleton.joint_names)
                        self.assertTrue(self.paths[index].is_file())

    def test_uncertified_or_mutated_canonical_topology_is_rejected(self):
        canonical, _ = decoded_bvh.load_decoded_bvh(self.paths[0])
        with self.assertRaisesRegex(ValueError, 'certification'):
            decoded_bvh.load_decoded_bvh(self.paths[1], canonical)
        certification = decoded_bvh.certify_soma_x2_topology(canonical)
        canonical.parent_indices[1] = -1
        with self.assertRaisesRegex(ValueError, 'current canonical'):
            decoded_bvh.load_decoded_bvh(self.paths[1], canonical, certification=certification)

    def test_different_joint_names_cannot_be_filled_from_canonical_reference(self):
        canonical, _ = decoded_bvh.load_decoded_bvh(self.paths[0])
        certification = decoded_bvh.certify_soma_x2_topology(canonical)
        original = f'JOINT {canonical.joint_names[1]}'
        self.paths[1].write_text(self.paths[1].read_text().replace(original, 'JOINT NonmatchingJoint', 1))
        for decoder in ('native', 'fast'):
            with self.subTest(decoder=decoder), self.assertRaisesRegex(ValueError, 'differ from canonical'):
                decoded_bvh.load_decoded_bvh(self.paths[1], canonical, decoder=decoder, certification=certification)

    def test_same_names_with_different_parents_cannot_enter_common_ik(self):
        canonical, _ = decoded_bvh.load_decoded_bvh(self.paths[0])
        certification = decoded_bvh.certify_soma_x2_topology(canonical)
        parents = self.layout.parent_indices.copy()
        index = next(index for index in range(2, len(parents))
                     if parents[index] >= 0 and parents[parents[index]] >= 0
                     and np.count_nonzero(parents == parents[index]) == 1)
        parents[index] = parents[parents[index]]
        changed = Skeleton(self.layout.num_joints, self.layout.joint_names, parents,
                           self.layout.reference_local_transforms)
        self.paths[1].write_text(_motion_fixture(changed))
        own, _ = native_load_bvh(str(self.paths[1]))
        self.assertEqual(own.joint_names, canonical.joint_names)
        self.assertFalse(np.array_equal(own.parent_indices, canonical.parent_indices))
        for decoder in ('native', 'fast'):
            with self.subTest(decoder=decoder), self.assertRaisesRegex(ValueError, 'parent indices'):
                decoded_bvh.load_decoded_bvh(self.paths[1], canonical, decoder=decoder, certification=certification)

    def test_zero_pose_incomplete_or_duplicate_names_cannot_be_certified(self):
        for replacement in ('UnknownJoint', self.layout.joint_names[0]):
            canonical, _ = decoded_bvh.load_decoded_bvh(self.paths[0])
            canonical.joint_names[1] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                decoded_bvh.certify_soma_x2_topology(canonical)

    def test_unknown_decoder_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'decoder'):
            decoded_bvh.load_decoded_bvh(self.paths[0], decoder='foreign-skeleton')


if __name__ == '__main__':
    unittest.main()
