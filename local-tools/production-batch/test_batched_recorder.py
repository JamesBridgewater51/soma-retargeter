import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import warp as wp

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from batched_recorder import BatchedRecorder

DEVICE = os.environ.get('RECORDER_TEST_DEVICE', 'cpu')


class RecorderStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device(DEVICE)

    def test_clip_mask_prefix_and_raw_base_match_native_copy_bytes(self):
        for lengths in ([1], [2, 3], [1, 2, 3]):
            for capture_base in (False, True):
                with self.subTest(lengths=lengths, capture_base=capture_base):
                    coordinates = wp.zeros((len(lengths), 38), dtype=wp.float32, device=DEVICE)
                    recorder = BatchedRecorder(coordinates, lengths, 2, capture_base)
                    final = [wp.empty((length, 38), dtype=wp.float32, device=DEVICE) for length in lengths]
                    base = [wp.empty_like(frames) for frames in final] if capture_base else []
                    for frame in range(2 + max(lengths) + 2):
                        values = (np.arange(len(lengths) * 38, dtype=np.float32).reshape(-1, 38) + frame * 100)
                        coordinates.assign(values)
                        recorder.record_base(frame, coordinates)
                        if capture_base and frame >= 2:
                            for env, length in enumerate(lengths):
                                if frame - 2 < length:
                                    wp.copy(base[env][frame - 2], coordinates[env])
                        coordinates.assign(values + np.float32(.125))
                        recorder.record_final(frame, coordinates)
                        if frame >= 2:
                            for env, length in enumerate(lengths):
                                if frame - 2 < length:
                                    wp.copy(final[env][frame - 2], coordinates[env])
                    for candidate, reference in zip(recorder.final_numpy_frames(), final, strict=True):
                        self.assertEqual(candidate.tobytes(), reference.numpy().tobytes())
                    for candidate, reference in zip(recorder.base_numpy_frames(), base, strict=True):
                        self.assertEqual(candidate.tobytes(), reference.numpy().tobytes())

    @unittest.skipUnless(DEVICE.startswith('cuda'), 'CUDA capture requires explicit GPU test device')
    def test_captured_kernel_reads_updated_coordinates(self):
        coordinates = wp.zeros((2, 38), dtype=wp.float32, device=DEVICE)
        recorder = BatchedRecorder(coordinates, [1, 2], 0)
        recorder.record_final(0, coordinates)
        wp.synchronize_device(DEVICE)
        with wp.ScopedCapture(device=DEVICE) as capture:
            recorder.record_final(0, coordinates)
        coordinates.assign(np.full((2, 38), 7.25, dtype=np.float32))
        wp.capture_launch(capture.graph)
        for frames in recorder.final_numpy_frames():
            np.testing.assert_array_equal(frames[0], np.full(38, 7.25, dtype=np.float32))


@unittest.skipUnless(os.environ.get('RECORDER_TEST_BVH'), 'Set RECORDER_TEST_BVH for real pipeline parity')
class RecorderPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import run_batch as batch
        batch.verify_local_imports()
        wp.init()
        wp.set_device(DEVICE)

    def test_full_native_outputs_and_base_prefixes_are_bitwise_identical(self):
        from smoke_batched_parity import trimmed_bvh
        from soma_retargeter.io.bvh import load_bvh
        from soma_retargeter.pipelines import utils
        from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
        from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

        class RecordedPipeline(SomaRetargetingPipeline):
            def _create_frame_recorder(self, joint_q, num_frames_to_remove):
                return BatchedRecorder(joint_q,
                    [len(targets) - num_frames_to_remove for targets in self.input_targets],
                    num_frames_to_remove, self.capture_base_motion)

        config = copy.deepcopy(utils.get_retargeter_config('soma', 'agibot_x2ultra'))
        config.update(json.loads(Path(os.environ['RECORDER_TEST_CONFIG']).read_text()))
        config['capture_base_motion'] = True
        source = Path(os.environ['RECORDER_TEST_BVH'])
        for lengths in ([1], [2, 3], [1, 2, 3], [8]):
            with self.subTest(lengths=lengths), tempfile.TemporaryDirectory(prefix='recorder-parity-') as directory:
                fixtures = [Path(directory) / f'clip-{index}.bvh' for index in range(len(lengths))]
                for index, (path, length) in enumerate(zip(fixtures, lengths, strict=True)):
                    trimmed_bvh(source, path, 16 + index * 8, length)
                skeleton, first = load_bvh(str(fixtures[0]))
                animations = [first] + [load_bvh(str(path), skeleton)[1] for path in fixtures[1:]]
                outputs = []
                for pipeline_type in (SomaRetargetingPipeline, RecordedPipeline):
                    pipeline = pipeline_type(skeleton, 'soma', 'agibot_x2ultra', retarget_config=copy.deepcopy(config))
                    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
                    pipeline.add_input_motions(animations, [transform] * len(animations), scale_animation=True)
                    final = pipeline.execute()
                    outputs.append(([buffer.data for buffer in pipeline.base_retargeted_motions],
                                    [buffer.data for buffer in final]))
                for stage in (0, 1):
                    for reference, candidate, length in zip(*[output[stage] for output in outputs], lengths, strict=True):
                        self.assertEqual(candidate.shape, (length, 38))
                        self.assertEqual(candidate.tobytes(), reference.tobytes())


if __name__ == '__main__':
    unittest.main()
