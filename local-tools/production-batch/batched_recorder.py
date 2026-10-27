"""Optional frame-major GPU recording without per-environment Python copies."""
import numpy as np
import warp as wp


@wp.kernel
def _record_frame(coordinates: wp.array2d(dtype=wp.float32),
                  destination: wp.array3d(dtype=wp.float32),
                  lengths: wp.array(dtype=wp.int32), frame: int):
    env, coord = wp.tid()
    if frame < lengths[env]:
        destination[env, frame, coord] = coordinates[env, coord]


class BatchedRecorder:
    """Preserve native frame stripping and raw-base/final recording boundaries."""

    def __init__(self, joint_q, retained_lengths, first_frame, capture_base=False):
        lengths = np.asarray(retained_lengths)
        if lengths.shape != (joint_q.shape[0],) or not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError('Expected one integer retained length per environment')
        if np.any(lengths < 0) or first_frame < 0:
            raise ValueError('Retained lengths and first frame must be nonnegative')
        self.lengths = lengths.astype(np.int32)
        self.first_frame = int(first_frame)
        self.device = joint_q.device
        self.shape = joint_q.shape
        self.lengths_wp = wp.array(self.lengths, dtype=wp.int32, device=self.device)
        shape = (joint_q.shape[0], int(lengths.max(initial=0)), joint_q.shape[1])
        # Zero padding is never exported, but makes the unchanged final clamp safe.
        self.final = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.base = wp.zeros(shape, dtype=wp.float32, device=self.device) if capture_base else None

    def _record(self, frame, joint_q, destination):
        retained_frame = frame - self.first_frame
        if 0 <= retained_frame < destination.shape[1]:
            wp.launch(_record_frame, dim=self.shape,
                      inputs=[joint_q, destination, self.lengths_wp, retained_frame], device=self.device)

    def record_base(self, frame, joint_q):
        if self.base is not None:
            self._record(frame, joint_q, self.base)

    def record_final(self, frame, joint_q):
        self._record(frame, joint_q, self.final)

    def final_frame_views(self):
        return [self.final[env] for env in range(len(self.lengths))]

    def final_numpy_frames(self):
        frames = self.final.numpy()
        return [frames[env, :length].copy() for env, length in enumerate(self.lengths)]

    def base_numpy_frames(self):
        if self.base is None:
            return []
        frames = self.base.numpy()
        return [frames[env, :length].copy() for env, length in enumerate(self.lengths)]
