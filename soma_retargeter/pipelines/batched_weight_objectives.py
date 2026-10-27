# SPDX-License-Identifier: Apache-2.0
"""Analytic Newton IK objectives with capture-stable per-problem weights."""

import newton.ik as ik
import warp as wp


@wp.kernel
def _scale_residuals(residuals: wp.array2d(dtype=float), weights: wp.array(dtype=float),
                     problem_idx: wp.array(dtype=int), start: int):
    row, component = wp.tid()
    residuals[row, start + component] *= weights[problem_idx[row]]


@wp.kernel
def _scale_jacobian(jacobian: wp.array3d(dtype=float), weights: wp.array(dtype=float),
                    rows_per_problem: int, start: int):
    row, component, dof = wp.tid()
    jacobian[row, start + component, dof] *= weights[row // rows_per_problem]


class _PerProblemWeights:
    """Post-scale native rows; only Newton's contiguous analytic batch layout."""

    def _set_weights(self, weights, count, device):
        if weights.dtype != wp.float32 or weights.ndim != 1 or weights.shape[0] != count:
            raise ValueError(f'per_problem_weights must be a float32 array of shape ({count},)')
        if weights.device != device:
            raise ValueError('Weights and targets must use the same device')
        self.per_problem_weights = weights

    def init_buffers(self, model, jacobian_mode):
        # In-place residual scaling is deliberately not recorded on an autodiff tape.
        if jacobian_mode != ik.IKJacobianType.ANALYTIC:
            raise ValueError('Per-problem weight objectives require ANALYTIC Jacobians')
        self._rows_per_problem(self.n_batch)
        if self.per_problem_weights.device != self.device:
            raise ValueError('Weights and solver must use the same device')
        super().init_buffers(model, jacobian_mode)

    def _rows_per_problem(self, count):
        base_count = self.per_problem_weights.shape[0]
        if not base_count or count % base_count:
            raise ValueError('Analytic batch must contain equally many contiguous rows per base problem')
        return count // base_count

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        super().compute_residuals(body_q, joint_q, model, residuals, start_idx, problem_idx)
        wp.launch(_scale_residuals, (body_q.shape[0], self.residual_dim()),
                  inputs=[residuals, self.per_problem_weights, problem_idx, start_idx], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        rows_per_problem = self._rows_per_problem(body_q.shape[0])
        super().compute_jacobian_analytic(body_q, joint_q, model, jacobian, joint_S_s, start_idx)
        wp.launch(_scale_jacobian, (body_q.shape[0], self.residual_dim(), model.joint_dof_count),
                  inputs=[jacobian, self.per_problem_weights, rows_per_problem, start_idx], device=self.device)


class BatchedPositionObjective(_PerProblemWeights, ik.IKObjectivePosition):
    """Native position residual times a per-base-problem float32 coefficient.

    Use with ANALYTIC IKSolver and its standard contiguous seed layout, not an
    optimizer with an arbitrary problem-index permutation. ``weight`` retains
    its native scalar meaning and multiplies ``per_problem_weights``. Update
    coefficients through ``per_problem_weights.assign(...)`` without rebinding
    the array, including when replaying captured CUDA graphs.
    """

    def __init__(self, link_index, link_offset, target_positions, weight=1.0, *, per_problem_weights):
        super().__init__(link_index, link_offset, target_positions, weight=weight)
        self._set_weights(per_problem_weights, target_positions.shape[0], target_positions.device)


class BatchedRotationObjective(_PerProblemWeights, ik.IKObjectiveRotation):
    """Native rotation residual with the same weight/layout contract as position."""

    def __init__(self, link_index, link_offset_rotation, target_rotations,
                 canonicalize_quat_err=True, weight=1.0, *, per_problem_weights):
        super().__init__(link_index, link_offset_rotation, target_rotations,
                         canonicalize_quat_err=canonicalize_quat_err, weight=weight)
        self._set_weights(per_problem_weights, target_rotations.shape[0], target_rotations.device)
