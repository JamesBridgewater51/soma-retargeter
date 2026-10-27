"""Reusable, fixed-target native BASE evaluations with explicit initial-state restore."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import warp as wp

from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.pipelines.ik_pass import BasePass
from soma_retargeter.pipelines.ik_solver import BaseIKSolver
from soma_retargeter.pipelines.temporal_ik import TemporalIK
from soma_retargeter.pipelines import utils as pipeline_utils


def _weightless_config(config):
    config = copy.deepcopy(config)
    for match in config['ik_match_table'].values():
        match['t_weight'] = 0.0
        match['r_weight'] = 0.0
    return json.dumps(config, sort_keys=True)


class _InitialArrays:
    """Restore buffers in place so previously captured graphs retain their pointers."""

    def __init__(self, owners, arrays):
        seen = set()
        self.copies = []
        for owner in owners:
            arrays.extend(buffer for buffer in vars(owner).values() if isinstance(buffer, wp.array))
        for buffer in arrays:
            key = (buffer.ptr, buffer.size, buffer.dtype)
            if buffer.size and key not in seen:
                seen.add(key)
                self.copies.append((buffer, wp.clone(buffer)))

    def restore(self):
        for destination, initial in self.copies:
            wp.copy(destination, initial)


class ReusableBaseEvaluator:
    """Reuse a prepared PerSequencePipeline with fixed targets and environment order.

    Only tracking coefficients may change. Native BASE evaluation must have
    post-processing disabled. Temporal histories are restored before every probe.
    This does not replace final POST conversion, support reordered target slots,
    or reuse an outer optimizer state.
    """

    def __init__(self, pipeline):
        if pipeline.post_processing_enabled:
            raise ValueError('Reusable BASE prototype requires post-processing disabled')
        self.pipeline = pipeline
        self.count = len(pipeline.input_targets)
        if not self.count or self.count != len(pipeline.per_sequence_configs):
            raise ValueError('Expected one fixed target and initial configuration per environment')
        self.config_signatures = [_weightless_config(config) for config in pipeline.per_sequence_configs]
        self.remove = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
        self.model = pipeline._build_model(self.count)
        self.state = self.model.state()
        self.positions, self.rotations, limits, self.smooth = pipeline._create_ik_objectives(
            self.count, self.model, self.state)
        extra = []
        if pipeline.joint_limit_weight > 0:
            extra.append(limits)
        if pipeline.smooth_joint_filter_weight > 0:
            extra.append(self.smooth)
        self.temporal = None
        temporal_owners = []
        if pipeline.temporal_ik_config.get('enabled', False):
            asset_info = pipeline_utils.get_target_asset_path(pipeline.target_type)
            needs_velocity = (pipeline.temporal_ik_config.get('hard_velocity', False)
                              or pipeline.temporal_ik_config.get('velocity_avoidance_weight', 0) > 0)
            if needs_velocity and 'urdf_path' not in asset_info:
                raise ValueError('Velocity constraints require a URDF with explicit velocity limits')
            self.temporal = TemporalIK(pipeline.ik_model, self.count, pipeline.input_sample_rates,
                                       pipeline.human_robot_scaler.reference_joint_q,
                                       pipeline.temporal_ik_config, asset_info.get('urdf_path'))
            if self.temporal.weight > 0:
                extra.append(self.temporal.objective)
            if self.temporal.limit_weight > 0:
                extra.append(self.temporal.limit_objective)
            if self.temporal.velocity_objective is not None:
                extra.append(self.temporal.velocity_objective)
            temporal_owners = [self.temporal, self.temporal.objective, self.temporal.limit_objective]
            if self.temporal.velocity_objective is not None:
                temporal_owners.append(self.temporal.velocity_objective)
            self.initial_previous = self.temporal.previous.copy()
            self.initial_older = self.temporal.older.copy()
        self.solver = BaseIKSolver(pipeline.ik_model, self.count, self.model,
                                  self.positions, self.rotations, extra,
                                  reference_joint_q=pipeline.human_robot_scaler.reference_joint_q)
        self.solver.temporal = self.temporal
        self.joint_q = wp.clone(self.solver.joint_q)
        # Snapshot before capture: a fresh native pass starts from this state.
        self.initial = _InitialArrays(
            [self.state, self.solver.solver, self.solver.solver._impl,
             *self.positions, *self.rotations, limits, self.smooth, *temporal_owners],
            [self.model.joint_q, self.model.joint_qd, self.solver.joint_q, self.joint_q])
        self.pass_ = BasePass(
            ik_solver=self.solver, joint_q=self.joint_q, num_envs=self.count,
            input_targets=pipeline.input_targets, position_objectives=self.positions,
            rotation_objectives=self.rotations, smooth_filter_objective=self.smooth,
            smooth_filter_weight=pipeline.smooth_joint_filter_weight,
            num_frames_to_remove=self.remove, ik_iterations=pipeline.ik_iterations, state=self.state)
        self.outputs = [wp.empty((len(targets) - self.remove, self.joint_q.shape[1]), dtype=wp.float32,
                                 device=self.model.device) for targets in pipeline.input_targets]
        self.last_timings = {}

    def evaluate(self, configs):
        started = time.monotonic()
        if len(configs) != self.count or [_weightless_config(config) for config in configs] != self.config_signatures:
            raise ValueError('Reusable evaluator only permits tracking-weight changes in fixed target slots')
        self.initial.restore()
        # This resets LM damping, selected seeds and RNG in addition to restoring scratch arrays.
        self.solver.solver.reset()
        if self.temporal is not None:
            self.temporal.previous = self.initial_previous.copy()
            self.temporal.older = self.initial_older.copy()
            self.temporal.frames = 0
        for effector, position, rotation in zip(self.pipeline.mapped_joints, self.positions, self.rotations, strict=True):
            position.per_problem_weights.assign(np.asarray(
                [config['ik_match_table'][effector]['t_weight'] for config in configs], dtype=np.float32))
            rotation.per_problem_weights.assign(np.asarray(
                [config['ik_match_table'][effector]['r_weight'] for config in configs], dtype=np.float32))
        for output in self.outputs:
            output.zero_()
        reset_done = time.monotonic()
        for frame in range(self.pipeline.max_frames):
            self.pass_.execute(frame)
            if self.temporal is not None:
                self.temporal.record(self.joint_q)
            if frame >= self.remove:
                for env, output in enumerate(self.outputs):
                    if frame - self.remove < output.shape[0]:
                        wp.copy(output[frame - self.remove], self.joint_q[env])
        if self.temporal is None or not self.temporal.hard_velocity:
            for output in self.outputs:
                self.pipeline.joint_limit_clamper.apply(output)
        buffers = [CSVAnimationBuffer.create_from_raw_data(output.numpy().copy(), rate)
                   for output, rate in zip(self.outputs, self.pipeline.input_sample_rates, strict=True)]
        self.last_timings = {'reset_seconds': reset_done - started,
                             'solve_export_seconds': time.monotonic() - reset_done,
                             'total_seconds': time.monotonic() - started}
        return buffers
