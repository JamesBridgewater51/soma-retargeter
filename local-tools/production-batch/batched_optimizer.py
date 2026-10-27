"""Batched forward solves with independent native singleton optimizer states."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
import time

import numpy as np
import warp as wp

from app.tools import ik_weight_optimizer as native
from soma_retargeter.pipelines.batched_weight_objectives import BatchedPositionObjective, BatchedRotationObjective
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter


class PerSequencePipeline(SomaRetargetingPipeline):
    """Keep native passes and constraints; replace only tracking-weight storage."""

    def __init__(self, skeleton, config, per_sequence_configs):
        self.per_sequence_configs = per_sequence_configs
        super().__init__(skeleton, "soma", "agibot_x2ultra", retarget_config=copy.deepcopy(config))

    @classmethod
    def from_base_template(cls, template, per_sequence_configs):
        """Share static geometry only; native execute still builds every solver/state fresh."""
        if template.post_processing_enabled:
            raise ValueError("Prepared geometry reuse is restricted to post-OFF optimizer evaluations")
        pipeline = cls.__new__(cls)
        pipeline.__dict__ = template.__dict__.copy()
        pipeline.per_sequence_configs = per_sequence_configs
        pipeline.temporal_ik_config = copy.deepcopy(template.temporal_ik_config)
        pipeline.clear()
        return pipeline

    def _create_ik_objectives(self, num_envs, model, state):
        positions, rotations, limits, smooth = super()._create_ik_objectives(num_envs, model, state)
        if num_envs != len(self.per_sequence_configs):
            raise ValueError("Each environment must have its own optimized configuration")
        position_batch, rotation_batch = [], []
        for effector, position, rotation in zip(self.mapped_joints, positions, rotations, strict=True):
            position_weights = wp.array([config["ik_match_table"][effector]["t_weight"]
                                         for config in self.per_sequence_configs], dtype=wp.float32,
                                        device=position.target_positions.device)
            rotation_weights = wp.array([config["ik_match_table"][effector]["r_weight"]
                                         for config in self.per_sequence_configs], dtype=wp.float32,
                                        device=rotation.target_rotations.device)
            position_batch.append(BatchedPositionObjective(
                position.link_index, position.link_offset, position.target_positions,
                per_problem_weights=position_weights))
            rotation_batch.append(BatchedRotationObjective(
                rotation.link_index, rotation.link_offset_rotation, rotation.target_rotations,
                canonicalize_quat_err=rotation.canonicalize_quat_err,
                per_problem_weights=rotation_weights))
        return position_batch, rotation_batch, limits, smooth


@dataclass
class ClipState:
    params: native.OptimizerParameters
    error_model: native.SourceAgnosticErrorModel
    anchor: native._EvaluationResult | None = None
    policy: native._ClipGuardPolicy | None = None
    initial_tracking_scale: float = 1.0
    smoothness_denom: float = 1.0
    initial_metrics: dict = field(default_factory=dict)
    no_improve: int = 0
    iterations: int = 0
    probes: int = 0
    accepts: int = 0
    done: bool = False
    modes: set = field(default_factory=set)


class IndependentBatchOptimizer:
    def __init__(self, skeleton, animations, names, settings, workspace):
        self.skeleton, self.animations, self.names = skeleton, animations, names
        self.config = copy.deepcopy(settings["optimizer"])
        self.reuse_geometry = bool(settings.get("reuse_geometry", False))
        self.snapshot_budgets = set(settings.get("snapshot_iterations", []))
        self.parameter_snapshots = {}
        self.snapshot_statistics = {}
        self.workspace = workspace
        self.ik = native.IkWeightUpdateConfig.from_config(self.config)
        self.line_search = native.LineSearchConfig.from_config(self.config)
        self.objective = native.ObjectiveConfig.from_config(self.config)
        self.symmetry = workspace.build_symmetry_pairs() if self.config.get("is_target_symmetric", True) else []
        identity = native.OptimizerParameters.identity(len(workspace.effector_names))
        baseline = self.retargeter(identity)
        started = time.monotonic()
        self.template = SomaRetargetingPipeline(skeleton, "soma", workspace.target_type, retarget_config=baseline)
        constructed = time.monotonic()
        transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
        self.template.add_input_motions(animations, [transform] * len(animations), scale_animation=True)
        targets_ready = time.monotonic()
        self.targets = self.template.input_targets
        self.sample_rates = self.template.input_sample_rates
        self.remove = self.template.num_initialization_frames + self.template.num_stabilization_frames
        self.evaluator = native.BodyEvaluator(self.template, fk_frame_batch_size=int(self.config.get("fk_frame_batch_size", 100)))
        evaluator_ready = time.monotonic()
        self.states = [ClipState(copy.deepcopy(identity), self.error_model(targets)) for targets in self.targets]
        self.setup_timings = {"template_construct_seconds": constructed - started,
                              "scaled_target_seconds": targets_ready - constructed,
                              "body_evaluator_construct_seconds": evaluator_ready - targets_ready,
                              "per_clip_error_model_seconds": time.monotonic() - evaluator_ready}
        self.evaluation_timings = []
        print(json.dumps({"event": "SETUP_PROFILE", **self.setup_timings}), flush=True)

    def retargeter(self, params):
        return native._build_retargeter_config(self.workspace, params, self.config.get("retargeter_overrides"))

    def error_model(self, targets):
        parents = native._pipeline_parent_indices(self.template)
        positions = np.asarray(targets[self.remove:, :, :3], dtype=np.float32)
        segments = [[] for _ in parents]
        for effector, parent in enumerate(parents):
            if parent >= 0:
                segments[effector].append(np.linalg.norm(positions[:, effector] - positions[:, parent], axis=-1))
        return native._build_error_model_streamed(self.template, segments, self.config)

    def evaluate(self, indices, parameters):
        started = time.monotonic()
        configs = [self.retargeter(params) for params in parameters]
        pipeline = (PerSequencePipeline.from_base_template(self.template, configs) if self.reuse_geometry
                    else PerSequencePipeline(self.skeleton, configs[0], configs))
        pipeline.input_targets = [self.targets[index] for index in indices]
        pipeline.input_sample_rates = [self.sample_rates[index] for index in indices]
        pipeline.max_frames = max(len(targets) for targets in pipeline.input_targets)
        constructed = time.monotonic()
        buffers = pipeline.execute()
        executed = time.monotonic()
        solved = native._buffers_to_arrays(buffers)
        transferred = time.monotonic()
        poses = self.evaluator.evaluate(solved)
        evaluated = time.monotonic()
        outcomes = [self.score(index, pose, coordinates)
                    for index, pose, coordinates in zip(indices, poses, solved, strict=True)]
        scored = time.monotonic()
        self.evaluation_timings.append({"clips": len(indices), "frames": sum(len(q) for q in solved),
                                        "seconds": scored - started,
                                        "construct_seconds": constructed - started,
                                        "execute_seconds": executed - constructed,
                                        "buffer_seconds": transferred - executed,
                                        "fk_seconds": evaluated - transferred,
                                        "score_seconds": scored - evaluated,
                                        "graph_setup_seconds": getattr(pipeline, "graph_setup_seconds", None),
                                        "frame_loop_seconds": getattr(pipeline, "frame_loop_seconds", None)})
        print(json.dumps({"event": "EVALUATION_PROFILE", **self.evaluation_timings[-1]}), flush=True)
        return outcomes

    def score(self, index, poses, coordinates):
        state = self.states[index]
        accumulator = native._TrackingErrorAccumulators.zeros(len(self.workspace.effector_names))
        metrics, frame_loss = native._compute_metrics_from_poses(
            poses, np.asarray(self.targets[index][self.remove:], dtype=np.float32),
            np.ones(len(coordinates), dtype=bool), state.error_model, accumulators=accumulator,
            unreachable_weight_decay=state.error_model.unreachable_weight_decay,
            hard_clip_emphasis=self.ik.hard_clip_emphasis)
        aggregate = native._aggregate_metrics([metrics])
        return native._EvaluationResult(
            tracking_loss=native._metric_value(aggregate, "tracking_loss"), aggregate=aggregate,
            per_motion_metrics=[metrics], ik_frame_loss_by_motion=[frame_loss], acc=accumulator,
            error_model=state.error_model, mapped_joints=list(self.template.mapped_joints))

    def breakdown(self, state, outcome):
        normalized = float(outcome.aggregate.get("ik_smoothness_loss", 0.0)) / max(state.smoothness_denom, 1e-8)
        scale = max(state.initial_tracking_scale, 1e-8)
        smoothness = self.objective.smoothness_balance * scale * normalized
        penalty = state.policy.penalty(outcome, scale)
        return native._ObjectiveBreakdown(outcome.tracking_loss + smoothness + penalty, normalized, smoothness, penalty)

    def stamp(self, state):
        components = self.breakdown(state, state.anchor)
        state.anchor.aggregate.update(normalized_smoothness_loss=components.normalized_smoothness_loss,
                                      objective_smoothness_term=components.smoothness_term,
                                      objective_clip_penalty_term=components.clip_penalty_term,
                                      objective_loss=components.objective_loss)
        state.modes.add(state.policy.active_mode)

    def initialize(self):
        indices = list(range(len(self.states)))
        outcomes = self.evaluate(indices, [state.params for state in self.states])
        for index, outcome in zip(indices, outcomes, strict=True):
            state = self.states[index]
            state.anchor = outcome
            state.initial_tracking_scale = max(outcome.tracking_loss, 1e-8)
            state.smoothness_denom = max(float(outcome.aggregate.get("ik_smoothness_loss", 0.0)), 1e-8)
            state.policy = native._ClipGuardPolicy(native.ClipGuardConfig.from_config(self.config),
                [self.names[index]], outcome.per_motion_metrics, outcome.ik_frame_loss_by_motion)
            self.stamp(state)
            state.initial_metrics = dict(outcome.aggregate)

    def propose(self, index, step_scale):
        state = self.states[index]
        return native._updated_params(state.params, state.anchor.acc, step_scale=step_scale,
                                      root_mask=state.error_model.parent_indices < 0,
                                      symmetry_pairs=self.symmetry, ik=self.ik)

    def outer_iteration(self, indices):
        pending = list(indices)
        anchor_objectives = {index: self.breakdown(self.states[index], self.states[index].anchor).objective_loss
                             for index in indices}
        accepted = set()
        probe_count = self.line_search.max_backtracks + 1 if self.line_search.enabled else 1
        scale = 1.0
        for _ in range(probe_count):
            if not pending:
                break
            proposals = [self.propose(index, scale)[0] for index in pending]
            trials = self.evaluate(pending, proposals)
            rejected = []
            for index, params, trial in zip(pending, proposals, trials, strict=True):
                state = self.states[index]
                state.probes += 1
                objective = self.breakdown(state, trial).objective_loss
                decision = state.policy.probe(state.anchor, trial)
                improved = not self.line_search.enabled or objective < anchor_objectives[index] - self.line_search.min_improvement
                if improved and not decision.blocked:
                    state.params, state.anchor = params, trial
                    state.no_improve = 0
                    state.accepts += 1
                    state.policy.note_accepted(trial)
                    accepted.add(index)
                else:
                    rejected.append(index)
            pending = rejected
            scale *= self.line_search.shrink
        self.finish_iteration(indices, accepted)

    def finish_iteration(self, indices, accepted):
        patience = int(self.config.get("convergence_patience", 1))
        for index in indices:
            state = self.states[index]
            state.iterations += 1
            if index not in accepted:
                state.no_improve += 1
            self.stamp(state)
            if index not in accepted and patience >= 0 and state.no_improve > patience:
                if state.policy.try_enter_penalty_phase():
                    state.no_improve = 0
                else:
                    state.done = True

    def capture_prefix(self, budget):
        self.parameter_snapshots[budget] = [native._build_retargeter_config(self.workspace, state.params)
                                           for state in self.states]
        self.snapshot_statistics[budget] = [{"effective_iterations": state.iterations, "probes": state.probes,
                                            "accepted_iterations": state.accepts, "converged": state.done,
                                            "metrics": finite_metrics(state.anchor.aggregate)} for state in self.states]

    def run(self):
        self.initialize()
        for iteration in range(int(self.config["outer_iterations"])):
            active = [index for index, state in enumerate(self.states) if not state.done]
            if not active:
                break
            self.outer_iteration(active)
            if iteration + 1 in self.snapshot_budgets:
                self.capture_prefix(iteration + 1)
            print(f"BATCH_OUTER iteration={iteration + 1} active={len(active)} "
                  f"converged={sum(state.done for state in self.states)}", flush=True)
        for budget in self.snapshot_budgets:
            if budget not in self.parameter_snapshots:
                self.capture_prefix(budget)
        return [native._build_retargeter_config(self.workspace, state.params) for state in self.states]

    def summaries(self):
        return [{"motion": name, "outer_iteration_budget": self.config["outer_iterations"],
                 "recorded_iterations": state.iterations, "probes": state.probes,
                 "accepted_iterations": state.accepts, "converged": state.done,
                 "initial_metrics": finite_metrics(state.initial_metrics),
                 "final_metrics": finite_metrics(state.anchor.aggregate),
                 "active_guard_modes": sorted(state.modes),
                 "temporal_ik": self.config.get("retargeter_overrides", {}).get("temporal_ik")}
                for name, state in zip(self.names, self.states, strict=True)]


def finite_metrics(metrics):
    return {key: float(number) for key, number in metrics.items()
            if isinstance(number, (int, float)) and np.isfinite(number)}
