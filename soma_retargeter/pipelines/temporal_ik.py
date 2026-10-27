# SPDX-License-Identifier: Apache-2.0
"""Opt-in temporal objectives and feasible, cost-checked IK iterations."""

import xml.etree.ElementTree as ET

import numpy as np
import warp as wp
import newton.ik as ik


@wp.kernel
def _temporal_residual(q: wp.array2d(dtype=float), target: wp.array2d(dtype=float),
                       coords: wp.array(dtype=int), weight: wp.array(dtype=float), start: int,
                       residual: wp.array2d(dtype=float)):
    env, i = wp.tid()
    residual[env, start + i] = weight[0] * (q[env, coords[i]] - target[env, coords[i]])


@wp.kernel
def _temporal_jacobian(dofs: wp.array(dtype=int), weight: wp.array(dtype=float), start: int,
                       jac: wp.array3d(dtype=float)):
    env, i = wp.tid()
    jac[env, start + i, dofs[i]] = weight[0]


@wp.kernel
def _project(q: wp.array2d(dtype=float), coords: wp.array(dtype=int),
             lower: wp.array2d(dtype=float), upper: wp.array2d(dtype=float)):
    env, i = wp.tid()
    c = coords[i]
    q[env, c] = wp.clamp(q[env, c], lower[env, i], upper[env, i])


@wp.kernel
def _accept_feasible(q: wp.array2d(dtype=float), previous: wp.array2d(dtype=float),
                     costs: wp.array(dtype=float), previous_costs: wp.array(dtype=float)):
    env, c = wp.tid()
    if not wp.isfinite(costs[env]) or costs[env] > previous_costs[env]:
        q[env, c] = previous[env, c]


@wp.kernel
def _reject_projected_damping(cost: wp.array(dtype=float), previous_cost: wp.array(dtype=float),
                               previous_lambda: wp.array(dtype=float), factor: float,
                               minimum: float, maximum: float, damping: wp.array(dtype=float)):
    env = wp.tid()
    if not wp.isfinite(cost[env]) or cost[env] > previous_cost[env]:
        damping[env] = wp.clamp(previous_lambda[env] * factor, minimum, maximum)


def _temporal_lm_optimizer(solver, q):
    optimizer = solver._impl
    if solver.optimizer_type != ik.IKOptimizer.LM or not isinstance(optimizer, ik.IKOptimizerLM):
        raise ValueError('Hard temporal IK requires the native Newton LM backend')
    if solver.sampler != ik.IKSampler.NONE or solver.n_seeds != 1 or solver.n_expanded != q.shape[0]:
        raise ValueError('Hard temporal IK requires identity single-seed batching')
    required = ('_step', 'lambda_values', 'lambda_initial', 'lambda_factor', 'lambda_min', 'lambda_max')
    if not all(hasattr(optimizer, name) for name in required) or not callable(optimizer._step):
        raise RuntimeError('Installed Newton LM private iteration contract changed')
    if optimizer.lambda_values.shape != (q.shape[0],) or optimizer.lambda_values.dtype != wp.float32:
        raise RuntimeError('Installed Newton LM damping buffer contract changed')
    damping = (optimizer.lambda_initial, optimizer.lambda_factor, optimizer.lambda_min, optimizer.lambda_max)
    if not all(np.isfinite(damping)) or not (0 < damping[2] <= damping[0] <= damping[3] and damping[1] > 1):
        raise ValueError('Hard temporal IK requires finite, valid adaptive LM damping parameters')
    return optimizer


class IKTemporalObjective(ik.IKObjective):
    """Scalar-joint pose or constant-velocity prediction residual, in radians/metres.

    The weight multiplies residuals (its square multiplies the least-squares
    cost). Floating-base quaternion coordinates are intentionally excluded.
    """

    def __init__(self, coords, dofs, target, weight):
        super().__init__()
        self.coords_np = np.asarray(coords, dtype=np.int32)
        self.dofs_np = np.asarray(dofs, dtype=np.int32)
        self.target = target
        self.weight = float(weight)

    def residual_dim(self):
        return len(self.coords_np)

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        self.coords = wp.array(self.coords_np, dtype=int, device=self.device)
        self.dofs = wp.array(self.dofs_np, dtype=int, device=self.device)
        self.weight_buffer = wp.array([self.weight], dtype=float, device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_temporal_residual, (joint_q.shape[0], self.residual_dim()),
                  inputs=[joint_q, self.target, self.coords, self.weight_buffer, start_idx, residuals],
                  device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_temporal_jacobian, (joint_q.shape[0], self.residual_dim()),
                  inputs=[self.dofs, self.weight_buffer, start_idx, jacobian], device=self.device)


@wp.kernel
def _limit_residual(q: wp.array2d(dtype=float), coords: wp.array(dtype=int),
                    lower: wp.array(dtype=float), upper: wp.array(dtype=float),
                    weight: float, start: int, residual: wp.array2d(dtype=float)):
    env, i = wp.tid()
    x = q[env, coords[i]]
    residual[env, start + i] = weight * (x - wp.clamp(x, lower[i], upper[i]))


@wp.kernel
def _limit_jacobian(q: wp.array2d(dtype=float), coords: wp.array(dtype=int),
                    dofs: wp.array(dtype=int), lower: wp.array(dtype=float),
                    upper: wp.array(dtype=float), weight: float, start: int,
                    jac: wp.array3d(dtype=float)):
    env, i = wp.tid()
    x = q[env, coords[i]]
    derivative = float(0.0)
    if x < lower[i] or x > upper[i]:
        derivative = weight
    jac[env, start + i, dofs[i]] = derivative


class IKNearLimitObjective(IKTemporalObjective):
    """Squared hinge outside an interior band; exact piecewise Jacobian."""

    def __init__(self, coords, dofs, lower, upper, margin, weight):
        super().__init__(coords, dofs, None, weight)
        inset = np.minimum(margin, np.maximum(0, (upper - lower) * 0.5))
        self.lower_np = lower + inset
        self.upper_np = upper - inset

    def init_buffers(self, model, jacobian_mode):
        super().init_buffers(model, jacobian_mode)
        self.lower = wp.array(self.lower_np, dtype=float, device=self.device)
        self.upper = wp.array(self.upper_np, dtype=float, device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_limit_residual, (joint_q.shape[0], self.residual_dim()),
                  inputs=[joint_q, self.coords, self.lower, self.upper, self.weight, start_idx, residuals],
                  device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_limit_jacobian, (joint_q.shape[0], self.residual_dim()),
                  inputs=[joint_q, self.coords, self.dofs, self.lower, self.upper,
                          self.weight, start_idx, jacobian], device=self.device)


@wp.kernel
def _velocity_avoidance_residual(q: wp.array2d(dtype=float), previous: wp.array2d(dtype=float),
                                 coords: wp.array(dtype=int), step: wp.array2d(dtype=float),
                                 activation: float, weight: wp.array(dtype=float), start: int,
                                 residual: wp.array2d(dtype=float)):
    env, i = wp.tid()
    ratio = (q[env, coords[i]] - previous[env, coords[i]]) / step[env, i]
    residual[env, start + i] = weight[0] * (ratio - wp.clamp(ratio, -activation, activation))


@wp.kernel
def _velocity_avoidance_jacobian(q: wp.array2d(dtype=float), previous: wp.array2d(dtype=float),
                                 coords: wp.array(dtype=int), dofs: wp.array(dtype=int),
                                 step: wp.array2d(dtype=float), activation: float,
                                 weight: wp.array(dtype=float), start: int,
                                 jacobian: wp.array3d(dtype=float)):
    env, i = wp.tid()
    ratio = (q[env, coords[i]] - previous[env, coords[i]]) / step[env, i]
    derivative = float(0.0)
    if wp.abs(ratio) > activation:
        derivative = weight[0] / step[env, i]
    jacobian[env, start + i, dofs[i]] = derivative


class IKVelocityAvoidanceObjective(IKTemporalObjective):
    """Dimensionless hinge on |delta q|/(scaled URDF vmax * dt).

    A margin fraction of 0.2 starts avoidance at 80% of the per-step bound.
    This term is independent of absolute-angle near-limit avoidance.
    """

    def __init__(self, coords, dofs, previous, step, margin_fraction, weight):
        super().__init__(coords, dofs, previous, weight)
        self.step = step
        self.activation = 1.0 - margin_fraction

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_velocity_avoidance_residual, (joint_q.shape[0], self.residual_dim()),
                  inputs=[joint_q, self.target, self.coords, self.step, self.activation,
                          self.weight_buffer, start_idx, residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_velocity_avoidance_jacobian, (joint_q.shape[0], self.residual_dim()),
                  inputs=[joint_q, self.target, self.coords, self.dofs, self.step, self.activation,
                          self.weight_buffer, start_idx, jacobian], device=self.device)


def scalar_joint_metadata(model, urdf_path=None):
    """Map scalar actuated joints to true DoF limits and URDF velocity units."""
    urdf_joints = {}
    if urdf_path is not None:
        urdf_joints = {j.attrib['name']: j for j in ET.parse(urdf_path).getroot().findall('joint')}
    q_start = model.joint_q_start.numpy()
    d_start = model.joint_qd_start.numpy()
    dims = model.joint_dof_dim.numpy()
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    coords, dofs, velocities = [], [], []
    for j, (lin, ang) in enumerate(dims):
        # A free/ball joint must never be treated as independent quaternion scalars.
        if int(lin + ang) != 1:
            continue
        name = model.joint_label[j].split('/')[-1]
        coords.append(int(q_start[j]))
        dofs.append(int(d_start[j]))
        if urdf_path is not None:
            joint = urdf_joints.get(name)
            limit = None if joint is None else joint.find('limit')
            if limit is None or 'velocity' not in limit.attrib:
                raise ValueError(f"Missing URDF velocity limit for scalar joint {name!r}")
            velocity = float(limit.attrib['velocity'])
            if not np.isfinite(velocity) or velocity <= 0:
                raise ValueError(f"Invalid URDF velocity limit for {name!r}: {velocity}")
            velocities.append(velocity)
    dofs = np.asarray(dofs, dtype=np.int32)
    return (np.asarray(coords, dtype=np.int32), dofs,
            lower[dofs].astype(np.float32), upper[dofs].astype(np.float32),
            np.asarray(velocities, dtype=np.float32))


def frame_bounds(previous, lower, upper, velocities, sample_rates, velocity_scale=1.0):
    """Intersect angle limits with per-environment velocity boxes; no root bounds."""
    rates = np.asarray(sample_rates, dtype=np.float64)
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError('Temporal IK requires finite positive sample rates (Hz)')
    step = np.asarray(velocities)[None, :] * velocity_scale / rates[:, None]
    lo = np.maximum(lower[None, :], previous - step)
    hi = np.minimum(upper[None, :], previous + step)
    if np.any(lo > hi):
        raise ValueError('Previous output is outside the reachable joint-limit interval')
    return lo.astype(np.float32), hi.astype(np.float32)


class TemporalIK:
    """Per-run history and projected descent; device buffers stay capture-stable."""

    def __init__(self, model, num_envs, sample_rates, reference, config, urdf_path=None):
        self.weight = float(config.get('weight', 0.0))
        self.mode = config.get('mode', 'position')
        self.hard_velocity = bool(config.get('hard_velocity', False))
        self.velocity_scale = float(config.get('velocity_scale', 1.0))
        self.limit_margin = float(config.get('limit_margin', 0.0))
        self.limit_weight = float(config.get('limit_weight', 0.0))
        self.velocity_avoidance_weight = float(config.get('velocity_avoidance_weight', 0.0))
        self.velocity_margin_fraction = float(config.get('velocity_margin_fraction', 0.2))
        if not np.isfinite(self.velocity_margin_fraction) or not 0 <= self.velocity_margin_fraction < 1:
            raise ValueError('temporal_ik.velocity_margin_fraction must lie in [0, 1)')
        if self.mode not in ('position', 'prediction'):
            raise ValueError("temporal_ik.mode must be 'position' or 'prediction'")
        for name in ('weight', 'velocity_scale', 'limit_margin', 'limit_weight', 'velocity_avoidance_weight'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == 'velocity_scale' and value == 0):
                raise ValueError(f'temporal_ik.{name} has invalid value {value}')
        self.coords, self.dofs, self.angle_lower, self.angle_upper, self.velocities = scalar_joint_metadata(
            model, urdf_path if self.hard_velocity or self.velocity_avoidance_weight > 0 else None)
        if not len(self.coords):
            raise ValueError('Temporal IK requires scalar actuated joints')
        self.sample_rates = sample_rates
        initial = np.tile(reference.numpy(), (num_envs, 1)).astype(np.float32)
        initial[:, self.coords] = np.clip(initial[:, self.coords], self.angle_lower, self.angle_upper)
        self.previous = initial.copy()
        self.older = initial.copy()
        self.frames = 0
        self.target = wp.array(initial, dtype=float, device=model.device)
        self.objective = IKTemporalObjective(self.coords, self.dofs, self.target, self.weight)
        self.limit_objective = IKNearLimitObjective(
            self.coords, self.dofs, self.angle_lower, self.angle_upper,
            self.limit_margin, self.limit_weight)
        self.velocity_objective = None
        if self.velocity_avoidance_weight > 0:
            # Reuse the hard-bound unit validation even when only avoidance is enabled.
            frame_bounds(self.previous[:, self.coords], self.angle_lower, self.angle_upper,
                         self.velocities, self.sample_rates, self.velocity_scale)
            step = self.velocities[None, :] * self.velocity_scale / np.asarray(sample_rates)[:, None]
            previous_target = wp.array(initial, dtype=float, device=model.device)
            self.velocity_objective = IKVelocityAvoidanceObjective(
                self.coords, self.dofs, previous_target,
                wp.array(step.astype(np.float32), dtype=float, device=model.device),
                self.velocity_margin_fraction, self.velocity_avoidance_weight)
        self.coords_wp = wp.array(self.coords, dtype=int, device=model.device)
        shape = (num_envs, len(self.coords))
        self.lower = wp.empty(shape, dtype=float, device=model.device)
        self.upper = wp.empty(shape, dtype=float, device=model.device)
        self.incumbent = wp.empty(initial.shape, dtype=float, device=model.device)
        self.incumbent_cost = wp.empty(num_envs, dtype=float, device=model.device)
        self._lambda_before = wp.empty(num_envs, dtype=float, device=model.device)
        self.device = model.device

    def reset_history(self):
        # Initialization padding is not part of the exported motion. Its pose
        # must not impose a speed constraint on the first retained sample.
        self.frames = 0
        self.older = self.previous.copy()

    def prepare(self):
        if self.weight > 0:
            history_ready = self.frames >= (2 if self.mode == 'prediction' else 1)
            self.objective.weight_buffer.assign(np.array([self.weight if history_ready else 0.0], dtype=np.float32))
        if self.velocity_objective is not None:
            self.velocity_objective.target.assign(self.previous)
            self.velocity_objective.weight_buffer.assign(np.array([
                self.velocity_avoidance_weight if self.frames > 0 else 0.0], dtype=np.float32))
        target = self.previous.copy()
        if self.mode == 'prediction' and self.frames >= 2:
            target[:, self.coords] += (self.previous - self.older)[:, self.coords]
        self.target.assign(target)
        if self.hard_velocity:
            # First frame has no predecessor; initialization is constrained only by angle.
            if self.frames == 0:
                lo = np.broadcast_to(self.angle_lower, self.lower.shape).copy()
                hi = np.broadcast_to(self.angle_upper, self.upper.shape).copy()
            else:
                lo, hi = frame_bounds(self.previous[:, self.coords], self.angle_lower,
                                      self.angle_upper, self.velocities, self.sample_rates,
                                      self.velocity_scale)
            self.lower.assign(lo)
            self.upper.assign(hi)

    def project(self, q):
        wp.launch(_project, self.lower.shape,
                  inputs=[q, self.coords_wp, self.lower, self.upper], device=self.device)

    def solve(self, solver, q, iterations):
        optimizer = _temporal_lm_optimizer(solver, q)
        self.project(q)
        # Both public Newton step layers reset damping. Initialize once per
        # solve, then retain per-environment adaptation between proposals.
        optimizer.lambda_values.fill_(optimizer.lambda_initial)
        for _ in range(iterations):
            wp.copy(self.incumbent, q)
            wp.copy(self.incumbent_cost, solver.compute_costs(q))
            wp.copy(self._lambda_before, optimizer.lambda_values)
            # Refresh residuals after our external projection or restoration.
            optimizer._step(q, iteration=0)
            self.project(q)
            costs = solver.compute_costs(q)
            wp.launch(_reject_projected_damping, q.shape[0],
                      inputs=[costs, self.incumbent_cost, self._lambda_before,
                              optimizer.lambda_factor, optimizer.lambda_min, optimizer.lambda_max,
                              optimizer.lambda_values], device=self.device)
            wp.launch(_accept_feasible, q.shape,
                      inputs=[q, self.incumbent, costs, self.incumbent_cost], device=self.device)

    def record(self, q):
        current = q.numpy().copy()
        if not np.all(np.isfinite(current)):
            raise ValueError('Temporal IK produced nonfinite output')
        if self.hard_velocity:
            scalar = current[:, self.coords]
            if np.any(scalar < self.lower.numpy() - 2e-6) or np.any(scalar > self.upper.numpy() + 2e-6):
                raise ValueError('Temporal IK final output violates hard bounds')
        else:
            # The legacy exporter clamps angles after the entire sequence.
            # Reference that exported pose, not its unclamped internal precursor.
            current[:, self.coords] = np.clip(current[:, self.coords], self.angle_lower, self.angle_upper)
        self.older, self.previous = self.previous, current.copy()
        self.frames += 1
