"""CPU tests for temporal coordinates, units, Jacobians and feasible IK descent."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp
import newton
import newton.ik as ik

from soma_retargeter.pipelines.temporal_ik import (
    IKTemporalObjective, IKNearLimitObjective, TemporalIK, frame_bounds,
    scalar_joint_metadata,
)


class TemporalIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device('cpu')

    def model(self):
        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0)
        free = builder.add_joint_free(parent=-1, child=root)
        tip = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=root, child=tip, axis=[0, 0, 1], label='hinge')
        builder.add_articulation([free, joint])
        builder.joint_limit_lower[-1] = -1.0
        builder.joint_limit_upper[-1] = 1.0
        return builder.finalize(device='cpu')

    def test_bounds_use_hz_and_independent_environments(self):
        lo, hi = frame_bounds(np.array([[0.9], [-0.5]]), np.array([-1]),
                              np.array([1]), np.array([2]), [10, 20])
        np.testing.assert_allclose(lo, [[0.7], [-0.6]])
        np.testing.assert_allclose(hi, [[1.0], [-0.4]])
        with self.assertRaises(ValueError):
            frame_bounds(np.zeros((1, 1)), np.array([-1]), np.array([1]), np.array([2]), [0])

    def test_metadata_excludes_root_and_reads_urdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.urdf'
            path.write_text('<robot name="test"><joint name="hinge"><limit velocity="2.5"/></joint></robot>')
            coords, dofs, lower, upper, velocity = scalar_joint_metadata(self.model(), path)
            np.testing.assert_array_equal(coords, [7])
            np.testing.assert_array_equal(dofs, [6])
            np.testing.assert_allclose([lower, upper, velocity], [[-1], [1], [2.5]])
            path.write_text('<robot name="test"/>')
            with self.assertRaises(ValueError):
                scalar_joint_metadata(self.model(), path)

    def test_objective_analytic_derivatives(self):
        model = self.model()
        q = wp.array(np.tile(model.joint_q.numpy(), (1, 1)), dtype=float)
        q_np = q.numpy().copy()
        q_np[0, 7] = 0.95
        q.assign(q_np)
        target = wp.zeros(q.shape, dtype=float)
        objectives = [IKTemporalObjective([7], [6], target, 2.0),
                      IKNearLimitObjective(np.array([7]), np.array([6]),
                                           np.array([-1]), np.array([1]), 0.2, 3.0)]
        for objective in objectives:
            q.assign(q_np)
            solver = ik.IKSolver(model, 1, [objective], jacobian_mode=ik.IKJacobianType.ANALYTIC)
            residual = wp.zeros((1, 1), dtype=float)
            jacobian = wp.zeros((1, 1, model.joint_dof_count), dtype=float)
            objective.compute_residuals(None, q, model, residual, 0, None)
            initial_residual = residual.numpy()[0, 0]
            objective.compute_jacobian_analytic(None, q, model, jacobian, None, 0)
            perturbed = q_np.copy()
            perturbed[0, 7] += 1e-3
            q.assign(perturbed)
            objective.compute_residuals(None, q, model, residual, 0, None)
            difference = (residual.numpy()[0, 0] - initial_residual) / 1e-3
            self.assertAlmostEqual(difference, jacobian.numpy()[0, 0, 6], places=3)
            q.assign(q_np)
            before = solver.compute_costs(q).numpy()[0]
            solver.step(q, q, iterations=8)
            after = solver.compute_costs(q).numpy()[0]
            self.assertLess(after, before)
            self.assertEqual(objective.dofs.numpy()[0], 6)

    def test_soft_history_matches_export_without_mutating_cpu_pose(self):
        model = self.model()
        temporal = TemporalIK(model, 1, [30], model.joint_q, {'weight': 1})
        pose = temporal.previous.copy()
        pose[0, 0] = -2.0
        pose[0, 7] = 1.5
        q = wp.array(pose, dtype=float)
        temporal.record(q)
        self.assertEqual(temporal.previous[0, 7], 1.0)
        self.assertEqual(temporal.previous[0, 0], -2.0)
        self.assertEqual(q.numpy()[0, 7], 1.5)
        q.zero_()
        self.assertEqual(temporal.previous[0, 7], 1.0)

    def test_projected_solve_and_final_history(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.urdf'
            path.write_text('<robot name="test"><joint name="hinge"><limit velocity="2"/></joint></robot>')
            temporal = TemporalIK(model, 2, [10, 20], model.joint_q,
                                  {'weight': 1, 'mode': 'prediction', 'hard_velocity': True}, path)
            solver = ik.IKSolver(model, 2, [temporal.objective], jacobian_mode=ik.IKJacobianType.ANALYTIC)
            q = wp.array(temporal.previous, dtype=float)
            temporal.prepare()
            temporal.record(q)
            temporal.prepare()
            q_np = q.numpy().copy()
            q_np[:, 7] = 0.9
            q.assign(q_np)
            temporal.solve(solver, q, 3)
            temporal.record(q)
            self.assertTrue(np.all(np.abs(q.numpy()[:, 7]) <= np.array([0.2, 0.1]) + 1e-6))
            final = q.numpy()
            final[:, 7] = [0.1, -0.05]
            q.assign(final)
            temporal.record(q)
            temporal.prepare()
            np.testing.assert_allclose(temporal.target.numpy()[:, 7],
                                       (2 * temporal.previous - temporal.older)[:, 7])
            np.testing.assert_allclose(temporal.previous, final)
            temporal.reset_history()
            temporal.prepare()
            np.testing.assert_allclose(temporal.lower.numpy(), [[-1], [-1]])
            np.testing.assert_allclose(temporal.upper.numpy(), [[1], [1]])
            self.assertEqual(temporal.objective.weight_buffer.numpy()[0], 0.0)


if __name__ == '__main__':
    unittest.main()
