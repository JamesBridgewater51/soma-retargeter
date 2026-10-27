"""Regression for one-step LM resets and capture-stable mixed-environment damping."""

import os
from pathlib import Path
import tempfile
import unittest

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from soma_retargeter.pipelines.temporal_ik import TemporalIK


@wp.kernel
def _cubic_residual(q: wp.array2d(dtype=float), start: int, residual: wp.array2d(dtype=float)):
    env = wp.tid()
    x = q[env, 0]
    residual[env, start] = x * x * x - 1.0


@wp.kernel
def _cubic_jacobian(q: wp.array2d(dtype=float), start: int, jacobian: wp.array3d(dtype=float)):
    env = wp.tid()
    x = q[env, 0]
    jacobian[env, start, 0] = 3.0 * x * x


class CubicObjective(ik.IKObjective):
    def residual_dim(self):
        return 1

    def supports_analytic(self):
        return True

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_cubic_residual, joint_q.shape[0], inputs=[joint_q, start_idx, residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_cubic_jacobian, joint_q.shape[0], inputs=[joint_q, start_idx, jacobian], device=self.device)


class TemporalDampingTests(unittest.TestCase):
    def scalar_model(self, device):
        builder = newton.ModelBuilder()
        link = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=[0, 0, 1], label='hinge')
        builder.add_articulation([joint])
        builder.joint_limit_lower[-1] = -1.0
        builder.joint_limit_upper[-1] = 1.0
        return builder.finalize(device=device)

    def exercise_mixed_environments(self, device, capture):
        with wp.ScopedDevice(device), tempfile.TemporaryDirectory() as directory:
            model = self.scalar_model(device)
            urdf = Path(directory) / 'robot.urdf'
            urdf.write_text('<robot name="test"><joint name="hinge"><limit velocity="2"/></joint></robot>')
            temporal = TemporalIK(model, 2, [10, 10], model.joint_q, {'hard_velocity': True}, urdf)
            solver = ik.IKSolver(model, 2, [CubicObjective()], jacobian_mode=ik.IKJacobianType.ANALYTIC)
            initial = np.array([[0.25], [0.9]], dtype=np.float32)
            temporal.previous = initial.copy()
            temporal.frames = 1
            temporal.prepare()
            q = wp.array(initial.copy(), dtype=float, device=device)
            before = solver.compute_costs(q).numpy().copy()
            # At q=.25 the initial LM proposal is rejected; at q=.9 it is accepted.
            # Re-entering public step(1) repeats the first rejection indefinitely.
            for _ in range(4):
                solver.step(q, q, iterations=1)
                temporal.project(q)
            self.assertEqual(q.numpy()[0, 0], initial[0, 0])
            self.assertGreater(q.numpy()[1, 0], initial[1, 0])
            q.assign(initial)
            temporal.solve(solver, q, 1)
            np.testing.assert_allclose(solver._impl.lambda_values.numpy(), [0.2, 0.05], atol=1e-7)
            q.assign(initial)
            temporal.solve(solver, q, 8)
            expected = q.numpy().copy()
            after = solver.compute_costs(q).numpy().copy()
            self.assertGreater(expected[0, 0], initial[0, 0] + 0.1)
            self.assertTrue(np.all(np.isfinite(expected)))
            self.assertTrue(np.all(after <= before))
            self.assertTrue(np.all(expected >= temporal.lower.numpy() - 2e-6))
            self.assertTrue(np.all(expected <= temporal.upper.numpy() + 2e-6))
            if capture:
                q.assign(initial)
                with wp.ScopedCapture() as captured:
                    temporal.solve(solver, q, 8)
                wp.capture_launch(captured.graph)
                np.testing.assert_array_equal(q.numpy(), expected)
                # Swap accepting/rejecting roles and update the SAME bound buffers.
                swapped = initial[::-1].copy()
                temporal.previous = swapped.copy()
                temporal.prepare()
                q.assign(swapped)
                temporal.solve(solver, q, 8)
                swapped_expected = q.numpy().copy()
                q.assign(swapped)
                wp.capture_launch(captured.graph)
                np.testing.assert_array_equal(q.numpy(), swapped_expected)
            lbfgs = ik.IKSolver(model, 2, [CubicObjective()], optimizer=ik.IKOptimizer.LBFGS,
                                jacobian_mode=ik.IKJacobianType.ANALYTIC)
            with self.assertRaisesRegex(ValueError, 'LM backend'):
                temporal.solve(lbfgs, q, 1)
            expanded = ik.IKSolver(model, 2, [CubicObjective()], sampler=ik.IKSampler.ROBERTS,
                                   n_seeds=2, jacobian_mode=ik.IKJacobianType.ANALYTIC)
            with self.assertRaisesRegex(ValueError, 'single-seed'):
                temporal.solve(expanded, q, 1)

    def test_cpu_rejected_environment_progresses_without_perturbing_other_damping(self):
        wp.init()
        self.exercise_mixed_environments('cpu', False)

    @unittest.skipUnless(os.environ.get('TEMPORAL_IK_TEST_CUDA') == '1', 'CUDA test requires an exclusive GPU slot')
    def test_cuda_capture_matches_uncaptured_after_mixed_environment_buffer_updates(self):
        wp.init()
        self.exercise_mixed_environments('cuda:0', True)


if __name__ == '__main__':
    unittest.main()
