"""Native objective equivalence on real Warp kernels; CPU by default."""

import os
import unittest

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from soma_retargeter.pipelines.batched_weight_objectives import (
    BatchedPositionObjective, BatchedRotationObjective,
)


class BatchedWeightObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = os.environ.get('BATCHED_WEIGHT_TEST_DEVICE', 'cpu')
        wp.set_device(cls.device)

    def setUp(self):
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=[0, 0, 1])
        builder.add_articulation([joint])
        self.model = builder.finalize(device=self.device)

    def objective(self, kind, weights=None, scalar=1.0):
        if kind == 'position':
            args = (0, wp.vec3(0.3, -0.2, 0.1), wp.array([[0.1, 0.2, 0], [-0.2, 0.1, 0.3]], dtype=wp.vec3))
            native, batched = ik.IKObjectivePosition, BatchedPositionObjective
        else:
            args = (0, wp.quat_identity(), wp.array([[0, 0, 0, 1], [0.1, 0, 0, np.sqrt(0.99)]], dtype=wp.vec4))
            native, batched = ik.IKObjectiveRotation, BatchedRotationObjective
        objective = native(*args, weight=scalar) if weights is None else batched(
            *args, weight=scalar, per_problem_weights=wp.array(weights, dtype=float))
        objective.bind_device(self.device)
        objective.set_batch_layout(8, 2, 4)
        objective.init_buffers(self.model, ik.IKJacobianType.ANALYTIC)
        return objective

    def evaluate(self, objective, mapping=(0, 0, 1, 1)):
        count = len(mapping)
        transforms = np.tile([0.2, -0.1, 0.3, 0, 0, np.sin(0.2), np.cos(0.2)], (count, 1, 1))
        body = wp.array(transforms, dtype=wp.transform)
        q = wp.zeros((count, 1), dtype=float)
        motion = wp.array(np.tile([0, 0, 1, 0, 0, 0], (count, 1, 1)), dtype=wp.spatial_vector)
        residual = wp.full((count, 8), 7.0, dtype=float)
        jacobian = wp.full((count, 8, 1), 11.0, dtype=float)
        objective.compute_residuals(body, q, self.model, residual, 2, wp.array(mapping, dtype=int))
        objective.compute_jacobian_analytic(body, q, self.model, jacobian, motion, 2)
        return residual.numpy(), jacobian.numpy()

    def test_uniform_weights_match_native_scalar_and_preserve_other_rows(self):
        for kind in ('position', 'rotation'):
            with self.subTest(kind=kind):
                actual = self.evaluate(self.objective(kind, weights=[2.5, 2.5]))
                expected = self.evaluate(self.objective(kind, scalar=2.5))
                for weighted, native in zip(actual, expected):
                    np.testing.assert_allclose(weighted, native, atol=2e-6)
                    np.testing.assert_array_equal(weighted[:, :2], native[:, :2])
                    np.testing.assert_array_equal(weighted[:, 5:], native[:, 5:])

    def test_independent_weights_zero_seed_expansion_and_in_place_update(self):
        for kind in ('position', 'rotation'):
            with self.subTest(kind=kind):
                objective = self.objective(kind, weights=[0, 3])
                pointer = objective.per_problem_weights.ptr
                baseline = self.evaluate(self.objective(kind))
                for weights in ([0, 3], [4, 0.5]):
                    objective.per_problem_weights.assign(np.array(weights, dtype=np.float32))
                    actual = self.evaluate(objective)
                    factors = np.repeat(weights, 2)
                    np.testing.assert_allclose(actual[0][:, 2:5], baseline[0][:, 2:5] * factors[:, None], atol=2e-6)
                    np.testing.assert_allclose(actual[1][:, 2:5], baseline[1][:, 2:5] * factors[:, None, None], atol=2e-6)
                    self.assertEqual(pointer, objective.per_problem_weights.ptr)

    def test_line_search_residuals_use_supplied_mapping(self):
        mapping = (1, 0, 1, 1, 0, 0)
        for kind in ('position', 'rotation'):
            objective = self.objective(kind, weights=[2, 5])
            baseline = self.objective(kind)
            count = len(mapping)
            body = wp.array(np.tile([0, 0, 0, 0, 0, 0, 1], (count, 1, 1)), dtype=wp.transform)
            q = wp.zeros((count, 1), dtype=float)
            indices = wp.array(mapping, dtype=int)
            expected = wp.zeros((count, 8), dtype=float)
            actual = wp.zeros_like(expected)
            baseline.compute_residuals(body, q, self.model, expected, 2, indices)
            objective.compute_residuals(body, q, self.model, actual, 2, indices)
            np.testing.assert_allclose(actual.numpy(), expected.numpy() * np.array([2, 5])[list(mapping), None], atol=2e-6)

    def test_native_solver_seeded_costs_match_scalar_objectives(self):
        for kind in ('position', 'rotation'):
            objective = self.objective(kind, weights=[2.5, 2.5])
            native = self.objective(kind, scalar=2.5)
            solver = ik.IKSolver(self.model, 2, [objective], n_seeds=2, sampler=ik.IKSampler.GAUSS,
                                 jacobian_mode=ik.IKJacobianType.ANALYTIC)
            reference = ik.IKSolver(self.model, 2, [native], n_seeds=2, sampler=ik.IKSampler.GAUSS,
                                    jacobian_mode=ik.IKJacobianType.ANALYTIC)
            q = wp.array([[0.3], [-0.4]], dtype=float)
            q_reference = wp.clone(q)
            solver.step(q, q, iterations=3)
            reference.step(q_reference, q_reference, iterations=3)
            np.testing.assert_allclose(q.numpy(), q_reference.numpy(), atol=2e-5)
            np.testing.assert_allclose(solver.costs_expanded.numpy(),
                                       reference.costs_expanded.numpy(), atol=2e-5)

    def test_cuda_graph_replay_reads_updated_weight_buffer(self):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest('CUDA graph replay requires a CUDA device')
        for kind in ('position', 'rotation'):
            objective = self.objective(kind, weights=[1, 1])
            body = wp.array(np.tile([0, 0, 0, 0, 0, 0, 1], (4, 1, 1)), dtype=wp.transform)
            q = wp.zeros((4, 1), dtype=float)
            mapping = wp.array([0, 0, 1, 1], dtype=int)
            residual = wp.zeros((4, 8), dtype=float)
            jacobian = wp.zeros((4, 8, 1), dtype=float)
            motion = wp.array(np.tile([0, 0, 1, 0, 0, 0], (4, 1, 1)), dtype=wp.spatial_vector)
            objective.compute_residuals(body, q, self.model, residual, 2, mapping)
            objective.compute_jacobian_analytic(body, q, self.model, jacobian, motion, 2)
            baseline = residual.numpy().copy()
            baseline_jacobian = jacobian.numpy().copy()
            with wp.ScopedCapture(device=self.device) as capture:
                objective.compute_residuals(body, q, self.model, residual, 2, mapping)
                objective.compute_jacobian_analytic(body, q, self.model, jacobian, motion, 2)
            objective.per_problem_weights.assign(np.array([2, 5], dtype=np.float32))
            wp.capture_launch(capture.graph)
            factors = np.array([2, 2, 5, 5])
            np.testing.assert_allclose(residual.numpy(), baseline * factors[:, None], atol=2e-6)
            np.testing.assert_allclose(jacobian.numpy(), baseline_jacobian * factors[:, None, None], atol=2e-6)

    def test_unsupported_backend_and_bad_weight_shape_fail(self):
        objective = self.objective('position', weights=[1, 1])
        with self.assertRaisesRegex(ValueError, 'ANALYTIC'):
            objective.init_buffers(self.model, ik.IKJacobianType.AUTODIFF)
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.objective('position', weights=[1])
        objective.set_batch_layout(8, 2, 3)
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            objective.init_buffers(self.model, ik.IKJacobianType.ANALYTIC)


if __name__ == '__main__':
    unittest.main()
