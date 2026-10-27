import copy
import unittest

import numpy as np

from metrics import tracking_metrics, trajectory_metrics
from selection import select
from plan import configurations
from run import build_config


NAMES = [f'joint_{index}' for index in range(31)]


class MetricsTests(unittest.TestCase):
    def test_derivatives_have_physical_units_and_exclude_root(self):
        q = np.zeros((10, 38))
        q[:, :7] = np.arange(10)[:, None] ** 4
        q[:, 7:] = np.arange(10)[:, None] * 0.02
        result = trajectory_metrics(q, 100, NAMES, np.ones(31))
        self.assertAlmostEqual(result['velocity_rad_s']['rms'], 2)
        self.assertLess(result['acceleration_rad_s2']['max'], 1e-10)
        self.assertLess(result['jerk_rad_s3']['max'], 1e-8)
        self.assertEqual(result['velocity_limits']['exceedance_samples'], 9 * 31)
        self.assertEqual(result['velocity_limits']['exceedance_frames'], 9)

    def test_spike_localization_includes_original_frame_interval(self):
        q = np.zeros((8, 38))
        q[4:, 12] = 1
        result = trajectory_metrics(q, 120, NAMES, np.full(31, 10))
        self.assertEqual(result['velocity_rad_s']['worst_joint'], 'joint_5')
        self.assertEqual(result['velocity_rad_s']['worst_frame_interval'], [3, 4])
        self.assertEqual(result['velocity_rad_s']['max'], 120)
        self.assertEqual(result['velocity_limits']['exceedance_samples'], 1)
        self.assertEqual(result['velocity_limits']['max_ratio'], 12)

    def test_csv_rounding_tolerance_is_distinct_from_strict_exceedance(self):
        q = np.zeros((8, 38))
        q[:, 7:] = np.arange(8)[:, None] * 1.00005 / 120
        limits = trajectory_metrics(q, 120, NAMES, np.ones(31))['velocity_limits']
        self.assertEqual(limits['exceedance_samples'], 7 * 31)
        self.assertEqual(limits['exceedance_samples_with_rounding_tolerance'], 0)

    def test_quaternion_sign_is_not_rotation_error(self):
        target = np.zeros((5, 2, 7))
        target[..., 6] = 1
        result = tracking_metrics(target[..., :3], -target[..., 3:], target)
        self.assertEqual(result['position_rmse_m'], 0)
        self.assertEqual(result['rotation_rmse_rad'], 0)

    def test_invalid_trajectory_is_rejected(self):
        q = np.zeros((8, 38))
        q[2, 11] = np.nan
        with self.assertRaises(ValueError):
            trajectory_metrics(q, 120, NAMES, np.ones(31))


class SelectionTests(unittest.TestCase):
    def record(self, variant, factor=1):
        q = np.zeros((8, 38))
        q[:, 7:] = np.arange(8)[:, None] ** 3 * 0.001 * factor
        temporal = trajectory_metrics(q, 120, NAMES, np.full(31, 100))
        return {
            'weight_set': 'optimized', 'variant': variant, 'motion': 'clip',
            'temporal_config': {'temporal_ik': {'enabled': variant != 'baseline', 'hard_velocity': False}},
            'controlled_config_sha256': 'fixed', 'input_sha256': 'same',
            'urdf_sha256': 'same-urdf', 'native_source_sha256': 'same-native',
            'exported_csv_temporal': temporal,
            'final': {'tracking': {'position_rmse_m': 0.01, 'rotation_rmse_rad': 0.1,
                'per_effector_position_rmse_m': [0.01, 0.01],
                'per_effector_rotation_rmse_rad': [0.1, 0.1]}, 'temporal': temporal},
        }

    def test_safe_improvement_can_win(self):
        result = select([self.record('baseline'), self.record('smooth', 0.8)], ['clip'])
        self.assertEqual(result['weight_sets']['optimized']['selection'], 'smooth')

    def test_tracking_regression_rejects_smoother_candidate(self):
        candidate = self.record('smooth', 0.8)
        candidate['final']['tracking']['position_rmse_m'] = 0.02
        result = select([self.record('baseline'), candidate], ['clip'])
        self.assertIsNone(result['weight_sets']['optimized']['selection'])

    def test_incomplete_candidate_cannot_win(self):
        baseline = self.record('baseline')
        second = copy.deepcopy(baseline)
        second['motion'] = 'second'
        result = select([baseline, second, self.record('smooth', 0.8)], ['clip', 'second'])
        self.assertEqual(result['weight_sets']['optimized']['candidates']['smooth']['status'], 'incomplete')

    def test_config_drift_rejected(self):
        candidate = self.record('smooth', 0.8)
        candidate['controlled_config_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            select([self.record('baseline'), candidate], ['clip'])

    def test_hard_velocity_requires_zero_not_merely_fewer_violations(self):
        baseline = self.record('baseline')
        candidate = self.record('limits', 0.8)
        baseline['exported_csv_temporal']['velocity_limits']['exceedance_samples_with_rounding_tolerance'] = 2
        candidate['exported_csv_temporal']['velocity_limits']['exceedance_samples_with_rounding_tolerance'] = 1
        candidate['temporal_config']['temporal_ik']['hard_velocity'] = True
        result = select([baseline, candidate], ['clip'])
        self.assertIsNone(result['weight_sets']['optimized']['selection'])
        failures = result['weight_sets']['optimized']['candidates']['limits']['failures']
        self.assertTrue(any(failure['metric'] == 'hard_velocity_feasibility_with_rounding_tolerance' for failure in failures))

    def test_no_improvement_does_not_force_winner(self):
        result = select([self.record('baseline'), self.record('same')], ['clip'])
        self.assertIsNone(result['weight_sets']['optimized']['selection'])


class ControlledConfigTests(unittest.TestCase):
    def test_families_isolate_smoothness_and_limits(self):
        variants = configurations([4])
        self.assertFalse(variants['baseline']['temporal_ik']['enabled'])
        self.assertEqual(variants['smooth-4']['temporal_ik']['weight'], 4)
        self.assertFalse(variants['smooth-4']['temporal_ik']['hard_velocity'])
        self.assertEqual(variants['limits-4']['temporal_ik']['weight'], 0)
        self.assertTrue(variants['limits-4']['temporal_ik']['hard_velocity'])
        self.assertEqual(variants['combined-4']['temporal_ik']['weight'], 4)
        self.assertTrue(variants['combined-4']['temporal_ik']['hard_velocity'])
        self.assertEqual(variants['limits-4']['temporal_ik']['velocity_avoidance_weight'], 4)
        self.assertEqual(variants['limits-4']['temporal_ik']['velocity_margin_fraction'], 0.2)
        self.assertEqual(variants['limits-4']['temporal_ik']['limit_weight'], 0)
        self.assertEqual(variants['smooth-4']['temporal_ik']['velocity_avoidance_weight'], 0)

    def test_only_tracking_weights_copy_from_existing_candidate(self):
        default = {'smooth_joint_filter_weight': 5.5, 'ik_match_table': {
            'Hips': {'t_weight': 1, 'r_weight': 2, 't_body': 'pelvis'}}}
        fixed = {'smooth_joint_filter_weight': 999, 'ik_match_table': {
            'Hips': {'t_weight': 3, 'r_weight': 4, 't_body': 'wrong'}}}
        _, baseline_hash = build_config(default, fixed, configurations([4])['baseline'])
        candidate, candidate_hash = build_config(default, fixed, configurations([4])['combined-4'])
        self.assertEqual(baseline_hash, candidate_hash)
        self.assertEqual(candidate['smooth_joint_filter_weight'], 5.5)
        self.assertEqual(candidate['ik_match_table']['Hips']['t_body'], 'pelvis')
        self.assertEqual(candidate['ik_match_table']['Hips']['t_weight'], 3)
        self.assertEqual(default['ik_match_table']['Hips']['t_weight'], 1)

    def test_outer_override_is_rejected(self):
        with self.assertRaises(ValueError):
            build_config({'ik_match_table': {}}, None, {'enable_post_processing': False})


if __name__ == '__main__':
    unittest.main()
