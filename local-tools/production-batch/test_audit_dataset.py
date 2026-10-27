"""Real-file, CPU/stdlib-only coverage and receipt binding regressions."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import audit_dataset as audit
import run_batch as batch


HEADERS = batch.ROOT_HEADER + [f'joint_{index}' for index in range(31)]


class DatasetFixture:
    def __init__(self, root):
        self.root = root
        self.sources = root / 'sources'
        self.sources.mkdir()
        self.manifest = root / 'manifest.jsonl'
        self.settings = {'weights': str(root / 'weights'), 'output': str(root / 'csv'),
                         'artifacts': str(root / 'artifacts'), 'optimizer': {'outer_iterations': 50},
                         'engine': 'native-independent-batched', 'reuse_geometry': True, 'fast_bvh': True,
                         'conversion_overrides': {}, 'provenance': {'selected-implementation': 'frozen-sha'},
                         'base_retargeter': {'human_robot_scaler_config': 'fixed.json',
                                            'ik_match_table': {'Foot': {'t_weight': 1.0, 'r_weight': 2.0}}}}
        self.entries = []
        for index, frames in enumerate((2, 3)):
            source = self.sources / f'clip-{index}.bvh'
            source.write_text(f'HIERARCHY\nMOTION\nFrames: {frames}\nFrame Time: 0.0333667\n' + '0 0 0\n' * frames)
            self.entries.append({'output_filename': source.name, 'archive_member': f'Uniform/{source.name}',
                                 'bytes': source.stat().st_size, 'sha256': batch.digest(source)})
        self.manifest.write_text(''.join(json.dumps(entry) + '\n' for entry in self.entries))
        self.jobs = list(batch.source_rows(self.manifest, self.sources))
        for job in self.jobs:
            self.publish(job)

    def paths(self, job):
        directory = Path(self.settings['weights']) / job['stem']
        return {'config': directory / 'retargeter.json', 'optimized': directory / 'optimized.done.json',
                'csv': Path(self.settings['output']) / f"{job['stem']}.csv",
                'done': Path(self.settings['artifacts']) / 'done' / f"{job['stem']}.json"}

    def publish(self, job):
        paths = self.paths(job)
        metadata = batch.bvh_metadata(job['source'])
        batch.atomic_json(paths['config'], copy.deepcopy(self.settings['base_retargeter']))
        optimized = {'content_key': batch.optimization_key(job, self.settings),
                     'sha256': batch.digest(paths['config']), 'source_sha256': job['sha256'], 'source': job['source']}
        batch.atomic_json(paths['optimized'], optimized)
        paths['csv'].parent.mkdir(parents=True, exist_ok=True)
        with paths['csv'].open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADERS)
            writer.writerows([[frame] + [0.1] * 37 for frame in range(metadata['frames'])])
        checked = batch.validate_csv(paths['csv'], HEADERS, metadata['frames'],
                                     batch.velocity_plan(self.settings, metadata['fps']))
        batch.atomic_json(paths['done'], {'content_key': batch.conversion_key_for(optimized, self.settings),
                          'sha256': batch.digest(paths['csv']), 'optimization_key': optimized['content_key'],
                          'weights_sha256': optimized['sha256'], 'source_sha256': job['sha256'],
                          'archive_member': job['archive_member'], 'headers': HEADERS,
                          'validation': checked, **metadata})

    def enable_hard_velocity(self, scale=0.8):
        from csv_velocity import load_velocity_contract
        urdf = self.root / 'robot.urdf'
        urdf.write_text('<robot name="synthetic">' + ''.join(
            f'<joint name="{name}" type="revolute"><limit velocity="2.0"/></joint>'
            for name in HEADERS[len(batch.ROOT_HEADER):]) + '</robot>')
        self.settings['conversion_overrides'] = {'temporal_ik': {'enabled': True, 'hard_velocity': True,
                                                               'velocity_scale': scale}}
        self.settings['velocity_validation'] = load_velocity_contract(urdf, scale)
        self.settings['provenance'][str(urdf)] = batch.digest(urdf)
        for job in self.jobs:
            self.publish(job)

    def run(self, workers=1):
        return audit.audit_dataset(self.manifest, self.sources, self.settings, expected_count=2, workers=workers)


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = DatasetFixture(Path(self.directory.name))

    def test_full_hash_bound_coverage_is_complete_with_summed_frames(self):
        report = self.fixture.run()
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(report['valid_count'], 2)
        self.assertEqual(report['expected_frames'], 5)
        self.assertEqual(report['validated_frames'], 5)
        self.assertEqual(report['headers'], HEADERS)
        self.assertEqual(report['products'][0]['fps'], 29.97)
        self.assertEqual(report['failure_categories'], {})
        json.dumps(report, allow_nan=False)

    def test_missing_products_cannot_be_hidden_by_success_progress(self):
        for kind in ('csv', 'done', 'config', 'optimized'):
            with self.subTest(kind=kind):
                fixture = self.fixture
                paths = fixture.paths(fixture.jobs[0])
                paths[kind].unlink()
                batch.atomic_json(Path(fixture.settings['artifacts']) / 'success.progress.json',
                                  {'completed': 2, 'failed': 0})
                report = fixture.run()
                self.assertEqual(report['status'], 'INCOMPLETE')
                self.assertEqual(report['valid_count'], 1)
                self.assertEqual(report['failed_count'], 1)
                fixture.publish(fixture.jobs[0])

    def test_changed_csv_content_is_rejected_without_numeric_row_reparse(self):
        for suffix in ('nan,0\n', '0,1\n', '2,' + ','.join(['0'] * 37) + '\n'):
            with self.subTest(suffix=suffix):
                paths = self.fixture.paths(self.fixture.jobs[0])
                with paths['csv'].open('a') as stream:
                    stream.write(suffix)
                report = self.fixture.run()
                self.assertEqual(report['status'], 'INCOMPLETE')
                self.assertIn('receipt_or_hash_mismatch', report['failure_categories'])
                self.fixture.publish(self.fixture.jobs[0])

    def test_selected_implementation_or_conversion_change_rejects_old_receipts(self):
        for key, changed in (('fast_bvh', False), ('reuse_geometry', False),
                             ('provenance', {'selected-implementation': 'different'}),
                             ('conversion_overrides', {'temporal_ik': {'enabled': True}})):
            with self.subTest(key=key):
                original = self.fixture.settings[key]
                self.fixture.settings[key] = changed
                self.assertEqual(self.fixture.run()['status'], 'INCOMPLETE')
                self.fixture.settings[key] = original

    def test_hard_velocity_selection_cannot_complete_without_native_attestation(self):
        self.fixture.settings['conversion_overrides'] = {'temporal_ik': {'enabled': True, 'hard_velocity': True}}
        for job in self.fixture.jobs:
            self.fixture.publish(job)
        report = self.fixture.run()
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertIn('velocity_plan', report['failure_categories'])

    def test_real_native_hard_attestations_complete_and_preserve_physical_evidence(self):
        self.fixture.enable_hard_velocity()
        report = self.fixture.run()
        self.assertEqual(report['status'], 'COMPLETE')
        for product in report['products']:
            velocity = product['validation']['velocity']
            self.assertEqual(velocity['status'], 'passed')
            self.assertEqual(velocity['velocity_scale'], 0.8)
            self.assertEqual(velocity['checked_pairs'], (product['frames'] - 1) * 31)
            self.assertEqual(velocity['violation_count'], 0)

    def test_corrupted_hard_attestation_fails_without_reparsing_numeric_rows(self):
        self.fixture.enable_hard_velocity()
        paths = self.fixture.paths(self.fixture.jobs[0])
        original = batch.load_json(paths['done'])
        mutations = [('schema', 9), ('status', 'failed'), ('enabled', False), ('frames', 99),
                     ('joint_count', 30), ('checked_pairs', 0), ('violation_count', 1),
                     ('exceedance_frames', 1), ('max_ratio', 1.1), ('max_ratio', float('nan')),
                     ('native_fps', 120), ('angle_units', 'rad'), ('limit_units', 'deg/s'),
                     ('relative_tolerance', 0.1), ('velocity_scale', 1.0),
                     ('urdf_sha256', '0' * 64), ('ordered_joint_names_sha256', '0' * 64),
                     ('velocity_limits_sha256', '0' * 64), ('effective_limits_sha256', '0' * 64),
                     ('fingerprint_schema', 'unknown'), ('worst_joint', 'missing'),
                     ('worst_frame_interval', [0, 99])]
        for key, changed in mutations:
            with self.subTest(key=key, changed=changed):
                corrupted = copy.deepcopy(original)
                corrupted['validation']['velocity'][key] = changed
                paths['done'].write_text(json.dumps(corrupted))
                report = self.fixture.run()
                self.assertEqual(report['status'], 'INCOMPLETE')
                self.assertIn('velocity_attestation', report['failure_categories'])
        batch.atomic_json(paths['done'], original)
        del self.fixture.settings['velocity_validation']['velocity_scale']
        self.assertIn('velocity_plan', self.fixture.run()['failure_categories'])

    def test_receipt_source_weights_metadata_and_header_mismatch_are_rejected(self):
        mutations = [('source_sha256', 'wrong'), ('archive_member', 'wrong'),
                     ('weights_sha256', 'wrong'), ('optimization_key', 'wrong'),
                     ('frames', 999), ('fps', 120), ('source_fps', 120),
                     ('source_frame_time', 0.01), ('headers', HEADERS[:-1]),
                     ('headers', HEADERS[:-1] + [HEADERS[-2]])]
        paths = self.fixture.paths(self.fixture.jobs[0])
        original = batch.load_json(paths['done'])
        for key, changed in mutations:
            with self.subTest(key=key, changed=changed):
                batch.atomic_json(paths['done'], {**original, key: changed})
                self.assertEqual(self.fixture.run()['status'], 'INCOMPLETE')
        batch.atomic_json(paths['done'], original)

    def test_hash_matching_nonfinite_or_nonweight_configuration_is_rejected(self):
        paths = self.fixture.paths(self.fixture.jobs[0])
        original = copy.deepcopy(self.fixture.settings['base_retargeter'])
        for bad_weight in (float('nan'), float('inf'), -1.0):
            with self.subTest(bad_weight=bad_weight):
                changed = copy.deepcopy(original)
                changed['ik_match_table']['Foot']['t_weight'] = bad_weight
                paths['config'].write_text(json.dumps(changed))
                receipt = batch.load_json(paths['optimized'])
                receipt['sha256'] = batch.digest(paths['config'])
                batch.atomic_json(paths['optimized'], receipt)
                report = self.fixture.run()
                self.assertEqual(report['status'], 'INCOMPLETE')
                self.assertIn('invalid_metadata_or_weights', report['failure_categories'])
                self.fixture.publish(self.fixture.jobs[0])

    def test_duplicate_inputs_count_shortfall_and_unexpected_products_are_incomplete(self):
        original = self.fixture.manifest.read_text()
        self.fixture.manifest.write_text(original + original.splitlines()[0] + '\n')
        report = self.fixture.run()
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertIn('input_manifest', report['failure_categories'])
        self.fixture.manifest.write_text(original.splitlines()[0] + '\n')
        report = self.fixture.run()
        self.assertIn('input_count', report['failure_categories'])
        self.assertIn('unexpected_products', report['failure_categories'])

    def test_source_size_change_and_data_aliases_are_rejected(self):
        source = Path(self.fixture.jobs[0]['source'])
        with source.open('a') as stream:
            stream.write('changed')
        self.assertIn('source_size', self.fixture.run()['failure_categories'])
        paths = self.fixture.paths(self.fixture.jobs[1])
        target = paths['csv'].with_suffix('.saved')
        paths['csv'].rename(target)
        paths['csv'].symlink_to(target)
        self.assertIn('data_alias', self.fixture.run()['failure_categories'])

    def test_cli_cannot_overwrite_production_csv_with_audit_report(self):
        settings_path = self.fixture.root / 'selected-settings.json'
        batch.atomic_json(settings_path, self.fixture.settings)
        csv_path = self.fixture.paths(self.fixture.jobs[0])['csv']
        original = csv_path.read_bytes()
        with self.assertRaisesRegex(audit.AuditFailure, 'must not overwrite'):
            audit.main(['--manifest', str(self.fixture.manifest), '--source-root', str(self.fixture.sources),
                        '--settings', str(settings_path), '--expected-count', '2', '--audit-output', str(csv_path)])
        self.assertEqual(csv_path.read_bytes(), original)

    def test_missing_manifest_is_incomplete_not_a_false_empty_success(self):
        self.fixture.manifest.unlink()
        report = self.fixture.run()
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['input_count'], 0)
        self.assertIsNone(report['input_manifest_sha256'])
        self.assertIn('input_manifest', report['failure_categories'])

    def test_parallel_results_equal_serial_for_valid_and_corrupted_hard_receipts(self):
        self.fixture.enable_hard_velocity()
        for corrupted in (False, True):
            with self.subTest(corrupted=corrupted):
                if corrupted:
                    paths = self.fixture.paths(self.fixture.jobs[0])
                    with paths['csv'].open('a') as stream:
                        stream.write('nan,0\n')
                    paths = self.fixture.paths(self.fixture.jobs[1])
                    done = batch.load_json(paths['done'])
                    done['validation']['velocity']['violation_count'] = 1
                    batch.atomic_json(paths['done'], done)
                serial = self.fixture.run()
                for key in ('elapsed_seconds', 'audit_workers'):
                    serial.pop(key)
                for workers in (2, 8, 16):
                    parallel = self.fixture.run(workers=workers)
                    for key in ('elapsed_seconds', 'audit_workers'):
                        parallel.pop(key)
                    self.assertEqual(parallel, serial)

    def test_invalid_worker_count_is_rejected_before_audit(self):
        for workers in (0, -1):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'workers must be positive'):
                self.fixture.run(workers=workers)

    def test_cli_writes_atomic_incomplete_report_and_returns_nonzero(self):
        paths = self.fixture.paths(self.fixture.jobs[0])
        paths['done'].unlink()
        settings_path = self.fixture.root / 'selected-settings.json'
        batch.atomic_json(settings_path, self.fixture.settings)
        report_path = Path(self.fixture.settings['artifacts']) / 'audit.json'
        exit_code = audit.main(['--manifest', str(self.fixture.manifest), '--source-root', str(self.fixture.sources),
                                '--settings', str(settings_path), '--expected-count', '2', '--workers', '8',
                                '--audit-output', str(report_path)])
        self.assertEqual(exit_code, 1)
        self.assertEqual(batch.load_json(report_path)['status'], 'INCOMPLETE')
        self.assertEqual(list(report_path.parent.glob('.*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
