"""Select only measured, tracking-safe candidates; emit JSON, never force a winner."""
import argparse
import json
from pathlib import Path

import numpy as np


POLICY = {
    'position_relative_tolerance': 0.02,
    'position_absolute_tolerance_m': 0.001,
    'rotation_relative_tolerance': 0.02,
    'rotation_absolute_tolerance_rad': 0.01,
    'per_clip_temporal_regression_tolerance': 0.05,
    'required_mean_jerk_rms_improvement': 0.05,
    'hard_velocity_relative_tolerance': 1e-4,
    'hard_velocity_required_exceedance_samples': 0,
}


def compare(candidate, baseline, policy):
    failures = []
    ratios = []
    for name, record in candidate.items():
        base = baseline[name]['final']
        final = record['final']
        for metric, relative, absolute in (
            ('position_rmse_m', policy['position_relative_tolerance'], policy['position_absolute_tolerance_m']),
            ('rotation_rmse_rad', policy['rotation_relative_tolerance'], policy['rotation_absolute_tolerance_rad']),
        ):
            old, new = base['tracking'][metric], final['tracking'][metric]
            if new > old + max(absolute, old * relative):
                failures.append({'motion': name, 'metric': metric, 'baseline': old, 'candidate': new})
            per_effector_key = 'per_effector_' + metric
            old_errors = np.asarray(base['tracking'][per_effector_key])
            new_errors = np.asarray(final['tracking'][per_effector_key])
            if old_errors.shape != new_errors.shape:
                raise ValueError('Effector arrays differ between candidate and baseline')
            for index in np.flatnonzero(new_errors > old_errors + np.maximum(absolute, old_errors * relative)):
                failures.append({'motion': name, 'metric': per_effector_key, 'effector_index': int(index),
                    'baseline': float(old_errors[index]), 'candidate': float(new_errors[index])})
        clip_ratios = []
        for derivative in ('velocity_rad_s', 'acceleration_rad_s2', 'jerk_rad_s3'):
            for statistic in ('rms', 'p99', 'max'):
                old = base['temporal'][derivative][statistic]
                new = final['temporal'][derivative][statistic]
                ratio = new / max(old, 1e-12) if old > 0 or new > 0 else 1.0
                if ratio > 1 + policy['per_clip_temporal_regression_tolerance']:
                    failures.append({'motion': name, 'metric': derivative + '.' + statistic, 'baseline': old, 'candidate': new})
                if statistic == 'rms':
                    clip_ratios.append(ratio)
        old = baseline[name]['exported_csv_temporal']['velocity_limits']['exceedance_samples_with_rounding_tolerance']
        new = record['exported_csv_temporal']['velocity_limits']['exceedance_samples_with_rounding_tolerance']
        if record['temporal_config']['temporal_ik'].get('hard_velocity', False) and new != 0:
            failures.append({'motion': name, 'metric': 'hard_velocity_feasibility_with_rounding_tolerance',
                'required': 0, 'candidate': new})
        if new > old:
            failures.append({'motion': name, 'metric': 'velocity_limit_exceedances', 'baseline': old, 'candidate': new})
        ratios.append(clip_ratios)
    mean_ratios = np.mean(ratios, axis=0)
    return {
        'strict_safe': not failures,
        'failures': failures,
        'equal_clip_mean_rms_ratio': dict(zip(('velocity', 'acceleration', 'jerk'), mean_ratios.tolist())),
        'meaningful_improvement': bool(mean_ratios[2] <= 1 - policy['required_mean_jerk_rms_improvement']),
    }


def select(records, expected_motions, policy=None):
    policy = POLICY if policy is None else policy
    grouped = {}
    for record in records:
        key = (record['weight_set'], record['variant'])
        group = grouped.setdefault(key, {})
        if record['motion'] in group:
            raise ValueError(f'Duplicate measurement: {key} / {record["motion"]}')
        group[record['motion']] = record
    report = {'policy': policy, 'aggregation': 'equal clip weight, never concatenate clips', 'weight_sets': {}}
    for weight_set in sorted({key[0] for key in grouped}):
        baseline = grouped.get((weight_set, 'baseline'), {})
        if set(baseline) != set(expected_motions):
            raise ValueError(f'Incomplete baseline for {weight_set}')
        candidates = {}
        for (weights, variant), clips in grouped.items():
            if weights != weight_set or variant == 'baseline':
                continue
            if set(clips) != set(expected_motions):
                candidates[variant] = {'status': 'incomplete', 'missing': sorted(set(expected_motions) - set(clips))}
                continue
            for name, record in clips.items():
                if record['controlled_config_sha256'] != baseline[name]['controlled_config_sha256']:
                    raise ValueError(f'Uncontrolled non-temporal config drift: {name} / {variant}')
                for field in ('input_sha256', 'urdf_sha256', 'native_source_sha256'):
                    if record[field] != baseline[name][field]:
                        raise ValueError(f'{field} drift: {name} / {variant}')
            candidates[variant] = compare(clips, baseline, policy)
        eligible = {k: v for k, v in candidates.items() if v.get('strict_safe') and v.get('meaningful_improvement')}
        pareto = []
        for variant, value in eligible.items():
            vector = list(value['equal_clip_mean_rms_ratio'].values())
            dominated = False
            for other, other_value in eligible.items():
                other_vector = list(other_value['equal_clip_mean_rms_ratio'].values())
                if other != variant and all(a <= b for a, b in zip(other_vector, vector)) and any(a < b for a, b in zip(other_vector, vector)):
                    dominated = True
            if not dominated:
                pareto.append(variant)
        report['weight_sets'][weight_set] = {
            'candidates': candidates,
            'pareto_safe_candidates': sorted(pareto),
            'selection': pareto[0] if len(pareto) == 1 else None,
            'reason': 'unique tracking-safe Pareto candidate' if len(pareto) == 1 else 'no safe improvement or unresolved tradeoff; do not force a winner',
        }
    if not report['weight_sets']:
        raise ValueError('No measurements')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in sorted(args.results.rglob('*.metrics.json'))]
    names = [entry['name'] for entry in json.loads(args.selection.read_text())]
    report = select(records, names)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
