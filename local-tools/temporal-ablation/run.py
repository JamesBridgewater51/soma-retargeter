"""Run one full-pipeline, fixed-weight temporal ablation (explicit invocation only)."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO))

import numpy as np
from metrics import tracking_metrics, trajectory_metrics, urdf_velocity_limits


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config(default, fixed, temporal):
    config = copy.deepcopy(default)
    if fixed is not None:
        if set(fixed['ik_match_table']) != set(config['ik_match_table']):
            raise ValueError('Fixed candidate effector set differs from native default')
        for effector, weights in config['ik_match_table'].items():
            for key in ('t_weight', 'r_weight'):
                value = float(fixed['ik_match_table'][effector][key])
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid fixed tracking weight: {effector}.{key}')
                weights[key] = value
    if not config.get('enable_post_processing', True) or not config.get('enable_contact_processing', True):
        raise ValueError('Outer post/contact processing must remain enabled')
    controlled_hash = digest(config)
    # Restrict override scope: no accidental tracking/outer/collision changes.
    if set(temporal) - {'temporal_ik'}:
        raise ValueError('Only temporal_ik overrides are allowed')
    allowed = {'enabled', 'mode', 'weight', 'hard_velocity', 'velocity_scale', 'limit_margin', 'limit_weight'}
    if not isinstance(temporal.get('temporal_ik'), dict) or set(temporal['temporal_ik']) - allowed:
        raise ValueError('Unknown or malformed temporal_ik configuration')
    config.update(copy.deepcopy(temporal))
    config['capture_base_motion'] = True
    return config, controlled_hash


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--weight-set', choices=('default', 'optimized'), required=True)
    parser.add_argument('--temporal-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, default=Path(os.environ.get('STORAGE_ROOT', '/data/liuminghao')))
    parser.add_argument('--fixed-weight-dir', type=Path)
    return parser.parse_args()


def selected_entry(args):
    selection_path = args.storage_root / 'datasets/bones-seed/metadata/x2-validation-selection.json'
    selection = json.loads(selection_path.read_text())
    entries = [entry for entry in selection if entry['name'] == args.motion]
    if len(entries) != 1:
        raise ValueError('Motion must name exactly one selected validation clip')
    return entries[0]


def fixed_weights(args):
    if args.weight_set == 'default':
        return None, None
    fixed_dir = args.fixed_weight_dir or args.storage_root / 'artifacts/soma-retargeter/agibot-x2ultra/per-sequence-comparison'
    fixed_path = fixed_dir / (args.motion + '.effective-retargeter.json')
    return json.loads(fixed_path.read_text()), fixed_path


def main():
    args = parse_args()
    os.environ['STORAGE_ROOT'] = str(args.storage_root)
    import newton
    import warp as wp
    from soma_retargeter.io import bvh
    from soma_retargeter.pipelines import utils
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils import newton_utils
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType

    root = args.storage_root
    entry = selected_entry(args)
    source = root / 'datasets/bones-seed/raw/soma-uniform/validation' / (entry['name'] + '.bvh')
    fixed, fixed_path = fixed_weights(args)
    default = utils.get_retargeter_config('soma', 'agibot_x2ultra')
    temporal = json.loads(args.temporal_config.read_text())
    config, controlled_hash = build_config(default, fixed, temporal)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'effective-config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    wp.init()
    skeleton, animation = bvh.load_bvh(str(source))
    pipeline = SomaRetargetingPipeline(skeleton, 'soma', 'agibot_x2ultra', retarget_config=config)
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    pipeline.add_input_motions([animation], [transform], True)
    final = pipeline.execute()[0]
    if not hasattr(pipeline, 'base_retargeted_motions') or not pipeline.base_retargeted_motions:
        raise RuntimeError('Native pre-post base capture is unavailable; do not silently substitute another solve')
    base = pipeline.base_retargeted_motions[0]
    names = newton_utils.get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    if len(names) != 31:
        raise ValueError('Unexpected actuator count')
    urdf_path = Path(utils.get_target_asset_path('agibot_x2ultra')['urdf_path'])
    limits = urdf_velocity_limits(urdf_path, names)
    record = {
        'motion': entry['name'], 'variant': args.variant, 'weight_set': args.weight_set,
        'frames': entry['frames'], 'fps': entry['fps'], 'temporal_config': temporal,
        'controlled_config_sha256': controlled_hash, 'input_sha256': file_hash(source),
        'urdf_sha256': file_hash(urdf_path), 'joint_names': names,
        'effector_names': list(config['ik_match_table']),
        'native_source_sha256': digest({name: file_hash(REPO / 'soma_retargeter/pipelines' / name)
            for name in ('soma_retargeting_pipeline.py', 'ik_solver.py', 'ik_pass.py', 'temporal_ik.py')}),
        'velocity_limits_rad_s': limits.tolist(), 'fixed_weights_source': str(fixed_path) if fixed_path else None,
        'base_stage': 'after base IK, before outer post-processing and final clamp',
        'final_stage': 'complete native pipeline including outer post/contact processing and clamp',
    }
    export_measurements(pipeline, {'base': base, 'final': final}, record, args.output_dir)
    print('TEMPORAL_ABLATION_READY ' + json.dumps({'motion': args.motion, 'variant': args.variant, 'output': str(args.output_dir)}), flush=True)


def export_measurements(pipeline, buffers, record, output_dir):
    from app.tools.ik_weight_optimizer import BodyEvaluator
    from soma_retargeter.io import csv as csv_io

    record = dict(record)
    evaluator = BodyEvaluator(pipeline)
    skip = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    targets = np.asarray(pipeline.input_targets[0])[skip:skip + record['frames']]
    trajectories = {}
    for stage, buffer in buffers.items():
        q = np.asarray(buffer.data)
        if q.shape != (record['frames'], 38) or abs(buffer.sample_rate - record['fps']) > 0.01:
            raise ValueError(f'Unexpected {stage} trajectory shape/sample rate')
        positions, rotations = evaluator.evaluate([q])[0]
        record[stage] = {
            'tracking': tracking_metrics(positions, rotations, targets),
            'temporal': trajectory_metrics(q, buffer.sample_rate, record['joint_names'], record['velocity_limits_rad_s']),
        }
        trajectories[stage] = q
    record['post_processing_change'] = {
        derivative: {statistic: record['final']['temporal'][derivative][statistic] - record['base']['temporal'][derivative][statistic]
            for statistic in ('rms', 'p95', 'p99', 'max')}
        for derivative in ('velocity_rad_s', 'acceleration_rad_s2', 'jerk_rad_s3')
    }
    np.savez_compressed(output_dir / 'trajectories.npz', **trajectories, fps=record['fps'])
    csv_io.save_csv(str(output_dir / 'final.csv'), record['joint_names'], buffers['final'])
    with (output_dir / 'measurement.metrics.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
