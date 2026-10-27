#!/usr/bin/env python3
"""Compare genuine outer-search prefixes on selected full-manifest BVHs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / 'temporal-ablation'))
import run_batch as batch

EXPECTED_NATIVE = '819777fc6dee90f52f14b6c2b94ee547c23c1b224114cb87aba863bd84656e4c'


def native_hash():
    names = ('soma_retargeting_pipeline.py', 'ik_solver.py', 'ik_pass.py', 'temporal_ik.py')
    hashes = {name: batch.digest(batch.REPO / 'soma_retargeter/pipelines' / name) for name in names}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-root', default=os.environ.get('STORAGE_ROOT', '/data/liuminghao'))
    parser.add_argument('--optimizer-config', type=Path, required=True)
    parser.add_argument('--conversion-overrides', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--pilot-name', default='outer-budget-pilot')
    return parser.parse_args()


def selected_jobs(args):
    from run_batched import inspect_motion
    root = Path(args.storage_root)
    manifest = args.manifest or root / 'datasets/bones-seed/metadata/soma-uniform-full/extracted-files.jsonl'
    source = args.source_root or root / 'datasets/bones-seed/raw/soma-uniform/full'
    selection = args.selection or root / 'datasets/bones-seed/metadata/x2-validation-selection.json'
    entries = json.loads(selection.read_text())
    expected = {entry['name']: entry for entry in entries}
    selected = {job['stem']: inspect_motion(job) for job in batch.source_rows(manifest, source)
                if job['stem'] in expected}
    if len(entries) != 11 or set(selected) != set(expected):
        raise ValueError('Pilot requires the exact eleven selected motions from the full manifest')
    jobs = [selected[entry['name']] for entry in entries]
    for job in jobs:
        if batch.digest(job['source']) != job['sha256']:
            raise ValueError('Full-source manifest SHA mismatch: ' + job['stem'])
        if job['frames'] != expected[job['stem']]['frames']:
            raise ValueError('Selected/full motion frame mismatch: ' + job['stem'])
    return jobs


def pilot_paths(args):
    if Path(args.pilot_name).name != args.pilot_name:
        raise ValueError('Pilot name must be one path component')
    root = Path(args.storage_root)
    paths = {
        'artifacts': root / 'artifacts/soma-retargeter/agibot-x2ultra' / args.pilot_name,
        'weights': root / 'optimizers/soma-retargeter/agibot-x2ultra' / args.pilot_name,
        'output': root / 'datasets/bones-seed/converted/agibot-x2ultra' / args.pilot_name,
        'logs': root / 'logs/soma-retargeter/agibot-x2ultra' / args.pilot_name,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    return paths


def checkpoint(optimizer, budget, started):
    from app.tools import ik_weight_optimizer as native
    configs = [native._build_retargeter_config(optimizer.workspace, state.params)
               for state in optimizer.states]
    summaries = copy.deepcopy(optimizer.summaries())
    for summary in summaries:
        summary['outer_iteration_budget'] = budget
    return {'budget': budget, 'configs': copy.deepcopy(configs),
            'optimization_seconds': time.monotonic() - started,
            'summaries': summaries,
            'evaluation_timings': copy.deepcopy(optimizer.evaluation_timings)}


def fit_prefixes(optimizer):
    started = time.monotonic()
    optimizer.initialize()
    snapshots = {}
    for iteration in range(1, 9):
        active = [index for index, state in enumerate(optimizer.states) if not state.done]
        if active:
            optimizer.outer_iteration(active)
        print(json.dumps({'event': 'BUDGET_PILOT_OUTER', 'iteration': iteration,
                          'active': len(active), 'seconds': time.monotonic() - started}), flush=True)
        if iteration in (4, 8):
            snapshots[iteration] = checkpoint(optimizer, iteration, started)
    return snapshots


def final_metrics(pipeline, buffers, jobs):
    import numpy as np
    from app.tools.ik_weight_optimizer import BodyEvaluator
    from metrics import tracking_metrics, trajectory_metrics, urdf_velocity_limits
    from soma_retargeter.pipelines import utils
    import newton
    from soma_retargeter.utils.newton_utils import get_filtered_joint_names
    names = get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    limits = urdf_velocity_limits(utils.get_target_asset_path('agibot_x2ultra')['urdf_path'], names)
    evaluator = BodyEvaluator(pipeline)
    poses = evaluator.evaluate([buffer.data for buffer in buffers])
    skip = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    measurements = []
    for index, (job, buffer, pose) in enumerate(zip(jobs, buffers, poses, strict=True)):
        targets = np.asarray(pipeline.input_targets[index])[skip:skip + job['frames']]
        if buffer.num_frames != job['frames'] or abs(buffer.sample_rate - job['fps']) > .01:
            raise ValueError('Final pilot frame/FPS mismatch')
        measurements.append({'motion': job['stem'], 'frames': job['frames'], 'fps': job['fps'],
            'source': job['source'], 'source_sha256': job['sha256'],
            'tracking': tracking_metrics(*pose, targets),
            'temporal': trajectory_metrics(buffer.data, buffer.sample_rate, names, limits)})
    return measurements, names


def export_checkpoint(snapshot, jobs, prepared, settings):
    import warp as wp
    from batched_optimizer import PerSequencePipeline
    from soma_retargeter.io.csv import save_csv
    from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter
    budget = snapshot['budget']
    config = copy.deepcopy(settings['base_retargeter'])
    config.update(copy.deepcopy(settings['conversion_overrides']))
    for flag in ('enable_post_processing', 'enable_contact_processing'):
        if config.get(flag, True) is not True:
            raise ValueError('Budget pilot final conversion must include post/contact processing')
    skeleton, animations = prepared
    pipeline = PerSequencePipeline(skeleton, config, snapshot['configs'])
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    pipeline.add_input_motions(animations, [transform] * len(animations), scale_animation=True)
    started = time.monotonic()
    buffers = pipeline.execute()
    conversion_seconds = time.monotonic() - started
    measurements, names = final_metrics(pipeline, buffers, jobs)
    for job, retargeter, buffer in zip(jobs, snapshot['configs'], buffers, strict=True):
        batch.verify_weights_only(retargeter, settings['base_retargeter'])
        weights = Path(settings['weights']) / f'budget-{budget}' / job['stem'] / 'retargeter.json'
        batch.atomic_json(weights, retargeter)
        output = Path(settings['output']) / f'budget-{budget}' / (job['stem'] + '.csv')
        output.parent.mkdir(parents=True, exist_ok=True)
        save_csv(str(output), names, buffer)
        batch.validate_csv(output, batch.ROOT_HEADER + names, job['frames'], batch.velocity_plan(settings, job['fps']))
    return {key: value for key, value in snapshot.items() if key != 'configs'} | {
        'full_pipeline_conversion_seconds': conversion_seconds, 'measurements': measurements,
        'clips': len(jobs), 'frames': sum(job['frames'] for job in jobs)}


def pooled(measurements):
    import numpy as np
    frames = sum(record['frames'] for record in measurements)
    summary = {metric: float(np.sqrt(sum(record['frames'] * record['tracking'][metric] ** 2
        for record in measurements) / frames)) for metric in ('position_rmse_m', 'rotation_rmse_rad')}
    for derivative in ('velocity_rad_s', 'acceleration_rad_s2', 'jerk_rad_s3'):
        samples = [record['temporal'][derivative] for record in measurements]
        summary[derivative] = {'rms': float(np.sqrt(sum(record['sample_count'] * record['rms'] ** 2
            for record in samples) / sum(record['sample_count'] for record in samples))),
            'max': max(record['max'] for record in samples)}
    return summary


def hierarchy_groups(jobs):
    groups = {}
    for job in jobs:
        groups.setdefault(job['hierarchy_sha256'], []).append(job)
    return list(groups.values())


def evaluate_group(jobs, settings, paths, group_index):
    from app.tools.ik_weight_optimizer import OptimizerConfigWorkspace
    from batched_optimizer import IndependentBatchOptimizer
    from run_batched import load_animations
    prepared = load_animations(jobs, settings)
    workspace = OptimizerConfigWorkspace('soma', 'agibot_x2ultra', paths['weights'])
    optimizer = IndependentBatchOptimizer(*prepared, [job['stem'] for job in jobs], settings, workspace)
    snapshots = fit_prefixes(optimizer)
    outcomes = {}
    for budget, snapshot in snapshots.items():
        if native_hash() != EXPECTED_NATIVE:
            raise ValueError('Native source changed during pilot')
        result = export_checkpoint(snapshot, jobs, prepared, settings)
        batch.atomic_json(paths['artifacts'] / f'group-{group_index}.budget-{budget}.json', result)
        outcomes[budget] = result
    return outcomes


def combine_groups(outcomes):
    results = {}
    for budget in (4, 8):
        groups = [outcome[budget] for outcome in outcomes]
        measurements = [record for group in groups for record in group['measurements']]
        seconds = sum(group['optimization_seconds'] for group in groups)
        results[str(budget)] = {'budget': budget, 'groups': groups, 'measurements': measurements,
            'pooled_final_metrics': pooled(measurements), 'clips': len(measurements),
            'frames': sum(group['frames'] for group in groups), 'optimization_seconds': seconds,
            'optimization_source_frames_per_second': sum(group['frames'] for group in groups) / seconds,
            'full_pipeline_conversion_seconds': sum(group['full_pipeline_conversion_seconds'] for group in groups)}
    return results


def main():
    args = parse_args()
    os.environ['STORAGE_ROOT'] = args.storage_root
    if native_hash() != EXPECTED_NATIVE:
        raise ValueError('Pilot requires the corrected frozen native snapshot')
    args.provenance_file = []
    settings = batch.prepare_settings(args)
    settings.update(engine='native-independent-batched', reuse_geometry=True, fast_bvh=True)
    settings['optimizer']['outer_iterations'] = 8
    paths = pilot_paths(args)
    settings.update({key: str(path) for key, path in paths.items()})
    jobs = selected_jobs(args)
    batch.atomic_json(paths['artifacts'] / 'settings.json', settings)
    batch.atomic_json(paths['artifacts'] / 'selected-full-manifest.json', jobs)
    batch.initialize_worker(args.gpu, settings['optimizer']['extra_robot_paths'])
    groups = hierarchy_groups(jobs)
    print(json.dumps({'event': 'BUDGET_PILOT_START', 'clips': len(jobs), 'hierarchy_groups': len(groups),
        'frames': sum(job['frames'] for job in jobs), 'budgets': [4, 8], 'prefix_run': True,
        'native_source_sha256': EXPECTED_NATIVE}), flush=True)
    outcomes = []
    for index, group in enumerate(groups):
        print(json.dumps({'event': 'BUDGET_PILOT_GROUP_START', 'group': index,
                          'clips': len(group), 'motions': [job['stem'] for job in group]}), flush=True)
        outcomes.append(evaluate_group(group, settings, paths, index))
    results = combine_groups(outcomes)
    for budget, result in results.items():
        batch.atomic_json(paths['artifacts'] / f'budget-{budget}.metrics.json', result)
        print(json.dumps({'event': 'BUDGET_PILOT_READY', 'budget': budget,
            'optimization_seconds': result['optimization_seconds'],
            'pooled_final_metrics': result['pooled_final_metrics']}), flush=True)
    batch.atomic_json(paths['artifacts'] / 'comparison.json', results)


if __name__ == '__main__':
    main()
