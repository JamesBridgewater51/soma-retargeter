"""Own-native BVH decoding followed by exact topology remapping, never loader skeleton forcing."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--device', default='cpu')
parser.add_argument('--frames', type=int, default=352)
parser.add_argument('--canonical', default='on_the_edge_003__A273_M.bvh')
parser.add_argument('--evidence-root', type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.repo), str(args.repo / 'local-tools/production-batch')]
os.environ['STORAGE_ROOT'] = '/data/liuminghao'
import numpy as np
import warp as wp
import newton
wp.init()
wp.set_device(args.device)
import run_batch as batch
from soma_retargeter.io.bvh import load_bvh
from soma_retargeter.io.csv import save_csv
from soma_retargeter.animation.animation_buffer import AnimationBuffer, create_animation_buffer_for_skeleton
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
from soma_retargeter.utils.newton_utils import get_filtered_joint_names

settings = batch.prepare_settings(argparse.Namespace(
    optimizer_config=args.repo / 'local-tools/production-batch/optimizer-native50.json',
    conversion_overrides=Path('/data/liuminghao/artifacts/soma-retargeter/x2-production/throughput/cluster-batch-large-s025/inputs/temporal-provisional-s025.json'),
    storage_root='/data/liuminghao', provenance_file=[]))
config = copy.deepcopy(settings['base_retargeter'])
config.update(copy.deepcopy(settings['conversion_overrides']))
config['capture_base_motion'] = True
source_root = Path('/data/liuminghao/datasets/bones-seed/raw/soma-uniform/full')
paths = [source_root / args.canonical, source_root / 'fire_in_the_hole_180_R_002__A473.bvh']
own = []
for path in paths:
    # No skeleton argument is supplied: every file's native OFFSETs are decoded independently.
    skeleton, full = load_bvh(str(path))
    assert full.num_frames >= args.frames
    own.append((skeleton, AnimationBuffer(skeleton, args.frames, full.sample_rate,
                                          full.local_transforms[:args.frames].copy())))
assert own[0][0].joint_names == own[1][0].joint_names
assert len(set(own[0][0].joint_names)) == own[0][0].num_joints == 78
np.testing.assert_array_equal(own[0][0].parent_indices, own[1][0].parent_indices)
rest_error = float(np.max(np.abs(own[0][0].reference_local_transforms - own[1][0].reference_local_transforms)))
assert rest_error > 0
transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
args.evidence_root.mkdir(parents=True, exist_ok=True)
references = []
result = {'device': args.device, 'frames': args.frames, 'own_native_loaded': True,
          'same_unique_names_and_parents': True, 'reference_pose_max_difference': rest_error,
          'sources': [str(path) for path in paths], 'temporal_config': config['temporal_ik'], 'orders': []}
print('REST_VARIANT_START', json.dumps(result), flush=True)
for index, (skeleton, animation) in enumerate(own):
    start = time.monotonic()
    pipeline = SomaRetargetingPipeline(skeleton, 'soma', 'agibot_x2ultra', retarget_config=copy.deepcopy(config))
    assert pipeline.post_processing_enabled and pipeline.contact_processing_enabled
    pipeline.add_input_motions([animation], [transform], scale_animation=True)
    buffers = pipeline.execute()
    wp.synchronize()
    names = get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
    csv_path = args.evidence_root / f'scalar-own-{index}.csv'
    save_csv(str(csv_path), names, buffers[0])
    references.append({'targets': pipeline.input_targets[0].copy(), 'q': buffers[0].data.copy(),
                       'base': pipeline.base_retargeted_motions[0].data.copy(),
                       'contacts': pipeline.contact_results[0].contacts.copy(),
                       'csv_bytes': csv_path.read_bytes(), 'seconds': time.monotonic() - start,
                       'landmark_config_path': pipeline._landmark_model.config_path})
    if index == 0:
        zero_names = set(pipeline.initialization_pose.skeleton.joint_names)
        result['initialization_missing_names'] = [name for name in skeleton.joint_names if name not in zero_names]
        try:
            pipeline.human_robot_scaler.compute_effectors_from_buffer(own[1][1], True, transform)
        except ValueError as error:
            result['direct_mixed_skeleton_guard'] = str(error)
        else:
            raise AssertionError('Expected native scaler identity guard for distinct Skeleton objects')
    print('REST_SCALAR_COMPLETE', index, references[-1]['seconds'], flush=True)
for order in [(0, 1), (1, 0)]:
    start = time.monotonic()
    skeleton = own[order[0]][0]
    animations = []
    for index in order:
        remapped = create_animation_buffer_for_skeleton(own[index][1], skeleton)
        np.testing.assert_array_equal(remapped.local_transforms, own[index][1].local_transforms)
        animations.append(remapped)
    pipeline = SomaRetargetingPipeline(skeleton, 'soma', 'agibot_x2ultra', retarget_config=copy.deepcopy(config))
    pipeline.add_input_motions(animations, [transform] * 2, scale_animation=True)
    comparisons = []
    for env, index in enumerate(order):
        target_error = float(np.max(np.abs(pipeline.input_targets[env] - references[index]['targets'])))
        np.testing.assert_array_equal(pipeline.input_targets[env], references[index]['targets'])
        contact_error = float(np.max(np.abs(pipeline.contact_results[env].contacts - references[index]['contacts'])))
        np.testing.assert_array_equal(pipeline.contact_results[env].contacts, references[index]['contacts'])
        comparisons.append({'source_index': index, 'target_max_abs_error': target_error,
                            'contacts_max_abs_error': contact_error})
    buffers = pipeline.execute()
    wp.synchronize()
    for env, index in enumerate(order):
        q = buffers[env].data
        base = pipeline.base_retargeted_motions[env].data
        q_error = float(np.max(np.abs(q - references[index]['q'])))
        base_error = float(np.max(np.abs(base - references[index]['base'])))
        np.testing.assert_allclose(q, references[index]['q'], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(base, references[index]['base'], rtol=1e-5, atol=1e-5)
        assert buffers[env].num_frames == args.frames and np.isfinite(q).all()
        path = args.evidence_root / f'batch-order-{order[0]}{order[1]}-source-{index}.csv'
        save_csv(str(path), names, buffers[env])
        comparisons[env].update(q_max_abs_error=q_error, base_q_max_abs_error=base_error,
                                csv_byte_identical=path.read_bytes() == references[index]['csv_bytes'])
    result['orders'].append({'order': list(order), 'comparisons': comparisons,
                             'seconds': time.monotonic() - start,
                             'landmark_config_path': pipeline._landmark_model.config_path})
    batch.atomic_json(args.evidence_root / 'parity.json', result)
    print('REST_ORDER_PASS', json.dumps(result['orders'][-1]), flush=True)
result['scalar_seconds'] = [reference['seconds'] for reference in references]
result['scalar_landmark_config_paths'] = [reference['landmark_config_path'] for reference in references]
result['result'] = 'PASS'
batch.atomic_json(args.evidence_root / 'parity.json', result)
print('REST_VARIANT_PARITY_PASS', json.dumps(result), flush=True)
