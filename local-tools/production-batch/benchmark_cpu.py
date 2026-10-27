#!/usr/bin/env python3
"""Measure native single-clip CPU retargeting, not outer-optimizer throughput."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--optimizer-config', type=Path, default=Path(__file__).with_name('optimizer-native50.json'))
    parser.add_argument('--conversion-overrides', type=Path, default=Path(__file__).with_name('probe-disabled.json'))
    parser.add_argument('--storage-root', default='/data/liuminghao')
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.max_frames < 1 or args.repeats < 1:
        raise ValueError('Frame limit and repeats must be positive')
    # Hide CUDA before importing Warp; thread limits apply before NumPy/BLAS loads.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['STORAGE_ROOT'] = str(Path(args.storage_root).resolve())
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_batch as batch
    import warp as wp
    import numpy as np
    wp.init()
    wp.set_device('cpu')
    args.provenance_file = []
    settings = batch.prepare_settings(args)
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    origin = batch.verified_module_path(sys.modules[SomaRetargetingPipeline.__module__])
    started = time.monotonic()
    skeleton, animation = load_bvh(str(args.motion.resolve()))
    metadata = batch.bvh_metadata(args.motion)
    config = copy.deepcopy(settings['base_retargeter'])
    config.update(copy.deepcopy(settings['conversion_overrides']))
    if config.get('temporal_ik', {}).get('enabled', False):
        raise ValueError('CPU baseline benchmark requires temporal IK disabled')
    pipeline = SomaRetargetingPipeline(skeleton, 'soma', 'agibot_x2ultra', retarget_config=config)
    if pipeline.ik_model.device.is_cuda:
        raise RuntimeError('Native pipeline did not honor the explicit CPU device')
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    pipeline.add_input_motions([animation], [transform], scale_animation=True)
    frames = min(args.max_frames, metadata['frames'])
    padding = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    pipeline.input_targets = [targets[:padding + frames] for targets in pipeline.input_targets]
    pipeline.max_frames = padding + frames
    wp.synchronize()
    setup_seconds = time.monotonic() - started
    print(json.dumps({'event': 'CPU_BENCHMARK_READY', 'frames': frames, 'padding_frames': padding,
                      'joint_dofs': pipeline.ik_model.joint_dof_count, 'setup_seconds': setup_seconds}), flush=True)
    timings = []
    for repeat in range(args.repeats):
        started = time.monotonic()
        outputs = pipeline.execute()
        wp.synchronize()
        elapsed = time.monotonic() - started
        if len(outputs) != 1 or outputs[0].num_frames != frames or not np.isfinite(outputs[0].data).all():
            raise ValueError('Native CPU benchmark returned invalid output')
        # Native BVH loading rounds frame rates to two decimals; preserve that rate.
        if abs(float(outputs[0].sample_rate) - float(animation.sample_rate)) > 1e-3:
            raise ValueError('Native CPU benchmark changed the loaded animation sample rate')
        measured = {'repeat': repeat, 'execute_seconds': elapsed, 'output_frames': frames,
                    'frames_per_second': frames / elapsed, 'first_execution_in_process': repeat == 0}
        timings.append(measured)
        print(json.dumps({'event': 'CPU_BENCHMARK_RESULT', **measured}), flush=True)
    batch.verify_protected_files(settings)
    evidence = {'device': 'cpu', 'thread_limit': 1, 'motion': str(args.motion.resolve()),
                'source_frames': metadata['frames'], 'source_header_fps': metadata['fps'],
                'native_loaded_fps': float(animation.sample_rate), 'measured_frames': frames, 'padding_frames': padding,
                'pipeline_module': origin, 'joint_dofs': pipeline.ik_model.joint_dof_count,
                'ik_iterations': pipeline.ik_iterations, 'post_processing': pipeline.post_processing_enabled,
                'contact_processing': pipeline.contact_processing_enabled, 'temporal_ik': False,
                'setup_seconds': setup_seconds, 'timings': timings,
                'warning': 'Measured native forward retargeting only; not an outer-optimizer ETA.'}
    evidence_path = Path(args.storage_root) / 'artifacts/soma-retargeter/cpu-throughput' / f'native-{uuid.uuid4().hex[:12]}.json'
    batch.atomic_json(evidence_path, evidence)
    print(json.dumps({'event': 'CPU_BENCHMARK_COMPLETE', 'evidence': str(evidence_path), **evidence}), flush=True)


if __name__ == '__main__':
    main()
