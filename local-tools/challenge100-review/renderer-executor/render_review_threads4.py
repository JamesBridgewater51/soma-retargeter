"""Render frozen production motions in the approved native shared scene."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(os.environ.get('STORAGE_ROOT', '/data/liuminghao'))
sys.path.insert(0, str(ROOT/'agibot-x2/soma-retargeter-full-release'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import legacy_scene as scene_api

np = scene_api.np
native = scene_api.native
ART = ROOT/'artifacts/soma-retargeter/agibot-x2ultra/challenge100-review'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def mapped_csv(path, joint_names, expected):
    with path.open() as stream:
        rows = csv.reader(stream)
        header = next(rows)
        names = [name.removesuffix('_dof') for name in header[7:]]
        if len(names) != len(set(names)) or set(names) != set(joint_names):
            raise ValueError(f'Joint-name set mismatch: {path}')
        frame_ids = [int(row[0]) for row in rows]
    if frame_ids != list(range(expected)):
        raise ValueError(f'Noncontinuous/incomplete source frame IDs: {path}')
    csv_frames = native.load_csv(str(path), fps=120.0).data
    if csv_frames.shape != (expected, 7+len(joint_names)) or not np.isfinite(csv_frames).all():
        raise ValueError(f'Incomplete or nonfinite CSV: {path}')
    order = [names.index(name) for name in joint_names]
    return np.concatenate([csv_frames[:, :7], csv_frames[:, 7:][:, order]], axis=1), order


def setup(entry, rig):
    name = entry['name']
    paths = [Path(entry[key]) for key in ['bvh_path', 'official_g1_path', 'x2_path']]
    hashes = [digest(path) for path in paths]
    if 'csv_validation' in entry:
        expected = [entry['source_sha256_verified'], entry['csv_validation']['official_g1']['sha256'],
                    entry['csv_validation']['production_x2']['sha256']]
        if hashes != expected:
            raise ValueError(f'Selection manifest input hash mismatch: {name}')
    done_path = ROOT/f'artifacts/soma-retargeter/x2-production/done/{name}.json'
    done = json.loads(done_path.read_text())
    if done['sha256'] != hashes[2] or done['source_sha256'] != hashes[0]:
        raise ValueError(f'Production receipt hash mismatch: {name}')
    weights = ROOT/f'optimizers/soma-retargeter/agibot-x2ultra/full-per-sequence/{name}'
    optimized = json.loads((weights/'optimized.done.json').read_text())
    if optimized['content_key'] != done['optimization_key'] or optimized['sha256'] != done['weights_sha256']:
        raise ValueError(f'Per-sequence optimizer receipt mismatch: {name}')
    if digest(weights/'retargeter.json') != done['weights_sha256']:
        raise ValueError(f'Per-sequence optimized config hash mismatch: {name}')
    skeleton, animation = native.load_bvh(str(paths[0]))
    with paths[0].open() as stream:
        raw_dt = next(float(line.split(':', 1)[1]) for line in stream if line.strip().startswith('Frame Time:'))
    if animation.num_frames != done['frames'] or not math.isclose(animation.sample_rate, 120.0):
        raise ValueError(f'Native production source frame/FPS mismatch: {name}')
    indices = np.minimum(np.rint(np.arange(math.ceil(animation.num_frames/120.0*30))*4).astype(int), animation.num_frames-1)
    transform = native.SpaceConverter(native.FacingDirectionType.MAYA).transform(native.wp.transform_identity())
    instance = native.SkeletonInstance(skeleton, native.COLOR, transform)
    renderer = native.SkeletalMeshRenderer(native.get_source_model_mesh(native.SourceType.SOMA, skeleton))
    motion = SimpleNamespace(name=name, path=paths[0], skeleton=skeleton, animation=animation, dt=1/120.0,
                             raw_dt=raw_dt, indices=indices, instance=instance, renderer=renderer)
    loaded = [mapped_csv(path, joints, animation.num_frames) for path, joints in zip(paths[1:], rig.joints)]
    clips = [item[0] for item in loaded]
    basis = scene_api.Rotation.from_euler('z', 90, degrees=True)
    clips[0][:, :3] = basis.apply(clips[0][:, :3])
    clips[0][:, 3:7] = (basis*scene_api.Rotation.from_quat(clips[0][:, 3:7])).as_quat()
    inputs = SimpleNamespace(clips=clips, paths=paths[1:], hashes=hashes[1:], g1_basis_xyzw=basis.as_quat().tolist())
    bounds = [native.source_bounds(motion)]+scene_api.robot_mesh_bounds(motion, rig, clips)
    lanes = scene_api.fixed_lanes(bounds)
    bounds = [actor+lane[None, None, :] for actor, lane in zip(bounds, lanes)]
    instance.xform = native.wp.mul(native.wp.transform(native.wp.vec3(*lanes[0]), native.wp.quat_identity()), instance.xform)
    return SimpleNamespace(motion=motion, rig=rig, inputs=inputs, paths=paths, hashes=hashes,
                           bounds=bounds, lanes=lanes, camera=scene_api.group_cameras(bounds),
                           mappings=[item[1] for item in loaded], production_receipt=done_path,
                           production_receipt_sha256=digest(done_path), entry=entry)


def render(scene, viewer, smoke=False):
    name = scene.motion.name
    final = ART/'videos'/f'{name}.mp4'
    partial = final.with_suffix('.partial.mp4')
    started = time.monotonic()
    if smoke:
        image = scene_api.draw_scene_frame(viewer, scene, len(scene.motion.indices)//2)
        preview = ART/'previews'/f'{name}__smoke.png'
        image.save(preview)
        print(json.dumps({'event':'REAL_SHARED_SCENE_SMOKE', 'preview':str(preview), 'name':name}), flush=True)
    command = [str(native.FFMPEG), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1920x1080', '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index, frame in enumerate(scene.motion.indices):
            image = scene_api.draw_scene_frame(viewer, scene, index)
            encoder.stdin.write(image.tobytes())
            if index in {0, len(scene.motion.indices)//2, len(scene.motion.indices)-1}:
                image.save(ART/'previews'/f'{name}__frame-{int(frame):04d}.png')
                if index == len(scene.motion.indices)//2:
                    image.save(ART/'previews'/f'{name}.png')
            if index == 0:
                print(json.dumps({'event':'FIRST_REAL_FRAME', 'name':name, 'frame':int(frame)}), flush=True)
    except BaseException:
        encoder.stdin.close()
        encoder.wait()
        raise
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f'Encoding failed: {name}')
    if [digest(path) for path in scene.paths] != scene.hashes:
        raise RuntimeError(f'Input changed while rendering: {name}')
    validation = subprocess.run([str(native.FFMPEG), '-v', 'error', '-i', str(partial),
                                 '-progress', 'pipe:1', '-f', 'null', '-'],
                                check=True, capture_output=True, text=True)
    decoded_frames = [int(line.split('=', 1)[1]) for line in validation.stdout.splitlines()
                      if line.startswith('frame=')][-1]
    if decoded_frames != len(scene.motion.indices):
        raise RuntimeError(f'Decoded frame-count mismatch: {name}')
    partial.replace(final)
    seconds = time.monotonic()-started
    receipt = {
        'name':name, 'status':'complete', 'video':str(final), 'output_sha256':digest(final),
        'width':1920, 'height':1080, 'fps':30, 'frames':len(scene.motion.indices), 'decoded_frames':decoded_frames,
        'source_frames':scene.motion.animation.num_frames, 'source_native_fps':120.0,
        'source_frame_time_raw':scene.motion.raw_dt, 'timing_policy':'all actors same source index; native-loaded 120FPS; no phase warp',
        'source_indices':scene.motion.indices.tolist(),
        'inputs':[{'path':str(path), 'sha256':value} for path,value in zip(scene.paths, scene.hashes)],
        'production_receipt':str(scene.production_receipt), 'production_receipt_sha256':scene.production_receipt_sha256,
        'joint_names_model_order':scene.rig.joints, 'csv_to_model_joint_indices':scene.mappings,
        'g1_root_basis_quaternion_xyzw':scene.inputs.g1_basis_xyzw,
        'g1_basis':'MUJOCO to MAYA fixed world-Z +90 degrees; root translation/orientation only',
        'x2_basis':'native MAYA/Newton; unchanged', 'source_representation':'native SOMA USD skinned mesh',
        'source_mesh_vertices':[mesh.num_points for mesh in scene.motion.renderer.skeletal_mesh.skinned_meshes],
        'single_viewport':True, 'single_ground_plane':True, 'ground_z':0,
        'fixed_xy_lane_offsets_metres':dict(zip(['source_soma','official_g1','production_x2'],scene.lanes.tolist())),
        'no_per_frame_root_alignment':True, 'no_height_correction':True, 'no_actor_rescaling':True,
        'scene_geometry':'one native flat floor only; corresponding source props/terrain not supplied',
        'camera':scene.camera, 'render_seconds':seconds, 'render_fps':len(scene.motion.indices)/seconds,
        'renderer_sha256':digest(Path(__file__)), 'scene_helper_sha256':digest(Path(scene_api.__file__)),
        'native_helper_sha256':digest(Path(native.__file__)), 'selection_entry':scene.entry,
    }
    atomic_json(ART/'receipts'/f'{name}.json', receipt)
    print(json.dumps({'event':'CLIP_COMPLETE', 'name':name, 'video':str(final), 'frames':receipt['frames'],
                      'seconds':seconds, 'render_fps':receipt['render_fps']}), flush=True)
    return receipt


def entries_from(path):
    proposal = json.loads(Path(path).read_text())
    motions = proposal if isinstance(proposal, list) else proposal['motions']
    return [dict(motion, name=motion['filename'], bvh_path=motion['soma_bvh'],
                 official_g1_path=motion['official_g1_csv'], x2_path=motion['production_x2_csv'])
            for motion in motions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest')
    parser.add_argument('--smoke-name')
    args = parser.parse_args()
    for directory in ['videos','previews','receipts']:
        (ART/directory).mkdir(parents=True, exist_ok=True)
    complete = json.loads((ROOT/'artifacts/soma-retargeter/x2-production/COMPLETE.json').read_text())
    if complete['status'] != 'COMPLETE' or digest(complete['selected_settings_file']) != complete['selected_settings_sha256']:
        raise ValueError('Selected production campaign completion/settings mismatch')
    if args.smoke_name:
        name = args.smoke_name
        entries = [{'name':name, 'bvh_path':str(ROOT/f'datasets/bones-seed/raw/soma-uniform/full/{name}.bvh'),
                    'official_g1_path':str(ROOT/f'datasets/bones-seed/raw/g1/csv/230324/{name}.csv'),
                    'x2_path':str(ROOT/f'datasets/bones-seed/converted/agibot-x2ultra/per-sequence-optimized/{name}.csv')}]
    else:
        entries = entries_from(args.manifest)
        if len(entries) != 100 or len({entry['name'] for entry in entries}) != 100:
            raise ValueError('Full review requires exactly 100 unique motions')
    rig = scene_api.merged_robot_model()
    viewer = scene_api.scene_viewer(SimpleNamespace(rig=rig))
    started = time.monotonic()
    try:
        for index, entry in enumerate(entries):
            print(json.dumps({'event':'CLIP_START', 'index':index, 'total':len(entries), 'name':entry['name']}), flush=True)
            scene = setup(entry, rig)
            receipt = render(scene, viewer, smoke=bool(args.smoke_name))
            atomic_json(ART/'progress.json', {'completed':index+1, 'total':len(entries), 'last_name':entry['name'],
                                            'elapsed_seconds':time.monotonic()-started})
            del scene, receipt
            gc.collect()
    finally:
        viewer.close()
    print(json.dumps({'event':'RENDER_BATCH_COMPLETE', 'count':len(entries), 'elapsed_seconds':time.monotonic()-started}), flush=True)


if __name__ == '__main__':
    main()
