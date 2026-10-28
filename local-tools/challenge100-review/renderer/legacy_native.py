"""Standalone native SOMA/X2 mesh videos with smooth close-follow cameras."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyglet
pyglet.options['headless'] = True
import warp as wp
import newton
from newton.viewer import ViewerGL
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.io.bvh import load_bvh
from soma_retargeter.io.csv import load_csv
from soma_retargeter.pipelines.utils import SourceType, create_robot_builder, get_source_model_mesh
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.utils.newton_utils import get_filtered_joint_names
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType

ROOT = Path('/data/liuminghao')
ART = ROOT/'artifacts/soma-retargeter/agibot-x2ultra-per-sequence-comparison'
FFMPEG = ROOT/'wallvideo/egoexo/bin/ffmpeg'
WIDTH, HEIGHT = 1280, 640
COLOR = wp.vec3(.63, .76, .88)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SkinBounds:
    def __init__(self):
        self.low = np.full(3, np.inf)
        self.high = np.full(3, -np.inf)

    def log_mesh(self, name, points, indices, color):
        vertices = points.numpy()
        self.low = np.minimum(self.low, vertices.min(axis=0))
        self.high = np.maximum(self.high, vertices.max(axis=0))


def source_motion(name):
    path = ROOT/f'datasets/bones-seed/raw/soma-uniform/validation/{name}.bvh'
    skeleton, animation = load_bvh(str(path))
    with path.open() as stream:
        dt = next(float(line.split(':', 1)[1]) for line in stream if line.strip().startswith('Frame Time:'))
    indices = np.minimum(np.rint(np.arange(math.ceil(animation.num_frames*dt*30))/30/dt).astype(int), animation.num_frames-1)
    transform = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    instance = SkeletonInstance(skeleton, COLOR, transform)
    renderer = SkeletalMeshRenderer(get_source_model_mesh(SourceType.SOMA, skeleton))
    return SimpleNamespace(name=name, path=path, skeleton=skeleton, animation=animation, dt=dt, indices=indices, instance=instance, renderer=renderer, source_transform={'translation':[float(v) for v in transform.p],'quaternion_xyzw':[float(v) for v in transform.q]})


def source_bounds(motion):
    bounds = []
    for frame in motion.indices:
        motion.instance.set_local_transforms(motion.animation.get_local_transforms(int(frame)))
        sink = SkinBounds()
        motion.renderer.draw(sink, motion.instance, COLOR, 0)
        bounds.append([sink.low, sink.high])
    return np.asarray(bounds)


def follow_cameras(bounds, padding=.06, fit_margin=1.08):
    low = bounds[:,0].copy() - padding
    high = bounds[:,1].copy() + padding
    low[:,2] = np.minimum(low[:,2], -.06)
    target = gaussian_filter1d((low+high)/2, sigma=3, axis=0, mode='nearest')
    pitch, yaw = -18., -135.
    direction = np.array([math.cos(math.radians(pitch))*math.cos(math.radians(yaw)), math.cos(math.radians(pitch))*math.sin(math.radians(yaw)), math.sin(math.radians(pitch))])
    right = np.cross(direction, [0.,0.,1.]); right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    tangent = math.tan(math.radians(45/2))
    required = []
    for lower, upper, center in zip(low, high, target):
        corners = np.array([[x,y,z] for x in [lower[0],upper[0]] for y in [lower[1],upper[1]] for z in [lower[2],upper[2]]])-center
        required.append(max(np.max(np.abs(corners@right)/(tangent*WIDTH/HEIGHT)-corners@direction), np.max(np.abs(corners@up)/tangent-corners@direction)))
    required = np.maximum(np.asarray(required)*fit_margin, .65)
    smooth = gaussian_filter1d(maximum_filter1d(required,size=15,mode='nearest'), sigma=3, mode='nearest')
    distance = np.maximum(required,smooth)
    return {'positions':(target-distance[:,None]*direction).tolist(), 'targets':target.tolist(), 'pitch':pitch, 'yaw':yaw, 'fov':45., 'ground_z':0., 'policy':'same world coordinates; smooth instantaneous body bounds, not whole trajectory; ground remains in frame'}


def configure_scene(viewer, model):
    viewer.set_model(model)
    viewer.set_world_offsets([0,0,0])


def set_camera(viewer, camera, index):
    viewer.camera.fov = camera['fov']
    viewer.set_camera(wp.vec3(*camera['positions'][index]), camera['pitch'], camera['yaw'])


def annotated_frame(viewer, motion, index, title):
    canvas = Image.new('RGB',(1280,720),'#121821')
    canvas.paste(Image.fromarray(viewer.get_frame().numpy()),(0,40))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    draw.text((14,7),title,fill='white',font=font)
    draw.text((14,690),f'{motion.name} | source frame {motion.indices[index]} | t={motion.indices[index]*motion.dt:.3f}s | close-follow camera; world motion unchanged',fill='white',font=small)
    return canvas


def encode_view(motion, camera, session, variant):
    output = ART/'videos'/f'{motion.name}__{variant}.mp4'
    output.parent.mkdir(parents=True,exist_ok=True)
    command = [str(FFMPEG),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    encoder = subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index, frame in enumerate(motion.indices):
            set_camera(session.viewer,camera,index)
            session.viewer.begin_frame(index/30)
            session.draw(int(frame))
            session.viewer.end_frame()
            image = annotated_frame(session.viewer,motion,index,session.title)
            encoder.stdin.write(image.tobytes())
            if index in {0,len(motion.indices)//2,len(motion.indices)-1}:
                previews = ART/'previews'; previews.mkdir(exist_ok=True)
                image.save(previews/f'{motion.name}__{variant}__frame-{int(frame):04d}.png')
    finally:
        encoder.stdin.close()
    if encoder.wait()!=0:
        raise RuntimeError(f'Video encoder failed: {output}')
    print('STANDALONE_VIDEO_COMPLETE',output,flush=True)
    return {'path':str(output),'sha256':sha256(output),'frames':len(motion.indices),'fps':30,'width':1280,'height':720}


def render_source(motion, viewer):
    builder = newton.ModelBuilder(); builder.add_ground_plane()
    model = builder.finalize(); state = model.state()
    camera = follow_cameras(source_bounds(motion))
    configure_scene(viewer,model)
    def draw_source(frame):
        motion.instance.set_local_transforms(motion.animation.get_local_transforms(frame))
        motion.renderer.draw(viewer,motion.instance,COLOR,0)
        viewer.log_state(state)
    session = SimpleNamespace(viewer=viewer,draw=draw_source,title='SOURCE SOMA | actual native USD skinned mesh')
    video = encode_view(motion,camera,session,'source-soma')
    return {'video':video,'camera':camera,'representation':'Native SOMA USD actual mesh with native SkeletalMeshRenderer; no reconstructed skeleton','vertex_counts':[mesh.num_points for mesh in motion.renderer.skeletal_mesh.skinned_meshes],'mesh_asset':str(ROOT/'assets/soma-retargeter/soma/soma_base_skel_minimal.usd')}


def robot_frames(path, joints, expected):
    with path.open() as stream:
        header = next(csv.reader(stream))
    if [name.removesuffix('_dof') for name in header[7:]]!=joints:
        raise ValueError(f'Native X2 joint order mismatch: {path}')
    frames = load_csv(str(path)).data
    if len(frames)!=expected or not np.isfinite(frames).all():
        raise ValueError(f'Incomplete/nonfinite X2 CSV: {path}')
    return frames


def visual_bound_points(model):
    points, bodies = [], []
    scales = model.shape_scale.numpy(); transforms = model.shape_transform.numpy()
    for index, (mesh, body, flags) in enumerate(zip(model.shape_source, model.shape_body.numpy(), model.shape_flags.numpy())):
        if body < 0 or not int(flags) & int(newton.ShapeFlags.VISIBLE):
            continue
        if mesh is None or not hasattr(mesh, 'vertices'):
            raise ValueError('Native X2 visible geometry is not a mesh')
        vertices = mesh.vertices * scales[index]
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        corners = np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
        points.extend(Rotation.from_quat(transforms[index,3:]).apply(corners)+transforms[index,:3])
        bodies.extend([body]*len(corners))
    return np.asarray(points), np.asarray(bodies,dtype=int)


def robot_bounds(model, frames, indices):
    state = model.state(); bounds=[]
    local_points, bodies = visual_bound_points(model)
    for frame in indices:
        model.joint_q.assign(frames[frame]); newton.eval_fk(model,model.joint_q,model.joint_qd,state)
        poses = state.body_q.numpy()[bodies]
        positions = Rotation.from_quat(poses[:,3:]).apply(local_points)+poses[:,:3]
        bounds.append([positions.min(axis=0), positions.max(axis=0)])
    return np.asarray(bounds)


def render_robots(motion, viewer):
    builder = create_robot_builder('agibot_x2ultra')
    joints = get_filtered_joint_names(builder,[newton.JointType.REVOLUTE])
    builder.add_ground_plane(); model=builder.finalize(); state=model.state()
    base = ROOT/'datasets/bones-seed/converted/agibot-x2ultra'
    paths = [base/f'{variant}/{motion.name}.csv' for variant in ['default-comparison','per-sequence-optimized-comparison']]
    input_hashes = [sha256(path) for path in paths]
    clips = [robot_frames(path,joints,motion.animation.num_frames) for path in paths]
    bounds = [robot_bounds(model,clip,motion.indices) for clip in clips]
    camera = follow_cameras(np.stack([np.minimum(bounds[0][:,0],bounds[1][:,0]),np.maximum(bounds[0][:,1],bounds[1][:,1])],axis=1),padding=.03,fit_margin=1.04)
    camera['bounds_representation']='Native X2 visual-mesh local bounds transformed by FK; shared default/optimized union'
    configure_scene(viewer,model); videos=[]
    for clip,variant,title in zip(clips,['default-x2','optimized-x2'],['DEFAULT X2 | full pipeline','PER-SEQUENCE OPTIMIZED X2 | full pipeline']):
        def draw_robot(frame):
            model.joint_q.assign(clip[frame]);newton.eval_fk(model,model.joint_q,model.joint_qd,state);viewer.log_state(state)
        videos.append(encode_view(motion,camera,SimpleNamespace(viewer=viewer,draw=draw_robot,title=title),variant))
    if input_hashes != [sha256(path) for path in paths]:
        raise RuntimeError('X2 input CSV changed during rendering')
    return {'videos':videos,'shared_camera':camera,'inputs':[{'path':str(path),'sha256':digest} for path,digest in zip(paths,input_hashes)],'representation':'Native configured agibot_x2ultra model; direct FK of unchanged full-pipeline CSV values'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--source-only',action='store_true');parser.add_argument('--robots-only',action='store_true')
    args=parser.parse_args()
    selection=json.loads((ROOT/'datasets/bones-seed/metadata/x2-validation-selection.json').read_text())
    if args.name not in {entry['name'] for entry in selection}:raise ValueError('Motion not in canonical11')
    if args.source_only and args.robots_only:parser.error('Choose source-only or robots-only, not both')
    motion=source_motion(args.name); source_hash=sha256(motion.path)
    viewer=ViewerGL(width=WIDTH,height=HEIGHT,headless=True); metadata={}
    try:
        if not args.robots_only:metadata['source']=render_source(motion,viewer)
        if not args.source_only:metadata['robots']=render_robots(motion,viewer)
    finally:
        viewer.close()
    assert sha256(motion.path)==source_hash,'Source changed while rendering'
    metadata.update({'name':motion.name,'source_input':{'path':str(motion.path),'sha256':source_hash},'source_frame_time':motion.dt,'source_indices':motion.indices.tolist(),'world_motion_modified':False,'source_facing_direction':'Maya','source_to_newton_transform':motion.source_transform,'source_basis_reference':'native app/tools/ik_weight_optimizer.py root_tx SpaceConverter(FacingDirectionType.MAYA).transform(identity)'})
    output=ART/'metadata';output.mkdir(parents=True,exist_ok=True)
    suffix='source' if args.source_only else ('robots' if args.robots_only else 'all')
    (output/f'{motion.name}__{suffix}.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
