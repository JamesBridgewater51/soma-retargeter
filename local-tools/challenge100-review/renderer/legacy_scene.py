"""One native 3D scene: SOMA USD mesh, our default G1 and default full-pipeline X2."""
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/data/liuminghao')
REPO = ROOT/'agibot-x2/soma-retargeter'
ART = ROOT/'artifacts/soma-retargeter/unitree-g1-comparison/soma-g1-x2-follow'
HELPER = Path(__file__).with_name('legacy_native.py')
spec = importlib.util.spec_from_file_location('approved_native_mesh_helpers',HELPER)
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
wp, newton = native.wp, native.newton
WIDTH, HEIGHT = 1920, 1080
YAW, PITCH = -135., -18.


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corners(bounds):
    lower, upper = bounds
    return np.array([[x,y,z] for x in [lower[0],upper[0]] for y in [lower[1],upper[1]] for z in [lower[2],upper[2]]])


def camera_axes():
    pitch, yaw = np.radians([PITCH,YAW])
    front = np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
    right = np.cross(front,[0.,0.,1.]); right /= np.linalg.norm(right)
    return front, right, np.cross(right,front)


def merged_robot_model():
    g1 = native.create_robot_builder('unitree_g1')
    x2 = native.create_robot_builder('agibot_x2ultra')
    joints = [native.get_filtered_joint_names(builder,[newton.JointType.REVOLUTE]) for builder in [g1,x2]]
    assert [len(names) for names in joints]==[29,31]
    builder = newton.ModelBuilder()
    builder.add_builder(g1,label_prefix='our_g1/')
    builder.add_builder(x2,label_prefix='our_x2/')
    # Native G1 already supplies the sole world-attached floor; no second plane.
    model = builder.finalize()
    assert model.joint_coord_count==74 and model.articulation_count==2
    assert np.count_nonzero(model.shape_type.numpy()==int(newton.GeoType.PLANE))==1
    assert len(np.unique(model.body_world.numpy()))==1
    return SimpleNamespace(model=model,state=model.state(),joints=joints,g1_q=36,g1_bodies=g1.body_count)


def robot_inputs(motion, rig):
    paths=[ROOT/f'datasets/bones-seed/converted/unitree-g1-default-comparison/{motion.name}.csv',ROOT/f'datasets/bones-seed/converted/agibot-x2ultra/default-comparison/{motion.name}.csv']
    hashes=[sha256(path) for path in paths]
    original=[native.robot_frames(path,joints,motion.animation.num_frames) for path,joints in zip(paths,rig.joints)]
    g1=original[0].copy(); x2=original[1].copy()
    basis=Rotation.from_euler('z',90,degrees=True)
    g1[:,:3]=basis.apply(g1[:,:3])
    g1[:,3:7]=(basis*Rotation.from_quat(g1[:,3:7])).as_quat()
    assert np.array_equal(g1[:,7:],original[0][:,7:])
    return SimpleNamespace(paths=paths,hashes=hashes,clips=[g1,x2],g1_basis_xyzw=basis.as_quat().tolist())


def set_robot_pose(rig, clips, frame, lanes):
    pose=np.concatenate([clips[0][frame].copy(),clips[1][frame].copy()])
    pose[:3]+=lanes[1]
    pose[rig.g1_q:rig.g1_q+3]+=lanes[2]
    rig.model.joint_q.assign(pose)
    newton.eval_fk(rig.model,rig.model.joint_q,rig.model.joint_qd,rig.state)


def robot_mesh_bounds(motion, rig, clips):
    local_points,bodies=native.visual_bound_points(rig.model)
    mask=bodies<rig.g1_bodies
    bounds=[[],[]]
    for frame in motion.indices:
        set_robot_pose(rig,clips,int(frame),np.zeros((3,3)))
        poses=rig.state.body_q.numpy()[bodies]
        points=Rotation.from_quat(poses[:,3:]).apply(local_points)+poses[:,:3]
        for index,selection in enumerate([mask,~mask]):
            bounds[index].append([points[selection].min(axis=0),points[selection].max(axis=0)])
    return [np.asarray(actor_bounds) for actor_bounds in bounds]


def fixed_lanes(actor_bounds):
    _,right,_=camera_axes()
    projected=[np.array([[np.min(corners(frame)@right),np.max(corners(frame)@right)] for frame in bounds]) for bounds in actor_bounds]
    gap=.16
    left=max(.65,float(np.max(projected[0][:,1]-projected[1][:,0]))+gap)
    rightward=max(.65,float(np.max(projected[1][:,1]-projected[2][:,0]))+gap)
    return np.array([-left*right,np.zeros(3),rightward*right])


def group_cameras(actor_bounds):
    front,right,up=camera_axes()
    lower=np.minimum.reduce([bounds[:,0] for bounds in actor_bounds])-.08
    upper=np.maximum.reduce([bounds[:,1] for bounds in actor_bounds])+.08
    lower[:,2]=np.minimum(lower[:,2],-.08)
    smooth=gaussian_filter1d((lower+upper)/2,sigma=3,axis=0,mode='nearest')
    future=np.minimum(np.arange(len(smooth))+3,len(smooth)-1)
    targets=.8*smooth+.2*smooth[future]
    tangent=np.tan(np.radians(45/2)); required=[]
    for index,target in enumerate(targets):
        projected_bounds=[]
        for actor in actor_bounds:
            low=actor[index,0].copy()-.08;high=actor[index,1].copy()+.08
            low[2]=min(low[2],-.08)
            projected_bounds.append(corners([low,high]))
        extent=np.concatenate(projected_bounds)-target
        required.append(max(np.max(np.abs(extent@right)/(tangent*WIDTH/HEIGHT)-extent@front),np.max(np.abs(extent@up)/tangent-extent@front)))
    required=np.maximum(np.asarray(required)*1.06,.8)
    distance=np.maximum(required,gaussian_filter1d(maximum_filter1d(required,size=15,mode='nearest'),sigma=3,mode='nearest'))
    return {'positions':(targets-distance[:,None]*front).tolist(),'targets':targets.tolist(),'yaw':YAW,'pitch':PITCH,'fov':45.,'policy':'Single smooth current union camera with 3-frame20percent lookahead; no full-trajectory fit','ground_z':0.,'fit_geometry':'Union of individual actor mesh-bound corners, NOT empty global world-AABB corners'}


def screen_position(position, camera, index):
    front,right,up=camera_axes();relative=position-np.asarray(camera['positions'][index])
    depth=relative@front; scale=HEIGHT/2/np.tan(np.radians(camera['fov']/2))
    return np.array([WIDTH/2+scale*(relative@right)/depth,HEIGHT/2-scale*(relative@up)/depth])


def annotate(frame, scene, index):
    image=Image.fromarray(frame);draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    labels=['SOURCE SOMA MESH','OFFICIAL G1','PRODUCTION OPTIMIZED X2']
    for label,bounds in zip(labels,scene.bounds):
        anchor=(bounds[index,0]+bounds[index,1])/2;anchor[2]=bounds[index,1,2]+.07
        point=screen_position(anchor,scene.camera,index)
        width=draw.textlength(label,font=font)+20;x=float(np.clip(point[0]-width/2,8,WIDTH-width-8));y=float(np.clip(point[1]-35,44,HEIGHT-90))
        draw.rectangle((x,y,x+width,y+34),fill='#152331');draw.text((x+10,y+2),label,font=font,fill='white')
    draw.rectangle((0,0,WIDTH,37),fill='#111923');draw.text((14,3),'ONE 3D SCENE | native SOMA mesh + OFFICIAL G1 + PRODUCTION OPTIMIZED X2',font=font,fill='white')
    draw.rectangle((0,HEIGHT-32,WIDTH,HEIGHT),fill='#111923');draw.text((14,HEIGHT-29),f'{scene.motion.name} | source frame {scene.motion.indices[index]} | t={scene.motion.indices[index]*scene.motion.dt:.3f}s | fixed XY lanes; unchanged input motions',font=small,fill='white')
    return image


def scene_setup(name):
    motion=native.source_motion(name);rig=merged_robot_model();inputs=robot_inputs(motion,rig)
    paths=[motion.path]+inputs.paths
    hashes=[sha256(motion.path)]+inputs.hashes
    bounds=[native.source_bounds(motion)]+robot_mesh_bounds(motion,rig,inputs.clips)
    lanes=fixed_lanes(bounds)
    bounds=[actor_bounds+lane[None,None,:] for actor_bounds,lane in zip(bounds,lanes)]
    motion.instance.xform=wp.mul(wp.transform(wp.vec3(*lanes[0]),wp.quat_identity()),motion.instance.xform)
    return SimpleNamespace(motion=motion,rig=rig,inputs=inputs,paths=paths,hashes=hashes,bounds=bounds,lanes=lanes,camera=group_cameras(bounds))


def scene_viewer(scene):
    viewer=native.ViewerGL(width=WIDTH,height=HEIGHT,headless=True)
    viewer.set_model(scene.rig.model);viewer.set_world_offsets([0,0,0]);viewer.camera.fov=45.
    return viewer


def draw_scene_frame(viewer, scene, index):
    frame=int(scene.motion.indices[index])
    set_robot_pose(scene.rig,scene.inputs.clips,frame,scene.lanes)
    scene.motion.instance.set_local_transforms(scene.motion.animation.get_local_transforms(frame))
    viewer.set_camera(wp.vec3(*scene.camera['positions'][index]),PITCH,YAW)
    viewer.begin_frame(index/30)
    scene.motion.renderer.draw(viewer,scene.motion.instance,native.COLOR,100)
    viewer.log_state(scene.rig.state);viewer.end_frame()
    return annotate(viewer.get_frame().numpy(),scene,index)


def preview_scene(scene):
    viewer=scene_viewer(scene)
    try:
        image=draw_scene_frame(viewer,scene,len(scene.motion.indices)//2)
        output=ART/'previews';output.mkdir(parents=True,exist_ok=True)
        image.save(output/f'{scene.motion.name}__tight-smoke.png')
        print('ONE_REAL_SCENE_TIGHT_PREVIEW',output,flush=True)
    finally:
        viewer.close()


def encode_scene(scene):
    viewer=scene_viewer(scene)
    output=ART/'videos'/f'{scene.motion.name}.mp4';output.parent.mkdir(parents=True,exist_ok=True)
    command=[str(native.FFMPEG),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index,frame in enumerate(scene.motion.indices):
            image=draw_scene_frame(viewer,scene,index)
            encoder.stdin.write(image.tobytes())
            if index in {0,len(scene.motion.indices)//2,len(scene.motion.indices)-1}:
                preview=ART/'previews';preview.mkdir(exist_ok=True)
                image.save(preview/f'{scene.motion.name}__frame-{int(frame):04d}.png')
    finally:
        encoder.stdin.close();viewer.close()
    if encoder.wait()!=0:raise RuntimeError('Shared-scene video encoder failed')
    if [sha256(path) for path in scene.paths]!=scene.hashes:raise RuntimeError('Input motion changed while rendering')
    return output


def record_scene(scene, video):
    metadata={'name':scene.motion.name,'output':str(video),'output_sha256':sha256(video),'width':WIDTH,'height':HEIGHT,'fps':30,'frames':len(scene.motion.indices),'source_indices':scene.motion.indices.tolist(),'source_frame_time':scene.motion.dt,'inputs':[{'path':str(path),'sha256':digest} for path,digest in zip(scene.paths,scene.hashes)],'render_method':'ONE Newton ViewerGL viewport; ONE merged G1+X2 model; native SOMA USD skinning drawn into SAME frame/depthbuffer; ONE existing native G1 floor; no panel captures or stitching','single_viewport':True,'single_ground_plane':True,'ground_plane_count':1,'merged_robot_bodies':scene.rig.model.body_count,'merged_robot_joint_coordinates':scene.rig.model.joint_coord_count,'merged_robot_articulations':2,'merged_body_world_ids':np.unique(scene.rig.model.body_world.numpy()).tolist(),'actuated_joint_counts':{'our_g1':29,'our_x2':31},'source_mesh_vertices':[mesh.num_points for mesh in scene.motion.renderer.skeletal_mesh.skinned_meshes],'source_basis':'Native optimizer MAYA','g1_display_basis':'Fixed +90degree worldZ on root position and root quaternion ONLY; CSV unchanged; joint angles unchanged','g1_root_basis_quaternion_xyzw':scene.inputs.g1_basis_xyzw,'x2_display_basis':'Already native MAYA/Newton-world CSV; no extra conversion','fixed_xy_lane_offsets_metres':dict(zip(['source_soma','our_g1','our_x2'],scene.lanes.tolist())),'lanes_are_time_invariant':True,'camera':scene.camera,'input_files_modified':False,'helper_path':str(HELPER),'helper_sha256':sha256(HELPER)}
    output=ART/'metadata';output.mkdir(parents=True,exist_ok=True)
    (output/f'{scene.motion.name}.json').write_text(json.dumps(metadata,indent=2))
    print('SINGLE_SHARED_SCENE_COMPLETE',video,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--preview-only',action='store_true');args=parser.parse_args()
    selection=json.loads((ROOT/'datasets/bones-seed/metadata/x2-validation-selection.json').read_text())
    if args.name not in {entry['name'] for entry in selection}:raise ValueError('Not a canonical11 sequence')
    scene=scene_setup(args.name)
    if args.preview_only:
        preview_scene(scene)
    else:
        video=encode_scene(scene);record_scene(scene,video)


if __name__=='__main__':main()
