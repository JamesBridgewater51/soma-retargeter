"""Disjoint continuation of the unchanged approved shared-scene renderer."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/liuminghao')
ART = ROOT/'artifacts/soma-retargeter/agibot-x2ultra/challenge100-review'
OPS = ART/'parallel-render'
SOURCE_DIR = ROOT/'agibot-x2/challenge100-review/renderer'
RENDERER = Path(__file__).with_name('render_review_threads4.py')
PYTHON = ROOT/'agibot-x2/soma-retargeter/.venv/bin/python'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, record):
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def emit(event, **fields):
    print(json.dumps({'event':event,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields}),flush=True)


def selection():
    path = OPS/'selection-frozen.json'
    launch = json.loads((OPS/'assignments.json').read_text())
    if digest(path) != launch['manifest_sha256']:
        raise ValueError('Frozen selection changed')
    return json.loads(path.read_text()), launch


def validated_receipt(entry):
    path = ART/'receipts'/f"{entry['filename']}.json"
    video = ART/'videos'/f"{entry['filename']}.mp4"
    if not path.exists() or not video.exists():
        return None
    receipt = json.loads(path.read_text())
    expected = math.ceil(entry['frames']/4)
    if (receipt.get('status') != 'complete' or receipt.get('source_frames') != entry['frames']
            or receipt.get('frames') != expected or receipt.get('decoded_frames') != expected
            or (receipt.get('width'),receipt.get('height'),receipt.get('fps')) != (1920,1080,30)):
        return None
    if digest(video) != receipt.get('output_sha256'):
        return None
    expected_paths = [entry[key] for key in ('soma_bvh','official_g1_csv','production_x2_csv')]
    if [row['path'] for row in receipt['inputs']] != expected_paths:
        return None
    if any(digest(row['path']) != row['sha256'] for row in receipt['inputs']):
        return None
    return receipt


def renderer_module():
    sys.path.insert(0,str(SOURCE_DIR))
    specification = importlib.util.spec_from_file_location('approved_parallel_renderer',RENDERER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def render_assigned(index):
    entries, launch = selection()
    names = set(launch['assignments'][str(index)])
    assigned = [entry for entry in entries if entry['filename'] in names]
    renderer = renderer_module()
    rig = renderer.scene_api.merged_robot_model()
    viewer = renderer.scene_api.scene_viewer(renderer.SimpleNamespace(rig=rig))
    emit('WORKER_REAL_START',worker=index,pid=os.getpid(),clips=len(assigned),manifest_sha256=launch['manifest_sha256'])
    session = renderer.SimpleNamespace(renderer=renderer,viewer=viewer,rig=rig)
    try:
        for entry in assigned:
            render_entry(session,entry,index)
    finally:
        viewer.close()
    emit('WORKER_COMPLETE',worker=index,clips=len(assigned))


def render_entry(session,entry,index):
    if validated_receipt(entry):
        emit('VALID_OUTPUT_SKIPPED',worker=index,name=entry['filename'])
        return
    name = entry['filename']
    final = ART/'videos'/f'{name}.mp4'
    if final.exists() or (ART/'receipts'/f'{name}.json').exists():
        raise ValueError(f'Existing unvalidated complete product requires explicit archival: {name}')
    prepared = dict(entry,name=name,bvh_path=entry['soma_bvh'],official_g1_path=entry['official_g1_csv'],x2_path=entry['production_x2_csv'])
    emit('CLIP_START',worker=index,name=name)
    scene = session.renderer.setup(prepared,session.rig)
    session.renderer.render(scene,session.viewer)
    if not validated_receipt(entry):
        raise ValueError(f'Post-render input/output validation failed: {name}')
    emit('CLIP_VERIFIED',worker=index,name=name)
    del scene
    session.renderer.gc.collect()


def published_count(entries):
    return sum((ART/'receipts'/f"{entry['filename']}.json").exists() for entry in entries)


def start_workers(launch):
    processes = []
    environment = dict(os.environ,STORAGE_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    for index in range(4):
        log = ROOT/f'logs/soma-retargeter/agibot-x2ultra/challenge100-review/parallel-worker-{index}.log'
        with log.open('ab') as stream:
            child = subprocess.Popen([str(PYTHON),'-I',str(Path(__file__)),'--worker',str(index)],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True,cwd='/')
        processes.append(child)
    atomic_json(OPS/'processes.json',{'supervisor_pid':os.getpid(),'worker_pids':[child.pid for child in processes],
                                    'manifest_sha256':launch['manifest_sha256'],'wrapper_sha256':digest(Path(__file__))})
    return processes


def audit_complete(entries,launch):
    products, failures = [], []
    for entry in entries:
        receipt = validated_receipt(entry)
        if receipt is None:
            failures.append(entry['filename'])
            continue
        products.append({'name':entry['filename'],'video_sha256':receipt['output_sha256'],'frames':receipt['frames']})
    report = {'status':'COMPLETE' if not failures and len(products)==100 else 'INCOMPLETE',
              'manifest_sha256':launch['manifest_sha256'],'count':len(products),'failures':failures,
              'frames':sum(row['frames'] for row in products),'products':products,
              'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(OPS/'coverage-audit.json',report)
    return report


def supervise():
    entries, launch = selection()
    processes = start_workers(launch)
    started = time.monotonic()
    emit('FOUR_DISJOINT_WORKERS_STARTED',pids=[child.pid for child in processes])
    while any(child.poll() is None for child in processes):
        atomic_json(ART/'progress.json',{'completed':published_count(entries),'total':100,
                    'elapsed_seconds':time.monotonic()-started,'mode':'four-disjoint-workers'})
        time.sleep(10)
    exits = [child.returncode for child in processes]
    report = audit_complete(entries,launch)
    success = not any(exits) and report['status']=='COMPLETE'
    atomic_json(OPS/'exit.json',{'returncode':0 if success else 1,'success':success,'worker_returncodes':exits,
                               'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    atomic_json(ART/'progress.json',{'completed':report['count'],'total':100,'status':report['status'],
                                   'elapsed_seconds':time.monotonic()-started,'mode':'four-disjoint-workers'})
    emit('RENDER_BATCH_COMPLETE' if success else 'RENDER_BATCH_FAILED',count=report['count'],failures=report['failures'])
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,choices=range(4))
    args = parser.parse_args()
    if args.worker is None:
        raise SystemExit(supervise())
    render_assigned(args.worker)
