"""Wait for the owned render continuation, then verify the exact selected media."""
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/liuminghao')
ART = ROOT/'artifacts/soma-retargeter/agibot-x2ultra/challenge100-review'
OPS = ART/'parallel-render'
FFMPEG = ROOT/'wallvideo/egoexo/bin/ffmpeg'
FFPROBE = Path('/usr/bin/ffprobe')
specification = importlib.util.spec_from_file_location('render_ops',Path(__file__).with_name('parallel_review.py'))
operations = importlib.util.module_from_spec(specification)
specification.loader.exec_module(operations)


def wait_terminal():
    started = time.monotonic()
    last = None
    while not (OPS/'exit.json').exists():
        if time.monotonic()-started > 7200:
            raise TimeoutError('Owned render continuation exceeded two hours')
        supervisor = json.loads((OPS/'processes.json').read_text())['supervisor_pid']
        if not (Path('/proc')/str(supervisor)).exists() and not (OPS/'exit.json').exists():
            raise RuntimeError('Owned render supervisor exited without a durable terminal receipt')
        progress = json.loads((ART/'progress.json').read_text())
        if progress.get('completed') != last:
            last = progress.get('completed')
            operations.emit('WAITING_FOR_ACTUAL_TERMINAL',completed=last,total=100)
        time.sleep(20)
    terminal = json.loads((OPS/'exit.json').read_text())
    if terminal['returncode'] != 0 or not terminal['success']:
        raise RuntimeError(f'Render continuation failed: {terminal}')
    return terminal


def stream_metadata(path):
    completed = subprocess.run([str(FFPROBE),'-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames','-of','json',str(path)],
        capture_output=True,text=True,check=True)
    streams = json.loads(completed.stdout)['streams']
    if len(streams) != 1:
        raise ValueError(f'Expected exactly one video stream: {path}')
    return streams[0]


def verify_stream(entry,receipt):
    stream = stream_metadata(ART/'videos'/f"{entry['filename']}.mp4")
    if (stream['codec_name'],stream['width'],stream['height'],stream['pix_fmt'],stream['avg_frame_rate']) != ('h264',1920,1080,'yuv420p','30/1'):
        raise ValueError(f'Unexpected codec/viewport/FPS: {entry["filename"]}')
    if int(stream['nb_frames']) != receipt['frames']:
        raise ValueError(f'Stream frame count mismatch: {entry["filename"]}')
    expected = [min(4*index,entry['frames']-1) for index in range(receipt['frames'])]
    if receipt['source_indices'] != expected:
        raise ValueError(f'Source frame mapping mismatch: {entry["filename"]}')
    if not all(receipt.get(key) is True for key in ('single_viewport','single_ground_plane',
            'no_per_frame_root_alignment','no_height_correction','no_actor_rescaling')):
        raise ValueError(f'Shared-scene contract mismatch: {entry["filename"]}')
    if any(offset[2] != 0 for offset in receipt['fixed_xy_lane_offsets_metres'].values()):
        raise ValueError(f'Unexpected vertical display correction: {entry["filename"]}')
    return stream


def representative_entries(entries):
    chosen = [next(entry for entry in entries if entry['filename']=='flip_360_002__A304')]
    for category in ('terrain','object_interaction','floor_recovery'):
        chosen.append(next(entry for entry in entries if entry['bucket']==category))
    return chosen


def compare_decoded_frame(entry,receipt):
    import numpy as np
    from PIL import Image
    middle = receipt['frames']//2
    path = ART/'videos'/f"{entry['filename']}.mp4"
    decoded = subprocess.run([str(FFMPEG),'-v','error','-i',str(path),'-vf',f'select=eq(n\\,{middle})',
        '-vsync','0','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True)
    frame = np.frombuffer(decoded.stdout,dtype=np.uint8).reshape(1080,1920,3)
    preview = np.asarray(Image.open(ART/'previews'/f"{entry['filename']}.png").convert('RGB'))
    error = float(np.abs(frame.astype(np.float32)-preview.astype(np.float32)).mean())
    if error > 8:
        raise ValueError(f'Video/inline-frame preview differ unexpectedly: {entry["filename"]}: {error}')
    directory = ART/'qa-video-frames';directory.mkdir(exist_ok=True)
    output = directory/f"{entry['filename']}__middle.png"
    Image.fromarray(frame).save(output)
    return {'name':entry['filename'],'video_frame':middle,'source_frame':receipt['source_indices'][middle],
            'decoded_vs_inline_preview_mean_abs_error':error,'decoded_frame_png':str(output)}


def verify_products(entries):
    products = []
    for entry in entries:
        receipt = operations.validated_receipt(entry)
        if receipt is None:
            raise ValueError(f'Invalid selected media: {entry["filename"]}')
        stream = verify_stream(entry,receipt)
        products.append({'name':entry['filename'],'video_sha256':receipt['output_sha256'],'frames':receipt['frames'],
                         'stream':stream,'receipt_sha256':operations.digest(ART/'receipts'/f"{entry['filename']}.json")})
    return products


def verify_representative_frames(entries):
    selected = representative_entries(entries)
    receipts = {entry['filename']:operations.validated_receipt(entry) for entry in selected}
    return [compare_decoded_frame(entry,receipts[entry['filename']]) for entry in selected]


def main():
    terminal = wait_terminal()
    entries, launch = operations.selection()
    if operations.digest(ART/'selection/challenge100-proposal.json') != launch['manifest_sha256']:
        raise ValueError('Published selection changed after the frozen render dispatch')
    products = verify_products(entries)
    previews = verify_representative_frames(entries)
    report = {'status':'COMPLETE','count':len(products),'frames':sum(row['frames'] for row in products),
              'manifest_sha256':launch['manifest_sha256'],'products':products,'representative_decoded_frames':previews,
              'terminal':terminal,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if len(products) != 100:
        raise ValueError('Review media coverage is not exactly one hundred')
    operations.atomic_json(OPS/'media-audit.json',report)
    operations.emit('HUNDRED_REVIEW_MEDIA_VERIFIED',count=100,frames=report['frames'],
                    report=str(OPS/'media-audit.json'),report_sha256=operations.digest(OPS/'media-audit.json'))


if __name__=='__main__':main()
