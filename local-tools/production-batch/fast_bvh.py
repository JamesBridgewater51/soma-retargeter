"""Optional BVH loader: vectorized payload preparation, native Warp pose math."""
from pathlib import Path
import re
import time
import warnings

import numpy as np
import warp as wp

from soma_retargeter.animation.animation_buffer import AnimationBuffer, create_animation_buffer_for_skeleton
from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.io.bvh import BVHImporter, load_bvh as native_load_bvh, wp_convert_frame_animation


# Keys include every hierarchy byte, including offsets, channels and End Sites.
_LAYOUTS = {}


def _hierarchy_layout(hierarchy, source_path):
    if hierarchy in _LAYOUTS:
        return _LAYOUTS[hierarchy]
    # Native owns hierarchy/offset/namespace/End Site semantics. Its first
    # payload parse is amortized across identical headers, not across paths.
    skeleton, root = BVHImporter().create_skeleton(source_path)
    channels = []

    def visit(joint):
        channels.append(tuple(joint.channels))
        for child in joint.children:
            visit(child)

    visit(root)
    if len(_LAYOUTS) >= 16:
        _LAYOUTS.pop(next(iter(_LAYOUTS)))
    _LAYOUTS[hierarchy] = skeleton, tuple(channels)
    return _LAYOUTS[hierarchy]


def _motion_sections(text):
    match = re.search(r'^\s*MOTION\s*$', text, re.MULTILINE)
    if match is None:
        raise ValueError('Malformed BVH: missing MOTION section')
    motion = text[match.end():]
    header = re.match(r'\s*Frames:\s*(\d+)\s*\n\s*Frame\s+Time:\s*(\S+)[^\S\n]*\n?', motion)
    if header is None:
        raise ValueError('Malformed BVH: missing frame count or frame time')
    frames, frame_time = int(header[1]), float(header[2])
    if frames == 0 or not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError('BVH requires at least one frame and positive finite frame time')
    return text[:match.start()], motion[header.end():], frames, round(1 / frame_time, 2)


def _channel_columns(channels):
    position_columns, rotation_columns, orders = [], [], []
    offset = 0
    for joint_channels in channels:
        p = [offset + i for i, c in enumerate(joint_channels) if 'position' in c]
        r = [offset + i for i, c in enumerate(joint_channels) if 'rotation' in c]
        if len(p) not in (0, 3) or len(r) != 3 or len(p) + len(r) != len(joint_channels):
            return None
        if any(c[0].lower() not in 'xyz' for c in joint_channels):
            return None
        position_columns.append(p)
        rotation_columns.append(r)
        orders.append(['xyz'.index(c[0].lower()) for c in joint_channels if 'rotation' in c])
        offset += len(joint_channels)
    return position_columns, rotation_columns, orders, offset


def _payload_array(payload, frames, channel_count):
    # Python float -> float32 is native's path. Direct decimal -> float32
    # parsing need not have identical double-rounding behavior.
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        try:
            values = np.fromstring(payload, sep=' ', dtype=np.float64)
        except (ValueError, DeprecationWarning) as exc:
            raise ValueError('Malformed BVH numeric payload') from exc
    if values.size != frames * channel_count or not np.isfinite(values).all():
        raise ValueError(f'Invalid BVH payload: expected {frames * channel_count} finite channel values, got {values.size}')
    with np.errstate(over='ignore'):
        values = values.reshape(frames, channel_count).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('BVH payload exceeds finite float32 range')
    return values


def _position_arrays(values, position_columns):
    frames = len(values)
    positions = np.zeros((frames, len(position_columns), 3), dtype=np.float32)
    has_positions = np.zeros((frames, len(position_columns)), dtype=np.bool_)
    for joint_index, columns in enumerate(position_columns):
        if columns:
            positions[:, joint_index] = values[:, columns]
            has_positions[:, joint_index] = True
    return positions, has_positions


def _native_conversion(skeleton, position_arrays, rotations, orders):
    positions, has_positions = position_arrays
    frames, joints = has_positions.shape
    output = wp.empty((frames, joints), dtype=wp.transform)
    wp.launch(wp_convert_frame_animation, dim=(frames, joints), inputs=[
        wp.array(skeleton.reference_local_transforms, dtype=wp.transform),
        wp.array(has_positions, dtype=wp.bool),
        wp.array(np.ones((frames, joints), dtype=np.bool_), dtype=wp.bool),
        wp.array(positions, dtype=wp.float32),
        wp.array(rotations, dtype=wp.float32),
        wp.array(np.arange(joints, dtype=np.int32), dtype=wp.int32),
        wp.array(np.asarray(orders, dtype=np.int32), dtype=wp.int32),
    ], outputs=[output])
    return output.numpy()


def load_bvh(bvh_file, input_skeleton=None, *, timings=None):
    """Return native Skeleton/AnimationBuffer; unsupported channels use native IO.

    ``timings`` receives read, prepare, and conversion seconds when requested.
    Empty clips are rejected explicitly (native cannot construct their frame rate).
    """
    started = time.perf_counter()
    path = Path(bvh_file)
    if not path.is_file() or path.suffix.lower() != '.bvh':
        raise ValueError(f'Invalid BVH file path: {bvh_file}')
    hierarchy, payload, frames, fps = _motion_sections(path.read_text(encoding='utf-8'))
    read_done = time.perf_counter()
    template, channels = _hierarchy_layout(hierarchy, str(path))
    # Native returns a fresh mutable Skeleton without input_skeleton. Never
    # expose the cached template to scalers or other downstream consumers.
    skeleton = Skeleton(template.num_joints, template.joint_names,
                        template.parent_indices.copy(), template.reference_local_transforms)
    columns = _channel_columns(channels)
    if columns is None or len(set(skeleton.joint_names)) != skeleton.num_joints:
        return native_load_bvh(str(path), input_skeleton)
    position_columns, rotation_columns, orders, channel_count = columns
    values = _payload_array(payload, frames, channel_count)
    position_arrays = _position_arrays(values, position_columns)
    rotations = np.ascontiguousarray(values[:, np.asarray(rotation_columns)])
    prepared = time.perf_counter()
    local_transforms = _native_conversion(skeleton, position_arrays, rotations, orders)
    animation = AnimationBuffer(skeleton, frames, fps, local_transforms)
    converted = time.perf_counter()
    if input_skeleton is not None:
        animation = create_animation_buffer_for_skeleton(animation, input_skeleton)
        skeleton = input_skeleton
    if timings is not None:
        timings.update(read_seconds=read_done-started, prepare_seconds=prepared-read_done,
                       convert_seconds=converted-prepared, total_seconds=time.perf_counter()-started)
    return skeleton, animation
