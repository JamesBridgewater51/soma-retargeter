"""CPU-only, clip-local metrics for X2 temporal ablations."""
import xml.etree.ElementTree as ET

import numpy as np


def urdf_velocity_limits(path, joint_names):
    joints = {joint.attrib['name']: joint for joint in ET.parse(path).getroot().findall('joint')}
    limits = []
    for name in joint_names:
        joint = joints[name]
        limit = joint.find('limit')
        if joint.attrib['type'] not in ('revolute', 'continuous') or limit is None:
            raise ValueError(f'No angular velocity limit for {name}')
        velocity = float(limit.attrib['velocity'])
        if not np.isfinite(velocity) or velocity <= 0:
            raise ValueError(f'Invalid velocity limit for {name}: {velocity}')
        limits.append(velocity)
    return np.asarray(limits)


def distribution(values, names, order):
    magnitude = np.abs(values)
    frame, joint = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    return {
        'rms': float(np.sqrt(np.mean(values ** 2))),
        'p95': float(np.percentile(magnitude, 95)),
        'p99': float(np.percentile(magnitude, 99)),
        'max': float(magnitude[frame, joint]),
        'worst_joint': names[joint],
        'worst_frame_interval': [int(frame), int(frame + order)],
        'sample_count': int(values.size),
    }


def trajectory_metrics(q, fps, joint_names, velocity_limits):
    q = np.asarray(q, dtype=np.float64)
    limits = np.asarray(velocity_limits, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 38 or len(joint_names) != 31 or q.shape[0] < 4:
        raise ValueError('Expected >=4 frames of root XYZ+quaternion and 31 angular joints')
    if not np.isfinite(q).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Non-finite trajectory or invalid sample rate')
    if limits.shape != (31,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Expected 31 positive URDF angular velocity limits in rad/s')
    # Never unwrap bounded joint angles: a genuine 2pi solver jump is a spike.
    angles = q[:, 7:]
    result = {}
    for order, key in enumerate(('velocity_rad_s', 'acceleration_rad_s2', 'jerk_rad_s3'), 1):
        values = np.diff(angles, n=order, axis=0) * fps ** order
        result[key] = distribution(values, joint_names, order)
    velocities = np.diff(angles, axis=0) * fps
    ratio = np.abs(velocities) / limits[None, :]
    exceeded = ratio > 1.0
    frame, joint = np.unravel_index(np.argmax(ratio), ratio.shape)
    result['velocity_limits'] = {
        'exceedance_samples': int(exceeded.sum()),
        'rounding_relative_tolerance': 1e-4,
        'exceedance_samples_with_rounding_tolerance': int((ratio > 1.0001).sum()),
        'exceedance_frames_with_rounding_tolerance': int((ratio > 1.0001).any(axis=1).sum()),
        'exceedance_frames': int(exceeded.any(axis=1).sum()),
        'exceedance_fraction': float(exceeded.mean()),
        'max_ratio': float(ratio[frame, joint]),
        'worst_joint': joint_names[joint],
        'worst_frame_interval': [int(frame), int(frame + 1)],
        'worst_joint_limit_rad_s': float(limits[joint]),
        'per_joint_exceedance_samples': dict(zip(joint_names, exceeded.sum(axis=0).astype(int).tolist())),
    }
    return result


def tracking_metrics(positions, rotations, targets):
    positions, rotations, targets = map(np.asarray, (positions, rotations, targets))
    if positions.shape != targets[..., :3].shape or rotations.shape != targets[..., 3:].shape:
        raise ValueError('Target and FK tracking arrays do not align')
    if not all(np.isfinite(value).all() for value in (positions, rotations, targets)):
        raise ValueError('Non-finite FK/target value')
    position_errors = np.linalg.norm(positions - targets[..., :3], axis=-1)
    q = rotations / np.linalg.norm(rotations, axis=-1, keepdims=True)
    target_q = targets[..., 3:] / np.linalg.norm(targets[..., 3:], axis=-1, keepdims=True)
    rotation_errors = 2 * np.arccos(np.clip(np.abs(np.sum(q * target_q, axis=-1)), 0, 1))
    if not np.isfinite(rotation_errors).all():
        raise ValueError('Invalid quaternion in FK tracking')
    return {
        'position_rmse_m': float(np.sqrt(np.mean(position_errors ** 2))),
        'rotation_rmse_rad': float(np.sqrt(np.mean(rotation_errors ** 2))),
        'per_effector_position_rmse_m': np.sqrt(np.mean(position_errors ** 2, axis=0)).tolist(),
        'per_effector_rotation_rmse_rad': np.sqrt(np.mean(rotation_errors ** 2, axis=0)).tolist(),
    }
