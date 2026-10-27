"""Own-reference BVH decoding followed by certified exact topology conformance."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from soma_retargeter.animation.animation_buffer import create_animation_buffer_for_skeleton
from soma_retargeter.animation.contact_phase import load_authored_foot_landmark_model
from soma_retargeter.io.bvh import BVHImporter, load_bvh as native_load_bvh
from soma_retargeter.pipelines import utils as pipeline_utils


def _topology_signature(skeleton):
    names = list(skeleton.joint_names)
    if len(names) != skeleton.num_joints or len(set(names)) != len(names):
        raise ValueError('Decoded topology requires unique ordered joint names')
    parents = np.asarray(skeleton.parent_indices, dtype=np.int32)
    if parents.shape != (len(names),):
        raise ValueError('Decoded topology has an invalid parent-index shape')
    encoded = json.dumps([names, parents.tolist()], separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SomaX2TopologyCertification:
    """In-process admission evidence, not a serialized replacement for live checks."""
    canonical_skeleton: object
    topology_sha256: str
    zero_pose_sha256: str
    contact_config_sha256: str


def certify_soma_x2_topology(skeleton):
    """Certify current native SOMA→X2 preprocessing, never arbitrary custom pipelines.

    The caller must use the native SOMA initialization/contact asset routing and
    unchanged X2 scaler configuration. Complete initialization coverage and an
    authored SOMA landmark model remove downstream reference-rest dependencies.
    A changed/custom source preprocessing configuration requires recertification.
    """
    signature = _topology_signature(skeleton)
    source = pipeline_utils.get_source_type_from_str('soma')
    zero_path = Path(pipeline_utils.get_source_zero_pose_asset_path(source))
    zero_skeleton, _ = BVHImporter().create_skeleton(str(zero_path))
    if skeleton.num_joints != 78 or set(skeleton.joint_names) != set(zero_skeleton.joint_names):
        raise ValueError('Native SOMA certification requires complete 78-joint zero-pose coverage')
    contact_path = Path(pipeline_utils.get_source_contact_detection_config_path(source))
    contact_bytes = contact_path.read_bytes()
    config = json.loads(contact_bytes)
    landmark_path = config.get('foot_landmarks_path')
    landmark_source = config.get('foot_landmarks') or (
        str(contact_path.parent / landmark_path) if landmark_path else None)
    if load_authored_foot_landmark_model(skeleton, 'soma', landmark_source) is None:
        raise ValueError('Topology batching requires authored SOMA landmarks, not rest-derived heuristics')
    return SomaX2TopologyCertification(skeleton, signature,
        hashlib.sha256(zero_path.read_bytes()).hexdigest(), hashlib.sha256(contact_bytes).hexdigest())


def _require_certification(canonical, certification):
    if certification is None:
        raise ValueError('Common-topology remapping requires native SOMA→X2 certification')
    if certification.canonical_skeleton is not canonical or certification.topology_sha256 != _topology_signature(canonical):
        raise ValueError('Certification does not match the current canonical Skeleton')


def load_decoded_bvh(path, canonical_skeleton=None, *, decoder='native', certification=None):
    """Return ``(Skeleton, AnimationBuffer)`` without decoding through foreign offsets.

    First load without a canonical Skeleton, then certify that Skeleton once and
    pass its certificate on subsequent loads. Only the known native/fast decoders
    are accepted, and both receive ONLY the original path. File names, frame count,
    sample rate, ordered joint names and every decoded local-transform byte survive.
    Rest transforms are deliberately NOT required to match after own decoding.
    """
    if decoder not in ('native', 'fast'):
        raise ValueError("decoder must be 'native' or 'fast'")
    if canonical_skeleton is not None:
        _require_certification(canonical_skeleton, certification)
    if decoder == 'fast':
        from fast_bvh import load_bvh
    else:
        load_bvh = native_load_bvh
    skeleton, animation = load_bvh(str(path))
    signature = _topology_signature(skeleton)
    if animation.skeleton is not skeleton:
        raise ValueError('Decoder returned inconsistent Skeleton/AnimationBuffer ownership')
    if canonical_skeleton is None:
        return skeleton, animation
    if signature != certification.topology_sha256:
        raise ValueError('Decoded joint names/order or parent indices differ from canonical topology')
    remapped = create_animation_buffer_for_skeleton(animation, canonical_skeleton)
    if (remapped.num_frames != animation.num_frames or remapped.sample_rate != animation.sample_rate
            or remapped.local_transforms.dtype != animation.local_transforms.dtype
            or remapped.local_transforms.shape != animation.local_transforms.shape
            or remapped.local_transforms.tobytes() != animation.local_transforms.tobytes()):
        raise RuntimeError('Native conformance changed decoded motion values or metadata')
    return canonical_skeleton, remapped
