"""Independent streaming validation of exported degree-valued joint angles.

Fingerprint encoding matches production ``key_for``: SHA-256 of UTF-8
``json.dumps(value, sort_keys=True, allow_nan=False)`` with default separators.
Velocity-limit fingerprints cover selected joint-name-to-limit mappings;
ordered-name fingerprints independently bind CSV column order.
"""

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

RELATIVE_TOLERANCE = 1e-4


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def _joint_name(name):
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Joint names must be nonempty strings without surrounding whitespace")
    return name


def _angular_limits(robot):
    if robot.tag != "robot":
        raise ValueError("Expected a URDF robot root element")
    limits = {}
    seen = set()
    for joint in robot.findall("joint"):
        name = _joint_name(joint.get("name"))
        if name in seen:
            raise ValueError(f"Duplicate URDF joint name: {name}")
        seen.add(name)
        kind = joint.get("type")
        if kind not in {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}:
            raise ValueError(f"Invalid URDF joint type for {name}")
        if kind in {"revolute", "continuous"}:
            entries = joint.findall("limit")
            if len(entries) != 1 or entries[0].get("velocity") is None:
                raise ValueError(f"Missing or ambiguous angular velocity limit for {name}")
            limits[name] = _positive_number(entries[0].get("velocity"), f"Velocity limit for {name}")
    if not limits:
        raise ValueError("URDF has no angular velocity limits")
    return limits


def load_velocity_contract(urdf_path, velocity_scale=1.0):
    """Read and fingerprint one real URDF; limits remain unscaled radians/second."""
    payload = Path(urdf_path).read_bytes()
    try:
        robot = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ValueError("Malformed URDF XML") from error
    return {"schema": 1, "urdf_sha256": hashlib.sha256(payload).hexdigest(),
            "limits_rad_s": _angular_limits(robot), "limit_units": "rad/s",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "velocity_scale": _positive_number(velocity_scale, "velocity_scale")}


def _ordered_contract(joint_names, plan):
    if not isinstance(plan, Mapping):
        raise ValueError("Velocity plan must be a mapping")
    if "schema" in plan and (type(plan["schema"]) is not int or plan["schema"] != 1):
        raise ValueError("Unsupported velocity contract schema")
    if plan.get("limit_units", "rad/s") != "rad/s":
        raise ValueError("URDF velocity limits must be in rad/s")
    if plan.get("relative_tolerance", RELATIVE_TOLERANCE) != RELATIVE_TOLERANCE:
        raise ValueError("CSV velocity validation requires relative_tolerance=1e-4")
    digest = plan.get("urdf_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Velocity contract requires a lowercase URDF SHA-256")
    if isinstance(joint_names, (str, bytes)):
        raise ValueError("Expected an ordered sequence of joint names")
    try:
        names = tuple(_joint_name(name) for name in joint_names)
    except TypeError as error:
        raise ValueError("Expected an ordered sequence of joint names") from error
    if not names or len(set(names)) != len(names):
        raise ValueError("CSV angular joint names must be nonempty and unique")
    mapping = plan.get("limits_rad_s")
    if not isinstance(mapping, Mapping) or any(name not in mapping for name in names):
        raise ValueError("CSV joint is missing a matching URDF angular velocity limit")
    limits = tuple(_positive_number(mapping[name], f"Velocity limit for {name}") for name in names)
    return names, limits, digest


class VelocityValidationError(ValueError):
    """Actual exported speed violations; the failure report is not a done receipt."""

    def __init__(self, attestation):
        self.attestation = attestation
        super().__init__("Exported CSV violates URDF speed limits: "
                         f"{attestation['violation_count']} samples, "
                         f"max_ratio={attestation['max_ratio']:.9g}, "
                         f"worst_joint={attestation['worst_joint']}, "
                         f"frames={attestation['worst_frame_interval']}")


class VelocityAccumulator:
    """Observe parsed angular columns once, then attest or reject their speeds.

    ``plan`` is ``load_velocity_contract(...)`` plus the clip's ``native_fps``.
    Angular observations are degrees, in the exact supplied header order. Frame
    intervals in the report are zero-based CSV frame indices. Ratios use the
    declared URDF limits multiplied by the effective ``velocity_scale``.
    """

    def __init__(self, joint_names, plan):
        self.names, self._limits, self._urdf_sha256 = _ordered_contract(joint_names, plan)
        self.fps = _positive_number(plan.get("native_fps"), "native_fps")
        self.velocity_scale = _positive_number(plan.get("velocity_scale", 1.0), "velocity_scale")
        effective = tuple(_positive_number(limit * self.velocity_scale, "Effective velocity limit")
                          for limit in self._limits)
        self._factors = tuple(_positive_number(math.pi / 180.0 * self.fps / limit, "Velocity ratio factor")
                              for limit in effective)
        self._declared_fingerprint = fingerprint(dict(zip(self.names, self._limits, strict=True)))
        self._effective_fingerprint = fingerprint(dict(zip(self.names, effective, strict=True)))
        self.frames = 0
        self._previous = None
        self._violation_count = 0
        self._exceedance_frames = 0
        self._max_ratio = 0.0
        self._worst_joint = None
        self._worst_interval = None

    def observe_angles(self, angle_degrees):
        if isinstance(angle_degrees, (str, bytes)):
            raise ValueError("Expected parsed angular columns, not a CSV string")
        try:
            angles = tuple(float(angle) for angle in angle_degrees)
        except (TypeError, ValueError) as error:
            raise ValueError("Angular row must contain parsed numeric degree values") from error
        if len(angles) != len(self.names) or not all(math.isfinite(angle) for angle in angles):
            raise ValueError("Angular row must have the declared width and finite degree values")
        if self._previous is not None:
            ratios = tuple(abs(angle - previous) * factor for angle, previous, factor
                           in zip(angles, self._previous, self._factors, strict=True))
            if not all(math.isfinite(ratio) for ratio in ratios):
                raise ValueError("Nonfinite exported angular-velocity calculation")
            self._observe_ratios(ratios)
        self._previous = angles
        self.frames += 1

    def _observe_ratios(self, ratios):
        violations = sum(ratio > 1.0 + RELATIVE_TOLERANCE for ratio in ratios)
        self._violation_count += violations
        self._exceedance_frames += int(violations > 0)
        worst = max(range(len(ratios)), key=ratios.__getitem__)
        if self._worst_joint is None or ratios[worst] > self._max_ratio:
            self._max_ratio = ratios[worst]
            self._worst_joint = self.names[worst]
            self._worst_interval = [self.frames - 1, self.frames]

    def _metadata(self):
        return {"schema": 1, "enabled": True, "joint_count": len(self.names),
                "native_fps": self.fps, "angle_units": "deg", "limit_units": "rad/s",
                "limit_basis": "URDF joint-name matched velocity attributes",
                "relative_tolerance": RELATIVE_TOLERANCE,
                "velocity_scale": self.velocity_scale, "urdf_sha256": self._urdf_sha256,
                "ordered_joint_names_sha256": fingerprint(list(self.names)),
                "velocity_limits_sha256": self._declared_fingerprint,
                "effective_limits_sha256": self._effective_fingerprint,
                "fingerprint_schema": "json-sha256-v1"}

    def finish(self):
        if self.frames < 1:
            raise ValueError("Cannot attest an empty CSV")
        report = {**self._metadata(), "status": "passed" if not self._violation_count else "failed",
                  "frames": self.frames, "checked_pairs": (self.frames - 1) * len(self.names),
                  "violation_count": self._violation_count, "exceedance_frames": self._exceedance_frames,
                  "max_ratio": self._max_ratio, "worst_joint": self._worst_joint,
                  "worst_frame_interval": list(self._worst_interval) if self._worst_interval is not None else None}
        if self._violation_count:
            raise VelocityValidationError(report)
        return report


def _receipt_number(report, name):
    number = report.get(name)
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ValueError(f"Velocity attestation {name} must be numeric")
    try:
        finite = math.isfinite(number)
    except OverflowError as error:
        raise ValueError(f"Velocity attestation {name} must be finite") from error
    if not finite:
        raise ValueError(f"Velocity attestation {name} must be finite")
    return number


def _validate_worst_interval(report, names, frames):
    if "worst_joint" not in report or "worst_frame_interval" not in report:
        raise ValueError("Velocity attestation is missing worst-interval fields")
    joint, interval = report["worst_joint"], report["worst_frame_interval"]
    if frames == 1:
        if joint is not None or interval is not None or report["max_ratio"] != 0:
            raise ValueError("One-frame velocity attestation must have no worst interval and zero ratio")
        return
    if joint not in names or not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("Velocity attestation has an invalid worst joint or interval")
    if (any(type(index) is not int for index in interval)
            or not 0 <= interval[0] < interval[1] < frames or interval[1] != interval[0] + 1):
        raise ValueError("Velocity attestation worst interval must be an adjacent in-range frame pair")


def validate_attestation(report, joint_names, frames, plan):
    """Validate a receipt against authoritative metadata without reading CSV data.

    The caller must bind ``plan['velocity_scale']`` to the selected raw temporal
    configuration and ``plan['native_fps']`` to the native per-clip metadata.
    Source/product hashes and the outer CSV validation remain the caller's job.
    """
    if not isinstance(report, Mapping) or type(frames) is not int or frames < 1:
        raise ValueError("Expected a velocity attestation and positive native frame count")
    if not isinstance(plan, Mapping) or "velocity_scale" not in plan:
        raise ValueError("Authoritative velocity plan must include the selected velocity_scale")
    contract = VelocityAccumulator(joint_names, plan)
    for field, expected in contract._metadata().items():
        if report.get(field) != expected:
            raise ValueError(f"Velocity attestation contract mismatch: {field}")
    counts = {"schema": 1, "frames": frames, "joint_count": len(contract.names),
              "checked_pairs": (frames - 1) * len(contract.names),
              "violation_count": 0, "exceedance_frames": 0}
    for field, expected in counts.items():
        if type(report.get(field)) is not int or report[field] != expected:
            raise ValueError(f"Velocity attestation counter mismatch: {field}")
    if report.get("enabled") is not True or report.get("status") != "passed":
        raise ValueError("Velocity attestation must be enabled and passed")
    for field in ("native_fps", "velocity_scale", "relative_tolerance"):
        _receipt_number(report, field)
    ratio = _receipt_number(report, "max_ratio")
    if not 0 <= ratio <= 1.0 + RELATIVE_TOLERANCE:
        raise ValueError("Velocity attestation reports a speed-limit exceedance")
    _validate_worst_interval(report, contract.names, frames)
