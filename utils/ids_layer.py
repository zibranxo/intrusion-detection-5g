"""
ids_layer.py
------------
Behavioural Intrusion Detection System (IDS) layer for the 5G MEC pipeline.

Sits on top of DeepSORT tracker output and classifies threat events from
skeleton keypoint trajectories — zero additional ML inference cost.

Threat types detected:
  LOITERING       — same track_id stationary in one zone for > LOITER_FRAMES
  CROWD_FORMATION — >= CROWD_THRESHOLD persons within CROWD_RADIUS pixels
  PERIMETER_BREACH— any person entering a defined restricted zone polygon
  LOW_CONFIDENCE  — pose confidence drops below threshold (occlusion / disguise)
  ANOMALOUS_GAIT  — hip-shoulder angle outside normal walking range

Output: structured ThreatEvent with timestamp, track_id, threat_type,
confidence, bbox — ready to serialise as a URLLC uplink packet.

Usage (imported by quantized_pipeline.py):
    from ids_layer import IDSLayer, ThreatEvent

    ids = IDSLayer(frame_shape=(480, 640))
    threats = ids.update(tracks, keypoints_by_id, frame_number)
    for threat in threats:
        print(threat)
"""

import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from config import (
    ALERT_COOLDOWN_FRAMES,
    CROWD_RADIUS_PX,
    CROWD_THRESHOLD,
    GAIT_ANGLE_MAX_DEG,
    GAIT_ANGLE_MIN_DEG,
    HISTORY_MAXLEN,
    IDS_CONFIDENCE_HIGH,
    IDS_CONFIDENCE_LOW,
    IDS_CONFIDENCE_MEDIUM,
    LOITER_DISTANCE_PX,
    LOITER_FRAMES,
    LOW_CONF_KEYPOINTS,
    LOW_CONF_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# Enums and data classes
# ──────────────────────────────────────────────────────────────────────────────

class ThreatType(str, Enum):
    LOITERING        = "LOITERING"
    CROWD_FORMATION  = "CROWD_FORMATION"
    PERIMETER_BREACH = "PERIMETER_BREACH"
    LOW_CONFIDENCE   = "LOW_CONFIDENCE"
    ANOMALOUS_GAIT   = "ANOMALOUS_GAIT"
    LEARNED_ANOMALY  = "LEARNED_ANOMALY"   # autoencoder reconstruction error spike


@dataclass
class ThreatEvent:
    """
    Structured threat alert — maps directly to a URLLC uplink packet payload.

    Fields:
        timestamp:    Unix epoch float (UTC).
        frame_number: Frame index in the current stream.
        track_id:     DeepSORT persistent track ID.
        threat_type:  ThreatType enum value.
        confidence:   IDS confidence score in [0, 1].
        bbox:         (x1, y1, x2, y2) bounding box in pixel coordinates.
        metadata:     Optional dict with threat-specific context.
    """
    timestamp:    float
    frame_number: int
    track_id:     int
    threat_type:  ThreatType
    confidence:   float
    bbox:         tuple[int, int, int, int]
    metadata:     dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["threat_type"] = self.threat_type.value
        return d

    def __str__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"[THREAT] {self.threat_type.value:<20} "
            f"track={self.track_id:>3}  "
            f"conf={self.confidence:.2f}  "
            f"bbox=({x1},{y1},{x2},{y2})  "
            f"frame={self.frame_number}"
        )


@dataclass
class TrackState:
    """
    Rolling state maintained per track_id across frames.

    Fields:
        centroids:      Recent (cx, cy) centroid positions.
        stationary_frames: Consecutive frames the centroid hasn't moved.
        last_centroid:  Most recent centroid.
        alert_cooldown: Frames remaining before re-alerting same threat type.
    """
    centroids:         deque = field(default_factory=lambda: deque(maxlen=HISTORY_MAXLEN))
    stationary_frames: int   = 0
    last_centroid:     Optional[tuple[float, float]] = None
    alert_cooldown:    dict  = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the (cx, cy) centre of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _point_in_polygon(point: tuple[float, float],
                       polygon: list[tuple[float, float]]) -> bool:
    """
    Ray-casting polygon containment test.

    Args:
        point:   (x, y) to test.
        polygon: List of (x, y) vertices (closed; last need not equal first).

    Returns:
        True if point is inside the polygon.
    """
    x, y     = point
    n        = len(polygon)
    inside   = False
    px, py   = polygon[-1]

    for vx, vy in polygon:
        if ((vy > y) != (py > y)) and (x < (px - vx) * (y - vy) / (py - vy + 1e-9) + vx):
            inside = not inside
        px, py = vx, vy

    return inside


def _hip_shoulder_angle(keypoints: np.ndarray) -> Optional[float]:
    """
    Compute the lateral angle between the hip midpoint and shoulder midpoint.

    A normal walking gait produces an oscillating lateral sway of ~10-35°.
    Extreme or absent sway is anomalous.

    Keypoint indices (COCO 17-point):
      5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip

    Args:
        keypoints: Array of shape (17, 3) with (x, y, confidence).

    Returns:
        Angle in degrees, or None if relevant keypoints are low-confidence.
    """
    REQUIRED_CONF = 0.4
    KP_L_SHOULDER = 5
    KP_R_SHOULDER = 6
    KP_L_HIP      = 11
    KP_R_HIP      = 12

    required = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
    if any(keypoints[i][2] < REQUIRED_CONF for i in required):
        return None

    shoulder_mid = (
        (keypoints[KP_L_SHOULDER][0] + keypoints[KP_R_SHOULDER][0]) / 2,
        (keypoints[KP_L_SHOULDER][1] + keypoints[KP_R_SHOULDER][1]) / 2,
    )
    hip_mid = (
        (keypoints[KP_L_HIP][0] + keypoints[KP_R_HIP][0]) / 2,
        (keypoints[KP_L_HIP][1] + keypoints[KP_R_HIP][1]) / 2,
    )

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    if abs(dy) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))


# ──────────────────────────────────────────────────────────────────────────────
# IDS Layer
# ──────────────────────────────────────────────────────────────────────────────

class IDSLayer:
    """
    Behavioural IDS that analyses DeepSORT tracker output each frame.

    Zero ML inference cost — all rules are geometric / statistical operations
    on keypoint coordinates and track histories.

    Usage:
        ids = IDSLayer(frame_shape=(480, 640))
        ids.add_restricted_zone([(100,100),(300,100),(300,300),(100,300)])
        threats = ids.update(tracks, keypoints_by_id, frame_number=42)
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        alert_cooldown_frames: int = 30,
    ) -> None:
        """
        Initialise the IDS layer.

        Args:
            frame_shape:            (height, width) of the video frame.
            alert_cooldown_frames:  Minimum frames between repeat alerts for
                                    the same track + threat type combination.
        """
        self._frame_h, self._frame_w = frame_shape
        self._cooldown = alert_cooldown_frames
        self._states:   dict[int, TrackState] = {}
        self._zones:    list[list[tuple[float, float]]] = []

    def add_restricted_zone(
        self, polygon: list[tuple[float, float]]
    ) -> None:
        """
        Register a polygonal perimeter zone.

        Any tracked person entering the zone triggers a PERIMETER_BREACH alert.

        Args:
            polygon: List of (x, y) pixel coordinates defining the zone boundary.
        """
        self._zones.append(polygon)

    def update(
        self,
        tracks: list,
        keypoints_by_id: dict[int, np.ndarray],
        frame_number: int,
    ) -> list[ThreatEvent]:
        """
        Analyse one frame of tracker output and return any new threat events.

        This is the hot path — must run in ≤2ms to stay within the 25ms budget.

        Args:
            tracks:          List of DeepSORT Track objects (confirmed only).
            keypoints_by_id: Mapping from track_id → (17, 3) keypoint array.
            frame_number:    Current frame index.

        Returns:
            List of ThreatEvent objects (may be empty).
        """
        threats:   list[ThreatEvent] = []
        now:       float             = time.time()
        centroids: dict[int, tuple[float, float]] = {}

        # ── Per-track analysis ────────────────────────────────────────────
        for track in tracks:
            track_id = track.track_id
            bbox     = tuple(map(int, track.to_tlbr()))  # (x1,y1,x2,y2)
            cx, cy   = _centroid(bbox)
            centroids[track_id] = (cx, cy)

            state = self._states.setdefault(track_id, TrackState())
            state.centroids.append((cx, cy))

            # Decrement cooldowns
            state.alert_cooldown = {
                k: max(0, v - 1) for k, v in state.alert_cooldown.items()
            }

            kp: Optional[np.ndarray] = keypoints_by_id.get(track_id)

            # ── Rule 1: Loitering ──────────────────────────────────────
            if state.last_centroid is not None:
                dist = _euclidean((cx, cy), state.last_centroid)
                if dist < LOITER_DISTANCE_PX:
                    state.stationary_frames += 1
                else:
                    state.stationary_frames = 0
            state.last_centroid = (cx, cy)

            if state.stationary_frames >= LOITER_FRAMES:
                if state.alert_cooldown.get(ThreatType.LOITERING, 0) == 0:
                    threats.append(ThreatEvent(
                        timestamp=now, frame_number=frame_number,
                        track_id=track_id, threat_type=ThreatType.LOITERING,
                        confidence=IDS_CONFIDENCE_HIGH, bbox=bbox,
                        metadata={"stationary_frames": state.stationary_frames},
                    ))
                    state.alert_cooldown[ThreatType.LOITERING] = self._cooldown

            # ── Rule 2: Perimeter breach ───────────────────────────────
            for zone in self._zones:
                if _point_in_polygon((cx, cy), zone):
                    if state.alert_cooldown.get(ThreatType.PERIMETER_BREACH, 0) == 0:
                        threats.append(ThreatEvent(
                            timestamp=now, frame_number=frame_number,
                            track_id=track_id,
                            threat_type=ThreatType.PERIMETER_BREACH,
                            confidence=IDS_CONFIDENCE_HIGH, bbox=bbox,
                        ))
                        state.alert_cooldown[ThreatType.PERIMETER_BREACH] = self._cooldown

            # ── Rule 3: Low-confidence keypoints (occlusion / disguise) ──
            if kp is not None:
                low_conf_count = sum(1 for kp_i in kp if kp_i[2] < LOW_CONF_THRESHOLD)
                if low_conf_count >= LOW_CONF_KEYPOINTS:
                    if state.alert_cooldown.get(ThreatType.LOW_CONFIDENCE, 0) == 0:
                        threats.append(ThreatEvent(
                            timestamp=now, frame_number=frame_number,
                            track_id=track_id,
                            threat_type=ThreatType.LOW_CONFIDENCE,
                            confidence=IDS_CONFIDENCE_MEDIUM, bbox=bbox,
                            metadata={"low_conf_keypoints": low_conf_count},
                        ))
                        state.alert_cooldown[ThreatType.LOW_CONFIDENCE] = self._cooldown

            # ── Rule 4: Anomalous gait angle ──────────────────────────
            if kp is not None:
                angle = _hip_shoulder_angle(kp)
                if angle is not None:
                    if not (GAIT_ANGLE_MIN_DEG <= angle <= GAIT_ANGLE_MAX_DEG):
                        if state.alert_cooldown.get(ThreatType.ANOMALOUS_GAIT, 0) == 0:
                            threats.append(ThreatEvent(
                                timestamp=now, frame_number=frame_number,
                                track_id=track_id,
                                threat_type=ThreatType.ANOMALOUS_GAIT,
                                confidence=IDS_CONFIDENCE_LOW, bbox=bbox,
                                metadata={"gait_angle_deg": round(angle, 1)},
                            ))
                            state.alert_cooldown[ThreatType.ANOMALOUS_GAIT] = self._cooldown

        # ── Rule 5: Crowd formation (cross-track) ─────────────────────────
        track_ids = list(centroids.keys())
        for i in range(len(track_ids)):
            nearby_count = 0
            focal_id     = track_ids[i]
            focal_pos    = centroids[focal_id]
            focal_bbox   = tuple(map(int, next(
                t.to_tlbr() for t in tracks if t.track_id == focal_id
            )))

            for j in range(len(track_ids)):
                if i == j:
                    continue
                if _euclidean(focal_pos, centroids[track_ids[j]]) < CROWD_RADIUS_PX:
                    nearby_count += 1

            if nearby_count >= CROWD_THRESHOLD - 1:
                state = self._states[focal_id]
                if state.alert_cooldown.get(ThreatType.CROWD_FORMATION, 0) == 0:
                    threats.append(ThreatEvent(
                        timestamp=now, frame_number=frame_number,
                        track_id=focal_id,
                        threat_type=ThreatType.CROWD_FORMATION,
                        confidence=IDS_CONFIDENCE_HIGH, bbox=focal_bbox,
                        metadata={"nearby_persons": nearby_count},
                    ))
                    state.alert_cooldown[ThreatType.CROWD_FORMATION] = self._cooldown

        # ── Clean up tracks that are no longer active ─────────────────────
        active_ids = {t.track_id for t in tracks}
        stale_ids  = [tid for tid in self._states if tid not in active_ids]
        for tid in stale_ids:
            del self._states[tid]

        return threats
