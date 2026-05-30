"""
test_ids_layer.py — unit and integration tests for the behavioural IDS layer.

Covers all five threat rules, alert cooldown logic, track cleanup, and
edge cases (empty input, single person, degenerate geometry).

Run with:
    pytest tests/test_ids_layer.py -v
"""

import math

import numpy as np
import pytest

# ── Test helpers ──────────────────────────────────────────────────────────────

class MockTrack:
    """Minimal DeepSORT track stub for testing the IDS layer."""

    def __init__(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        confirmed: bool = True,
    ) -> None:
        self.track_id = track_id
        self._bbox = bbox  # (x1, y1, x2, y2)
        self._confirmed = confirmed

    def to_tlbr(self) -> list[int]:
        return list(self._bbox)

    def is_confirmed(self) -> bool:
        return self._confirmed


def _make_keypoints(
    x: float = 320.0,
    y: float = 240.0,
    conf: float = 0.9,
    shoulder_angle_deg: float = 20.0,
) -> np.ndarray:
    """
    Build a synthetic (17, 3) keypoint array for one person.

    The person is centred at (cx, cy) with all joints at high confidence.
    Shoulder/hip positions are rotated by `shoulder_angle_deg` to control
    the gait angle computation.

    Args:
        x, y:                  Approximate centre of the person.
        conf:                  Confidence for all keypoints.
        shoulder_angle_deg:    Desired hip-shoulder lateral angle (0 = upright).

    Returns:
        Float32 array of shape (17, 3).
    """
    kp = np.full((17, 3), conf, dtype=np.float32)

    # Place shoulders and hips to produce a controlled gait angle.
    angle_rad = math.radians(shoulder_angle_deg)
    # Vertical distance between shoulder_mid and hip_mid in _make_keypoints:
    #   shoulder_mid.y = y - 80,  hip_mid.y = y - 20  →  dy = 60 px
    #   angle = atan(half_sway / 60)  →  half_sway = 60 * tan(angle)
    half_sway = 60.0 * math.tan(angle_rad)  # lateral offset for given angle

    # Shoulders  (COCO indices 5=L, 6=R)
    kp[5] = [x - 20 - half_sway, y - 80, conf]
    kp[6] = [x + 20 - half_sway, y - 80, conf]
    # Hips       (COCO indices 11=L, 12=R) — placed directly below
    kp[11] = [x - 15, y - 20, conf]
    kp[12] = [x + 15, y - 20, conf]

    # Remaining joints spread around the centre for visual plausibility.
    # Arms
    kp[7]  = [x - 50, y - 120, conf]
    kp[9]  = [x - 70, y - 160, conf]
    kp[8]  = [x + 50, y - 120, conf]
    kp[10] = [x + 70, y - 160, conf]
    # Legs
    kp[13] = [x - 25, y + 40, conf]
    kp[15] = [x - 30, y + 100, conf]
    kp[14] = [x + 25, y + 40, conf]
    kp[16] = [x + 30, y + 100, conf]
    # Nose, eyes, ears
    kp[0]  = [x,     y - 160, conf]
    kp[1]  = [x - 8, y - 170, conf]
    kp[2]  = [x + 8, y - 170, conf]
    kp[3]  = [x - 20, y - 165, conf]
    kp[4]  = [x + 20, y - 165, conf]

    return kp


def _ids_layer(cooldown: int = 30) -> "IDSLayer":
    """Create an IDSLayer instance at 640×480 resolution."""
    from utils.ids_layer import IDSLayer
    return IDSLayer(frame_shape=(480, 640), alert_cooldown_frames=cooldown)


# ══════════════════════════════════════════════════════════════════════════════
# Rule 1 — Loitering
# ══════════════════════════════════════════════════════════════════════════════

class TestLoitering:
    """Verify that a stationary person triggers LOITERING after the threshold."""

    def test_stationary_triggers_alert(self):
        ids = _ids_layer()
        track = MockTrack(1, (300, 200, 340, 400))
        kp = _make_keypoints(x=320, y=300)

        # Feed the same track for LOITER_FRAMES — no alert until threshold.
        from config import LOITER_FRAMES

        # Frame 0 initialises state with last_centroid=None (no increment).
        # Frames 1..LOITER_FRAMES each increment stationary_frames by 1.
        # At frame LOITER_FRAMES the counter reaches LOITER_FRAMES and fires.
        for fn in range(LOITER_FRAMES):
            threats = ids.update([track], {1: kp}, frame_number=fn)
            assert len(threats) == 0, f"no alert expected at frame {fn}"

        # One more frame should trigger the alert.
        threats = ids.update([track], {1: kp}, frame_number=LOITER_FRAMES)
        assert len(threats) == 1
        assert threats[0].threat_type.value == "LOITERING"
        assert threats[0].track_id == 1
        assert threats[0].confidence > 0.5

    def test_moving_person_does_not_trigger(self):
        ids = _ids_layer()

        for fn in range(0, 200, 2):
            # Move the person 60 px each frame — above LOITER_DISTANCE_PX.
            x = 100 + fn * 3
            track = MockTrack(1, (x, 200, x + 40, 400))
            kp = _make_keypoints(x=x + 20, y=300)
            threats = ids.update([track], {1: kp}, frame_number=fn)

        # No loitering should ever trigger.
        loitering = [t for t in threats if t.threat_type.value == "LOITERING"]
        assert len(loitering) == 0, "moving person should not trigger loitering"

    def test_resets_on_movement(self):
        """Stationary counter should reset when the person moves."""
        ids = _ids_layer()
        track = MockTrack(1, (300, 200, 340, 400))
        kp = _make_keypoints(x=320, y=300)

        # Build up 80 stationary frames.
        for fn in range(80):
            ids.update([track], {1: kp}, frame_number=fn)

        # Move the person.
        track_moved = MockTrack(1, (400, 200, 440, 400))
        kp_moved = _make_keypoints(x=420, y=300)
        ids.update([track_moved], {1: kp_moved}, frame_number=81)

        # Continue stationary at new position — should need full threshold again.
        for fn in range(82, 82 + 80):
            threats = ids.update([track_moved], {1: kp_moved}, frame_number=fn)
            # It takes LOITER_FRAMES from the reset point, so 80 frames is not enough.
            loitering = [t for t in threats if t.threat_type.value == "LOITERING"]
            assert len(loitering) == 0, f"counter should have reset; frame {fn}"


# ══════════════════════════════════════════════════════════════════════════════
# Rule 2 — Perimeter breach
# ══════════════════════════════════════════════════════════════════════════════

class TestPerimeterBreach:
    """Verify that entering a restricted polygon zone triggers an alert."""

    def test_person_inside_zone_triggers_alert(self):
        ids = _ids_layer()
        # Define a zone covering the left half of the frame.
        ids.add_restricted_zone([(0, 0), (320, 0), (320, 480), (0, 480)])

        # Person inside the zone.
        track = MockTrack(1, (100, 200, 140, 400))
        kp = _make_keypoints(x=120, y=300)
        threats = ids.update([track], {1: kp}, frame_number=0)

        assert len(threats) == 1
        assert threats[0].threat_type.value == "PERIMETER_BREACH"

    def test_person_outside_zone_no_alert(self):
        ids = _ids_layer()
        ids.add_restricted_zone([(0, 0), (320, 0), (320, 480), (0, 480)])

        # Person outside the zone (right half).
        track = MockTrack(1, (400, 200, 440, 400))
        kp = _make_keypoints(x=420, y=300)
        threats = ids.update([track], {1: kp}, frame_number=0)

        assert len(threats) == 0

    def test_multiple_zones(self):
        ids = _ids_layer()
        ids.add_restricted_zone([(0, 0), (100, 0), (100, 100), (0, 100)])
        ids.add_restricted_zone([(540, 380), (640, 380), (640, 480), (540, 480)])

        # Person in second zone.
        track = MockTrack(1, (560, 400, 600, 470))
        kp = _make_keypoints(x=580, y=430)
        threats = ids.update([track], {1: kp}, frame_number=0)

        assert len(threats) == 1
        assert threats[0].threat_type.value == "PERIMETER_BREACH"


# ══════════════════════════════════════════════════════════════════════════════
# Rule 3 — Low-confidence keypoints  (occlusion / disguise)
# ══════════════════════════════════════════════════════════════════════════════

class TestLowConfidence:
    """Verify that low-confidence keypoints trigger an alert."""

    def test_low_confidence_triggers_alert(self):
        ids = _ids_layer()
        track = MockTrack(1, (300, 200, 340, 400))

        # Build keypoints where 5 joints are below LOW_CONF_THRESHOLD.
        kp = _make_keypoints(x=320, y=300, conf=0.9)
        for i in range(5):
            kp[i][2] = 0.1  # well below 0.25 threshold

        threats = ids.update([track], {1: kp}, frame_number=0)
        assert len(threats) == 1
        assert threats[0].threat_type.value == "LOW_CONFIDENCE"
        assert threats[0].metadata["low_conf_keypoints"] >= 4

    def test_high_confidence_no_alert(self):
        ids = _ids_layer()
        track = MockTrack(1, (300, 200, 340, 400))
        kp = _make_keypoints(x=320, y=300, conf=0.9)

        threats = ids.update([track], {1: kp}, frame_number=0)
        assert len(threats) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Rule 4 — Anomalous gait
# ══════════════════════════════════════════════════════════════════════════════

class TestAnomalousGait:
    """Verify that extreme or absent lateral sway triggers ANOMALOUS_GAIT."""

    def test_normal_gait_no_alert(self):
        ids = _ids_layer()
        track = MockTrack(1, (280, 100, 360, 360))
        kp = _make_keypoints(x=320, y=240, shoulder_angle_deg=20.0)

        threats = ids.update([track], {1: kp}, frame_number=0)
        assert len(threats) == 0, "20° is within normal range"

    def test_excessive_sway_triggers_alert(self):
        ids = _ids_layer()
        track = MockTrack(1, (280, 100, 360, 360))
        kp = _make_keypoints(x=320, y=240, shoulder_angle_deg=50.0)

        threats = ids.update([track], {1: kp}, frame_number=0)
        assert len(threats) == 1
        assert threats[0].threat_type.value == "ANOMALOUS_GAIT"
        assert "gait_angle_deg" in threats[0].metadata

    def test_near_vertical_gait_triggers_alert(self):
        """0° (perfectly vertical) should trigger — no sway."""
        ids = _ids_layer()
        track = MockTrack(1, (280, 100, 360, 360))
        kp = _make_keypoints(x=320, y=240, shoulder_angle_deg=0.0)

        threats = ids.update([track], {1: kp}, frame_number=0)
        assert len(threats) == 1
        assert threats[0].threat_type.value == "ANOMALOUS_GAIT"

    def test_low_confidence_keypoints_skip_gait(self):
        """If shoulder/hip keypoints are low confidence, gait is skipped."""
        ids = _ids_layer()
        track = MockTrack(1, (280, 100, 360, 360))
        kp = _make_keypoints(x=320, y=240, shoulder_angle_deg=50.0)
        # Drop confidence on required joints (5, 6, 11, 12).
        for idx in (5, 6, 11, 12):
            kp[idx][2] = 0.1

        threats = ids.update([track], {1: kp}, frame_number=0)
        gait_alerts = [t for t in threats if t.threat_type.value == "ANOMALOUS_GAIT"]
        assert len(gait_alerts) == 0, "low-conf joints should suppress gait check"


# ══════════════════════════════════════════════════════════════════════════════
# Rule 5 — Crowd formation
# ══════════════════════════════════════════════════════════════════════════════

class TestCrowdFormation:
    """Verify that 4+ people in close proximity trigger crowd alerts."""

    def test_crowd_triggers_alert(self):
        ids = _ids_layer()
        kp = _make_keypoints(x=320, y=300)

        # Four people within 100 px of each other.
        tracks = [
            MockTrack(1, (300, 280, 340, 400)),
            MockTrack(2, (350, 280, 390, 400)),
            MockTrack(3, (300, 330, 340, 450)),
            MockTrack(4, (350, 330, 390, 450)),
        ]
        kp_by_id = {i: kp for i in range(1, 5)}

        threats = ids.update(tracks, kp_by_id, frame_number=0)
        crowd_alerts = [t for t in threats if t.threat_type.value == "CROWD_FORMATION"]
        assert len(crowd_alerts) >= 1, "should detect crowd of 4"

    def test_three_people_no_crowd_alert(self):
        ids = _ids_layer()
        kp = _make_keypoints(x=320, y=300)

        tracks = [
            MockTrack(1, (300, 280, 340, 400)),
            MockTrack(2, (350, 280, 390, 400)),
            MockTrack(3, (300, 330, 340, 450)),
        ]
        kp_by_id = {i: kp for i in range(1, 4)}

        threats = ids.update(tracks, kp_by_id, frame_number=0)
        crowd_alerts = [t for t in threats if t.threat_type.value == "CROWD_FORMATION"]
        assert len(crowd_alerts) == 0, "3 people is below threshold"

    def test_spread_out_people_no_alert(self):
        ids = _ids_layer()
        kp = _make_keypoints(x=320, y=300)

        # Four people, but spread across the frame (>> 200 px apart).
        tracks = [
            MockTrack(1, (50,  50,  90,  150)),
            MockTrack(2, (550, 50,  590, 150)),
            MockTrack(3, (50,  350, 90,  450)),
            MockTrack(4, (550, 350, 590, 450)),
        ]
        kp_by_id = {i: _make_keypoints(x=t.to_tlbr()[0] + 20, y=t.to_tlbr()[1] + 50)
                    for i, t in enumerate(tracks, start=1)}

        threats = ids.update(tracks, kp_by_id, frame_number=0)
        crowd_alerts = [t for t in threats if t.threat_type.value == "CROWD_FORMATION"]
        assert len(crowd_alerts) == 0, "people too far apart"


# ══════════════════════════════════════════════════════════════════════════════
# Cross-cutting behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertCooldown:
    """Alert cooldown should suppress repeat alerts for the same track+type."""

    def test_cooldown_suppresses_repeat_alerts(self):
        ids = _ids_layer(cooldown=10)
        ids.add_restricted_zone([(0, 0), (640, 0), (640, 480), (0, 480)])
        track = MockTrack(1, (300, 200, 340, 400))
        kp = _make_keypoints(x=320, y=300)

        # First alert fires.
        threats = ids.update([track], {1: kp}, frame_number=0)
        assert len(threats) == 1

        # Next 9 frames should be suppressed.
        for fn in range(1, 10):
            threats = ids.update([track], {1: kp}, frame_number=fn)
            assert len(threats) == 0, f"cooldown should suppress frame {fn}"

        # Frame 10 (after cooldown) should fire again.
        threats = ids.update([track], {1: kp}, frame_number=10)
        assert len(threats) == 1

    def test_different_threat_types_fire_independently(self):
        """Cooldown is per threat type — perimeter + low-conf should both fire."""
        ids = _ids_layer(cooldown=10)
        ids.add_restricted_zone([(0, 0), (640, 0), (640, 480), (0, 480)])
        track = MockTrack(1, (300, 200, 340, 400))

        # Low-confidence + inside perimeter = two threat types at once.
        kp = _make_keypoints(x=320, y=300, conf=0.9)
        for i in range(6):
            kp[i][2] = 0.1  # trigger LOW_CONFIDENCE

        threats = ids.update([track], {1: kp}, frame_number=0)
        threat_types = {t.threat_type.value for t in threats}
        assert "PERIMETER_BREACH" in threat_types
        assert "LOW_CONFIDENCE" in threat_types


class TestTrackCleanup:
    """Stale tracks should be evicted from internal state."""

    def test_missing_track_removed(self):
        ids = _ids_layer()
        track = MockTrack(1, (300, 200, 340, 400))
        kp = _make_keypoints(x=320, y=300)

        # Register the track.
        ids.update([track], {1: kp}, frame_number=0)
        assert 1 in ids._states

        # Feed an empty track list — track 1 should be evicted.
        ids.update([], {}, frame_number=1)
        assert 1 not in ids._states, "stale track should be cleaned up"


class TestEdgeCases:
    """Empty input, missing keypoints, and degenerate inputs."""

    def test_empty_tracks_no_crash(self):
        ids = _ids_layer()
        threats = ids.update([], {}, frame_number=0)
        assert threats == []

    def test_unconfirmed_tracks_still_analysed(self):
        """
        The IDS layer does NOT filter unconfirmed tracks itself — the
        caller (quantized_pipeline.py) filters to confirmed-only before
        calling update().  Passing an unconfirmed track should still
        run all rules against it.
        """
        ids = _ids_layer()
        ids.add_restricted_zone([(0, 0), (640, 0), (640, 480), (0, 480)])
        track = MockTrack(1, (300, 200, 340, 400), confirmed=False)
        kp = _make_keypoints(x=320, y=300)

        threats = ids.update([track], {1: kp}, frame_number=0)
        # The IDS layer trusts the caller; unconfirmed tracks are still processed.
        assert len(threats) >= 1, "IDS processes unconfirmed tracks — caller filters"

    def test_track_without_keypoints_no_crash(self):
        """Track present but no keypoints in the dict — no crash."""
        ids = _ids_layer()
        track = MockTrack(1, (300, 200, 340, 400))

        threats = ids.update([track], {}, frame_number=0)
        # Should not crash — just no keypoint-based alerts.
        assert all(t.threat_type.value not in ("LOW_CONFIDENCE", "ANOMALOUS_GAIT")
                   for t in threats)

    def test_single_person_no_false_positives(self):
        """A single person walking normally should generate zero alerts."""
        ids = _ids_layer()
        ids.add_restricted_zone([(0, 0), (100, 0), (100, 100), (0, 100)])

        for fn in range(0, 100, 2):
            x = 320 + fn * 2  # moving steadily
            track = MockTrack(1, (x, 200, x + 40, 400))
            kp = _make_keypoints(x=x + 20, y=300, shoulder_angle_deg=20.0)
            threats = ids.update([track], {1: kp}, frame_number=fn)
            assert len(threats) == 0, f"false positive at frame {fn}: {threats}"


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestGeometryHelpers:
    """Verify the pure geometry utilities used by the IDS layer."""

    def test_point_in_polygon_inside(self):
        from utils.ids_layer import _point_in_polygon
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon((5, 5), square) is True

    def test_point_in_polygon_outside(self):
        from utils.ids_layer import _point_in_polygon
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon((15, 5), square) is False

    def test_point_in_polygon_on_edge(self):
        from utils.ids_layer import _point_in_polygon
        triangle = [(0, 0), (10, 0), (5, 10)]
        # On the bottom edge — behaviour depends on ray-cast implementation,
        # but it should not crash and should return a bool.
        result = _point_in_polygon((5, 0), triangle)
        assert isinstance(result, bool)

    def test_hip_shoulder_angle_normal(self):
        from utils.ids_layer import _hip_shoulder_angle
        kp = _make_keypoints(x=320, y=240, shoulder_angle_deg=20.0)
        angle = _hip_shoulder_angle(kp)
        assert angle is not None
        assert 15.0 <= angle <= 25.0, f"expected ~20°, got {angle:.1f}°"

    def test_hip_shoulder_angle_returns_none_on_low_conf(self):
        from utils.ids_layer import _hip_shoulder_angle
        kp = _make_keypoints(x=320, y=240, shoulder_angle_deg=20.0)
        kp[5][2] = 0.1  # drop left shoulder confidence
        angle = _hip_shoulder_angle(kp)
        assert angle is None

    def test_centroid(self):
        from utils.ids_layer import _centroid
        cx, cy = _centroid((100, 200, 300, 400))
        assert cx == 200.0
        assert cy == 300.0

    def test_euclidean(self):
        from utils.ids_layer import _euclidean
        assert _euclidean((0, 0), (3, 4)) == 5.0
        assert _euclidean((1, 1), (1, 1)) == 0.0
