"""
utils/draw.py
-------------
Shared OpenCV drawing utilities for the 5G IDS pipeline.

Used by quantized_pipeline.py for frame annotation. All functions
operate in-place on BGR numpy arrays and return None.

Colour palette targets visibility on dark surveillance footage:
high-contrast green for skeleton, red for threats, grey for HUD.
"""

from typing import Optional
import cv2
import numpy as np

from ids_layer import ThreatEvent

# ──────────────────────────────────────────────────────────────────────────────
# Colour constants (BGR)
# ──────────────────────────────────────────────────────────────────────────────

CLR_SKELETON  = (0, 220, 100)
CLR_KEYPOINT  = (0, 255, 160)
CLR_BBOX_OK   = (0, 200, 80)
CLR_BBOX_ALERT= (0, 80, 255)
CLR_ID_LABEL  = (255, 255, 255)
CLR_THREAT    = (0, 80, 255)
CLR_HUD_OK    = (160, 220, 160)
CLR_HUD_WARN  = (0, 80, 255)
CLR_HUD_BG    = (20, 20, 20)

TARGET_LATENCY_MS = 25.0

# COCO 17-point skeleton edge pairs
SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 6),                                   # shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (11, 12),                                 # hips
    (5, 11), (6, 12),                         # torso
]


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton + keypoints
# ──────────────────────────────────────────────────────────────────────────────

def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.5,
    colour: tuple[int, int, int] = CLR_SKELETON,
) -> None:
    """
    Draw COCO 17-point skeleton edges and joint circles on a frame.

    Args:
        frame:          BGR frame, modified in-place.
        keypoints:      Shape (17, 3): (x, y, confidence) per joint.
        conf_threshold: Minimum confidence to draw a keypoint or edge.
        colour:         BGR colour for lines and dots.
    """
    for pair in SKELETON_EDGES:
        kp_a = keypoints[pair[0]]
        kp_b = keypoints[pair[1]]
        if kp_a[2] > conf_threshold and kp_b[2] > conf_threshold:
            pt1 = (int(kp_a[0]), int(kp_a[1]))
            pt2 = (int(kp_b[0]), int(kp_b[1]))
            cv2.line(frame, pt1, pt2, colour, 2, cv2.LINE_AA)

    for kp in keypoints:
        if kp[2] > conf_threshold:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, CLR_KEYPOINT, -1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Track ID label
# ──────────────────────────────────────────────────────────────────────────────

def draw_track_label(
    frame: np.ndarray,
    track_id: int,
    bbox: tuple[int, int, int, int],
    threat_active: bool = False,
) -> None:
    """
    Draw a persistent track-ID chip above the bounding box.

    Background is filled black for readability on any background colour.
    Red label when a threat is active for this track.

    Args:
        frame:         BGR frame, modified in-place.
        track_id:      DeepSORT track identifier.
        bbox:          (x1, y1, x2, y2) in pixel coordinates.
        threat_active: If True, renders label in threat colour.
    """
    x1, y1 = bbox[0], bbox[1]
    label_colour = CLR_THREAT if threat_active else CLR_BBOX_OK
    text  = f"ID {track_id}"

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad = 3

    # Background chip
    cv2.rectangle(
        frame,
        (x1, y1 - th - baseline - pad * 2),
        (x1 + tw + pad * 2, y1),
        CLR_HUD_BG, -1,
    )
    cv2.putText(
        frame, text,
        (x1 + pad, y1 - baseline - pad),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_colour, 1, cv2.LINE_AA,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Threat alert banners
# ──────────────────────────────────────────────────────────────────────────────

def draw_threat_banner(
    frame: np.ndarray,
    threats: list[ThreatEvent],
    max_visible: int = 3,
) -> None:
    """
    Render recent threat alerts as stacked banners at the bottom of the frame.

    Shows the most recent `max_visible` events. Each banner includes the
    threat type, track ID, and IDS confidence score.

    Args:
        frame:       BGR frame, modified in-place.
        threats:     All threat events so far (most recent shown first).
        max_visible: Maximum number of banners to render.
    """
    h = frame.shape[0]
    recent = threats[-max_visible:]

    for i, threat in enumerate(reversed(recent)):
        y_pos = h - 12 - i * 22
        label = (
            f"[ALERT] {threat.threat_type.value:<20} "
            f"ID {threat.track_id:<3}  "
            f"conf {threat.confidence:.2f}"
        )
        # Shadow for readability
        cv2.putText(frame, label, (7, y_pos + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (7, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_THREAT, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Latency HUD
# ──────────────────────────────────────────────────────────────────────────────

def draw_latency_hud(
    frame: np.ndarray,
    stage_times: dict[str, float],
    fps: float,
    model_variant: str,
    backend: str,
) -> None:
    """
    Overlay a per-stage latency HUD in the top-right corner.

    Total latency is coloured green when under 25ms (TARGET_LATENCY_MS)
    and red when over budget — instant visual feedback during demos.

    Args:
        frame:         BGR frame, modified in-place.
        stage_times:   Mapping of stage name → elapsed milliseconds.
        fps:           Current rolling-window FPS.
        model_variant: e.g. 'fp16', 'int8' — shown in the header line.
        backend:       e.g. 'cuda', 'mps' — shown alongside model variant.
    """
    h, w = frame.shape[:2]
    total_ms = sum(stage_times.values())
    hud_colour = CLR_HUD_OK if total_ms <= TARGET_LATENCY_MS else CLR_HUD_WARN

    header = f"{model_variant.upper()} / {backend}  {total_ms:.1f}ms  {fps:.0f}fps"
    lines  = [header] + [f"  {k:<14} {v:>5.1f}ms" for k, v in stage_times.items()]

    x0 = w - 215
    for i, line in enumerate(lines):
        y  = 18 + i * 17
        colour = hud_colour if i == 0 else CLR_ID_LABEL
        # Drop shadow
        cv2.putText(frame, line, (x0 + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Restricted zone overlay
# ──────────────────────────────────────────────────────────────────────────────

def draw_restricted_zones(
    frame: np.ndarray,
    zones: list[list[tuple[float, float]]],
    alpha: float = 0.25,
) -> None:
    """
    Draw semi-transparent red polygons for each restricted zone.

    Uses a blended overlay so the underlying scene remains visible.

    Args:
        frame:  BGR frame, modified in-place.
        zones:  List of polygons, each a list of (x, y) vertex tuples.
        alpha:  Blend factor for the fill (0 = invisible, 1 = opaque).
    """
    if not zones:
        return

    overlay = frame.copy()
    for polygon in zones:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], (0, 0, 180))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ──────────────────────────────────────────────────────────────────────────────
# Inference-mode indicator
# ──────────────────────────────────────────────────────────────────────────────

def draw_frame_mode_indicator(
    frame: np.ndarray,
    was_full_inference: bool,
) -> None:
    """
    Draw a small indicator showing whether this frame used full inference
    or optical-flow propagation (alternate-frame mode).

    Full inference  → solid green dot
    Propagated      → hollow grey dot

    Args:
        frame:              BGR frame, modified in-place.
        was_full_inference: True if the model ran this frame.
    """
    colour = CLR_SKELETON if was_full_inference else (120, 120, 120)
    thickness = -1 if was_full_inference else 1
    cv2.circle(frame, (10, 10), 5, colour, thickness, cv2.LINE_AA)
