"""
config.py — centralised configuration for the 5G MEC IDS pipeline.

Single source of truth for all tunable hyperparameters, model paths,
and geometric constants. Imported by every module in the project.

All values are documented with units and rationale. Changing a value
here propagates to every consumer — no duplicate constants.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Model paths
# ══════════════════════════════════════════════════════════════════════════════

YOLO_WEIGHTS   = "yolov8n-pose.pt"           # Ultralytics FP32 baseline
ONNX_FP32_PATH = "models/yolov8n-pose-fp32.onnx"   # intermediate export
ONNX_INT8_PATH = "models/yolov8n-pose-int8.onnx"   # static-quantised INT8
FP16_PT_PATH   = "models/yolov8n-pose-fp16.pt"     # half-precision PyTorch

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

INPUT_SIZE        = (640, 480)   # (width, height) — all frames resized to this
TARGET_LATENCY_MS = 25.0         # URLLC end-to-end budget (P95)
RESULTS_DIR       = Path("results")
CALIBRATION_FRAMES = 300  # frames used for INT8 static calibration

# ══════════════════════════════════════════════════════════════════════════════
# Alternate-frame inference  (Lucas-Kanade optical-flow propagation)
# ══════════════════════════════════════════════════════════════════════════════

ALT_FRAME_INTERVAL         = 2    # run full inference every N frames
FLOW_WIN_SIZE              = (15, 15)
FLOW_MAX_LEVEL             = 2
FLOW_MAX_CORNERS           = 8    # Shi-Tomasi corners per bbox
FLOW_QUALITY_LEVEL         = 0.3
FLOW_MIN_DISTANCE          = 7
FLOW_STALE_THRESHOLD       = 3    # max consecutive flow failures before dropping bbox

# ══════════════════════════════════════════════════════════════════════════════
# Behavioural IDS  (all in pixel space at INPUT_SIZE resolution)
# ══════════════════════════════════════════════════════════════════════════════

LOITER_FRAMES            = 90    # ~3 s at 30 fps before loitering alert
LOITER_DISTANCE_PX       = 50    # max centroid drift to still count as "stationary"
CROWD_THRESHOLD          = 4     # persons within CROWD_RADIUS_PX to trigger crowd alert
CROWD_RADIUS_PX          = 200   # pixel radius for crowd formation detection
LOW_CONF_THRESHOLD       = 0.25  # keypoint confidence below this = suspicious
LOW_CONF_KEYPOINTS       = 4     # how many low-conf keypoints needed to trigger
GAIT_ANGLE_MIN_DEG       = 10.0  # min normal hip-shoulder lateral angle
GAIT_ANGLE_MAX_DEG       = 35.0  # max normal hip-shoulder lateral angle
HISTORY_MAXLEN           = 120   # frames of position history per track
IDS_CONFIDENCE_HIGH       = 0.90
IDS_CONFIDENCE_MEDIUM     = 0.70
IDS_CONFIDENCE_LOW        = 0.50
ALERT_COOLDOWN_FRAMES     = 30    # min frames between repeat alerts for same track+type

# ══════════════════════════════════════════════════════════════════════════════
# Skeleton  (COCO 17-point edge pairs)
# ══════════════════════════════════════════════════════════════════════════════

SKELETON_EDGES: list[tuple[int, int]] = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6),                                      # shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
    (11, 12),                                    # hips
    (5, 11), (6, 12),                            # torso
]
