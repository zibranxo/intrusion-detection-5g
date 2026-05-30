"""
quantized_pipeline.py
---------------------
Main 5G MEC IDS demo pipeline.

Integrates:
  - YOLOv8n-pose (INT8 or FP16, switchable at runtime)
  - Alternate-frame inference with optical-flow bbox propagation
  - DeepSORT persistent re-identification
  - Behavioural IDS threat detection (ids_layer.py)
  - Per-stage latency overlay on the output frame
  - URLLC-style threat event logging

Target: sub-25ms end-to-end latency at 640×480 on CUDA / Apple Metal.

Apple / Core ML reframe:
  The alternate-frame strategy and FP16 autocast are the same techniques
  Core ML uses for real-time camera inference on M-series chips. The
  quantization path mirrors Neural Engine INT8 acceleration.

Usage:
    python quantized_pipeline.py --source test.mp4 --model fp16
    python quantized_pipeline.py --source 0 --model int8 --backend cuda
    python quantized_pipeline.py --source cross.mp4 --model fp16 --show-latency
"""

import argparse
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.ids_layer import IDSLayer, ThreatEvent, ThreatType
from utils.draw import (
    draw_skeleton,
    draw_track_label,
    draw_threat_banner,
    draw_latency_hud,
    draw_frame_mode_indicator,
    draw_restricted_zones,
)
from utils.logger import ThreatLogger, MetricsLogger
from utils.trajectory_exporter import TrajectoryExporter
from utils.learned_ids import LearnedIDSLayer

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  — all constants are imported from config.py
# ──────────────────────────────────────────────────────────────────────────────

from config import (
    ALT_FRAME_INTERVAL,
    FLOW_MAX_CORNERS,
    FLOW_MAX_LEVEL,
    FLOW_MIN_DISTANCE,
    FLOW_QUALITY_LEVEL,
    FLOW_WIN_SIZE,
    FP16_PT_PATH,
    FRAME_BUDGET_MS,
    FRAME_DROP_ENABLED,
    INPUT_SIZE,
    KEYPOINT_DIST_FALLBACK,
    KEYPOINT_IOU_MIN,
    MAX_CONSECUTIVE_DROPS,
    ONNX_INT8_PATH,
    RESULTS_DIR,
    TARGET_LATENCY_MS,
    YOLO_WEIGHTS,
)

ModelVariant = Literal["fp32", "fp16", "int8"]
Backend      = Literal["cuda", "mps", "cpu"]


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

class PoseModel:
    """
    Unified wrapper around YOLO pose models (FP32, FP16, INT8 ONNX).

    Provides a single `infer(frame)` method that returns keypoints and bboxes
    regardless of the underlying backend.
    """

    def __init__(
        self,
        variant: ModelVariant,
        backend: Backend,
    ) -> None:
        """
        Load the specified model variant onto the target device.

        Args:
            variant: 'fp32', 'fp16', or 'int8'.
            backend: 'cuda', 'mps', or 'cpu'.
        """
        self.variant = variant
        self.backend = backend
        self._device = self._resolve_device(backend)

        if variant == "int8":
            self._session = self._load_ort_session()
            self._torch_model = None
        else:
            self._torch_model = self._load_torch_model()
            self._session = None

        print(f"[pipeline] PoseModel loaded: variant={variant}  backend={backend}  device={self._device}")

    @staticmethod
    def _resolve_device(backend: Backend) -> str:
        if backend == "cuda" and torch.cuda.is_available():
            return "cuda"
        if backend == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_ort_session(self):
        import onnxruntime as ort

        if not Path(ONNX_INT8_PATH).exists():
            raise FileNotFoundError(
                f"INT8 model not found at {ONNX_INT8_PATH}. "
                "Run export_quantize.py first."
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.backend == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        return ort.InferenceSession(ONNX_INT8_PATH, providers=providers)

    def _load_torch_model(self):
        from ultralytics import YOLO

        use_fp16 = self.variant == "fp16"
        weights  = FP16_PT_PATH if (use_fp16 and Path(FP16_PT_PATH).exists()) else YOLO_WEIGHTS

        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        model = YOLO(weights)
        if use_fp16 and self._device != "cpu":
            model.model = model.model.half().to(self._device)
        else:
            model.model = model.model.to(self._device)
        return model

    def infer(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int,int,int,int]]]:
        """
        Run pose inference on a single BGR frame.

        Args:
            frame: BGR frame at INPUT_SIZE.

        Returns:
            Tuple of:
              - keypoints_list: list of (17, 3) numpy arrays, one per detected person
              - bboxes_list:    list of (x1, y1, x2, y2) bounding boxes
        """
        if self._session is not None:
            return self._infer_ort(frame)
        return self._infer_torch(frame)

    def _infer_torch(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[int,int,int,int]]]:
        """Run inference via Ultralytics / PyTorch."""
        use_fp16 = self.variant == "fp16" and self._device != "cpu"

        if use_fp16:
            with torch.autocast(
                device_type="cuda" if self._device == "cuda" else "cpu"
            ):
                results = self._torch_model(frame, verbose=False)
        else:
            results = self._torch_model(frame, verbose=False)

        keypoints_list: list[np.ndarray]            = []
        bboxes_list:    list[tuple[int,int,int,int]] = []

        for r in results:
            if r.keypoints is None:
                continue
            for idx, pose in enumerate(r.keypoints.data):
                kp_arr = pose.cpu().numpy()  # (17, 3)
                keypoints_list.append(kp_arr)

                # Tight bbox from valid keypoints
                valid = [(int(kp[0]), int(kp[1])) for kp in kp_arr if kp[2] > 0.5]
                if valid:
                    x1 = max(0, min(p[0] for p in valid))
                    y1 = max(0, min(p[1] for p in valid))
                    x2 = min(INPUT_SIZE[0], max(p[0] for p in valid))
                    y2 = min(INPUT_SIZE[1], max(p[1] for p in valid))
                    bboxes_list.append((x1, y1, x2, y2))
                elif r.boxes is not None and idx < len(r.boxes):
                    b = r.boxes.xyxy[idx].cpu().numpy().astype(int)
                    bboxes_list.append((b[0], b[1], b[2], b[3]))

        return keypoints_list, bboxes_list

    def _infer_ort(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[int,int,int,int]]]:
        """Run inference via ONNX Runtime (INT8 path)."""
        resized     = cv2.resize(frame, INPUT_SIZE)
        rgb         = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw         = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        nchw        = np.expand_dims(chw, axis=0)
        input_name  = self._session.get_inputs()[0].name
        outputs     = self._session.run(None, {input_name: nchw})
        raw         = outputs[0][0].T  # (N, 56)

        CONF_THRESH     = 0.5
        KP_OFFSET       = 5
        keypoints_list  = []
        bboxes_list     = []

        for pred in raw:
            if pred[4] < CONF_THRESH:
                continue
            kp_arr = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                base         = KP_OFFSET + i * 3
                kp_arr[i, 0] = pred[base]
                kp_arr[i, 1] = pred[base + 1]
                kp_arr[i, 2] = pred[base + 2]

            keypoints_list.append(kp_arr)
            x1, y1, w, h = int(pred[0] - pred[2]/2), int(pred[1] - pred[3]/2), int(pred[2]), int(pred[3])
            bboxes_list.append((x1, y1, x1 + w, y1 + h))

        return keypoints_list, bboxes_list


# ──────────────────────────────────────────────────────────────────────────────
# Alternate-frame inference with per-person optical-flow propagation
# ──────────────────────────────────────────────────────────────────────────────

class AltFrameInference:
    """
    Runs full pose inference every ALT_FRAME_INTERVAL frames.

    On intermediate frames, propagates each person's bounding box
    independently using sparse Lucas-Kanade optical flow.  Each bbox
    has its own set of Shi-Tomasi corner points and its own flow vector
    — this fixes the original single-global-vector bug where all bboxes
    were shifted together regardless of individual motion.

    People whose flow tracking fails (occlusion, exit frame, low texture)
    keep their last known bbox position.  This is safe because DeepSORT's
    max_age parameter handles stale tracks.
    """

    def __init__(self, model: PoseModel) -> None:
        self._model         = model
        self._frame_idx     = 0
        self._last_kps:     list[np.ndarray]            = []
        self._last_bboxes:  list[tuple[int,int,int,int]] = []
        self._prev_gray:    Optional[np.ndarray]         = None

        # Per-person state — one entry per bbox, indices aligned.
        # Each entry is an (N, 1, 2) float32 corner array, or None.
        self._corners_per_bbox: list[Optional[np.ndarray]] = []

    # ── Public interface ─────────────────────────────────────────────────

    def process(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[int,int,int,int]], bool]:
        """
        Process one frame — full inference or per-person flow propagation.

        Args:
            frame: BGR frame at INPUT_SIZE resolution.

        Returns:
            (keypoints_list, bboxes_list, was_full_inference).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        run_full = (self._frame_idx % ALT_FRAME_INTERVAL == 0)
        self._frame_idx += 1

        if run_full or self._prev_gray is None:
            kps, bboxes = self._model.infer(frame)
            self._last_kps    = kps
            self._last_bboxes = bboxes
            self._prev_gray   = gray
            self._corners_per_bbox = self._extract_corners_per_bbox(bboxes, frame)
            return kps, bboxes, True

        # ── Flow frame: propagate each person independently ──────────
        propagated = self._propagate_bboxes_per_person(gray)
        self._last_bboxes = propagated
        self._prev_gray   = gray
        return self._last_kps, self._last_bboxes, False

    # ── Corner extraction (per bbox) ─────────────────────────────────────

    def _extract_corners_per_bbox(
        self,
        bboxes: list[tuple[int,int,int,int]],
        frame: np.ndarray,
    ) -> list[Optional[np.ndarray]]:
        """
        Extract Shi-Tomasi corner points independently for each bbox.

        Returns a list with one entry per bbox: either a (N, 1, 2)
        float32 corner array with at least 2 points, or None if the
        bbox region has no trackable texture.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list: list[Optional[np.ndarray]] = []

        for x1, y1, x2, y2 in bboxes:
            roi = gray[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                corners_list.append(None)
                continue

            pts = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY_LEVEL,
                minDistance=FLOW_MIN_DISTANCE,
            )
            if pts is not None and len(pts) >= 2:
                # Shift back to full-frame coordinates
                pts[:, 0, 0] += x1
                pts[:, 0, 1] += y1
                corners_list.append(pts.astype(np.float32))
            else:
                corners_list.append(None)

        return corners_list

    # ── Per-person propagation ───────────────────────────────────────────

    def _propagate_bboxes_per_person(
        self, curr_gray: np.ndarray
    ) -> list[tuple[int,int,int,int]]:
        """
        Propagate each bounding box independently using its own optical flow.

        For every person we compute a separate Lucas-Kanade flow vector
        from their tracked corner points.  People whose flow tracking
        fails (occlusion, exit frame, low texture) keep their last known
        bbox position — the tracker's max_age handles eventual expiry.

        Returns:
            Propagated bbox list, always same length as _last_bboxes.
        """
        n = len(self._last_bboxes)
        if n == 0:
            return []

        # Guard: corner list length must match bbox list length.
        # This holds because _extract_corners_per_bbox returns one entry
        # per bbox, and we never change length on flow frames.
        if len(self._corners_per_bbox) != n:
            return list(self._last_bboxes)

        propagated: list[tuple[int,int,int,int]] = []
        new_corners_list: list[Optional[np.ndarray]] = []

        lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03
        )

        for i in range(n):
            bbox = self._last_bboxes[i]
            prev_corners = self._corners_per_bbox[i]

            # ── No trackable corners for this person ────────────────
            if prev_corners is None or len(prev_corners) < 2:
                propagated.append(bbox)
                new_corners_list.append(None)
                continue

            # ── Lucas-Kanade for this person's points ───────────────
            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, curr_gray,
                prev_corners, None,
                winSize=FLOW_WIN_SIZE,
                maxLevel=FLOW_MAX_LEVEL,
                criteria=lk_criteria,
            )

            if new_corners is None or status is None:
                propagated.append(bbox)
                new_corners_list.append(None)
                continue

            good_new = new_corners[status.flatten() == 1]
            good_old = prev_corners[status.flatten() == 1]

            if len(good_new) < 2:
                # Lost tracking for this person
                propagated.append(bbox)
                new_corners_list.append(None)
                continue

            # ── Per-person median flow → shift this bbox ────────────
            flow = good_new - good_old
            dx = float(np.median(flow[:, 0]))
            dy = float(np.median(flow[:, 1]))

            x1, y1, x2, y2 = bbox
            propagated.append((
                int(x1 + dx), int(y1 + dy),
                int(x2 + dx), int(y2 + dy),
            ))
            new_corners_list.append(
                good_new.reshape(-1, 1, 2).astype(np.float32)
            )

        self._corners_per_bbox = new_corners_list
        return propagated


# ──────────────────────────────────────────────────────────────────────────────
# Keypoint-to-track association  (IoU-based, replaces centroid matching)
# ──────────────────────────────────────────────────────────────────────────────

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over Union for two (x1,y1,x2,y2) bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _associate_keypoints_iou(
    tracks: list,
    kps_list: list[np.ndarray],
    bboxes_list: list[tuple[int, int, int, int]],
    iou_min: float = 0.3,
    dist_fallback: float = 80.0,
) -> dict[int, np.ndarray]:
    """
    Greedy IoU-based association of detection keypoints to tracker tracks.

    IoU is a better metric than centroid distance because two people with
    overlapping boxes but different IoU scores won't have keypoints swapped
    (the original centroid-distance bug).  Falls back to centroid distance
    when no pair meets the IoU threshold.

    Args:
        tracks:        Confirmed DeepSORT Track objects.
        kps_list:      Keypoint arrays from the pose model (one per detection).
        bboxes_list:   Bounding boxes from the pose model (aligned with kps_list).
        iou_min:       Minimum IoU to accept an association.
        dist_fallback: Max centroid pixel distance for fallback matching.

    Returns:
        Mapping from track_id → (17,3) keypoint array.
    """
    kp_by_id: dict[int, np.ndarray] = {}
    used: set[int] = set()  # indices of kps_list / bboxes_list already claimed

    for track in tracks:
        track_bbox = tuple(map(int, track.to_tlbr()))

        # ── Pass 1: IoU-based association ────────────────────────────
        best_iou = iou_min
        best_idx = -1
        for i, det_bbox in enumerate(bboxes_list):
            if i in used:
                continue
            score = _iou(track_bbox, det_bbox)
            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_idx >= 0:
            kp_by_id[track.track_id] = kps_list[best_idx]
            used.add(best_idx)
            continue

        # ── Pass 2: centroid-distance fallback ───────────────────────
        tcx = (track_bbox[0] + track_bbox[2]) / 2
        tcy = (track_bbox[1] + track_bbox[3]) / 2
        best_dist = dist_fallback
        best_idx = -1

        for i, det_bbox in enumerate(bboxes_list):
            if i in used:
                continue
            bcx = (det_bbox[0] + det_bbox[2]) / 2
            bcy = (det_bbox[1] + det_bbox[3]) / 2
            dist = ((tcx - bcx) ** 2 + (tcy - bcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            kp_by_id[track.track_id] = kps_list[best_idx]
            used.add(best_idx)

    return kp_by_id


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    source: str | int,
    model_variant: ModelVariant,
    backend: Backend,
    show_latency: bool,
    save_output: bool,
    log_threats: bool,
    drop_frames: bool = True,
    export_trajectories: bool = False,
    simulate_urllc: bool = False,
    learned_ids_enabled: bool = False,
) -> None:
    """
    Run the full 5G MEC IDS pipeline.

    Stages per frame (targeting ≤25ms total):
      1. Ingest + alternate-frame inference + optical-flow propagation
      2. DeepSORT tracker update
      3. IDS behavioural analysis
      4. Annotated frame render + optional display

    Args:
        source:               Video file path or webcam index.
        model_variant:        'fp32', 'fp16', or 'int8'.
        backend:              'cuda', 'mps', or 'cpu'.
        show_latency:         Overlay latency HUD on output frames.
        save_output:          Write annotated video to results/.
        log_threats:          Write threat events to results/threats.jsonl.
        drop_frames:          Skip display when over latency budget.
        export_trajectories:  Write per-track trajectories for offline analysis.
        simulate_urllc:       Simulate 5G URLLC uplink with latency/packet loss.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Init components ───────────────────────────────────────────────────
    pose_model = PoseModel(variant=model_variant, backend=backend)
    alt_engine = AltFrameInference(pose_model)
    tracker    = DeepSort(max_age=30)
    ids_layer  = IDSLayer(frame_shape=(INPUT_SIZE[1], INPUT_SIZE[0]))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps_in     = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = None

    if save_output:
        out_path   = str(RESULTS_DIR / f"output_{model_variant}.mp4")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, INPUT_SIZE)
        print(f"[pipeline] Saving output to {out_path}")

    threat_logger  = ThreatLogger(
        str(RESULTS_DIR / "threats.jsonl"), overwrite=True
    ) if log_threats else None
    metrics_logger = MetricsLogger(
        str(RESULTS_DIR / "metrics.json"),
        model_variant=model_variant,
        backend=backend,
    )

    # ── Trajectory export ─────────────────────────────────────────────────
    traj_exporter = None
    if export_trajectories:
        from config import TRAJECTORY_EXPORT_DIR
        TRAJECTORY_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        traj_path = TRAJECTORY_EXPORT_DIR / f"session_{model_variant}.jsonl"
        traj_exporter = TrajectoryExporter(str(traj_path))
        print(f"[pipeline] Trajectory export → {traj_path}")

    # ── URLLC simulated transport ─────────────────────────────────────────
    urllc_receiver = None
    urllc_transport = None
    if simulate_urllc:
        from urllc import URLLCReceiver, URLLCTransport
        urllc_path = str(RESULTS_DIR / "urllc_received.jsonl")
        urllc_receiver = URLLCReceiver(urllc_path)
        urllc_transport = URLLCTransport(urllc_receiver)
        print(f"[pipeline] URLLC simulation active → {urllc_path}")

    # ── Learned IDS (autoencoder anomaly detection) ──────────────────────
    learned_ids = None
    if learned_ids_enabled:
        from config import AE_ONNX_PATH
        learned_ids = LearnedIDSLayer(onnx_path=AE_ONNX_PATH)
        print(f"[pipeline] Learned IDS active → {AE_ONNX_PATH}")

    frame_number   = 0
    fps_tracker    = FPSTracker(window=30)
    all_threats:   list[ThreatEvent] = []

    # Frame-dropping state
    frame_budget_s   = FRAME_BUDGET_MS / 1000.0
    consecutive_drops = 0
    frames_dropped    = 0

    print(f"[pipeline] Starting  source={source}  variant={model_variant}  backend={backend}")
    print(f"[pipeline] Press 'q' to quit, 's' to toggle latency HUD.\n")

    while True:
        frame_start = time.perf_counter()

        ret, raw_frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(raw_frame, INPUT_SIZE)

        # ── Stage 1: Alternate-frame inference ────────────────────────────
        t0     = time.perf_counter()
        kps_list, bboxes_list, was_full = alt_engine.process(frame)
        t1     = time.perf_counter()

        # ── Stage 2: DeepSORT tracker update ─────────────────────────────
        detections = [
            ([x1, y1, x2 - x1, y2 - y1], 0.9, 0)   # (ltwh, conf, class)
            for x1, y1, x2, y2 in bboxes_list
        ]
        tracks = tracker.update_tracks(detections, frame=frame)
        t2     = time.perf_counter()

        # ── Stage 3: IDS analysis ─────────────────────────────────────────
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]

        # ── Map keypoints to track IDs by IoU (avoids centroid-swap bug) ──
        kp_by_id = _associate_keypoints_iou(
            confirmed_tracks, kps_list, bboxes_list,
            iou_min=KEYPOINT_IOU_MIN, dist_fallback=KEYPOINT_DIST_FALLBACK,
        )

        threats = ids_layer.update(confirmed_tracks, kp_by_id, frame_number)
        t3      = time.perf_counter()

        # ── Persist threats ───────────────────────────────────────────────
        if threats:
            all_threats.extend(threats)
            for threat in threats:
                print(threat)
            if threat_logger:
                for threat in threats:
                    threat_logger.log(threat)
            # URLLC uplink simulation
            if urllc_transport:
                for threat in threats:
                    urllc_transport.send(threat)

        # ── Learned IDS (autoencoder) ─────────────────────────────────
        if learned_ids:
            learned_threats = learned_ids.score_frame(
                confirmed_tracks, kp_by_id, frame_number,
                frame_shape=(INPUT_SIZE[1], INPUT_SIZE[0]),
            )
            if learned_threats:
                threats.extend(learned_threats)
                for t in learned_threats:
                    print(t)
                if threat_logger:
                    for t in learned_threats:
                        threat_logger.log(t)
                if urllc_transport:
                    for t in learned_threats:
                        urllc_transport.send(t)

        # ── Trajectory export ─────────────────────────────────────────────
        if traj_exporter:
            traj_exporter.record(frame_number, confirmed_tracks,
                                 kp_by_id, threats)

        # ── Stage 4: Annotate frame ───────────────────────────────────────
        t4 = time.perf_counter()
        alerted_ids = {t.track_id for t in threats}

        for kp_arr in kps_list:
            draw_skeleton(frame, kp_arr)

        for track in confirmed_tracks:
            bbox = tuple(map(int, track.to_tlbr()))
            draw_track_label(frame, track.track_id, bbox,
                             threat_active=track.track_id in alerted_ids)

        draw_threat_banner(frame, all_threats)
        draw_frame_mode_indicator(frame, was_full)

        stage_times = {
            "infer" + (" *" if was_full else " ~"): (t1 - t0) * 1000,
            "tracker":                               (t2 - t1) * 1000,
            "IDS":                                   (t3 - t2) * 1000,
            "render":                                (time.perf_counter() - t4) * 1000,
        }

        if show_latency:
            draw_latency_hud(frame, stage_times, fps_tracker.fps, model_variant, backend)

        metrics_logger.record(stage_times, fps_tracker.fps)
        fps_tracker.tick()

        # ── Frame-dropping: skip display when over latency budget ─────
        elapsed = time.perf_counter() - frame_start
        over_budget = (FRAME_DROP_ENABLED and drop_frames
                       and elapsed > frame_budget_s)

        should_display = True
        if over_budget:
            consecutive_drops += 1
            if consecutive_drops > MAX_CONSECUTIVE_DROPS:
                # Force a display frame so the window doesn't hang.
                consecutive_drops = 0
            else:
                should_display = False
                frames_dropped += 1

        # Always write to output video (even when skipping display).
        if out_writer:
            out_writer.write(frame)

        if should_display:
            cv2.imshow(f"5G MEC IDS — {model_variant.upper()} | {backend}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                show_latency = not show_latency
            consecutive_drops = 0

        frame_number += 1

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    if traj_exporter:
        traj_exporter.close()
    if urllc_receiver:
        urllc_receiver.close()
    if learned_ids:
        pass  # no explicit close needed; ONNX session cleaned up on GC
    if threat_logger:
        threat_logger.close()
    metrics_logger.close()

    dropped_str = f"  |  {frames_dropped} display frames dropped" if frames_dropped else ""
    print(f"\n[pipeline] Done. {frame_number} frames{dropped_str}  |  {len(all_threats)} threat events")


# ──────────────────────────────────────────────────────────────────────────────
# FPS tracker utility
# ──────────────────────────────────────────────────────────────────────────────

class FPSTracker:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> None:
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="5G MEC IDS — quantized on-device inference pipeline"
    )
    parser.add_argument("--source",       default="0",    help="Video path or '0' for webcam")
    parser.add_argument("--model",        default="fp16", choices=["fp32", "fp16", "int8"],
                        help="Model variant (default: fp16)")
    parser.add_argument("--backend",      default="cuda", choices=["cuda", "mps", "cpu"],
                        help="Inference backend")
    parser.add_argument("--show-latency", action="store_true", help="Overlay latency HUD")
    parser.add_argument("--save",         action="store_true", help="Save annotated video to results/")
    parser.add_argument("--log-threats",  action="store_true", help="Log threats to results/threats.jsonl")
    parser.add_argument("--drop-frames",  action="store_true",
                        help="Skip display when over latency budget")
    parser.add_argument("--export-trajectories", action="store_true",
                        help="Export per-track trajectories for offline analysis")
    parser.add_argument("--simulate-urllc", action="store_true",
                        help="Simulate 5G URLLC uplink with latency and packet loss")
    parser.add_argument("--learned-ids", action="store_true",
                        help="Enable learned anomaly detection via trajectory autoencoder")
    args = parser.parse_args()

    source: str | int = int(args.source) if args.source.isdigit() else args.source

    run_pipeline(
        source=source,
        model_variant=args.model,
        backend=args.backend,
        show_latency=args.show_latency,
        save_output=args.save,
        log_threats=args.log_threats,
        drop_frames=args.drop_frames,
        export_trajectories=args.export_trajectories,
        simulate_urllc=args.simulate_urllc,
        learned_ids_enabled=args.learned_ids,
    )


if __name__ == "__main__":
    main()
