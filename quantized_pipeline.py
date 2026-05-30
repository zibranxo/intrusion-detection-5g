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

from ids_layer import IDSLayer, ThreatEvent, ThreatType
from utils.draw import (
    draw_skeleton,
    draw_track_label,
    draw_threat_banner,
    draw_latency_hud,
    draw_frame_mode_indicator,
    draw_restricted_zones,
)
from utils.logger import ThreatLogger, MetricsLogger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

YOLO_WEIGHTS        = "yolov8n-pose.pt"
ONNX_INT8_PATH      = "models/yolov8n-pose-int8.onnx"
FP16_PT_PATH        = "models/yolov8n-pose-fp16.pt"
INPUT_SIZE          = (640, 480)
TARGET_LATENCY_MS   = 25.0

# Alternate-frame inference — run full inference every N frames,
# propagate bboxes on intermediate frames via Lucas-Kanade optical flow
ALT_FRAME_INTERVAL  = 2

# Skeleton edges (COCO 17-point pairs)
SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 6),                               # shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (11, 12),                             # hips
    (5, 11), (6, 12),                     # torso
]

# Colours and drawing constants live in utils/draw.py

ModelVariant = Literal["fp32", "fp16", "int8"]
Backend      = Literal["cuda", "mps", "cpu"]

RESULTS_DIR  = Path("results")


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
# Alternate-frame inference with optical-flow propagation
# ──────────────────────────────────────────────────────────────────────────────

class AltFrameInference:
    """
    Runs full pose inference every ALT_FRAME_INTERVAL frames.
    On intermediate frames, propagates bounding boxes using sparse
    Lucas-Kanade optical flow — same approach as Core ML camera pipelines.

    This roughly halves inference cost while maintaining temporal coherence.
    """

    def __init__(self, model: PoseModel) -> None:
        self._model         = model
        self._frame_idx     = 0
        self._last_kps:     list[np.ndarray]            = []
        self._last_bboxes:  list[tuple[int,int,int,int]] = []
        self._prev_gray:    Optional[np.ndarray]         = None
        self._prev_corners: Optional[np.ndarray]         = None

    def process(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[int,int,int,int]], bool]:
        """
        Process one frame, using full inference or flow propagation.

        Args:
            frame: BGR frame.

        Returns:
            Tuple of (keypoints_list, bboxes_list, was_full_inference).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        run_full = (self._frame_idx % ALT_FRAME_INTERVAL == 0)
        self._frame_idx += 1

        if run_full or self._prev_gray is None:
            kps, bboxes = self._model.infer(frame)
            self._last_kps    = kps
            self._last_bboxes = bboxes
            self._prev_gray   = gray
            self._prev_corners = self._compute_corners(bboxes, frame)
            return kps, bboxes, True

        # Propagate bboxes via optical flow
        if self._prev_corners is not None and len(self._prev_corners) > 0:
            propagated = self._propagate_bboxes(gray)
            if propagated:
                self._last_bboxes = propagated

        self._prev_gray = gray
        return self._last_kps, self._last_bboxes, False

    def _compute_corners(
        self, bboxes: list[tuple[int,int,int,int]], frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute Shi-Tomasi corners within all detected bboxes for flow tracking."""
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_points = []

        for x1, y1, x2, y2 in bboxes:
            roi = gray[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                continue
            corners = cv2.goodFeaturesToTrack(
                roi, maxCorners=8, qualityLevel=0.3, minDistance=7
            )
            if corners is not None:
                corners[:, 0, 0] += x1
                corners[:, 0, 1] += y1
                all_points.extend(corners.reshape(-1, 2))

        if not all_points:
            return None
        return np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

    def _propagate_bboxes(
        self, curr_gray: np.ndarray
    ) -> Optional[list[tuple[int,int,int,int]]]:
        """
        Use Lucas-Kanade sparse flow to propagate bboxes one frame forward.

        Returns:
            Propagated bbox list, or None on tracking failure.
        """
        if self._prev_corners is None or len(self._prev_corners) < 2:
            return None

        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray,
            self._prev_corners, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        if new_corners is None or status is None:
            return None

        good_new = new_corners[status.flatten() == 1]
        good_old = self._prev_corners[status.flatten() == 1]

        if len(good_new) < 2:
            return None

        # Estimate translation vector from median flow
        flow    = good_new - good_old
        dx      = float(np.median(flow[:, 0]))
        dy      = float(np.median(flow[:, 1]))

        propagated = [
            (
                int(x1 + dx), int(y1 + dy),
                int(x2 + dx), int(y2 + dy),
            )
            for x1, y1, x2, y2 in self._last_bboxes
        ]
        self._prev_corners = good_new.reshape(-1, 1, 2)
        return propagated


# ──────────────────────────────────────────────────────────────────────────────
# Drawing utilities
# ──────────────────────────────────────────────────────────────────────────────


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
) -> None:
    """
    Run the full 5G MEC IDS pipeline.

    Stages per frame (targeting ≤25ms total):
      1. Ingest + alternate-frame inference + optical-flow propagation
      2. DeepSORT tracker update
      3. IDS behavioural analysis
      4. Annotated frame render + optional display

    Args:
        source:        Video file path or webcam index.
        model_variant: 'fp32', 'fp16', or 'int8'.
        backend:       'cuda', 'mps', or 'cpu'.
        show_latency:  Overlay latency HUD on output frames.
        save_output:   Write annotated video to results/.
        log_threats:   Write threat events to results/threats.jsonl.
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

    frame_number   = 0
    fps_tracker    = FPSTracker(window=30)
    all_threats:   list[ThreatEvent] = []

    print(f"[pipeline] Starting  source={source}  variant={model_variant}  backend={backend}")
    print(f"[pipeline] Press 'q' to quit, 's' to toggle latency HUD.\n")

    while True:
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

        # Map keypoints to track IDs by proximity of centroids
        kp_by_id: dict[int, np.ndarray] = {}
        for track in confirmed_tracks:
            tx1, ty1, tx2, ty2 = map(int, track.to_tlbr())
            tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            best_dist, best_kp = float("inf"), None
            for kp_arr, (bx1, by1, bx2, by2) in zip(kps_list, bboxes_list):
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                dist = ((tcx - bcx) ** 2 + (tcy - bcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_kp = dist, kp_arr
            if best_kp is not None and best_dist < 80:
                kp_by_id[track.track_id] = best_kp

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

        if out_writer:
            out_writer.write(frame)

        cv2.imshow(f"5G MEC IDS — {model_variant.upper()} | {backend}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            show_latency = not show_latency

        frame_number += 1

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    if threat_logger:
        threat_logger.close()
    metrics_logger.close()

    print(f"\n[pipeline] Done. {frame_number} frames  |  {len(all_threats)} threat events")


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
    args = parser.parse_args()

    source: str | int = int(args.source) if args.source.isdigit() else args.source

    run_pipeline(
        source=source,
        model_variant=args.model,
        backend=args.backend,
        show_latency=args.show_latency,
        save_output=args.save,
        log_threats=args.log_threats,
    )


if __name__ == "__main__":
    main()
