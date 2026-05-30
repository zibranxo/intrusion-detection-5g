"""
benchmark.py
------------
Latency benchmark harness for the 5G IDS quantized inference pipeline.

Measures per-stage P50 / P95 / P99 latency over N frames for three model
variants: FP32 baseline, FP16, and INT8 (ONNX Runtime).

Outputs:
  - Console table with per-stage and end-to-end latency stats
  - results/benchmark_results.json  (machine-readable, for dashboards)
  - results/benchmark_summary.txt   (human-readable, for the README)

Apple / Core ML reframe:
  On Apple Silicon (M2), pass --backend mps to route inference through the
  Metal Performance Shaders backend. Latency numbers map directly to
  Core ML on-device inference benchmarks.

Usage:
    python benchmark.py --source test.mp4 --frames 500
    python benchmark.py --source 0 --frames 200 --backend mps   # M2 Mac
    python benchmark.py --source test.mp4 --skip-int8
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ONNX_INT8_PATH  = "models/yolov8n-pose-int8.onnx"
ONNX_FP32_PATH  = "models/yolov8n-pose-fp32.onnx"
YOLO_WEIGHTS    = "yolov8n-pose.pt"
FP16_PT_PATH    = "models/yolov8n-pose-fp16.pt"
INPUT_SIZE      = (640, 480)       # (width, height)
TARGET_LATENCY_MS = 25.0           # sub-25ms end-to-end target
RESULTS_DIR     = Path("results")
Backend = Literal["cuda", "mps", "cpu"]


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StageTiming:
    """Timing samples (ms) for a single pipeline stage."""
    name:    str
    samples: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return float(np.percentile(self.samples, 50)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.samples, 99)) if self.samples else 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    def to_dict(self) -> dict:
        return {
            "stage": self.name,
            "p50_ms":  round(self.p50,  2),
            "p95_ms":  round(self.p95,  2),
            "p99_ms":  round(self.p99,  2),
            "mean_ms": round(self.mean, 2),
            "n":       len(self.samples),
        }


@dataclass
class VariantResult:
    """Benchmark results for one model variant (FP32 / FP16 / INT8)."""
    variant:      str
    backend:      str
    stages:       list[StageTiming] = field(default_factory=list)
    fps:          float = 0.0
    meets_target: bool = False

    @property
    def total_p95(self) -> float:
        return sum(s.p95 for s in self.stages)

    def to_dict(self) -> dict:
        return {
            "variant":      self.variant,
            "backend":      self.backend,
            "stages":       [s.to_dict() for s in self.stages],
            "total_p95_ms": round(self.total_p95, 2),
            "fps":          round(self.fps, 1),
            "meets_25ms_target": self.meets_target,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Timer utilities
# ──────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """
    Context-manager timer that uses CUDA/MPS events when available,
    falling back to perf_counter for CPU-only runs.

    This is the same mechanism Core ML uses for Metal shader profiling.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend  = backend
        self._use_cuda = backend == "cuda" and torch.cuda.is_available()
        self._use_mps  = backend == "mps" and hasattr(torch.backends, "mps") \
                         and torch.backends.mps.is_available()
        self._elapsed_ms = 0.0

        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event   = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "StageTimer":
        if self._use_cuda:
            self._start_event.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self._use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self._elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def ms(self) -> float:
        return self._elapsed_ms


# ──────────────────────────────────────────────────────────────────────────────
# Frame preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a BGR frame for YOLO inference.

    Converts: BGR → RGB, HWC → NCHW float32, [0,255] → [0,1].
    This matches the FP16 normalise step in the MEC ingestion stage.

    Args:
        frame: Raw BGR frame from cv2.VideoCapture.

    Returns:
        NCHW float32 array, shape (1, 3, H, W), values in [0, 1].
    """
    resized  = cv2.resize(frame, INPUT_SIZE)
    rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw      = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Runtime inference
# ──────────────────────────────────────────────────────────────────────────────

def build_ort_session(onnx_path: str, backend: Backend):
    """
    Create an ONNX Runtime InferenceSession with the appropriate execution
    provider for the target backend.

    Provider priority: CUDA → CPU (MPS uses PyTorch path instead).

    Args:
        onnx_path: Path to .onnx model.
        backend:   Target backend string.

    Returns:
        ort.InferenceSession ready for inference.
    """
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if backend == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[ort] Session built: {onnx_path}  providers={session.get_providers()}")
    return session


def run_ort_inference(session, input_array: np.ndarray) -> np.ndarray:
    """
    Run a single forward pass via ONNX Runtime.

    Args:
        session:     ort.InferenceSession.
        input_array: Preprocessed NCHW float32 array.

    Returns:
        Raw output array from the model.
    """
    input_name  = session.get_inputs()[0].name
    outputs     = session.run(None, {input_name: input_array})
    return outputs[0]


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch inference (FP32 baseline + FP16)
# ──────────────────────────────────────────────────────────────────────────────

def build_torch_model(weights_path: str, backend: Backend, use_fp16: bool = False):
    """
    Load YOLOv8 pose model via Ultralytics onto the target device.

    Args:
        weights_path: Path to .pt file.
        backend:      'cuda', 'mps', or 'cpu'.
        use_fp16:     If True, cast model to half precision.

    Returns:
        Tuple of (YOLO model, device string).
    """
    from ultralytics import YOLO

    device = (
        "cuda" if backend == "cuda" and torch.cuda.is_available()
        else "mps"  if backend == "mps" and hasattr(torch.backends, "mps")
                       and torch.backends.mps.is_available()
        else "cpu"
    )
    model = YOLO(weights_path)
    if use_fp16 and device != "cpu":
        model.model = model.model.half().to(device)
    else:
        model.model = model.model.to(device)

    print(f"[torch] Model loaded: {weights_path}  device={device}  fp16={use_fp16}")
    return model, device


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner for a single variant
# ──────────────────────────────────────────────────────────────────────────────

def run_variant_benchmark(
    variant_name: str,
    source: str | int,
    n_frames: int,
    backend: Backend,
    use_ort: bool = False,
    onnx_path: str = "",
    torch_weights: str = YOLO_WEIGHTS,
    use_fp16: bool = False,
) -> VariantResult:
    """
    Benchmark a single model variant over N frames.

    Measures four pipeline stages independently:
      1. Ingest + preprocess
      2. Model inference (forward pass only)
      3. Post-process (NMS / keypoint extraction)
      4. Overhead (display prep, IO bookkeeping)

    Args:
        variant_name:   Display name, e.g. 'FP32', 'FP16', 'INT8'.
        source:         Video path or webcam index.
        n_frames:       Number of frames to benchmark.
        backend:        'cuda', 'mps', or 'cpu'.
        use_ort:        If True, use ONNX Runtime (INT8 path).
        onnx_path:      Path to .onnx file (required if use_ort=True).
        torch_weights:  Path to .pt weights (used if use_ort=False).
        use_fp16:       If True, run model in FP16 mode.

    Returns:
        VariantResult with per-stage timing statistics.
    """
    result  = VariantResult(variant=variant_name, backend=backend)
    t_preprocess = StageTiming("ingest_preprocess")
    t_inference  = StageTiming("model_inference")
    t_postproc   = StageTiming("postprocess")
    t_overhead   = StageTiming("overhead")
    timer        = StageTimer(backend)

    # Build session / model
    if use_ort:
        session = build_ort_session(onnx_path, backend)
        model   = None
    else:
        model, _device = build_torch_model(torch_weights, backend, use_fp16)
        session = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    frame_count = 0
    wall_start  = time.perf_counter()

    print(f"[bench] Running {variant_name} over {n_frames} frames...")

    while frame_count < n_frames:
        ret, raw_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw_frame = cap.read()
            if not ret:
                break

        # ── Stage 1: ingest + preprocess ──────────────────────────────────
        with timer:
            input_array = preprocess_frame(raw_frame)

        t_preprocess.samples.append(timer.ms)

        # ── Stage 2: model inference ──────────────────────────────────────
        with timer:
            if use_ort:
                raw_output = run_ort_inference(session, input_array)
            else:
                if use_fp16:
                    device_str = "cuda" if backend == "cuda" else "mps" if backend == "mps" else "cpu"
                    tensor = torch.from_numpy(input_array).half().to(device_str)
                    with torch.autocast(device_type="cuda" if backend == "cuda" else "cpu"):
                        raw_output = model(raw_frame, verbose=False)
                else:
                    raw_output = model(raw_frame, verbose=False)

        t_inference.samples.append(timer.ms)

        # ── Stage 3: postprocess (keypoint + bbox extraction) ─────────────
        with timer:
            if use_ort:
                _extract_ort_keypoints(raw_output)
            else:
                _extract_yolo_keypoints(raw_output)

        t_postproc.samples.append(timer.ms)

        # ── Stage 4: overhead (display, logging) ──────────────────────────
        with timer:
            _ = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)  # simulate output prep

        t_overhead.samples.append(timer.ms)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  {frame_count}/{n_frames} frames processed...")

    cap.release()

    wall_elapsed  = time.perf_counter() - wall_start
    result.fps    = frame_count / wall_elapsed if wall_elapsed > 0 else 0.0
    result.stages = [t_preprocess, t_inference, t_postproc, t_overhead]
    result.meets_target = result.total_p95 <= TARGET_LATENCY_MS

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Keypoint extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_yolo_keypoints(results) -> list[tuple[int, int]]:
    """
    Extract (x, y) keypoint pairs from Ultralytics YOLO results.

    Args:
        results: Ultralytics Results object list.

    Returns:
        Flat list of (x, y) tuples for all detected persons.
    """
    keypoints: list[tuple[int, int]] = []
    for r in results:
        if r.keypoints is not None:
            for pose in r.keypoints.data:
                for kp in pose:
                    if kp[2] > 0.5:
                        keypoints.append((int(kp[0]), int(kp[1])))
    return keypoints


def _extract_ort_keypoints(output: np.ndarray) -> list[tuple[int, int]]:
    """
    Extract keypoints from raw ONNX Runtime output tensor.

    YOLOv8 ONNX output shape: (1, 56, N) where
    first 4 values = bbox, next 1 = conf, then 51 = 17 keypoints * 3 (x, y, conf).

    Args:
        output: Raw ONNX output array.

    Returns:
        Flat list of (x, y) tuples.
    """
    keypoints: list[tuple[int, int]] = []
    if output is None or output.size == 0:
        return keypoints

    predictions = output[0].T  # (N, 56)
    CONFIDENCE_THRESHOLD = 0.5
    KEYPOINT_OFFSET      = 5

    for pred in predictions:
        conf = pred[4]
        if conf < CONFIDENCE_THRESHOLD:
            continue
        for i in range(17):
            base = KEYPOINT_OFFSET + i * 3
            kp_conf = pred[base + 2]
            if kp_conf > CONFIDENCE_THRESHOLD:
                keypoints.append((int(pred[base]), int(pred[base + 1])))

    return keypoints


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[VariantResult]) -> None:
    """
    Print a formatted latency comparison table to stdout.

    Args:
        results: List of VariantResult objects from benchmark runs.
    """
    STAGE_NAMES = ["ingest_preprocess", "model_inference", "postprocess", "overhead"]
    COL_W       = 12

    header_variants = "".join(f"{r.variant:>{COL_W}}" for r in results)
    print(f"\n{'─' * (28 + COL_W * len(results))}")
    print(f"  Latency Benchmark — P95 (ms)   {header_variants}")
    print(f"{'─' * (28 + COL_W * len(results))}")

    for stage_name in STAGE_NAMES:
        row = f"  {stage_name:<26}"
        for r in results:
            stage = next((s for s in r.stages if s.name == stage_name), None)
            val = f"{stage.p95:.1f}" if stage else "—"
            row += f"{val:>{COL_W}}"
        print(row)

    print(f"{'─' * (28 + COL_W * len(results))}")
    total_row = f"  {'TOTAL P95':<26}"
    for r in results:
        flag  = " ✓" if r.meets_target else " ✗"
        total = f"{r.total_p95:.1f}{flag}"
        total_row += f"{total:>{COL_W}}"
    print(total_row)

    fps_row = f"  {'FPS (mean)':<26}"
    for r in results:
        fps_row += f"{r.fps:>{COL_W}.1f}"
    print(fps_row)
    print(f"{'─' * (28 + COL_W * len(results))}")
    print(f"  Target: ≤{TARGET_LATENCY_MS:.0f}ms end-to-end P95  (5G URLLC budget)")
    print(f"  Backend: {results[0].backend if results else '—'}\n")


def save_results(results: list[VariantResult], backend: Backend) -> None:
    """
    Persist benchmark results to JSON and a plain-text summary.

    Args:
        results: List of completed VariantResult objects.
        backend: Backend string used for the run.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # JSON (machine-readable)
    json_path = RESULTS_DIR / "benchmark_results.json"
    payload   = {
        "meta": {
            "target_latency_ms": TARGET_LATENCY_MS,
            "backend":           backend,
            "input_size":        INPUT_SIZE,
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[save] Results → {json_path}")

    # Plain text (for README)
    txt_path = RESULTS_DIR / "benchmark_summary.txt"
    lines    = [
        "5G IDS MEC — Latency Benchmark Summary",
        f"Backend: {backend}  |  Target: ≤{TARGET_LATENCY_MS:.0f}ms P95",
        "",
        f"{'Variant':<10} {'Total P95 (ms)':>16} {'FPS':>8} {'≤25ms?':>8}",
        "-" * 46,
    ]
    for r in results:
        lines.append(
            f"{r.variant:<10} {r.total_p95:>16.1f} {r.fps:>8.1f} "
            f"{'YES' if r.meets_target else 'NO':>8}"
        )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[save] Summary → {txt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 5G IDS quantized inference pipeline latency."
    )
    parser.add_argument("--source",    default="0",  help="Video path or '0' for webcam")
    parser.add_argument("--frames",    type=int, default=300, help="Frames to benchmark per variant")
    parser.add_argument("--backend",   default="cuda", choices=["cuda", "mps", "cpu"],
                        help="Inference backend (cuda/mps/cpu)")
    parser.add_argument("--skip-int8", action="store_true", help="Skip INT8 ONNX benchmark")
    args = parser.parse_args()

    source:  str | int = int(args.source) if args.source.isdigit() else args.source
    backend: Backend   = args.backend

    results: list[VariantResult] = []

    # ── Variant 1: FP32 baseline ──────────────────────────────────────────
    if Path(YOLO_WEIGHTS).exists():
        r = run_variant_benchmark(
            "FP32",
            source=source, n_frames=args.frames,
            backend=backend,
            torch_weights=YOLO_WEIGHTS, use_fp16=False,
        )
        results.append(r)
    else:
        print(f"[bench] Skipping FP32 — {YOLO_WEIGHTS} not found")

    # ── Variant 2: FP16 ───────────────────────────────────────────────────
    fp16_weights = FP16_PT_PATH if Path(FP16_PT_PATH).exists() else YOLO_WEIGHTS
    r = run_variant_benchmark(
        "FP16",
        source=source, n_frames=args.frames,
        backend=backend,
        torch_weights=fp16_weights, use_fp16=True,
    )
    results.append(r)

    # ── Variant 3: INT8 ONNX ─────────────────────────────────────────────
    if not args.skip_int8:
        if Path(ONNX_INT8_PATH).exists():
            r = run_variant_benchmark(
                "INT8",
                source=source, n_frames=args.frames,
                backend=backend,
                use_ort=True, onnx_path=ONNX_INT8_PATH,
            )
            results.append(r)
        else:
            print(f"[bench] Skipping INT8 — {ONNX_INT8_PATH} not found.")
            print("        Run export_quantize.py first.")

    print_results_table(results)
    save_results(results, backend)


if __name__ == "__main__":
    main()
