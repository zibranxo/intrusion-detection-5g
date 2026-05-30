"""
export_quantize.py
------------------
One-time model preparation script for the 5G IDS pipeline.

Exports YOLOv8n-pose to ONNX, then produces two optimised variants:
  - INT8 (static quantization via ONNX Runtime calibration)
  - FP16 (half-precision PyTorch export)

Run once before benchmark.py or quantized_pipeline.py.

Usage:
    python export_quantize.py --source <calibration_video.mp4>
    python export_quantize.py --source 0   # webcam calibration
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  — all constants are imported from config.py
# ──────────────────────────────────────────────────────────────────────────────

from config import (
    CALIBRATION_FRAMES,
    FP16_PT_PATH,
    INPUT_SIZE,
    ONNX_FP32_PATH,
    ONNX_INT8_PATH,
    YOLO_WEIGHTS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration data collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_calibration_frames(source: str | int, n_frames: int) -> list[np.ndarray]:
    """
    Capture frames from a video source for INT8 calibration.

    Args:
        source:   Path to video file, or 0 for webcam.
        n_frames: Number of frames to collect.

    Returns:
        List of BGR frames, each resized to INPUT_SIZE.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    frames: list[np.ndarray] = []
    print(f"[calibration] Collecting {n_frames} frames from '{source}'...")

    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            # Loop the video if we run out of frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        resized = cv2.resize(frame, INPUT_SIZE)
        frames.append(resized)

    cap.release()
    print(f"[calibration] Collected {len(frames)} frames.")
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# ONNX FP32 export
# ──────────────────────────────────────────────────────────────────────────────

def export_to_onnx(weights_path: str, output_path: str) -> None:
    """
    Export YOLOv8 pose model to ONNX FP32 format.

    Args:
        weights_path: Path to .pt weights file.
        output_path:  Destination .onnx path.
    """
    from ultralytics import YOLO

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading weights: {weights_path}")
    model = YOLO(weights_path)

    print(f"[export] Exporting to ONNX → {output_path}")
    model.export(
        format="onnx",
        imgsz=INPUT_SIZE[::-1],  # YOLO expects (height, width)
        opset=17,
        simplify=True,
        dynamic=False,
    )
    # Ultralytics saves to same dir as .pt; move to models/
    generated = Path(weights_path).with_suffix(".onnx")
    if generated.exists() and str(generated) != output_path:
        generated.rename(output_path)

    print(f"[export] ✓ ONNX FP32 saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# INT8 static quantization via ONNX Runtime
# ──────────────────────────────────────────────────────────────────────────────

def quantize_int8(fp32_onnx_path: str, int8_onnx_path: str,
                   calibration_frames: list[np.ndarray]) -> None:
    """
    Apply static INT8 quantization to an ONNX model using calibration data.

    Uses ONNX Runtime's quantize_static with entropy calibration — better
    accuracy than minmax for pose models.

    Args:
        fp32_onnx_path:      Path to input FP32 ONNX model.
        int8_onnx_path:      Path to write the quantized INT8 ONNX model.
        calibration_frames:  Preprocessed frames for calibration.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.calibrate import CalibrationMethod
    except ImportError:
        print("[quantize] onnxruntime-tools not installed. Skipping INT8.")
        print("           Install: pip install onnxruntime-tools")
        return

    class PoseCalibrationReader(CalibrationDataReader):
        """Feeds preprocessed frames into the ONNX calibration pipeline."""

        def __init__(self, frames: list[np.ndarray]) -> None:
            self._frames = iter(self._preprocess(frames))

        @staticmethod
        def _preprocess(frames: list[np.ndarray]) -> list[dict]:
            batch = []
            for frame in frames:
                # BGR → RGB, HWC → NCHW, [0,255] → [0,1]
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chw   = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
                nchw  = np.expand_dims(chw, axis=0)
                batch.append({"images": nchw})
            return batch

        def get_next(self) -> dict | None:
            return next(self._frames, None)

    Path(int8_onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[quantize] Running INT8 static calibration ({len(calibration_frames)} frames)...")
    t0 = time.perf_counter()

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=PoseCalibrationReader(calibration_frames),
        quant_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Entropy,
        per_channel=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"[quantize] ✓ INT8 model saved to {int8_onnx_path}  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# FP16 export (PyTorch half-precision)
# ──────────────────────────────────────────────────────────────────────────────

def export_fp16(weights_path: str, output_path: str) -> None:
    """
    Save a FP16 (half-precision) version of the YOLOv8 pose model.

    FP16 inference is handled at runtime via torch.autocast. This function
    saves the half-precision state dict so the model loads in FP16 from disk,
    halving VRAM usage and improving throughput on Tensor Core GPUs.

    Args:
        weights_path: Path to FP32 .pt weights.
        output_path:  Destination path for FP16 .pt.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[fp16] Converting {weights_path} → FP16...")

    # Load raw state dict (avoid Ultralytics model overhead)
    checkpoint = torch.load(weights_path, map_location="cpu")

    # Handle both raw state_dict and Ultralytics checkpoint formats
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_obj = checkpoint["model"]
        if hasattr(model_obj, "half"):
            checkpoint["model"] = model_obj.half()
        elif isinstance(model_obj, dict):
            checkpoint["model"] = {
                k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in model_obj.items()
            }
    elif isinstance(checkpoint, dict):
        checkpoint = {
            k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in checkpoint.items()
        }

    torch.save(checkpoint, output_path)
    print(f"[fp16] ✓ FP16 model saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Size reporter
# ──────────────────────────────────────────────────────────────────────────────

def report_model_sizes() -> None:
    """Print file sizes for all generated model artifacts."""
    paths = {
        "FP32 ONNX": ONNX_FP32_PATH,
        "INT8 ONNX": ONNX_INT8_PATH,
        "FP16 PT":   FP16_PT_PATH,
        "Original":  YOLO_WEIGHTS,
    }
    print("\n[sizes] Model artifact sizes:")
    print(f"  {'Variant':<14} {'Path':<42} {'Size':>8}")
    print("  " + "-" * 66)
    for label, path in paths.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {label:<14} {path:<42} {size_mb:>6.1f} MB")
        else:
            print(f"  {label:<14} {path:<42} {'MISSING':>8}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and quantize YOLOv8n-pose for 5G MEC IDS pipeline."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Calibration video path or '0' for webcam (default: 0)",
    )
    parser.add_argument(
        "--skip-int8",
        action="store_true",
        help="Skip INT8 quantization (faster, useful for FP16-only benchmarks)",
    )
    parser.add_argument(
        "--weights",
        default=YOLO_WEIGHTS,
        help=f"Path to YOLOv8 pose weights (default: {YOLO_WEIGHTS})",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    # Step 1: ONNX FP32
    export_to_onnx(args.weights, ONNX_FP32_PATH)

    # Step 2: INT8 calibration + quantization
    if not args.skip_int8:
        frames = collect_calibration_frames(source, CALIBRATION_FRAMES)
        quantize_int8(ONNX_FP32_PATH, ONNX_INT8_PATH, frames)
    else:
        print("[quantize] Skipping INT8 (--skip-int8 flag set)")

    # Step 3: FP16 PyTorch export
    export_fp16(args.weights, FP16_PT_PATH)

    # Report
    report_model_sizes()
    print("\n[done] Model preparation complete. Run benchmark.py next.")


if __name__ == "__main__":
    main()
