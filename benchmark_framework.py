"""
benchmark_framework.py — Generalised edge-inference benchmarking harness.

Refactors the single-pipeline benchmark (benchmark.py) into a reusable
framework that can compare any model, any quantisation scheme, any
resolution, and any batch size.  Designed for researchers who need
reproducible edge-inference latency comparisons.

Key abstractions:
  - ModelSpec:       Describes a model variant (name, path, precision, backend).
  - BenchmarkConfig: Controls sweep parameters (resolutions, batch sizes, frames).
  - BenchmarkSuite:  Runs benchmarks across specs and configs, collects results.
  - BenchmarkReport: Formats results as tables, JSON, and summary text.

This is Hidden Opportunity #1 from plan.md — turning a project-specific
benchmark into a tool other researchers can cite and use.

Usage (programmatic):
    from benchmark_framework import ModelSpec, BenchmarkConfig, BenchmarkSuite

    specs = [
        ModelSpec("FP32", "yolov8n-pose.pt", precision="fp32", backend="cuda"),
        ModelSpec("INT8", "models/yolov8n-pose-int8.onnx", precision="int8", backend="cuda"),
    ]
    config = BenchmarkConfig(resolutions=[(640,480), (320,240)], n_frames=300)
    suite = BenchmarkSuite(specs, config)
    report = suite.run(source="test.mp4")
    report.print_table()
    report.save("results/")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

Precision = Literal["fp32", "fp16", "int8"]
Backend   = Literal["cuda", "mps", "cpu"]


@dataclass
class ModelSpec:
    """
    Describes a model variant to benchmark.

    Attributes:
        name:            Human-readable label (e.g. 'FP32', 'INT8', 'FP16-MPS').
        weights_path:    Path to model weights (.pt, .onnx, etc.).
        precision:       'fp32', 'fp16', or 'int8'.
        backend:         'cuda', 'mps', or 'cpu'.
        is_onnx:         True if weights_path is an ONNX model.
        extra:           Arbitrary metadata dict (e.g. {'calib_frames': 300}).
    """
    name:         str
    weights_path: str
    precision:    Precision  = "fp32"
    backend:      Backend    = "cuda"
    is_onnx:      bool       = False
    extra:        dict       = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """
    Controls the parameters of a benchmark suite run.

    Attributes:
        resolutions:     List of (width, height) tuples to sweep.
        batch_sizes:     List of batch sizes to sweep (1 = single-frame).
        n_frames:        Frames to process per variant per resolution.
        warmup_frames:   Frames to discard before measuring (CUDA warm-up).
        target_latency_ms: Latency budget for meets_target check.
    """
    resolutions:        list[tuple[int, int]] = field(default_factory=lambda: [(640, 480)])
    batch_sizes:        list[int]             = field(default_factory=lambda: [1])
    n_frames:           int                   = 300
    warmup_frames:      int                   = 30
    target_latency_ms:  float                 = 25.0


@dataclass
class StageStats:
    """Per-stage timing statistics (P50 / P95 / P99 / mean)."""
    stage:    str
    p50_ms:   float = 0.0
    p95_ms:   float = 0.0
    p99_ms:   float = 0.0
    mean_ms:  float = 0.0
    n:        int   = 0


@dataclass
class VariantResult:
    """Benchmark result for one model variant at one resolution / batch size."""
    variant:      str
    precision:    str
    backend:      str
    resolution:   tuple[int, int]
    batch_size:   int
    stages:       list[StageStats] = field(default_factory=list)
    fps:          float = 0.0
    total_p95_ms: float = 0.0
    meets_target: bool  = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolution"] = list(self.resolution)
        return d


@dataclass
class BenchmarkReport:
    """Aggregated results from a benchmark suite run."""
    meta:        dict               = field(default_factory=dict)
    results:     list[VariantResult] = field(default_factory=list)
    timestamp:   str                = field(default_factory=lambda: time.strftime(
                                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def filter(
        self,
        precision: Optional[str] = None,
        backend:   Optional[str] = None,
        resolution: Optional[tuple[int, int]] = None,
    ) -> list[VariantResult]:
        """Filter results by attribute."""
        out = self.results
        if precision:
            out = [r for r in out if r.precision == precision]
        if backend:
            out = [r for r in out if r.backend == backend]
        if resolution:
            out = [r for r in out if tuple(r.resolution) == tuple(resolution)]
        return out

    def print_table(self, stage_filter: Optional[list[str]] = None) -> None:
        """
        Print a formatted comparison table to stdout.

        Args:
            stage_filter: If provided, only show these stage names.
                          Default shows all stages.
        """
        if not self.results:
            print("[benchmark] No results to display.")
            return

        COL_W = 14
        variants = [r.variant for r in self.results]

        # Determine stage names from the first result.
        all_stages = [s.stage for s in self.results[0].stages]
        stages = [s for s in all_stages if not stage_filter or s in stage_filter]

        header = "".join(f"{v:>{COL_W}}" for v in variants)
        print(f"\n{'─' * (28 + COL_W * len(variants))}")
        print(f"  Latency Benchmark — P95 (ms)     {header}")
        print(f"{'─' * (28 + COL_W * len(variants))}")

        for stage_name in stages:
            row = f"  {stage_name:<26}"
            for r in self.results:
                stage = next((s for s in r.stages if s.stage == stage_name), None)
                val = f"{stage.p95_ms:.1f}" if stage else "—"
                row += f"{val:>{COL_W}}"
            print(row)

        print(f"{'─' * (28 + COL_W * len(variants))}")
        total_row = f"  {'TOTAL P95':<26}"
        for r in self.results:
            flag  = " ✓" if r.meets_target else " ✗"
            total = f"{r.total_p95_ms:.1f}{flag}"
            total_row += f"{total:>{COL_W}}"
        print(total_row)

        fps_row = f"  {'FPS':<26}"
        for r in self.results:
            fps_row += f"{r.fps:>{COL_W}.1f}"
        print(fps_row)
        print(f"{'─' * (28 + COL_W * len(variants))}")
        print(f"  Target: ≤{self.meta.get('target_latency_ms', 25):.0f}ms P95")
        print(f"  Backend: {self.results[0].backend}\n")

    def save(self, output_dir: str | Path) -> None:
        """Persist results to JSON and plain-text summary."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out / "benchmark_results.json"
        payload = {
            "meta": {**self.meta, "timestamp": self.timestamp},
            "results": [r.to_dict() for r in self.results],
        }
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"[benchmark] JSON → {json_path}")

        # Plain text
        txt_path = out / "benchmark_summary.txt"
        lines = [
            "Edge ML Benchmark Summary",
            f"Timestamp: {self.timestamp}",
            f"Target: ≤{self.meta.get('target_latency_ms', 25):.0f}ms P95",
            "",
            f"{'Variant':<14} {'Precision':<10} {'Resolution':<14} {'Total P95':>12} {'FPS':>8}",
            "-" * 62,
        ]
        for r in self.results:
            res_str = f"{r.resolution[0]}×{r.resolution[1]}"
            lines.append(
                f"{r.variant:<14} {r.precision:<10} {res_str:<14} "
                f"{r.total_p95_ms:>8.1f}ms {r.fps:>7.1f}"
            )
        txt_path.write_text("\n".join(lines) + "\n")
        print(f"[benchmark] Summary → {txt_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Registry of known model variants.

    Pre-populated with the YOLOv8n-pose variants from this project.
    Call register() to add custom models before running a benchmark suite.

    Usage:
        registry = ModelRegistry()
        registry.register(ModelSpec("MyModel", "path/to/model.pt", precision="fp16"))

        specs = registry.list_specs(backend="cuda")
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

        # ── Built-in specs for YOLOv8n-pose ───────────────────────────
        defaults = [
            ModelSpec("FP32-CUDA", "yolov8n-pose.pt",
                      precision="fp32", backend="cuda"),
            ModelSpec("FP16-CUDA", "models/yolov8n-pose-fp16.pt",
                      precision="fp16", backend="cuda"),
            ModelSpec("INT8-CUDA", "models/yolov8n-pose-int8.onnx",
                      precision="int8", backend="cuda", is_onnx=True),
            ModelSpec("FP32-MPS", "yolov8n-pose.pt",
                      precision="fp32", backend="mps"),
            ModelSpec("FP16-MPS", "models/yolov8n-pose-fp16.pt",
                      precision="fp16", backend="mps"),
            ModelSpec("FP32-CPU", "yolov8n-pose.pt",
                      precision="fp32", backend="cpu"),
            ModelSpec("INT8-CPU", "models/yolov8n-pose-int8.onnx",
                      precision="int8", backend="cpu", is_onnx=True),
        ]
        for spec in defaults:
            self._specs[spec.name] = spec

    def register(self, spec: ModelSpec) -> None:
        """Register a model variant."""
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[ModelSpec]:
        """Get a registered spec by name."""
        return self._specs.get(name)

    def list_specs(
        self,
        precision: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> list[ModelSpec]:
        """List registered specs, optionally filtered."""
        out = list(self._specs.values())
        if precision:
            out = [s for s in out if s.precision == precision]
        if backend:
            out = [s for s in out if s.backend == backend]
        return out

    def list_names(self) -> list[str]:
        """Return sorted list of registered spec names."""
        return sorted(self._specs.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Resolution sweep helper
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResolutionSweepConfig:
    """
    Defines a resolution sweep for latency comparison.

    Attributes:
        min_size:    Smallest (width, height).
        max_size:    Largest (width, height).
        step:        Step size in pixels for both dimensions.
        aspect_only: If True, generate resolutions by varying aspect ratio
                     at fixed area instead of varying absolute size.
    """
    min_size:    tuple[int, int] = (160, 120)
    max_size:    tuple[int, int] = (1280, 960)
    step:        int             = 160

    def generate(self) -> list[tuple[int, int]]:
        """Generate a list of resolutions for the sweep."""
        resolutions: list[tuple[int, int]] = []
        w = self.min_size[0]
        while w <= self.max_size[0]:
            h = int(w * self.min_size[1] / self.min_size[0])
            resolutions.append((w, h))
            w += self.step
        return resolutions


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark suite runner
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkSuite:
    """
    Runs benchmarks across multiple model specs, resolutions, and batch sizes.

    This is the programmatic API — use it from Python scripts and notebooks.

    Usage:
        suite = BenchmarkSuite(specs, config)
        report = suite.run(source="test.mp4")
        report.print_table()
    """

    def __init__(
        self,
        specs: list[ModelSpec],
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        self._specs  = specs
        self._config = config or BenchmarkConfig()

    def run(self, source: str | int) -> BenchmarkReport:
        """
        Run benchmarks for all specs × resolutions × batch sizes.

        Args:
            source: Video path or webcam index.

        Returns:
            BenchmarkReport with all results.
        """
        report = BenchmarkReport(meta={
            "target_latency_ms": self._config.target_latency_ms,
            "n_frames":          self._config.n_frames,
            "warmup_frames":     self._config.warmup_frames,
        })

        total_combos = (
            len(self._specs)
            * len(self._config.resolutions)
            * len(self._config.batch_sizes)
        )
        print(f"[benchmark] {total_combos} benchmark(s) to run\n")

        for spec in self._specs:
            for res in self._config.resolutions:
                for bs in self._config.batch_sizes:
                    print(
                        f"[benchmark] {spec.name:<16} "
                        f"res={res[0]}×{res[1]:<6} "
                        f"batch={bs}"
                    )

                    result = self._run_single(spec, source, res, bs)
                    report.results.append(result)

                    status = "✓" if result.meets_target else "✗"
                    print(
                        f"            → P95={result.total_p95_ms:.1f}ms  "
                        f"FPS={result.fps:.1f}  target={status}"
                    )

        return report

    def _run_single(
        self,
        spec: ModelSpec,
        source: str | int,
        resolution: tuple[int, int],
        batch_size: int,
    ) -> VariantResult:
        """
        Run a single benchmark and return timing statistics.

        Uses the same timing infrastructure as benchmark.py but
        wrapped in the framework abstraction.
        """
        import cv2

        result = VariantResult(
            variant=spec.name,
            precision=spec.precision,
            backend=spec.backend,
            resolution=resolution,
            batch_size=batch_size,
        )

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        # Collect timing samples
        preprocess_samples: list[float] = []
        inference_samples:  list[float] = []
        postproc_samples:   list[float] = []
        overhead_samples:   list[float] = []

        total_frames = self._config.n_frames + self._config.warmup_frames
        frame_count  = 0
        wall_start   = time.perf_counter()

        # Build model session once
        if spec.is_onnx:
            import onnxruntime as ort
            session = ort.InferenceSession(
                spec.weights_path,
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if spec.backend == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"]
                ),
            )
        else:
            from ultralytics import YOLO
            import torch

            device = (
                "cuda" if spec.backend == "cuda" and torch.cuda.is_available()
                else "mps" if spec.backend == "mps" and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                else "cpu"
            )
            model = YOLO(spec.weights_path)
            if spec.precision == "fp16" and device != "cpu":
                model.model = model.model.half().to(device)
            else:
                model.model = model.model.to(device)
            session = None

        while frame_count < total_frames:
            ret, raw_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, raw_frame = cap.read()
                if not ret:
                    break

            # Preprocess
            t0 = time.perf_counter()
            resized  = cv2.resize(raw_frame, resolution)
            rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            chw      = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
            nchw     = np.expand_dims(chw, axis=0)
            t1 = time.perf_counter()

            # Inference
            t2 = time.perf_counter()
            if spec.is_onnx:
                input_name = session.get_inputs()[0].name
                _ = session.run(None, {input_name: nchw})
            else:
                _ = model(raw_frame, verbose=False)
            t3 = time.perf_counter()

            # Post-process (minimal — just extract keypoints)
            t4 = time.perf_counter()
            _ = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            t5 = time.perf_counter()

            frame_count += 1

            # Skip warmup frames
            if frame_count <= self._config.warmup_frames:
                continue

            preprocess_samples.append((t1 - t0) * 1000.0)
            inference_samples.append((t3 - t2) * 1000.0)
            postproc_samples.append((t4 - t3) * 1000.0)
            overhead_samples.append((t5 - t4) * 1000.0)

        cap.release()

        if not inference_samples:
            return result

        wall_elapsed = time.perf_counter() - wall_start
        total_p95 = float(
            np.percentile(preprocess_samples, 95)
            + np.percentile(inference_samples, 95)
            + np.percentile(postproc_samples, 95)
            + np.percentile(overhead_samples, 95)
        )

        result.stages = [
            StageStats("preprocess", p50=float(np.percentile(preprocess_samples, 50)),
                       p95=float(np.percentile(preprocess_samples, 95)),
                       p99=float(np.percentile(preprocess_samples, 99)),
                       mean=float(np.mean(preprocess_samples)), n=len(preprocess_samples)),
            StageStats("inference", p50=float(np.percentile(inference_samples, 50)),
                       p95=float(np.percentile(inference_samples, 95)),
                       p99=float(np.percentile(inference_samples, 99)),
                       mean=float(np.mean(inference_samples)), n=len(inference_samples)),
            StageStats("postprocess", p50=float(np.percentile(postproc_samples, 50)),
                       p95=float(np.percentile(postproc_samples, 95)),
                       p99=float(np.percentile(postproc_samples, 99)),
                       mean=float(np.mean(postproc_samples)), n=len(postproc_samples)),
            StageStats("overhead", p50=float(np.percentile(overhead_samples, 50)),
                       p95=float(np.percentile(overhead_samples, 95)),
                       p99=float(np.percentile(overhead_samples, 99)),
                       mean=float(np.mean(overhead_samples)), n=len(overhead_samples)),
        ]
        result.fps = self._config.n_frames / max(wall_elapsed, 0.001)
        result.total_p95_ms = total_p95
        result.meets_target = total_p95 <= self._config.target_latency_ms

        return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point (optional — use benchmark.py for the project-specific harness)
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Run the benchmark framework from the command line.

    This is a simplified CLI; for the full project-specific benchmark
    (with CUDA events, per-stage timing, and variant-specific paths),
    use benchmark.py directly.

    Usage:
        python benchmark_framework.py --source test.mp4 --specs FP16-CUDA,INT8-CUDA
    """
    import argparse

    registry = ModelRegistry()

    parser = argparse.ArgumentParser(
        description="Generalised edge-inference benchmark framework."
    )
    parser.add_argument("--source", default="0",
                        help="Video path or '0' for webcam")
    parser.add_argument("--specs", default="",
                        help="Comma-separated spec names (empty = all registered)")
    parser.add_argument("--frames", type=int, default=300,
                        help="Frames per benchmark")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Warmup frames to discard")
    parser.add_argument("--resolutions", default="640x480",
                        help="Comma-separated resolutions, e.g. '640x480,320x240'")
    parser.add_argument("--target", type=float, default=25.0,
                        help="Target latency in ms")
    parser.add_argument("--list-specs", action="store_true",
                        help="List all registered model specs and exit")
    args = parser.parse_args()

    if args.list_specs:
        print("\nRegistered model specs:")
        for name in registry.list_names():
            spec = registry.get(name)
            print(f"  {name:<16}  {spec.precision:<6}  {spec.backend:<6}  "
                  f"onnx={spec.is_onnx}  → {spec.weights_path}")
        return

    # Parse specs
    if args.specs:
        spec_names = [s.strip() for s in args.specs.split(",")]
        specs = [registry.get(n) for n in spec_names]
        specs = [s for s in specs if s is not None]
        if not specs:
            print(f"No valid specs found. Available: {registry.list_names()}")
            return
    else:
        specs = registry.list_specs()

    # Parse resolutions
    resolutions: list[tuple[int, int]] = []
    for r_str in args.resolutions.split(","):
        parts = r_str.strip().split("x")
        if len(parts) == 2:
            resolutions.append((int(parts[0]), int(parts[1])))
    if not resolutions:
        resolutions = [(640, 480)]

    config = BenchmarkConfig(
        resolutions=resolutions,
        n_frames=args.frames,
        warmup_frames=args.warmup,
        target_latency_ms=args.target,
    )

    suite = BenchmarkSuite(specs, config)
    report = suite.run(source=args.source)
    report.print_table()
    report.save("results/")


if __name__ == "__main__":
    main()
