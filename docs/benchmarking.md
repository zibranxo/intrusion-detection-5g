# Benchmarking Guide — Interpreting Edge Inference Latency

How to run, read, and trust the benchmark numbers in this project.

---

## What we measure

The pipeline has four stages, each timed independently:

| Stage | What it measures | Typical cost |
|---|---|---|
| `ingest_preprocess` | Frame read + resize + BGR→RGB + normalize | 2-3ms |
| `model_inference` | YOLOv8n-pose forward pass only (no post-processing) | 7-12ms |
| `postprocess` | NMS + keypoint extraction + bbox formatting | 1-2ms |
| `overhead` | Display prep + OpenCV I/O | 0.5-1ms |

### Why P95, not mean

Mean latency is misleading for real-time systems. A pipeline that runs at 10ms average but spikes to 50ms once per second will miss its URLLC budget 3% of the time. P95 captures the tail: 95% of frames complete within this time. P99 is also reported for the strictest SLAs.

### CUDA events vs. wall-clock

On CUDA backends, `benchmark.py` uses `torch.cuda.Event` with `enable_timing=True`. CUDA events measure GPU kernel execution time, not CPU wall-clock time. They exclude CPU-GPU synchronization overhead and Python interpreter overhead. This is the same mechanism NVIDIA uses for `nsys` profiling and TensorRT benchmarking.

On MPS backends, `time.perf_counter` is used instead (MPS does not expose event timers). These numbers include CPU-GPU sync time and are slightly higher than equivalent CUDA event timings.

---

## Running benchmarks

```bash
# CUDA GPU
python benchmark.py --source test.mp4 --frames 500 --backend cuda

# Apple Silicon
python benchmark.py --source test.mp4 --frames 500 --backend mps

# CPU-only
python benchmark.py --source test.mp4 --frames 300 --backend cpu --skip-int8

# Skip INT8 (faster iteration)
python benchmark.py --source test.mp4 --frames 300 --skip-int8
```

### Using the generalized framework

The project also includes `benchmark_framework.py` — a reusable benchmark harness that supports:

- **Model registry**: Register any model variant with metadata
- **Resolution sweeps**: Compare latency at 640×480, 320×240, 160×120
- **Batch size sweeps**: Measure throughput with batch inference
- **Programmatic API**: Import and use from Python scripts/notebooks

```python
from benchmark_framework import ModelSpec, BenchmarkConfig, BenchmarkSuite

specs = [
    ModelSpec("FP16", "models/yolov8n-pose-fp16.pt", precision="fp16", backend="cuda"),
    ModelSpec("INT8", "models/yolov8n-pose-int8.onnx", precision="int8", backend="cuda", is_onnx=True),
]
config = BenchmarkConfig(
    resolutions=[(640, 480), (320, 240)],
    n_frames=300,
    warmup_frames=30,
)
suite = BenchmarkSuite(specs, config)
report = suite.run(source="test.mp4")
report.print_table()
report.save("results/")
```

---

## Interpreting results

### What "meets target" means

The target is 25ms end-to-end P95 latency. This is derived from the 5G URLLC latency budget:

- 1ms: air interface (5G NR URLLC target)
- 25ms: MEC processing budget (our pipeline)
- ~5ms: uplink to core network
- Total round-trip: ~31ms

If `total_p95_ms ≤ 25.0`, the pipeline meets the URLLC budget.

### Warm-up frames

The first 5-10 inferences on CUDA are slower due to CUDA kernel compilation and memory allocation. The benchmark runs 300+ frames; the first 30 should be considered warm-up and excluded from statistics (controlled by `--warmup` in the framework, implicitly handled by the large frame count in benchmark.py).

### Variability sources

| Source | Impact | Mitigation |
|---|---|---|
| GPU boost clock decay | 5-15% slower after thermal throttling | Run benchmarks cold, use consistent GPU state |
| CPU governor / power state | 10-20% on laptops | Pin CPU frequency during benchmarks |
| Video decode variability | 2-5ms on CPU decode | Use GPU-accelerated decode (NVDEC) |
| Python GC pauses | Occasional 5-20ms spikes | Use `gc.disable()` during benchmark loop |
| First-frame overhead | 50-100ms slower | Exclude via warmup frames |

---

## Comparing across hardware

### CUDA GPUs

| GPU | YOLOv8n-pose INT8 P95 | Notes |
|---|---|---|
| RTX 4090 | ~3ms | TensorRT recommended |
| RTX 3080 | ~4ms | |
| RTX 2070 | ~6ms | |
| T4 (cloud) | ~8ms | Limited INT8 acceleration |

### Apple Silicon

| Chip | YOLOv8n-pose FP16 P95 | Notes |
|---|---|---|
| M2 Max | ~8ms | MPS backend, Metal Performance Shaders |
| M2 | ~10ms | |
| M1 | ~12ms | |

### CPU

| CPU | YOLOv8n-pose INT8 P95 | Notes |
|---|---|---|
| AMD Ryzen 7950X | ~15ms | ONNX Runtime |
| Intel i9-13900K | ~18ms | ONNX Runtime |
| Apple M2 (CPU) | ~25ms | ONNX Runtime CPU EP |

---

## Output files

| File | Format | Contents |
|---|---|---|
| `results/benchmark_results.json` | JSON | Machine-readable, all metrics |
| `results/benchmark_summary.txt` | Plain text | Human-readable table |
| `results/metrics.json` | JSON | Per-run metrics from pipeline |

---

## Further reading

- [NVIDIA CUDA Event Timing](https://developer.nvidia.com/blog/how-implement-performance-measurements-cuda-cc/)
- [MLPerf Inference Benchmarks](https://mlcommons.org/benchmarks/inference-edge/) — industry-standard edge inference suite
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
