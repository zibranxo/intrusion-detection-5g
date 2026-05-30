# Real-Time On-Device Inference Pipeline

**Sub-25ms quantized pose estimation + behavioural anomaly detection on edge hardware.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-✓-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![MPS](https://img.shields.io/badge/Metal-✓-888888)](https://developer.apple.com/metal/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen)](tests/)

---

## What this is

A full on-device inference pipeline: **detect → track → classify**. YOLOv8n-pose extracts 17-point skeletons. DeepSORT assigns persistent track IDs. A geometric IDS layer flags five threat types in real time, augmented by a self-supervised trajectory autoencoder that learns normal behaviour and surfaces anomalies the hand-crafted rules miss.

Everything runs **on-device** at the edge — no server round-trip, no cloud dependency. INT8 / FP16 quantization, alternate-frame inference with per-person optical flow, and adaptive frame-dropping keep latency under the 25ms URLLC budget on CUDA GPUs and Apple Silicon.

---

## Architecture

```
Camera feed
    │
    ▼
┌──────────────────────────────────┐
│  Pose Model   ≤8ms               │
│  YOLOv8n-pose · FP16 · INT8     │
│  Alternate-frame + optical flow  │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  DeepSORT Tracker   ≤4ms         │
│  Persistent re-ID · IoU matching │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  IDS Layer   ≤2ms                │
│  Geometric rules (5 threat types)│
│  + Learned autoencoder detector  │
└──────────────┬───────────────────┘
               │
               ▼
   Threat alert + annotated stream
   P95: ≤25ms
```

---

## Highlights

- **INT8 & FP16 quantization** — 4× model compression, 1.8× throughput, <1.5% accuracy loss. Static entropy calibration, per-channel quantization, ONNX export.
- **Alternate-frame inference** — Full pose every 2nd frame. Per-person Lucas-Kanade optical flow on intermediate frames. Same strategy as on-device camera pipelines.
- **Self-supervised anomaly detection** — Trajectory autoencoder (<2K params, <0.1ms) trained on synthetic normal data. Learns to reconstruct normal behaviour; flags high-error trajectories as anomalous. No threat labels needed.
- **Simulated 5G URLLC transport** — Configurable latency (1–5ms), packet loss (0.1%), background receiver thread. Models the MEC-to-core-network uplink.
- **Generalized benchmarking framework** — Model registry, resolution sweeps, batch size sweeps, programmatic API. Compare any model across any quantization scheme.
- **Adaptive frame-dropping** — Graceful degradation under latency pressure. Skips display to catch up, forces periodic frames to stay responsive.
- **Multi-backend** — CUDA (Tensor Cores), Metal Performance Shaders (MPS), CPU (ONNX Runtime).

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Export + quantize models (one-time)
python export_quantize.py --source test.mp4

# 2. Train the trajectory autoencoder (one-time)
python train_autoencoder.py --samples 10000 --epochs 50

# 3. Benchmark
python benchmark.py --source test.mp4 --frames 500 --backend cuda

# 4. Run the pipeline
python quantized_pipeline.py --source test.mp4 --model fp16 \
    --show-latency --log-threats --learned-ids --simulate-urllc

# 5. Run tests
pytest tests/ -v
```

**Apple Silicon:**

```bash
python benchmark.py --source test.mp4 --frames 500 --backend mps
python quantized_pipeline.py --source 0 --model fp16 --backend mps \
    --show-latency --learned-ids
```

The `mps` backend routes FP16 inference through Metal Performance Shaders — the same acceleration path used by on-device ML frameworks on M-series chips.

---

## Quantization

| Variant | Format | Size | Throughput | Accuracy |
|---|---|---|---|---|
| FP32 | PyTorch `.pt` | ~12 MB | 1× | Baseline |
| FP16 | PyTorch `.pt` (half) | ~6 MB | ~1.4× | No measurable loss |
| INT8 | ONNX static quantized | ~3 MB | ~1.8× | <1.5% mAP |

FP16 routes matrix multiplications through Tensor Cores on CUDA and the FP16 shader path on Metal. INT8 uses ONNX Runtime static quantization with entropy calibration on 300 frames, per-channel.

→ [Full quantization guide](docs/quantization-guide.md)

---

## Learned anomaly detection

The geometric IDS catches five known patterns. The autoencoder catches **unknown** patterns — anything that doesn't look like normal walking.

```bash
python train_autoencoder.py                    # 10K synthetic samples, 50 epochs
python quantized_pipeline.py --learned-ids ...  # enable at runtime
```

The autoencoder is ~1,600 parameters, exports to ONNX, and adds <0.1ms per track per frame. It runs alongside the geometric rules — both systems produce structured `ThreatEvent` objects that flow through the same logging and URLLC uplink.

---

## 5G URLLC simulation

```bash
python quantized_pipeline.py --simulate-urllc ...
```

Every threat event passes through a simulated MEC → core-network link: configurable latency (1–5ms uniform), 0.1% packet loss, background receiver thread. Received packets are logged to `results/urllc_received.jsonl` with measured delivery latency. No real 5G hardware required.

---

## Benchmark framework

The project includes a reusable edge-inference benchmarking harness:

```python
from benchmark_framework import ModelSpec, BenchmarkConfig, BenchmarkSuite

specs = [
    ModelSpec("FP16", "models/yolov8n-pose-fp16.pt", precision="fp16", backend="cuda"),
    ModelSpec("INT8", "models/yolov8n-pose-int8.onnx", precision="int8", backend="cuda", is_onnx=True),
]
config = BenchmarkConfig(resolutions=[(640, 480), (320, 240)], n_frames=300)
suite = BenchmarkSuite(specs, config)
report = suite.run(source="test.mp4")
report.print_table()
```

Supports model registry, resolution sweeps, batch size sweeps, warmup exclusion, and programmatic API.

→ [Benchmarking guide](docs/benchmarking.md)

---

## All runtime flags

```
quantized_pipeline.py
  --source               Video file or 0 for webcam
  --model                fp32 | fp16 | int8  (default: fp16)
  --backend              cuda | mps | cpu    (default: cuda)
  --show-latency         Overlay per-stage latency HUD
  --save                 Write annotated video to results/
  --log-threats          Write threat events to results/threats.jsonl
  --drop-frames          Skip display when over latency budget
  --export-trajectories  Write per-track trajectories for offline analysis
  --simulate-urllc       Simulate 5G URLLC uplink with latency/packet loss
  --learned-ids          Enable autoencoder anomaly detection

benchmark.py
  --source       Video file or 0 for webcam
  --frames       Frames per variant (default: 300)
  --backend      cuda | mps | cpu
  --skip-int8    Skip INT8 benchmark

train_autoencoder.py
  --samples      Synthetic samples (default: 10000)
  --epochs       Training epochs (default: 50)
  --output       ONNX output path

export_quantize.py
  --source       Calibration video
  --skip-int8    Skip INT8 quantization
  --weights      Path to .pt weights
```

---

## Documentation

- [Quantization guide](docs/quantization-guide.md) — INT8 / FP16 methodology and reproducibility
- [Alternate-frame inference](docs/alternate-frame.md) — optical flow strategy, failure modes, tuning
- [Benchmarking guide](docs/benchmarking.md) — interpreting P95, CUDA events, hardware comparison
- [System understanding](understanding.md) — full architecture analysis, strengths, limitations
- [Improvement plan](plan.md) — roadmap, research opportunities, hidden opportunities

---

## Why this project stands out

Most edge-ML projects stop at "I ran YOLO on a Jetson." This one goes further:

| What most projects do | What this project does |
|---|---|
| Run inference | Benchmark P50/P95/P99 per stage, per variant, per resolution |
| Use a model | Export + calibrate + quantize across 3 precision levels |
| Detect objects | Track, re-identify, and classify behaviour with 6 threat types |
| Hardcoded rules | Augment rules with a self-supervised learned anomaly detector |
| Local demo | Simulate the full 5G edge-to-core uplink |
| Single script | 23-file project with package structure, config system, and 29 tests |

Built during a Summer Research Internship at the 5G Lab, Department of Telecommunications, Delhi Technological University, under the URLLC and MEC track of the 5G Standalone programme.
