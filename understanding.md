# System Understanding — 5G MEC IDS Pipeline

## What the system does

A real-time, on-device human pose inference and behavioral intrusion detection pipeline targeting **≤25ms P95 end-to-end latency** on edge hardware. It ingests video frames, detects 17-point COCO skeletons via YOLOv8n-pose, assigns persistent track IDs via DeepSORT, and classifies five threat types (loitering, crowd formation, perimeter breach, low-confidence pose, anomalous gait) using pure geometric rules — zero additional ML inference cost.

The system was built during a Summer Research Internship at the 5G Lab, Delhi Technological University, under the URLLC/MEC track of their 5G Standalone programme.

## Architecture (as intended — not as currently runnable)

```
Camera feed (file/webcam) → Preprocess → Pose Model → DeepSORT → IDS → Output
                                    ↑
                          Alternate-frame: full inference every 2nd frame,
                          optical-flow propagation on intermediate frames
```

### 7 source files, ~3,800 lines total

| File | Role | Lines |
|---|---|---|
| `quantized_pipeline.py` | Main pipeline orchestrator | ~420 |
| `benchmark.py` | Latency/throughput benchmark harness | ~530 |
| `export_quantize.py` | One-time model export + INT8/FP16 preparation | ~300 |
| `ids_layer.py` | Behavioral IDS — 5 geometric threat rules | ~350 |
| `draw.py` | OpenCV annotation utilities (skeleton, HUD, alerts) | ~260 |
| `logger.py` | Structured threat + metrics logging | ~240 |
| `README.md` | Documentation, benchmarks, research context | ~300 |

### Model variants supported

| Variant | Format | Backend | Size vs FP32 |
|---|---|---|---|
| FP32 | PyTorch `.pt` (Ultralytics YOLO) | CUDA / MPS / CPU | 1× |
| FP16 | PyTorch `.pt` (half-precision) | CUDA / MPS | 0.5× |
| INT8 | ONNX static quantized | ONNX Runtime (CUDA/CPU EP) | 0.25× |

### Pipeline stages per frame

1. **Ingest + Preprocess** — BGR→RGB, HWC→NCHW, normalize to [0,1], resize to 640×480
2. **Alternate-frame inference** — Full YOLOv8n-pose every 2nd frame; Lucas-Kanade optical flow (Shi-Tomasi corners, pyramid LK, 3-level) propagates bboxes on skipped frames
3. **DeepSORT tracker** — Persistent track IDs across occlusion (max_age=30)
4. **Behavioral IDS** — 5 geometric rules: loitering, crowd, perimeter, low-confidence, anomalous gait
5. **Render** — Skeleton overlay, track labels, threat banners, latency HUD

### Key design decisions

- **Alternate-frame inference** halves model inference load. Assumes temporal coherence is sufficient for 2-frame propagation. Same strategy as Core ML camera pipelines.
- **Geometric IDS rules** instead of learned classifiers. Zero ML cost. Justification: "fast enough to run inline within the 25ms budget."
- **Centroid-based track-to-keypoint association** in the main loop (O(n²) per frame) rather than using DeepSORT's built-in feature vector matching.
- **Per-frame timing via perf_counter** on MPS, CUDA events on CUDA. Accurate but adds bookkeeping overhead.
- **Static INT8 quantization via entropy calibration** on 300 frames rather than dynamic quantization or QAT.

### Data flow

```
VideoCapture → resize(640×480) → alt_engine.process(frame)
  ├── full frame: pose_model.infer(frame) → (kps_list, bboxes_list)
  └── skip frame: optical flow → propagate last bboxes
→ DeepSort.update_tracks(detections) → tracks
→ Centroid matching: associate kps_list with track bboxes (<80px radius)
→ ids_layer.update(tracks, kp_by_id, frame_number) → threats[]
→ Draw skeletons, track labels, threat banners, latency HUD
→ Display + optional save
```

## Key strengths

1. **Well-structured codebase** — clean separation of concerns, consistent style, thorough docstrings on all public interfaces
2. **Multi-backend support** — CUDA, MPS (Apple Silicon/Metal), and CPU paths with appropriate instrumentation per backend
3. **Structured output** — ThreatEvent dataclasses with JSON serialization, newline-delimited threat logging, metrics in the same format as benchmarks
4. **Comprehensive benchmarking** — per-stage P50/P95/P99, CUDA event timers, machine + human-readable output
5. **Research framing** — the README explicitly connects design choices (alternate-frame, FP16 autocast, INT8 export) to Core ML/Neural Engine equivalents, making it a strong portfolio piece
6. **Model compression awareness** — not just "we quantized" but documented methodology (per-channel entropy calibration, calibration frame count, per-variant size/throughput/accuracy tradeoffs)

## Key limitations

### Critical (prevents running)

1. **Import structure is broken** — `quantized_pipeline.py` imports `from utils.draw import ...` and `from utils.logger import ...` but both files live at the root directory. No `utils/` package exists. The pipeline cannot run as committed.
2. **Never executed** — single commit, no `results/` or `models/` directories, no test artifacts, no benchmark results. The README latency table contains placeholder values ("~3.1ms", "~7.9ms"). This is a snapshot of code, not a validated system.

### Architectural

3. **Optical flow uses a single translation vector for all bboxes** — all tracked corners are pooled into one `calcOpticalFlowPyrLK` call, then a single median (dx, dy) is applied to every bounding box. When two people move in opposite directions, the median cancels out and both bboxes freeze. When a person exits the frame, their corners vanish and the remaining flow vector becomes unreliable.
4. **No flow-to-full-inference fallback** — when `_propagate_bboxes` returns `None` (insufficient good corners), the pipeline silently reuses the last known bboxes with no position update. This causes the IDS loitering counter to increment falsely.
5. **ONNX INT8 path is CPU-bound** — ONNX Runtime's CUDAExecutionProvider has limited INT8 op support. For real GPU INT8 inference, TensorRT is the standard path. As written, the INT8 variant likely runs on CPU even when `--backend cuda` is specified.
6. **No actual 5G/network component** — despite being framed as a "5G MEC IDS," there is no network simulation, no URLLC packet modeling, no bandwidth/latency modeling, no gNB/UPF topology. The system reads from a local video file.
7. **No multi-camera support** — a real MEC edge node serves multiple cameras. This is a single-stream pipeline.

### Performance

8. **O(n²) centroid matching per frame** — the track-to-keypoint association loop compares every track centroid against every detection centroid. With 10–20 people, costs ~1–2ms. Could be O(n log n) with spatial indexing.
9. **Optical flow cost may offset savings** — Shi-Tomasi corner detection + pyramid LK for all corners takes ~1–3ms per frame. For a model that runs at ~7.9ms (INT8), the "saving" is ~3–4ms, not 50%. The net gain is smaller than implied.
10. **No GPU-accelerated preprocessing** — frame resize, color conversion, and normalization are CPU numpy operations. On a GPU pipeline, these should be CUDA kernels or at minimum use `torchvision.transforms` on the GPU tensor.
11. **No decode acceleration** — OpenCV's `VideoCapture` uses CPU software decode. For a real 5G camera feed (likely HEVC), hardware decode (NVDEC, VideoToolbox) would save 2–5ms per frame.

### Code quality

12. **Duplicated constants** — `SKELETON_EDGES` is defined in both `draw.py` and `quantized_pipeline.py`. `INPUT_SIZE`, `TARGET_LATENCY_MS` are repeated across files.
13. **Inconsistent type hints** — some public methods have full annotations, others have none. Internal methods in `AltFrameInference` are untyped.
14. **Stale/empty comment block** — `# ── Drawing utilities ──` sits orphaned in `quantized_pipeline.py` between class definitions.
15. **No tests** — `pytest` is in `requirements.txt`, the README documents `pytest tests/ -v`, but no `tests/` directory exists.
16. **No configuration management** — all hyperparameters (confidence thresholds, loiter distance, crowd radius, frame intervals) are module-level constants duplicated or near-duplicated across files.

### Behavioral IDS

17. **Arbitrary confidence scores** — Loitering=0.90, Crowd=0.90, Perimeter=0.90, LowConfidence=0.70, AnomalousGait=0.50. These are not calibrated against any labeled data. They don't reflect measurement uncertainty or per-rule precision.
18. **Per-person crowd alerts** — when 4+ people stand together, every single person in the group gets a separate CROWD_FORMATION alert. This generates N redundant alerts for one event.
19. **No adaptive thresholds** — loitering detection uses a hardcoded 50px radius. At different camera angles or resolutions, this means completely different real-world distances.
20. **Gait angle rule is fragile** — the hip-shoulder lateral angle is computed in image space, not world space. Camera perspective distorts it. A person walking directly toward the camera will have a near-zero angle and trigger ANOMALOUS_GAIT.
