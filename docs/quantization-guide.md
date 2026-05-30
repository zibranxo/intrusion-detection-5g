# Quantization Guide — 5G MEC IDS Pipeline

How model quantization works in this project, why we use INT8 and FP16, and how to reproduce the results.

---

## Overview

The pipeline supports three precision levels for YOLOv8n-pose:

| Variant | Format | Size vs FP32 | Throughput vs FP32 | Accuracy impact |
|---|---|---|---|---|
| FP32 | PyTorch `.pt` | 1× (~12 MB) | 1× | Baseline |
| FP16 | PyTorch `.pt` (half) | 0.5× (~6 MB) | ~1.4× | None measurable |
| INT8 | ONNX static quantized | 0.25× (~3 MB) | ~1.8× | <1.5% mAP |

---

## FP16 (Half-Precision)

### What happens

The model weights are cast from `float32` to `float16`. On CUDA hardware, matrix multiplications are routed through Tensor Cores (via `torch.autocast`). On Apple Silicon, FP16 maps to Metal Performance Shaders' FP16 path — the same path Core ML uses for Neural Engine acceleration.

### Code path

```python
# In export_quantize.py:
model = YOLO(weights_path)
checkpoint["model"] = model_obj.half()  # cast all float tensors
torch.save(checkpoint, "models/yolov8n-pose-fp16.pt")

# At inference (in PoseModel):
model.model = model.model.half().to(device)
with torch.autocast(device_type="cuda"):
    results = model(frame, verbose=False)
```

### When to use

- CUDA GPUs with Tensor Cores (Volta+, Turing+, Ampere+)
- Apple Silicon (M1/M2/M3) via MPS backend
- When you need ~40% latency reduction with zero accuracy loss

### When NOT to use

- CPU-only inference (FP16 has no CPU acceleration)
- Pre-Tensor Core GPUs (Pascal and older)

---

## INT8 (Static Quantization)

### Methodology

1. **Export to ONNX FP32** — via Ultralytics' built-in exporter (`opset=17`, graph simplification enabled)
2. **Calibration** — 300 representative frames are collected from the target video source
3. **Static quantization** — ONNX Runtime's `quantize_static` with entropy calibration (not minmax — entropy preserves more dynamic range for pose models)
4. **Per-channel quantization** — convolutional layer weights are quantized per-channel rather than per-tensor, preserving per-filter dynamic range

### Why entropy calibration vs. minmax

Minmax calibration fits the [min, max] range of each tensor. Entropy calibration fits the distribution to minimize KL divergence between the FP32 and INT8 output distributions. For pose estimation, where keypoint heatmaps have long-tailed distributions, entropy calibration retains ~1-2% more mAP than minmax.

### Why per-channel

Pose models have filters that respond to different spatial frequencies (some detect heads, others detect ankles). Per-tensor quantization would force all filters in a layer to share the same scale, compressing the dynamic range of low-magnitude filters. Per-channel quantization gives each filter its own scale.

### Code path

```python
# Calibration data collection
frames = collect_calibration_frames("test.mp4", n_frames=300)

# Static quantization
quantize_static(
    model_input="models/yolov8n-pose-fp32.onnx",
    model_output="models/yolov8n-pose-int8.onnx",
    calibration_data_reader=PoseCalibrationReader(frames),
    quant_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.Entropy,
    per_channel=True,
)
```

### Known limitation

ONNX Runtime's CUDAExecutionProvider has limited INT8 op support. For production GPU INT8 inference, TensorRT is the standard path (see plan.md Phase 5). The current INT8 path runs on CPU via ONNX Runtime, which is still fast (~8ms) but not GPU-accelerated.

---

## Reproducing the benchmarks

```bash
# Step 1: Export and quantize (run once)
python export_quantize.py --source test.mp4

# Step 2: Benchmark all variants
python benchmark.py --source test.mp4 --frames 500 --backend cuda

# Step 3: Run the pipeline with your chosen variant
python quantized_pipeline.py --source test.mp4 --model fp16 --show-latency
```

### Apple Silicon

```bash
python benchmark.py --source test.mp4 --frames 500 --backend mps
python quantized_pipeline.py --source test.mp4 --model fp16 --backend mps --show-latency
```

---

## Model size comparison

```bash
$ ls -lh models/
yolov8n-pose.pt             12M   # FP32 baseline
yolov8n-pose-fp16.pt         6M   # FP16 (half)
yolov8n-pose-fp32.onnx      12M   # ONNX intermediate
yolov8n-pose-int8.onnx       3M   # INT8 (quantized)
```

---

## Further reading

- [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/quantization.html)
- [Ultralytics YOLO export guide](https://docs.ultralytics.com/modes/export/)
- [NVIDIA TensorRT INT8 calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-calib)
