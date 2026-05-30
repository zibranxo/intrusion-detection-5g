# Improvement Plan — 5G MEC IDS Pipeline

---

## Critical Review

### Architectural Issues

**1. Import structure prevents execution.** The pipeline imports from `utils.draw` and `utils.logger` but no `utils/` package exists. This is a one-line fix but a showstopper that means the code has never been run. This also suggests the code was refactored from a different directory layout without updating imports — a common artifact of solo research projects where the author mentally held the intended structure.

**2. Optical flow is applied globally, not per-person.** The current approach extracts Shi-Tomasi corners from all bounding boxes, pools them, runs one `calcOpticalFlowPyrLK`, computes a single median (dx, dy), and shifts every bbox by that vector. This is incorrect for multi-person scenes. It only works when everyone moves in the same direction (e.g., a crowd walking uniformly). When two people diverge, the median is a compromise that fits neither. When one person exits the frame, the remaining flow is contaminated. The fix — running flow independently per bbox — is straightforward and would dramatically improve propagation accuracy.

**3. Optical flow failure silently freezes bboxes.** When `_propagate_bboxes` returns `None` (< 2 good corners), `AltFrameInference.process()` falls through to `return self._last_kps, self._last_bboxes, False`. The bboxes don't move. The IDS loitering rule sees zero centroid movement and starts counting stationary frames — generating false positive LOITERING alerts on the next full-inference frame.

**4. ONNX INT8 is not GPU-accelerated.** The `build_ort_session` function correctly requests `CUDAExecutionProvider`, but ONNX Runtime's CUDA EP has limited support for INT8 quantized ops. In practice, many INT8 ops fall back to CPU. For production GPU INT8 inference, TensorRT is the standard. The claim of "~7.9ms INT8 inference" in the README is aspirational — it would require TensorRT, not ONNX Runtime.

**5. No real 5G integration.** The README frames this as a "5G MEC IDS" with URLLC uplink, but there is zero networking code. No packet serialization, no bandwidth model, no latency simulation, no gNB/UPF topology. The project is a local computer vision pipeline with a 5G narrative layered on top. This is fine for a CV internship but limits the project's uniqueness — without networking, it's just another YOLO + DeepSORT + rules pipeline.

### Performance Bottlenecks

**6. O(n²) centroid matching.** The pipeline manually matches DeepSORT track centroids to detection centroids using a double loop with `<80px` radius threshold. DeepSORT already performs feature-vector matching internally. The manual re-matching is redundant and expensive at high person counts.

**7. Optical flow cost is non-trivial.** Shi-Tomasi corner detection + pyramid LK on every other frame costs ~1–3ms. With INT8 inference at ~7.9ms, the net saving from alternate-frame is ~3–4ms per frame pair — about a 25% reduction, not the "halving" implied. For FP16 at ~11.4ms, the saving is ~3–4ms — about 17%. The technique is sound but the overhead is underreported.

**8. No GPU preprocessing.** Resize + BGR→RGB + HWC→NCHW + normalize are all CPU numpy operations. On a GPU pipeline, these should be CUDA kernels or PyTorch tensor operations on the GPU. At high throughput, this CPU-GPU synchronization becomes a bottleneck.

### Scalability Concerns

**9. Single-stream only.** A real MEC node handles 4–16 cameras. This pipeline handles one. No multi-stream architecture, no shared model across streams, no stream multiplexing.

**10. No frame-dropping strategy.** If inference falls behind real-time (e.g., the model takes 30ms), the pipeline has no mechanism to drop frames and catch up. It will accumulate latency indefinitely.

### Reliability Concerns

**11. No model warm-up.** The first 5–10 inferences on CUDA are always slower (CUDA kernel compilation, memory allocation). The benchmark runs 300 frames but doesn't discard warm-up samples. P95 numbers are inflated if warm-up is included.

**12. No graceful degradation on source failure.** `cv2.VideoCapture` errors cause unhandled exceptions. For a real deployment, this would crash the edge node.

### Code Quality Concerns

**13. Duplicated constants** (`SKELETON_EDGES`, `INPUT_SIZE`, `TARGET_LATENCY_MS` appear in 2+ files).
**14. No tests exist** despite pytest in requirements.
**15. No configuration file** — all hyperparameters are hardcoded module-level constants.

---

## Highest-Impact Improvements

Ranked by impact-to-effort ratio. Each includes impact, complexity, tradeoffs, and suitability category.

---

### #1: Fix Import Structure (Make the Pipeline Runnable)

**What:** Create a `utils/` package or change imports to reference root-level modules. I recommend moving `draw.py` and `logger.py` into `utils/__init__.py` equivalents (or `utils/draw.py` and `utils/logger.py`) since the code is already written against that structure.

**Why it matters:** The pipeline literally cannot run. This is blocking all validation, benchmarking, and demonstration. No improvement matters until this is fixed.

**Expected impact:** Unblocks all further work.

**Implementation complexity:** Trivial. Create `utils/` directory, move two files, add `__init__.py`.

**Tradeoffs:** None. The alternative (changing imports to root-level) is equally valid but diverges from the intended architecture the original author clearly had in mind.

**Suitability:** Research, Production, Resume — all require working code.

---

### #2: Per-Person Optical Flow (Fix the Single-Vector Bug)

**What:** Rewrite `AltFrameInference._propagate_bboxes` to compute per-bbox flow vectors instead of one global median. Track corners per bbox, compute `cv2.calcOpticalFlowPyrLK` independently for each bbox, propagate each bbox with its own (dx, dy).

**Why it matters:** This is the central mechanical flaw in the alternate-frame strategy. Without per-person flow, the system generates wrong bboxes 50% of the time in multi-person scenes, which cascades into wrong track associations and false IDS alerts. The entire alternate-frame strategy's correctness depends on this.

**Expected impact:** Drastically reduces bbox drift on intermediate frames. Prevents false loitering signals. Makes the alternate-frame approach actually viable for multi-person surveillance.

**Implementation complexity:** Moderate (1–2 hours). The key change is tracking corners as `dict[int, np.ndarray]` keyed by bbox index, running LK per group, and computing per-group flow vectors. Also need to handle the case where a person's corners are lost (fall back to global median for that person, or mark bbox as stale).

**Tradeoffs:** Per-bbox LK calls are more expensive than one global call. With 5–10 people, the flow computation goes from ~1ms to ~3–5ms. This partially offsets the alternate-frame savings. Mitigation: skip flow if the person count exceeds a threshold and fall back to full inference on those frames.

**Suitability:** Research value (the technique itself is interesting for edge deployment papers), Production value (necessary for correctness), Resume value (shows understanding of CV fundamentals).

---

### #3: Add Proper Frame-Dropping and Graceful Degradation

**What:** Add a watchdog timer to the pipeline loop. If `time.perf_counter() - frame_start > 1/fps`, skip the display/render stage and proceed to the next frame. Also add a back-pressure mechanism: if the inference queue exceeds 2 frames of latency, drop incoming frames until the pipeline catches up.

**Why it matters:** A real camera produces frames at a fixed rate (30fps = 33.3ms period). If the pipeline takes 35ms on a frame, it must either drop the next frame or accumulate unbounded latency. This is the difference between a research demo that "runs" and a production system that stays within latency SLA.

**Expected impact:** Guarantees the pipeline never exceeds the latency budget over extended runs. Prevents the "drift" where cumulative latency grows from 25ms to 200ms after 100 frames.

**Implementation complexity:** Low (30 minutes). A simple skip condition in the main loop, using `time.perf_counter()` and a configurable `max_frame_latency_ms`.

**Tradeoffs:** Dropped frames mean lost IDS coverage for those frames. In practice, at 30fps, dropping every 10th frame is invisible to the IDS (most threat patterns evolve over seconds, not frames). The tradeoff is frame coverage vs. latency guarantee — latency guarantee is the right priority for URLLC.

**Suitability:** Production value (essential), Resume value (shows real-time systems thinking).

---

### #4: Add GPU-Accelerated Preprocessing

**What:** Replace numpy-based preprocessing with PyTorch tensor operations on GPU. Use `torchvision.transforms.functional.resize` and tensor-based normalization, or use CUDA streams to overlap preprocessing with the previous frame's inference.

**Why it matters:** The current preprocessing path (CPU resize → numpy transpose → CPU normalize → GPU transfer) creates a GPU stall on every frame. With CUDA preprocessing, the GPU can process frame N+1 while still finishing inference on frame N. At 30fps, saving even 1–2ms is meaningful within a 25ms budget.

**Expected impact:** 1–3ms reduction in preprocessing latency per frame, plus elimination of the GPU starvation gap between frames.

**Implementation complexity:** Moderate (2–3 hours). Requires careful CUDA stream management or using PyTorch's async copy + tensor preprocessing pipeline.

**Tradeoffs:** Adds PyTorch dependency for the preprocessing path (currently it's pure OpenCV/numpy). This is acceptable since PyTorch is already a dependency. For the INT8 ONNX path, the tensor still needs to convert back to numpy — but the conversion can be scheduled earlier in the pipeline.

**Suitability:** Production value, Resume value.

---

### #5: Replace ONNX INT8 with TensorRT

**What:** Export the FP32 ONNX model to TensorRT with INT8 calibration, replacing ONNX Runtime for GPU inference. TensorRT provides real GPU INT8 acceleration with kernel fusion, layer fusion, and optimized memory layout.

**Why it matters:** The current "INT8" path likely runs on CPU through ONNX Runtime. TensorRT INT8 on CUDA would give 2–3× the throughput of the ONNX Runtime INT8 path and is the industry standard for edge GPU inference. Without this, the INT8 numbers in the README are unachievable on GPU.

**Expected impact:** Real INT8 GPU inference at ~4–6ms (vs the stated ~7.9ms ONNX RT INT8 or ~11.4ms FP16). This makes the sub-25ms target achievable with significant headroom.

**Implementation complexity:** High (4–6 hours). TensorRT has a steep learning curve, requires building a TensorRT engine (min/max calibration, builder config, serialization), and the API changes between versions. However, Ultralytics models have well-documented TensorRT export paths.

**Tradeoffs:** TensorRT engines are hardware-specific — you can't move them between GPU architectures. ONNX Runtime is more portable. For a research project, this is acceptable. The TensorRT engine path adds a third inference backend (PyTorch, ONNX RT, TensorRT) to maintain.

**Suitability:** Production value (industry standard), Resume value (TensorRT experience is highly sought-after).

---

### #6: Eliminate O(n²) Centroid Matching

**What:** Instead of manually re-matching DeepSORT track centroids to detection centroids via double loop, use the track's `to_tlbr()` bbox directly and associate keypoints by IoU with the detection that spawned the track (DeepSORT already returns detection-to-track associations internally via the `track.det_class` and track metadata). If DeepSORT's internal association isn't exposed, fall back to a spatial hash grid or kd-tree for O(n log n) association.

**Why it matters:** With 20+ people, the double loop runs ~400 distance calculations per frame. Each is cheap (~microseconds) but they add up. More importantly, this is fragile: two people whose bboxes overlap by <80px (common in crowds) will have their keypoints swapped, causing the IDS to analyze the wrong skeleton.

**Expected impact:** ~0.5–1ms savings at high person counts. More importantly, eliminates keypoint-swapping bugs in crowded scenes.

**Implementation complexity:** Low (1 hour). The simplest fix: use the DeepSORT track's detection confidence and class to infer which detection it was associated with, rather than re-matching by distance.

**Tradeoffs:** None. The current approach is strictly worse than using DeepSORT's built-in associations.

**Suitability:** Production value (correctness), Resume value (attention to algorithmic detail).

---

### #7: Create a Centralized Configuration System

**What:** Create a `config.py` (or `config.yaml`) with all tunable parameters: `INPUT_SIZE`, `TARGET_LATENCY_MS`, `ALT_FRAME_INTERVAL`, `LOITER_FRAMES`, `CROWD_RADIUS_PX`, confidence thresholds, etc. Import from a single source of truth.

**Why it matters:** Currently, changing `INPUT_SIZE` from 640×480 to 320×240 requires editing 3 files. Tuning IDS thresholds requires finding scattered constants in `ids_layer.py`. A single config source enables rapid experimentation and prevents configuration drift.

**Expected impact:** Eliminates a whole class of bugs (divergent constants). Enables systematic hyperparameter sweeps.

**Implementation complexity:** Trivial (30 minutes). A dataclass or YAML file.

**Tradeoffs:** None. Pure win.

**Suitability:** Research value (enables experiments), Production value (configuration management).

---

### #8: Write Integration Tests

**What:** Add a `tests/` directory with:
- `test_ids_layer.py` — verify each of 5 rules with synthetic inputs (known centroids, known keypoint arrays)
- `test_alt_frame.py` — verify that a bbox moving at a known speed is correctly propagated
- `test_export_quantize.py` — verify that exported models have expected shapes and dtypes
- `test_pipeline.py` — end-to-end smoke test with a 10-frame synthetic video

**Why it matters:** The project has zero tests. Every change risks breaking something silently. The IDS layer is particularly testable — it's pure functions operating on geometry, perfect for unit testing. Tests are also a strong signal for resume review: researchers who test their code are rare and valued.

**Expected impact:** Confidence to refactor. Catches regressions. Demonstrates engineering maturity.

**Implementation complexity:** Moderate (3–4 hours for a good test suite covering the critical paths).

**Tradeoffs:** Time investment that doesn't directly improve the pipeline's capabilities. The payoff is in maintainability and correctness assurance during the improvements above.

**Suitability:** Production value, Resume value.

---

## Research Opportunities

These are ideas that could plausibly become a paper, conference poster, internship talking point, or strong portfolio differentiator. None are incremental; each represents a genuine research contribution at the intersection of edge ML and computer vision.

---

### Research #1: Self-Supervised Trajectory Anomaly Detection for Edge Deployment

**The idea:** Replace the 5 hand-crafted geometric IDS rules with a lightweight autoencoder or normalizing flow trained on normal trajectory embeddings. Train only on "normal" trajectories (people walking, standing, moving through the scene). At inference time, flag trajectories with high reconstruction error as anomalies. This is self-supervised — no labeled threat data needed.

**Why it's a paper:** Anomaly detection on trajectories usually runs on server GPUs with complex models (transformers, diffusion). Showing that a tiny autoencoder (<100K params) running in <1ms can outperform hand-crafted rules on real surveillance data is a genuine contribution. The constraint (must run within the 25ms edge budget) makes it interesting.

**What exists already:** The IDS layer already computes and stores per-track centroids, keypoint confidences, and gait angles. These are trajectory features. The `TrackState` class already maintains a rolling history (`deque` of 120 frames). The data structure for trajectory-based learning is already in place.

**Implementation path:**
1. Extract trajectory features from the existing pipeline: centroid velocity, centroid acceleration, bbox aspect ratio change, mean keypoint confidence, gait angle, inter-person distances
2. Train a small autoencoder (3-layer MLP, ~50K params) on "normal" walking trajectories from a public dataset (MOT17, MOT20, or the project's own footage)
3. Replace the 5 hard-coded rules with reconstruction error thresholding
4. Compare false-positive rate, detection latency, and computational cost against the geometric rules
5. Export the autoencoder to ONNX for inference in the pipeline

**Research questions this addresses:**
- Can learned trajectory embeddings generalize across camera viewpoints better than hand-tuned geometric thresholds?
- What is the minimum model size needed to match rule-based IDS accuracy?
- Does self-supervised pretraining on unlabeled trajectory data transfer to novel scenes?

**Effort:** High (2–4 weeks). Requires data collection/labeling, model training, integration, and evaluation.

**Suitability:** Research value (paper potential), Resume value (ML + CV + edge deployment in one project).

---

### Research #2: Lightweight Neural Optical Flow for Edge Pose Propagation

**The idea:** Replace Lucas-Kanade optical flow with a distilled/tiny neural flow network that runs in <2ms. Options: a heavily pruned RAFT, FlowNet-S variant, or a custom 2-frame convolutional flow estimator with <1M parameters. The neural flow would handle occlusion better than LK and could be co-designed with the pose model.

**Why it's a paper:** Neural optical flow is standard at CVPR/ECCV, but always at high compute (RAFT takes 100ms+). Showing that a distilled flow network running at edge latency can replace classical LK for pose propagation is a novel application. The paper is about the distillation methodology + the application, not the flow architecture itself.

**Implementation path:**
1. Record paired frames + ground-truth bbox displacements from full YOLO inference (run full inference on every frame as ground truth)
2. Train a tiny CNN (3–5 conv layers) to predict per-bbox flow from two consecutive frames
3. Compare against LK in terms of bbox IoU, latency, and robustness to occlusion
4. Export to ONNX, run at edge

**Key advantage the project already has:** The `AltFrameInference` class is a clean abstraction. You can swap the flow method without touching the rest of the pipeline. The benchmarking harness (`benchmark.py`) already measures per-stage latency. This is an ideal experimental setup.

**Effort:** Very high (4–6 weeks). Model design, training, distillation, and thorough evaluation.

**Suitability:** Research value (strong paper potential), Resume value (deep learning + classical CV).

---

### Research #3: Multi-Camera Person Re-Identification Under Edge Latency Constraints

**The idea:** Extend the pipeline to handle 2–4 cameras simultaneously, with cross-camera track association. The research contribution is not the re-ID model (use a pretrained OSNet or similar) but the architecture for running multi-camera re-ID within the unified latency budget — deciding when to run re-ID, which camera pairs to associate, and how to schedule inference to stay under budget.

**Why it's a paper:** Multi-camera tracking is well-studied. Multi-camera tracking **under a hard real-time edge compute budget** is not. The scheduling problem is novel: you have N cameras, M model variants, and a total latency budget B. Which camera gets full vs. alternate-frame inference? When do you run cross-camera re-ID? This becomes a constrained optimization problem.

**Implementation path:**
1. Instantiate N pipeline instances (one per camera), sharing the pose model
2. Add a cross-camera re-ID module (OSNet-lite or a learned appearance embedding)
3. Implement a simple scheduler: round-robin full inference across cameras, run re-ID on a slower cadence (every 10 frames)
4. Extend `ThreatEvent` with `camera_id` field for multi-camera threat correlation
5. Evaluate tracking consistency across camera handoffs

**Effort:** Very high (6–8 weeks). Requires access to multi-camera surveillance data and significant systems engineering.

**Suitability:** Research value (novel scheduling problem), Resume value (large-scale systems + CV), Production value (MEC nodes serve multiple cameras).

---

## Hidden Opportunities

These are non-obvious directions or unique advantages the project already possesses but is not exploiting. They require minimal new code but significantly amplify the project's value.

---

### Hidden #1: This Is Already a General Edge Inference Benchmarking Framework

The `benchmark.py` module is unusually well-designed for a research project. It has:
- Per-stage timing with CUDA events (not just wall-clock)
- P50/P95/P99 percentile computation
- Structured JSON output
- A `VariantResult` abstraction that can represent any model variant
- Backend-agnostic design

With minimal refactoring, `benchmark.py` could become a general-purpose edge inference benchmark harness that compares **any** model across **any** quantization scheme. Add support for:
- Plug-in model definitions (a registry pattern)
- Multiple input resolutions
- Batch size sweeps
- Power measurement (nvidia-smi / powermetrics integration)

**What this becomes:** An open-source tool that other researchers use to benchmark their edge models. This gets citations, GitHub stars, and demonstrates systems thinking far beyond a typical summer internship project.

**Effort:** Moderate (1 week of refactoring + documentation).

---

### Hidden #2: Add a Simulated 5G Link (Without Real Hardware)

The project's unique angle is "5G MEC." Currently that's only in the README. Add a **simulated 5G link layer** between the pipeline and a mock core network:

- A `URLLCTransport` class that serializes `ThreatEvent` objects, applies a configurable latency distribution (1–5ms URLLC budget), adds packet loss (0.001%), and delivers to a mock "core network" receiver
- A bandwidth model that limits how many threat events per second can be uplinked
- A simple dashboard that shows threats arriving at the "core" after the simulated network delay

**What this becomes:** A genuinely unique project combining CV + edge ML + 5G networking. Almost nobody in CV internships builds this. It turns "just another YOLO + DeepSORT pipeline" into "an end-to-end 5G edge security system with simulated network integration."

**Effort:** Low (1–2 days). Pure Python, no real hardware. The `ThreatEvent` dataclass is already designed to be serializable.

---

### Hidden #3: The TrackState History Is a Trajectory Dataset Generator

The `TrackState` class stores 120 frames of centroid history per track. The `ThreatLogger` writes every event with metadata. Together, these produce labeled trajectory data **for free** during any pipeline run.

Add a `--save-trajectories` flag that dumps the track state history + threat labels to a structured format (Parquet or HDF5). After running the pipeline on a few hours of surveillance footage, you have a labeled trajectory anomaly dataset — exactly what Research #1 needs.

**What this becomes:** A data flywheel. Each run of the pipeline produces training data for the next iteration. This is how production ML systems work and demonstrates data engineering maturity.

**Effort:** Low (1–2 hours). Serialize the `TrackState.centroids` deque with timestamps and threat labels.

---

### Hidden #4: The Lucid-Comment Style IS a Feature

The codebase has unusually thorough inline documentation — ASCII art architecture diagrams, "Apple / Core ML reframe" sections, explicit rationale for every design choice. This style is rare in research code.

Lean into this. The codebase is already 80% of the way to being a **tutorial-style educational resource** for edge ML inference. Add a `docs/` directory with:
- `docs/quantization-guide.md` — explains the INT8/FP16 methodology with the existing code as reference
- `docs/alternate-frame.md` — explains the optical flow strategy with diagrams
- `docs/benchmarking.md` — how to interpret P95 vs mean latency

**What this becomes:** A portfolio piece that demonstrates not just coding ability but technical communication — one of the most sought-after skills in ML engineering.

**Effort:** Moderate (2–3 days of writing). The content already exists in the comments and README.

---

## Roadmap

If I were the technical lead on this project, here are the next steps in priority order, with justification for each.

### Phase 1: Make It Run, Make It Correct (Week 1)

| Step | Effort | Rationale |
|---|---|---|
| Fix import structure (Plan #1) | 30 min | Unblock everything |
| Per-person optical flow (Plan #2) | 2 hrs | Fix the central algorithm bug |
| Run the pipeline end-to-end, collect real benchmarks | 1 hr | Replace README placeholder numbers with real data |
| Write integration tests for IDS layer (Plan #8) | 3 hrs | Establish correctness baseline before further changes |
| Centralize configuration (Plan #7) | 30 min | Prevent configuration drift as we add features |

**Why this order:** You can't improve what you can't measure. Getting the pipeline running with real numbers is the prerequisite for all further work. The optical flow fix is essential because incorrect bboxes corrupt every downstream measurement. Tests lock in correctness.

### Phase 2: Performance & Production Readiness (Week 2)

| Step | Effort | Rationale |
|---|---|---|
| GPU preprocessing (Plan #4) | 3 hrs | Latency headroom for Phase 3 features |
| Frame-dropping and graceful degradation (Plan #3) | 30 min | The system needs to run reliably for hours, not seconds |
| Fix centroid matching (Plan #6) | 1 hr | Correctness at high person counts |
| Hidden #2: simulated 5G link | 2 days | **This is the differentiator.** Turns a CV pipeline into a 5G project |
| Hidden #3: trajectory dataset export | 2 hrs | Builds the data flywheel for Research #1 |

**Why this order:** Phase 2 makes the system production-grade AND adds the unique 5G angle that distinguishes it from the thousands of other YOLO+DeepSORT projects on GitHub. The trajectory export is low-effort and immediately useful.

### Phase 3: Research Value (Weeks 3–4)

| Step | Effort | Rationale |
|---|---|---|
| Research #1: trajectory autoencoder IDS | 2 weeks | Highest value-to-effort ratio among the research ideas. Builds on infrastructure from Phase 2 |
| Hidden #1: generalize benchmark harness | 1 week | Tool-building that generates citations |
| Hidden #4: educational documentation | 3 days | Portfolio and visibility |

**Why this order:** Research #1 is the most achievable paper/capstone project. It's self-supervised (no labeling cost), the data infrastructure exists (Hidden #3), and the computational constraints (sub-2ms, <100K params) make it genuinely novel. The benchmarking framework and documentation amplify the project's reach.

### What NOT to do now

- **TensorRT integration (Plan #5):** Skip for now. ONNX Runtime INT8 is acceptable for the research phase. TensorRT is a production optimization that can be done later if the project transitions to deployment.
- **Neural optical flow (Research #2):** High effort, uncertain payoff. Do this only if Research #1 succeeds and the trajectory anomaly detection paper needs a companion contribution.
- **Multi-camera re-ID (Research #3):** Highest complexity. Only pursue if the project transitions to a funded research project with access to real multi-camera infrastructure.

### Strategic pivot opportunity

The strongest single move this project could make: **position it as an open-source edge ML benchmarking + IDS framework rather than a one-off pipeline.**

The code already has the architecture for this — the benchmarking harness, the model variant abstraction, the structured logging. Reframing from "my summer internship project" to "a framework other researchers use for edge inference benchmarking" multiplies the project's impact by 10× with relatively modest engineering effort. The 5G + IDS angle then becomes the flagship demo within that framework rather than the entire project.

This pivot also makes the resume narrative stronger: "I built a framework, not just a pipeline." That's the difference between "implemented YOLO + DeepSORT" and "designed a system for reproducible edge ML research."
