# Alternate-Frame Inference — Optical Flow Propagation

How the pipeline halves inference load while maintaining temporal coherence.

---

## The problem

Running YOLOv8n-pose on every frame is expensive. At 30fps, a 25ms budget leaves exactly 0ms for everything else (tracking, IDS, display). We need to reduce inference cost without losing tracking quality.

## The solution

Run full pose inference every **2nd frame**. On intermediate frames, propagate bounding boxes forward using sparse Lucas-Kanade optical flow. The key insight: human motion is smooth between consecutive frames (16ms at 60fps, 33ms at 30fps), so a bbox shift estimated from corner-point flow is accurate enough for the tracker.

This is the same approach Core ML uses in its `MLMultiArray` streaming camera pipeline on iOS/macOS.

---

## How it works

### Frame N (full inference)

1. YOLOv8n-pose detects persons and estimates 17-point skeletons
2. Shi-Tomasi corner detection extracts 8 trackable corner points **per person** inside each bounding box
3. The corners are stored for the next frame

### Frame N+1 (flow propagation)

1. Lucas-Kanade pyramid optical flow tracks each person's corners independently
2. A per-person median flow vector (dx, dy) is computed from the corner displacements
3. Each bounding box is shifted by its own flow vector
4. Keypoints from frame N are reused (no new pose estimation)

**Critical design decision (fixed in Phase 1):** We track corners **per person**, not globally. The original implementation pooled all corners and computed a single global median flow — this broke when two people moved in different directions. Now each person has their own corner set and their own flow vector.

---

## Configuration

All optical flow parameters live in `config.py`:

```python
ALT_FRAME_INTERVAL  = 2      # run full inference every N frames
FLOW_MAX_CORNERS    = 8      # Shi-Tomasi corners per bbox
FLOW_QUALITY_LEVEL  = 0.3    # corner quality threshold
FLOW_MIN_DISTANCE   = 7      # minimum pixel distance between corners
FLOW_WIN_SIZE       = (15, 15)  # LK search window
FLOW_MAX_LEVEL      = 2      # pyramid levels
FLOW_STALE_THRESHOLD = 3     # max consecutive failures before treating as lost
```

---

## Failure modes

### Person exits the frame

Corners move outside the image boundary → corner count drops below 2 → flow fails. The bbox stays at its last known position. DeepSORT's `max_age` parameter handles eventual track expiry.

### Occlusion

Another person walks in front → corners are occluded → flow fails. Same handling as above.

### Low-texture regions

A person wearing a plain dark coat has few trackable corners → Shi-Tomasi returns < 2 corners → bbox is not propagated (stays in place). This is safe but means the alternate-frame strategy provides no benefit for that person.

### Fast motion

If a person moves > 15 pixels between frames (the LK window size), flow tracking may lose accuracy. The pyramid LK (3 levels) handles moderate speeds, but sprinting (> 300 px/frame) will break tracking.

---

## Performance tradeoff

| Approach | Inference cost | Tracking quality |
|---|---|---|
| Every-frame inference | 2× (full cost every frame) | Best |
| Alternate-frame (global flow, buggy) | 1.5× (global flow is cheap but wrong) | Poor (multi-person) |
| Alternate-frame (per-person flow) | 1.5× (per-person flow costs ~1-3ms) | Good (temporal coherence preserved) |

The per-person flow approach achieves most of the savings of alternate-frame inference without the correctness issues of the global-flow approach.

---

## When to tune

| Symptom | Likely cause | Fix |
|---|---|---|
| Bboxes drift during turns | Flow can't track rotation | Increase `ALT_FRAME_INTERVAL` to 1 (disable) or use neural flow (Research #2) |
| Bboxes freeze on dark clothing | No trackable corners | Lower `FLOW_QUALITY_LEVEL` to 0.1 |
| Flow computation too slow (> 3ms) | Too many corners or people | Reduce `FLOW_MAX_CORNERS` to 4 |
| False loitering alerts | Frozen bboxes during flow failures | Reduce `FLOW_STALE_THRESHOLD` or tune IDS cooldown |

---

## Further reading

- [Lucas-Kanade 20 Years On: A Unifying Framework](https://link.springer.com/article/10.1023/B:VISI.0000011205.11775.fd) — Baker & Matthews, IJCV 2004
- [Pyramidal Implementation of the Lucas Kanade Feature Tracker](http://robots.stanford.edu/cs223b04/algo_tracking.pdf) — Bouguet, Intel 2001
- [Core ML Video](https://developer.apple.com/documentation/vision) — Apple's on-device camera pipeline (uses same alternate-frame strategy)
