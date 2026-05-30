"""
utils/autoencoder.py — Lightweight trajectory autoencoder for anomaly detection.

A self-supervised approach to behavioural intrusion detection: train a
tiny autoencoder (<2K parameters) on synthetic "normal" trajectory features,
then flag real trajectories with high reconstruction error as anomalies.

This replaces (or augments) the five hand-crafted geometric IDS rules with
a learned model that can discover novel threat patterns the rules miss.

Architecture:
    Input (8-d) → Encoder (32→16→4) → Bottleneck (4-d)
                → Decoder (4→16→32) → Reconstruction (8-d)

    Total params: ~1,600  |  Inference: <0.1 ms on CPU  |  ONNX exportable

Training is self-supervised — only "normal" data is needed (no threat
labels).  Synthetic normal trajectories are generated in train_autoencoder.py
or real trajectories from the Phase 2 trajectory exporter can be used.

Usage:
    from utils.autoencoder import TrajectoryAutoencoder, AnomalyScorer

    model = TrajectoryAutoencoder()
    scorer = AnomalyScorer(model, threshold=0.05)

    features = extract_features(centroid, bbox, kp, prev_centroid, frame_shape)
    score = scorer.score(features)
    if scorer.is_anomalous(features):
        print("Anomaly detected!")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_DIM = 8  # must match the autoencoder input size


def extract_features(
    centroid: tuple[float, float],
    bbox: tuple[int, int, int, int],
    kp_confidences: np.ndarray,               # shape (17,) — keypoint confs
    prev_centroid: Optional[tuple[float, float]],
    gait_angle_deg: Optional[float],
    frame_shape: tuple[int, int],              # (height, width)
) -> np.ndarray:
    """
    Convert per-track per-frame state into an 8-d normalised feature vector.

    Features:
      0. centroid_x_norm    — [0, 1]
      1. centroid_y_norm    — [0, 1]
      2. velocity_x         — pixel delta since last frame (0 if first)
      3. velocity_y
      4. bbox_area_norm     — [0, 1]  (relative to frame area)
      5. bbox_aspect_ratio  — w / h
      6. mean_kp_conf       — [0, 1]
      7. gait_angle_norm    — [0, 1]  (0 if unknown)

    Args:
        centroid:       (cx, cy) in pixel coords.
        bbox:           (x1, y1, x2, y2) in pixel coords.
        kp_confidences: Array of 17 keypoint confidence values.
        prev_centroid:  Centroid from previous frame (None on first frame).
        gait_angle_deg: Hip-shoulder lateral angle or None.
        frame_shape:    (height, width) of the video frame.

    Returns:
        Float32 array of shape (8,).
    """
    h, w = frame_shape
    frame_area = float(w * h)

    # Normalised centroid
    cx_norm = centroid[0] / max(w, 1.0)
    cy_norm = centroid[1] / max(h, 1.0)

    # Velocity
    if prev_centroid is not None:
        vx = centroid[0] - prev_centroid[0]
        vy = centroid[1] - prev_centroid[1]
    else:
        vx = 0.0
        vy = 0.0

    # Bbox features
    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, float(bx2 - bx1))
    bh = max(1.0, float(by2 - by1))
    area_norm = (bw * bh) / max(frame_area, 1.0)
    aspect = bw / bh

    # Keypoint confidence
    mean_conf = float(np.mean(kp_confidences)) if len(kp_confidences) > 0 else 0.0

    # Gait angle (normalised to [0, 1] — 90° is maximum plausible)
    gait_norm = 0.0
    if gait_angle_deg is not None:
        gait_norm = min(gait_angle_deg / 90.0, 1.0)

    return np.array(
        [cx_norm, cy_norm, vx, vy, area_norm, aspect, mean_conf, gait_norm],
        dtype=np.float32,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Autoencoder model  (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryAutoencoder:
    """
    Lightweight autoencoder for trajectory anomaly detection.

    Wraps a small PyTorch MLP with a 4-dim bottleneck.  Trained
    self-supervised on normal trajectories; anomalies produce high
    reconstruction error.

    This is a pure-numpy wrapper that can optionally load a PyTorch
    state dict or run via ONNX.  The actual training happens in
    train_autoencoder.py and the exported ONNX model is used at
    inference time in the pipeline.
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        torch_state_path: Optional[str] = None,
    ) -> None:
        """
        Load the autoencoder from ONNX or PyTorch state dict.

        Args:
            onnx_path:        Path to exported ONNX model.
            torch_state_path: Path to PyTorch state dict (.pth).
        """
        self._session = None
        self._torch_model = None

        if onnx_path and Path(onnx_path).exists():
            self._load_onnx(onnx_path)
        elif torch_state_path and Path(torch_state_path).exists():
            self._load_torch(torch_state_path)
        else:
            # Untrained — will produce random reconstruction errors.
            # Call train_autoencoder.py first.
            self._torch_model = _build_torch_autoencoder()

    def _load_onnx(self, path: str) -> None:
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

    def _load_torch(self, path: str) -> None:
        import torch
        self._torch_model = _build_torch_autoencoder()
        self._torch_model.load_state_dict(
            torch.load(path, map_location="cpu")
        )
        self._torch_model.eval()

    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        """
        Reconstruct a batch of feature vectors through the autoencoder.

        Args:
            features: (N, 8) float32 array of trajectory feature vectors.

        Returns:
            (N, 8) float32 array of reconstructed features.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._session is not None:
            input_name = self._session.get_inputs()[0].name
            out = self._session.run(None, {input_name: features.astype(np.float32)})
            return out[0]

        if self._torch_model is not None:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(features.astype(np.float32))
                recon = self._torch_model(t)
                return recon.numpy()

        raise RuntimeError("No model loaded. Run train_autoencoder.py first.")

    def reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """
        Compute per-sample MSE reconstruction error.

        Args:
            features: (N, 8) float32 array.

        Returns:
            (N,) float32 array of reconstruction errors.
        """
        recon = self.reconstruct(features)
        return np.mean((features - recon) ** 2, axis=1)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None or self._torch_model is not None


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly scorer
# ──────────────────────────────────────────────────────────────────────────────

class AnomalyScorer:
    """
    Threshold-based anomaly detector wrapping the autoencoder.

    Computes reconstruction error and compares against a percentile
    threshold determined during training.
    """

    def __init__(
        self,
        autoencoder: TrajectoryAutoencoder,
        threshold: float = 0.05,
    ) -> None:
        self._ae = autoencoder
        self._threshold = threshold

    def score(self, features: np.ndarray) -> float:
        """
        Compute anomaly score for a single feature vector.

        Args:
            features: (8,) float32 feature vector.

        Returns:
            Anomaly score (reconstruction error), higher = more anomalous.
        """
        err = self._ae.reconstruction_error(features.reshape(1, -1))
        return float(err[0])

    def is_anomalous(self, features: np.ndarray) -> bool:
        """Return True if reconstruction error exceeds threshold."""
        return self.score(features) > self._threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value


# ──────────────────────────────────────────────────────────────────────────────
# Internal: PyTorch model builder  (used during training)
# ──────────────────────────────────────────────────────────────────────────────

def _build_torch_autoencoder():
    """Build the PyTorch autoencoder module (requires torch)."""
    import torch
    import torch.nn as nn

    class _AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 4),    # bottleneck
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return _AutoEncoder()


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities  (used by train_autoencoder.py)
# ──────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    model,
    dataloader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """
    Train the autoencoder on a DataLoader of normal trajectory features.

    Args:
        model:      PyTorch nn.Module (from _build_torch_autoencoder).
        dataloader: DataLoader yielding (batch, 8) tensors.
        epochs:     Number of training epochs.
        lr:         Learning rate for Adam.
        device:     'cpu' or 'cuda'.
        verbose:    Print progress per epoch.

    Returns:
        List of per-epoch average losses.
    """
    import torch

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}")

    return losses


def compute_anomaly_threshold(
    model,
    dataloader,
    percentile: float = 95.0,
    device: str = "cpu",
) -> float:
    """
    Compute the anomaly threshold as a percentile of reconstruction
    errors on a validation set of normal trajectories.

    Args:
        model:      Trained autoencoder.
        dataloader: DataLoader of normal validation features.
        percentile: Percentile to use as threshold (default 95th).
        device:     'cpu' or 'cuda'.

    Returns:
        Reconstruction error threshold.
    """
    import torch

    model = model.to(device)
    model.eval()
    all_errors: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon = model(batch)
            err = torch.mean((batch - recon) ** 2, dim=1)
            all_errors.extend(err.cpu().tolist())

    threshold = float(np.percentile(all_errors, percentile))
    print(
        f"[autoencoder] Anomaly threshold: {threshold:.6f}  "
        f"(p{percentile:.0f} of {len(all_errors)} normal samples)"
    )
    return threshold


def export_to_onnx(model, output_path: str, device: str = "cpu") -> None:
    """
    Export the trained autoencoder to ONNX for edge inference.

    Args:
        model:       Trained PyTorch autoencoder.
        output_path: Destination .onnx file path.
        device:      'cpu' or 'cuda'.
    """
    import torch

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    model.eval()

    dummy = torch.randn(1, 8, device=device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["features"],
        output_names=["reconstruction"],
        opset_version=17,
        dynamic_axes={
            "features": {0: "batch"},
            "reconstruction": {0: "batch"},
        },
    )
    print(f"[autoencoder] Exported to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation  (for training without collected data)
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_trajectories(
    n_samples: int = 10000,
    frame_shape: tuple[int, int] = (480, 640),
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    Generate synthetic "normal" trajectory feature vectors for training.

    Simulates people walking at realistic speeds with natural motion
    patterns.  Each sample is an independent frame — the autoencoder
    learns a per-frame representation, not sequence modelling.

    Args:
        n_samples:   Number of feature vectors to generate.
        frame_shape: (height, width) of the virtual scene.
        noise_std:   Gaussian noise standard deviation added to features.

    Returns:
        (n_samples, 8) float32 array of feature vectors.
    """
    rng = np.random.RandomState(42)
    h, w = frame_shape
    frame_area = float(w * h)
    samples: list[np.ndarray] = []

    # Simulate multiple short trajectories (walks) and sample frames from them.
    n_trajectories = max(1, n_samples // 50)
    samples_per_traj = n_samples // n_trajectories

    for _ in range(n_trajectories):
        # Random start position — bias toward centre (people walk through scenes)
        cx = rng.uniform(w * 0.1, w * 0.9)
        cy = rng.uniform(h * 0.2, h * 0.8)

        # Random velocity (realistic walking speed: 50-200 px/s at 30fps = 1.7-6.7 px/frame)
        vx = rng.uniform(-5.0, 5.0)
        vy = rng.uniform(-5.0, 5.0)

        # Random bbox size (person at typical surveillance distance)
        bw = rng.uniform(30, 100)
        bh = rng.uniform(80, 200)
        aspect = bw / bh

        for _ in range(samples_per_traj):
            # Add slight random acceleration
            vx += rng.uniform(-0.5, 0.5)
            vy += rng.uniform(-0.5, 0.5)
            # Speed limit
            speed = np.sqrt(vx * vx + vy * vy)
            if speed > 8.0:
                vx *= 8.0 / speed
                vy *= 8.0 / speed

            cx += vx
            cy += vy

            # Bounce off frame edges
            if cx < 10:
                cx = 10
                vx = abs(vx)
            elif cx > w - 10:
                cx = w - 10
                vx = -abs(vx)
            if cy < 10:
                cy = 10
                vy = abs(vy)
            elif cy > h - 10:
                cy = h - 10
                vy = -abs(vy)

            # Slightly vary bbox size (perspective change)
            bw += rng.uniform(-1, 1)
            bh += rng.uniform(-2, 2)
            bw = max(20, min(bw, 120))
            bh = max(60, min(bh, 250))
            area_norm = (bw * bh) / frame_area
            aspect = bw / bh

            # Build feature vector
            feat = np.array([
                cx / w,                    # centroid_x_norm
                cy / h,                    # centroid_y_norm
                vx * 0.1,                  # velocity_x (scaled down)
                vy * 0.1,                  # velocity_y (scaled down)
                area_norm,
                aspect * 0.5,              # aspect_ratio (scaled)
                rng.uniform(0.6, 0.95),    # mean_kp_conf (normal range)
                rng.uniform(0.1, 0.35),    # gait_angle_norm (~10-35°)
            ], dtype=np.float32)

            # Add Gaussian noise
            feat += rng.normal(0, noise_std, size=feat.shape)

            samples.append(feat)

    # Trim to exact count
    result = np.array(samples[:n_samples], dtype=np.float32)
    return result
