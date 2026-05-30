"""
utils/learned_ids.py — Learned anomaly detection augmenting geometric IDS rules.

Integrates the trajectory autoencoder (utils/autoencoder.py) into the
pipeline as an additional threat detector.  Operates alongside the five
geometric rules in IDSLayer — each track gets scored by both systems.

The autoencoder learns a compact representation of "normal" trajectories.
When a real trajectory deviates from the learned manifold, reconstruction
error spikes, producing a LEARNED_ANOMALY threat event.

This is Research #1 from plan.md: self-supervised trajectory anomaly
detection for edge deployment.

Usage:
    from utils.learned_ids import LearnedIDSLayer

    learned_ids = LearnedIDSLayer(onnx_path="models/trajectory_ae.onnx",
                                   threshold=0.05)
    # inside the frame loop:
    threats = learned_ids.score_frame(tracks, kp_by_id, frame_number, frame_shape)
"""

from pathlib import Path
from typing import Optional

import numpy as np

from config import INPUT_SIZE, TRAJECTORY_EXPORT_DIR
from utils.autoencoder import (
    AnomalyScorer,
    TrajectoryAutoencoder,
    extract_features,
)
from utils.ids_layer import ThreatEvent, ThreatType


class LearnedIDSLayer:
    """
    Learned anomaly detector wrapping the trajectory autoencoder.

    For each confirmed track, extracts an 8-d feature vector and
    scores it through the autoencoder.  Tracks with reconstruction
    error above the threshold generate LEARNED_ANOMALY threats.

    Maintains per-track state (previous centroid for velocity
    computation) across frames.
    """

    def __init__(
        self,
        onnx_path: str = "models/trajectory_ae.onnx",
        threshold: Optional[float] = None,
        alert_cooldown_frames: int = 30,
    ) -> None:
        """
        Initialise the learned IDS layer.

        Args:
            onnx_path:              Path to trained autoencoder ONNX model.
            threshold:              Anomaly threshold.  If None, loaded from
                                    {onnx_path}.threshold.txt.
            alert_cooldown_frames:  Min frames between repeat alerts per track.
        """
        self._cooldown = alert_cooldown_frames

        # Load threshold from companion file if not explicitly provided.
        if threshold is None:
            threshold_path = Path(onnx_path).with_suffix(".threshold.txt")
            if threshold_path.exists():
                threshold = float(threshold_path.read_text().strip())
            else:
                threshold = 0.05  # fallback

        # Load the autoencoder.
        ae = TrajectoryAutoencoder(onnx_path=onnx_path)
        if not ae.is_loaded:
            print(
                f"[learned_ids] WARNING: No autoencoder found at {onnx_path}.  "
                "Run train_autoencoder.py first.  Learned IDS will be inactive."
            )
        self._scorer = AnomalyScorer(ae, threshold=threshold)

        # Per-track state.
        self._prev_centroids: dict[int, tuple[float, float]] = {}
        self._alert_cooldowns: dict[int, dict[ThreatType, int]] = {}

        print(
            f"[learned_ids] Loaded  "
            f"threshold={threshold:.6f}  "
            f"onnx={onnx_path}"
        )

    @property
    def is_active(self) -> bool:
        return self._scorer is not None

    def score_frame(
        self,
        tracks: list,
        kp_by_id: dict[int, np.ndarray],
        frame_number: int,
        frame_shape: tuple[int, int] = (INPUT_SIZE[1], INPUT_SIZE[0]),
    ) -> list[ThreatEvent]:
        """
        Score every confirmed track through the autoencoder.

        Args:
            tracks:       List of confirmed DeepSORT Track objects.
            kp_by_id:     Mapping from track_id → (17, 3) keypoint array.
            frame_number: Current frame index.
            frame_shape:  (height, width) for feature normalisation.

        Returns:
            List of ThreatEvent objects for tracks flagged as anomalous.
        """
        import time

        if not self._scorer:
            return []

        threats: list[ThreatEvent] = []
        now = time.time()

        for track in tracks:
            tid = track.track_id
            bbox = tuple(map(int, track.to_tlbr()))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            kp = kp_by_id.get(tid)
            kp_confs = kp[:, 2] if kp is not None else np.zeros(17, dtype=np.float32)

            # Compute gait angle for the feature vector.
            gait = None
            if kp is not None:
                gait = self._compute_gait_angle(kp)

            prev = self._prev_centroids.get(tid)
            feat = extract_features(
                centroid=(cx, cy),
                bbox=bbox,
                kp_confidences=kp_confs,
                prev_centroid=prev,
                gait_angle_deg=gait,
                frame_shape=frame_shape,
            )

            self._prev_centroids[tid] = (cx, cy)

            # ── Anomaly check ─────────────────────────────────────────
            score = self._scorer.score(feat)
            if self._scorer.is_anomalous(feat):
                cooldowns = self._alert_cooldowns.setdefault(tid, {})
                cd = cooldowns.get(ThreatType.LEARNED_ANOMALY, 0)
                if cd == 0:
                    threats.append(ThreatEvent(
                        timestamp=now,
                        frame_number=frame_number,
                        track_id=tid,
                        threat_type=ThreatType.LEARNED_ANOMALY,
                        confidence=min(score / self._scorer.threshold * 0.5, 0.95),
                        bbox=bbox,
                        metadata={
                            "recon_error": round(float(score), 6),
                            "threshold": self._scorer.threshold,
                        },
                    ))
                    cooldowns[ThreatType.LEARNED_ANOMALY] = self._cooldown

            # Decrement cooldowns.
            for ttype in list(cooldowns.keys()):
                cooldowns[ttype] = max(0, cooldowns[ttype] - 1)

        # Clean up stale tracks.
        active_ids = {t.track_id for t in tracks}
        for tid in list(self._prev_centroids.keys()):
            if tid not in active_ids:
                del self._prev_centroids[tid]
        for tid in list(self._alert_cooldowns.keys()):
            if tid not in active_ids:
                del self._alert_cooldowns[tid]

        return threats

    @staticmethod
    def _compute_gait_angle(keypoints: np.ndarray) -> Optional[float]:
        """Compute hip-shoulder lateral angle (same logic as ids_layer.py)."""
        import math

        REQUIRED_CONF = 0.4
        if any(keypoints[i][2] < REQUIRED_CONF for i in (5, 6, 11, 12)):
            return None

        shoulder_mid = (
            (keypoints[5][0] + keypoints[6][0]) / 2,
            (keypoints[5][1] + keypoints[6][1]) / 2,
        )
        hip_mid = (
            (keypoints[11][0] + keypoints[12][0]) / 2,
            (keypoints[11][1] + keypoints[12][1]) / 2,
        )
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        if abs(dy) < 1e-6:
            return 90.0
        return abs(math.degrees(math.atan2(abs(dx), abs(dy))))
