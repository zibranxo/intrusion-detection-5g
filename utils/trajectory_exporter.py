"""
utils/trajectory_exporter.py
-----------------------------
Structured export of per-track trajectories from the IDS pipeline.

Every frame, the exporter records each confirmed track's centroid,
bounding box, keypoint confidence, and threat labels (if any).  The
output is a newline-delimited JSON file — one line per track per frame —
that can be ingested directly by trajectory-based anomaly detectors.

This turns the pipeline into a data flywheel: each run produces
labelled trajectory data suitable for training a learned IDS (Research
#1 in plan.md).

Usage:
    from utils.trajectory_exporter import TrajectoryExporter

    exporter = TrajectoryExporter("results/trajectories/session_01.jsonl")
    # inside the frame loop:
    exporter.record(frame_number, tracks, kp_by_id, active_threats)
    # on exit:
    exporter.close()
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from utils.ids_layer import ThreatEvent


class TrajectoryExporter:
    """
    Append-only JSONL logger for per-track trajectory features.

    Each line is a self-contained JSON record:

        {
          "frame": 847,
          "track_id": 3,
          "centroid": [142.5, 279.0],
          "bbox": [100, 88, 185, 471],
          "mean_kp_conf": 0.83,
          "n_low_conf_kp": 1,
          "gait_angle_deg": 22.3,
          "threats": ["LOITERING"]
        }

    Writes are buffered and flushed every `flush_every` frames to
    minimise I/O overhead on the hot path.
    """

    def __init__(
        self,
        path: str | Path,
        flush_every: int = 30,
        overwrite: bool = True,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._buffer: list[str] = []
        self._count = 0

        mode = "w" if overwrite else "a"
        self._fh = self._path.open(mode, encoding="utf-8", buffering=1)

        # Metadata header (comment line — ignored by JSONL parsers but
        # documents the session for downstream consumers).
        header = {
            "_comment": "Trajectory export — 5G MEC IDS pipeline",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": 1,
        }
        self._fh.write(json.dumps(header) + "\n")
        self._fh.flush()

    # ── Public API ────────────────────────────────────────────────────────

    def record(
        self,
        frame_number: int,
        tracks: list,                          # DeepSORT Track objects
        kp_by_id: dict[int, np.ndarray],       # track_id → (17,3) keypoints
        active_threats: list[ThreatEvent],
    ) -> None:
        """
        Record one frame's worth of trajectory features.

        Args:
            frame_number:   Current frame index.
            tracks:         List of confirmed DeepSORT Track objects.
            kp_by_id:       Mapping from track_id to (17, 3) keypoint array.
            active_threats: Threat events generated this frame.
        """
        from config import LOW_CONF_THRESHOLD

        # Build a quick lookup: track_id → set of threat types this frame.
        threat_by_track: dict[int, set[str]] = {}
        for t in active_threats:
            threat_by_track.setdefault(t.track_id, set()).add(
                t.threat_type.value
            )

        for track in tracks:
            tid = track.track_id
            bbox = tuple(map(int, track.to_tlbr()))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            kp = kp_by_id.get(tid)
            mean_conf   = float(np.mean(kp[:, 2])) if kp is not None else 0.0
            n_low_conf  = int(np.sum(kp[:, 2] < LOW_CONF_THRESHOLD)) if kp is not None else 0

            # Compute gait angle if keypoints are available.
            gait_angle = None
            if kp is not None:
                gait_angle = self._compute_gait_angle(kp)

            record = {
                "frame": frame_number,
                "track_id": tid,
                "centroid": [round(cx, 1), round(cy, 1)],
                "bbox": list(bbox),
                "mean_kp_conf": round(mean_conf, 3),
                "n_low_conf_kp": n_low_conf,
                "gait_angle_deg": round(gait_angle, 1) if gait_angle is not None else None,
                "threats": sorted(threat_by_track.get(tid, [])),
            }
            self._buffer.append(json.dumps(record))
            self._count += 1

        if self._count % self._flush_every == 0:
            self._flush()

    def close(self) -> None:
        """Flush remaining buffer and close file handle."""
        self._flush()
        self._fh.close()
        print(f"[trajectory] Export closed: {self._path}  ({self._count} records)")

    # ── Internals ─────────────────────────────────────────────────────────

    def _flush(self) -> None:
        if self._buffer:
            self._fh.write("\n".join(self._buffer) + "\n")
            self._fh.flush()
            self._buffer.clear()

    @staticmethod
    def _compute_gait_angle(keypoints: np.ndarray) -> Optional[float]:
        """
        Compute hip-shoulder lateral angle from a (17,3) keypoint array.

        Returns angle in degrees, or None if required keypoints are
        below confidence threshold.
        """
        import math

        REQUIRED_CONF = 0.4
        KP_L_SHOULDER = 5
        KP_R_SHOULDER = 6
        KP_L_HIP      = 11
        KP_R_HIP      = 12

        required = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
        if any(keypoints[i][2] < REQUIRED_CONF for i in required):
            return None

        shoulder_mid_x = (keypoints[KP_L_SHOULDER][0] + keypoints[KP_R_SHOULDER][0]) / 2
        shoulder_mid_y = (keypoints[KP_L_SHOULDER][1] + keypoints[KP_R_SHOULDER][1]) / 2
        hip_mid_x = (keypoints[KP_L_HIP][0] + keypoints[KP_R_HIP][0]) / 2
        hip_mid_y = (keypoints[KP_L_HIP][1] + keypoints[KP_R_HIP][1]) / 2

        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y

        if abs(dy) < 1e-6:
            return 90.0
        return abs(math.degrees(math.atan2(abs(dx), abs(dy))))

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "TrajectoryExporter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
