"""
utils/logger.py
---------------
Structured logging for the 5G IDS pipeline.

Two responsibilities:
  1. ThreatLogger  — writes threat events as newline-delimited JSON
                     (one event per line, append-only, zero latency impact)
  2. MetricsLogger — accumulates per-frame latency and FPS samples,
                     flushes a summary JSON on close.

Both loggers are designed to add <0.1ms per frame — all I/O is buffered
and flushed periodically, not on every frame.

Usage:
    from utils.logger import ThreatLogger, MetricsLogger

    threat_log = ThreatLogger("results/threats.jsonl")
    metrics    = MetricsLogger("results/metrics.json")

    # in the frame loop:
    threat_log.log(threat_event)
    metrics.record(stage_times, fps)

    # on exit:
    threat_log.close()
    metrics.close()
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .ids_layer import ThreatEvent


# ──────────────────────────────────────────────────────────────────────────────
# Threat logger
# ──────────────────────────────────────────────────────────────────────────────

class ThreatLogger:
    """
    Append-only newline-delimited JSON log of ThreatEvent objects.

    Each line is a self-contained JSON object, making the log trivially
    parseable by any downstream SIEM or dashboard tool.

    Writes are buffered in memory and flushed to disk every `flush_every`
    events to avoid per-event fsync overhead.
    """

    def __init__(
        self,
        path: str,
        flush_every: int = 10,
        overwrite: bool = True,
    ) -> None:
        """
        Open the threat log file.

        Args:
            path:        Output file path (will be created if missing).
            flush_every: Flush buffer to disk every N events.
            overwrite:   If True, truncate existing file on open.
        """
        self._path       = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._buffer:    list[str] = []
        self._count      = 0

        mode = "w" if overwrite else "a"
        self._fh = self._path.open(mode, encoding="utf-8", buffering=1)

    def log(self, event: ThreatEvent) -> None:
        """
        Buffer one threat event for writing.

        Args:
            event: ThreatEvent from IDSLayer.update().
        """
        self._buffer.append(json.dumps(event.to_dict()))
        self._count += 1

        if self._count % self._flush_every == 0:
            self._flush()

    def _flush(self) -> None:
        """Write buffered events to disk."""
        if self._buffer:
            self._fh.write("\n".join(self._buffer) + "\n")
            self._fh.flush()
            self._buffer.clear()

    def close(self) -> None:
        """Flush remaining buffer and close file handle."""
        self._flush()
        self._fh.close()
        print(f"[logger] Threat log closed: {self._path}  ({self._count} events)")

    @property
    def event_count(self) -> int:
        """Total number of threat events logged so far."""
        return self._count

    def __enter__(self) -> "ThreatLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Metrics logger
# ──────────────────────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Accumulates per-frame stage latency samples and FPS readings.

    On close(), computes P50/P95/P99 statistics and writes a summary
    JSON to disk — the same format as benchmark.py output, so results
    can be compared directly.
    """

    def __init__(self, path: str, model_variant: str, backend: str) -> None:
        """
        Initialise the metrics logger.

        Args:
            path:          Output JSON path.
            model_variant: e.g. 'fp16'.
            backend:       e.g. 'cuda'.
        """
        self._path          = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._model_variant = model_variant
        self._backend       = backend
        self._stage_samples: defaultdict[str, list[float]] = defaultdict(list)
        self._fps_samples:   list[float] = []
        self._start_time:    float       = time.time()

    def record(
        self,
        stage_times: dict[str, float],
        fps: float,
    ) -> None:
        """
        Record one frame's worth of timing data.

        Args:
            stage_times: Mapping of stage name → elapsed ms.
            fps:         Current rolling FPS reading.
        """
        for stage, ms in stage_times.items():
            self._stage_samples[stage].append(ms)
        if fps > 0:
            self._fps_samples.append(fps)

    def close(self) -> None:
        """Compute statistics and write the summary JSON."""
        stage_stats = []
        for stage, samples in self._stage_samples.items():
            arr = np.array(samples)
            stage_stats.append({
                "stage":   stage,
                "n":       len(arr),
                "mean_ms": round(float(arr.mean()), 2),
                "p50_ms":  round(float(np.percentile(arr, 50)), 2),
                "p95_ms":  round(float(np.percentile(arr, 95)), 2),
                "p99_ms":  round(float(np.percentile(arr, 99)), 2),
            })

        total_samples = [
            sum(self._stage_samples[s][i] for s in self._stage_samples)
            for i in range(min(len(v) for v in self._stage_samples.values()) if self._stage_samples else [0])
        ]

        fps_arr     = np.array(self._fps_samples) if self._fps_samples else np.array([0.0])
        total_arr   = np.array(total_samples) if total_samples else np.array([0.0])
        elapsed     = time.time() - self._start_time

        summary = {
            "meta": {
                "model_variant":     self._model_variant,
                "backend":           self._backend,
                "wall_time_seconds": round(elapsed, 1),
                "total_frames":      len(self._fps_samples),
            },
            "stages":           stage_stats,
            "total_p95_ms":     round(float(np.percentile(total_arr, 95)), 2) if len(total_arr) > 0 else 0,
            "mean_fps":         round(float(fps_arr.mean()), 1),
            "meets_25ms_target": float(np.percentile(total_arr, 95)) <= 25.0 if len(total_arr) > 0 else False,
        }

        self._path.write_text(json.dumps(summary, indent=2))
        print(f"[logger] Metrics saved: {self._path}")
        print(f"[logger] Total P95: {summary['total_p95_ms']}ms  |  "
              f"Mean FPS: {summary['mean_fps']}  |  "
              f"≤25ms: {'YES' if summary['meets_25ms_target'] else 'NO'}")

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Session summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_session_summary(
    threat_logger: ThreatLogger,
    metrics_logger: MetricsLogger,
) -> None:
    """
    Print a concise end-of-session summary to stdout.

    Args:
        threat_logger:  Closed ThreatLogger instance.
        metrics_logger: Closed MetricsLogger instance.
    """
    print("\n" + "─" * 50)
    print(f"  Session complete")
    print(f"  Threat events logged : {threat_logger.event_count}")
    print(f"  Threat log           : {threat_logger._path}")
    print(f"  Metrics              : {metrics_logger._path}")
    print("─" * 50 + "\n")
