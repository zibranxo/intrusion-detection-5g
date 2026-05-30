"""
urllc.py — Simulated 5G URLLC transport layer.

Models the MEC → core-network uplink path for threat alerts with
configurable latency distribution, packet loss, and bandwidth cap.
No real 5G hardware required — this is a pure-Python simulation that
makes the pipeline genuinely unique among edge-CV projects.

Architecture:

    quantized_pipeline.py
           │
           ▼
    URLLCTransport.send(threat_event)
           │
           ▼  [simulated URLLC link: 1-5 ms latency, 0.1 % loss]
           │
           ▼
    URLLCReceiver  (logs received alerts to console + JSONL)

The receiver runs in a background thread, simulating the core-network
endpoint that a real 5G MEC deployment would uplink to.

Usage:
    from urllc import URLLCTransport, URLLCReceiver

    receiver = URLLCReceiver("results/urllc_received.jsonl")
    transport = URLLCTransport(receiver)

    # inside the frame loop:
    for threat in threats:
        transport.send(threat)

    # on exit:
    transport.close()
    receiver.close()
"""

import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    URLLC_LATENCY_MS_MAX,
    URLLC_LATENCY_MS_MIN,
    URLLC_PACKET_LOSS_RATE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class URLLCPacket:
    """A single URLLC uplink packet — one threat event over the wire."""

    payload:        dict           # serialised ThreatEvent.to_dict()
    send_time:      float          # time.perf_counter() when send() was called
    scheduled_time: float          # when this packet should be delivered
    dropped:        bool = False   # True if randomly dropped (packet loss)


@dataclass
class URLLCStats:
    """Telemetry for the simulated link."""

    sent:       int = 0
    delivered:  int = 0
    dropped:    int = 0
    latencies:  list[float] = field(default_factory=list)

    @property
    def loss_rate(self) -> float:
        return self.dropped / max(self.sent, 1)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        import numpy as np
        return float(np.percentile(self.latencies, 95))


# ──────────────────────────────────────────────────────────────────────────────
# Receiver  (core-network endpoint)
# ──────────────────────────────────────────────────────────────────────────────

class URLLCReceiver:
    """
    Simulates the core-network endpoint that receives URLLC uplink packets.

    Runs a background thread that polls for newly-"delivered" packets
    (those whose scheduled_time has elapsed) and logs them.
    """

    def __init__(
        self,
        log_path: str | Path,
        poll_interval_ms: float = 0.5,
    ) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._poll_interval = poll_interval_ms / 1000.0
        self._stats = URLLCStats()

        self._lock = threading.Lock()
        self._queue: deque[URLLCPacket] = deque()
        self._running = True

        self._fh = self._log_path.open("w", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    # ── Background poll loop ──────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Continuously check for packets ready to be delivered."""
        while self._running:
            now = time.perf_counter()
            delivered: list[URLLCPacket] = []

            with self._lock:
                remaining: deque[URLLCPacket] = deque()
                while self._queue:
                    pkt = self._queue.popleft()
                    if pkt.dropped:
                        self._stats.dropped += 1
                        continue
                    if pkt.scheduled_time <= now:
                        delivered.append(pkt)
                    else:
                        remaining.appendleft(pkt)
                        break  # packets are in-order by scheduled_time
                # Put back any un-delivered packets
                self._queue.extendleft(reversed(remaining))

            for pkt in delivered:
                latency = (now - pkt.send_time) * 1000.0
                self._stats.delivered += 1
                self._stats.latencies.append(latency)

                # Write to log
                record = {
                    "received_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.gmtime()
                    ),
                    "latency_ms": round(latency, 3),
                    **pkt.payload,
                }
                self._fh.write(json.dumps(record) + "\n")
                self._fh.flush()

            time.sleep(self._poll_interval)

    # ── Internal — called by URLLCTransport ───────────────────────────────

    def _enqueue(self, packet: URLLCPacket) -> None:
        with self._lock:
            # Insert in scheduled_time order (or append and rely on poll loop).
            self._queue.append(packet)
            self._stats.sent += 1

    # ── Public ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> URLLCStats:
        with self._lock:
            return self._stats

    def close(self) -> None:
        """Stop the poll loop, flush remaining packets, and close."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # Drain remaining queue
        now = time.perf_counter()
        with self._lock:
            while self._queue:
                pkt = self._queue.popleft()
                if not pkt.dropped:
                    latency = (now - pkt.send_time) * 1000.0
                    self._stats.delivered += 1
                    self._stats.latencies.append(latency)
                    record = {
                        "received_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%S", time.gmtime()
                        ),
                        "latency_ms": round(latency, 3),
                        **pkt.payload,
                    }
                    self._fh.write(json.dumps(record) + "\n")

        self._fh.close()

        print(
            f"[urllc] Receiver closed — "
            f"sent={self._stats.sent}  "
            f"delivered={self._stats.delivered}  "
            f"dropped={self._stats.dropped}  "
            f"loss={self._stats.loss_rate:.4f}  "
            f"mean_lat={self._stats.mean_latency_ms:.1f}ms  "
            f"p95_lat={self._stats.p95_latency_ms:.1f}ms"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Transport  (MEC edge node → core network)
# ──────────────────────────────────────────────────────────────────────────────

class URLLCTransport:
    """
    Simulated URLLC uplink transport from the MEC edge node.

    Each send() call:
      1. Serialises the ThreatEvent to a dict
      2. Draws a random network latency from uniform(URLLC_LATENCY_MS_MIN,
         URLLC_LATENCY_MS_MAX)
      3. With probability URLLC_PACKET_LOSS_RATE, marks the packet as dropped
      4. Enqueues the packet for delivery at (now + latency)

    The packet is "delivered" when the receiver's poll loop notices its
    scheduled_time has elapsed.
    """

    def __init__(self, receiver: URLLCReceiver) -> None:
        self._receiver = receiver

    def send(self, threat_event) -> None:
        """
        Send a ThreatEvent over the simulated URLLC link.

        Args:
            threat_event: ThreatEvent from IDSLayer.update().
        """
        from utils.ids_layer import ThreatEvent as TE

        now = time.perf_counter()

        # ── Packet loss ────────────────────────────────────────────────
        dropped = random.random() < URLLC_PACKET_LOSS_RATE

        # ── Network latency ────────────────────────────────────────────
        latency_s = (
            random.uniform(URLLC_LATENCY_MS_MIN, URLLC_LATENCY_MS_MAX)
            / 1000.0
        )

        packet = URLLCPacket(
            payload=threat_event.to_dict(),
            send_time=now,
            scheduled_time=now + latency_s,
            dropped=dropped,
        )
        self._receiver._enqueue(packet)

    def close(self) -> None:
        """No-op — the receiver owns the queue and thread."""
        pass
