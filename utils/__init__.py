"""
utils/ — shared utilities for the 5G MEC IDS pipeline.

Sub-modules (import individually to avoid loading unnecessary deps):
  - utils.ids_layer  : behavioural intrusion detection rules  (numpy only)
  - utils.draw       : OpenCV annotation  (requires cv2)
  - utils.logger     : structured threat + metrics logging  (numpy only)

Usage:
    from utils.ids_layer import IDSLayer, ThreatEvent
    from utils.draw import draw_skeleton, draw_latency_hud
    from utils.logger import ThreatLogger
"""
