#!/usr/bin/env python3
"""
detectors/__init__.py — Detector Registry and Factory
"""

from typing import Any, Dict, List

from .base import BaseDetector, DetectorContext, wrap_angle_180
from .depth_shape import DepthShapeDetector
from .yolo_seg import YoloSegDetector
from .sam_anomaly import SamAnomalyDetector
from .hybrid_fusion import HybridFusionDetector
from .data_collect import DataCollectDetector
from .cascade_qc import CascadeQCDetector

DETECTOR_REGISTRY = {
    'depth_shape': DepthShapeDetector,
    'yolo_seg': YoloSegDetector,
    'sam_anomaly': SamAnomalyDetector,
    'hybrid_fusion': HybridFusionDetector,
    'data_collect': DataCollectDetector,
    'cascade_qc': CascadeQCDetector,
}

AVAILABLE_MODES: List[str] = list(DETECTOR_REGISTRY.keys())


def create_detector(mode_name: str, config: Dict[str, Any], logger: Any = None) -> BaseDetector:
    key = mode_name.strip().lower()
    cls = DETECTOR_REGISTRY.get(key)
    if cls is None:
        if logger is not None:
            logger.warn(f"ไม่รู้จักโหมด '{mode_name}' — ใช้ 'depth_shape' แทน")
        cls = DepthShapeDetector
    return cls(config=config, logger=logger)


__all__ = [
    'AVAILABLE_MODES',
    'DETECTOR_REGISTRY',
    'BaseDetector',
    'DetectorContext',
    'wrap_angle_180',
    'create_detector',
]
