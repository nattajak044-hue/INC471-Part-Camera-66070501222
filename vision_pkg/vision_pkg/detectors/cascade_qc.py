#!/usr/bin/env python3
"""
cascade_qc.py — Mode 6: Two-Stage Cascade QC
Stage 1 (Real-time): HybridFusion (Depth Guard + YOLOv8-seg)
Stage 2 (On LOCK / INSPECT trigger): EfficientAD fine surface defect check
on GOOD / HITL crops.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .base import BaseDetector, DetectorContext
from .hybrid_fusion import HybridFusionDetector
from .sam_anomaly import SamAnomalyDetector


class CascadeQCDetector(BaseDetector):
    mode_name = 'cascade_qc'

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        super().__init__(config, logger)
        self.stage1 = HybridFusionDetector(config, logger)
        self.stage2 = SamAnomalyDetector(config, logger)
        self.model_ready = True
        self.status_note = 'INIT'

    def load_model(self):
        self.stage1.ensure_loaded()
        self.stage2.ensure_loaded()
        self.model_ready = True
        self.status_note = f"S1:{self.stage1.status_note} | S2:{self.stage2.status_note}"

    def detect(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        ctx: DetectorContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        self.ensure_loaded()
        detections, fg_mask = self.stage1.detect(cv_color, cv_depth, ctx)

        run_stage2 = ctx.trigger_inspect or ctx.snapshot_locked
        if run_stage2 and self.stage2.anomaly_model is not None:
            h, w = cv_color.shape[:2]
            anomaly_thresh = float(ctx.config.get('anomaly_threshold', 0.50))
            anomaly_hitl_low = float(ctx.config.get('anomaly_hitl_low', 0.35))

            for det in detections:
                if det['decision'] in ('GOOD', 'HITL'):
                    x, y, bw, bh = cv2.boundingRect(det['box'])
                    x1, y1 = max(0, x - 6), max(0, y - 6)
                    x2, y2 = min(w, x + bw + 6), min(h, y + bh + 6)
                    score = self.stage2.score_crop_anomaly(cv_color[y1:y2, x1:x2])
                    det['anomaly_score'] = round(score, 4) if score is not None else None
                    if score is not None:
                        if score >= anomaly_thresh:
                            det['decision'] = 'WASTE'
                            det['decision_reason'] = f'cascade_s2_defect_{score:.2f}'
                        elif score >= anomaly_hitl_low:
                            det['decision'] = 'HITL'
                            det['decision_reason'] = f'cascade_s2_borderline_{score:.2f}'
                        else:
                            det['decision'] = 'GOOD'
                            det['decision_reason'] = f'cascade_s2_verified_{score:.2f}'

        for det in detections:
            det['source_mode'] = self.mode_name

        return detections, fg_mask
