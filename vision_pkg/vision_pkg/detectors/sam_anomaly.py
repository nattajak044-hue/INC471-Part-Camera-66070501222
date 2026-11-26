#!/usr/bin/env python3
"""
sam_anomaly.py — Mode 3: FastSAM Segmentation + EfficientAD Anomaly Detection
Designed for on-demand / snapshot-triggered or throttled fine inspection so the
30 FPS ROS 2 camera callback never freezes.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .base import (
    BaseDetector,
    DetectorContext,
    compute_3d_from_mask,
    sort_and_assign_ids,
)
from .depth_shape import DepthShapeDetector


class SamAnomalyDetector(BaseDetector):
    mode_name = 'sam_anomaly'

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        super().__init__(config, logger)
        self.fastsam_model = None
        self.anomaly_model = None
        self.depth_fallback = DepthShapeDetector(config, logger)
        self.model_ready = False
        self.status_note = 'NOT_LOADED'
        self._last_run_time = 0.0
        self._cached_detections: List[Dict[str, Any]] = []
        self._cached_mask: Optional[np.ndarray] = None

    def load_model(self):
        fastsam_path = os.path.expanduser(str(self.config.get('fastsam_model_path', '')).strip())
        anomaly_path = os.path.expanduser(str(self.config.get('anomaly_model_path', '')).strip())

        loaded_parts = []
        if fastsam_path and os.path.isfile(fastsam_path):
            try:
                from ultralytics import FastSAM
                self.fastsam_model = FastSAM(fastsam_path)
                loaded_parts.append('FastSAM')
                self.log_info(f"โหลด FastSAM สำเร็จ: {fastsam_path}")
            except Exception as e:
                self.log_error(f"โหลด FastSAM ไม่สำเร็จ: {e}")

        if anomaly_path and os.path.isfile(anomaly_path):
            try:
                import torch
                self.anomaly_model = torch.jit.load(anomaly_path, map_location='cpu')
                self.anomaly_model.eval()
                loaded_parts.append('EfficientAD')
                self.log_info(f"โหลด EfficientAD สำเร็จ: {anomaly_path}")
            except Exception as e:
                self.log_warn(f"โหลด EfficientAD ไม่สำเร็จ (ตรวจรูปแบบไฟล์ .pt/.onnx): {e}")

        if loaded_parts:
            self.model_ready = True
            self.status_note = '+'.join(loaded_parts)
        else:
            self.model_ready = False
            self.status_note = 'NO_WEIGHTS_FALLBACK_DEPTH'
            self.log_warn(
                "ยังไม่พบไฟล์โมเดล FastSAM / EfficientAD — "
                "โหมด sam_anomaly พร้อมโครงสร้างแล้ว (ใช้ Depth Mask ชั่วคราวเมื่อกด INSPECT)"
            )

    def score_crop_anomaly(self, crop_bgr: np.ndarray) -> Optional[float]:
        """Score a single object RGB crop using EfficientAD (0.0 = good, 1.0 = defective)."""
        if self.anomaly_model is None or crop_bgr.size == 0:
            return None
        try:
            import torch
            img = cv2.resize(crop_bgr, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                out = self.anomaly_model(tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                score = float(out.mean().item())
            return max(0.0, min(1.0, score))
        except Exception as e:
            self.log_warn(f"Anomaly scoring error: {e}")
            return None

    def detect(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        ctx: DetectorContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        self.ensure_loaded()
        min_interval = float(ctx.config.get('sam_interval_sec', 0.5))
        now = time.time()

        # Run immediately on explicit INSPECT/LOCK trigger, otherwise throttle
        should_run = ctx.trigger_inspect or (now - self._last_run_time >= min_interval)
        if not should_run and self._cached_mask is not None:
            return self._cached_detections, self._cached_mask

        self._last_run_time = now
        h, w = cv_color.shape[:2]
        anomaly_thresh = float(ctx.config.get('anomaly_threshold', 0.50))
        anomaly_hitl_low = float(ctx.config.get('anomaly_hitl_low', 0.35))

        # Step 1: Segment objects (FastSAM if available, else DepthShape fallback)
        if self.fastsam_model is not None:
            rx1, ry1, rx2, ry2 = ctx.roi_rect
            roi_crop = cv_color[ry1:ry2, rx1:rx2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            detections: List[Dict[str, Any]] = []
            try:
                results = self.fastsam_model(roi_crop, verbose=False)
                if results and results[0].masks is not None:
                    min_area = float(ctx.config.get('min_contour_area', 350.0))
                    max_area = float(ctx.config.get('max_contour_area', 35000.0))
                    for m in results[0].masks.data:
                        m_np = cv2.resize(
                            m.cpu().numpy(), (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_NEAREST
                        )
                        obj_mask = np.zeros((h, w), dtype=np.uint8)
                        obj_mask[ry1:ry2, rx1:rx2] = np.where(m_np > 0.5, 255, 0).astype(np.uint8)
                        obj_mask = cv2.bitwise_and(obj_mask, ctx.roi_mask)
                        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not cnts:
                            continue
                        cnt = max(cnts, key=cv2.contourArea)
                        area = float(cv2.contourArea(cnt))
                        if area < min_area or area > max_area:
                            continue
                        geom = compute_3d_from_mask(obj_mask, cnt, cv_depth, ctx)
                        if geom is None:
                            continue
                        combined_mask = cv2.bitwise_or(combined_mask, obj_mask)
                        detections.append({
                            'class_name': 'SAM_OBJ',
                            'decision': 'UNCLASSIFIED',
                            'decision_reason': 'fastsam_seg',
                            'conf': 0.90,
                            'anomaly_score': None,
                            'is_overlapped': False,
                            'u': geom['u'],
                            'v': geom['v'],
                            'u_sub': geom['u_sub'],
                            'v_sub': geom['v_sub'],
                            'depth_mm': geom['depth_mm'],
                            'height_mm': geom['height_mm'],
                            'roll_deg': geom['roll_deg'],
                            'cam_xyz_m': geom['cam_xyz_m'],
                            'box': geom['box'],
                            'area': geom['area'],
                            'contour': cnt,
                            'source_mode': self.mode_name,
                        })
            except Exception as e:
                self.log_error(f"FastSAM inference error: {e}")
        else:
            detections, combined_mask = self.depth_fallback.detect(cv_color, cv_depth, ctx)
            for det in detections:
                det['source_mode'] = self.mode_name

        # Step 2: Run EfficientAD anomaly check on each detected object crop
        for det in detections:
            x, y, bw, bh = cv2.boundingRect(det['box'])
            x1, y1 = max(0, x - 6), max(0, y - 6)
            x2, y2 = min(w, x + bw + 6), min(h, y + bh + 6)
            crop = cv_color[y1:y2, x1:x2]
            score = self.score_crop_anomaly(crop)
            det['anomaly_score'] = round(score, 4) if score is not None else None
            if score is not None:
                if score >= anomaly_thresh:
                    det['decision'] = 'WASTE'
                    det['decision_reason'] = f'anomaly_high_{score:.2f}'
                elif score >= anomaly_hitl_low:
                    det['decision'] = 'HITL'
                    det['decision_reason'] = f'anomaly_borderline_{score:.2f}'
                else:
                    det['decision'] = 'GOOD'
                    det['decision_reason'] = f'anomaly_ok_{score:.2f}'
            else:
                det['decision'] = 'HITL'
                det['decision_reason'] = 'anomaly_model_not_loaded'

        self._cached_detections = sort_and_assign_ids(detections, prioritize_height=True)
        self._cached_mask = combined_mask
        return self._cached_detections, self._cached_mask
