#!/usr/bin/env python3
"""
hybrid_fusion.py — Mode 4: Production Hybrid Fusion
Combines Depth Guard (physical safety net) + YOLOv8-seg + HITL Routing.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .base import BaseDetector, DetectorContext, sort_and_assign_ids
from .depth_shape import DepthShapeDetector
from .yolo_seg import YoloSegDetector


class HybridFusionDetector(BaseDetector):
    mode_name = 'hybrid_fusion'

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        super().__init__(config, logger)
        self.depth_guard = DepthShapeDetector(config, logger)
        self.yolo = YoloSegDetector(config, logger)
        self.model_ready = True
        self.status_note = 'INIT'

    def load_model(self):
        self.depth_guard.ensure_loaded()
        self.yolo.ensure_loaded()
        if self.yolo.model_ready:
            self.model_ready = True
            self.status_note = f"FUSION_READY ({self.yolo.status_note})"
        else:
            # Depth Guard still operates safely, routing objects to HITL
            self.model_ready = True
            self.status_note = 'DEPTH_GUARD_ONLY (NO_YOLO_PT)'

    def detect(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        ctx: DetectorContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        self.ensure_loaded()

        # 1. Depth Guard: physically detect every raised object on the belt
        depth_dets, depth_mask = self.depth_guard.detect(cv_color, cv_depth, ctx)

        # 2. YOLOv8-seg: semantic GOOD/WASTE + overlap separation
        yolo_dets, yolo_mask = self.yolo.detect(cv_color, cv_depth, ctx)

        merged_mask = depth_mask
        if yolo_mask is not None and depth_mask is not None:
            merged_mask = cv2.bitwise_or(depth_mask, yolo_mask)

        # If YOLO model is not loaded yet, Depth Guard protects the line and routes to HITL
        if not self.yolo.model_ready:
            for d in depth_dets:
                d['decision'] = 'HITL'
                d['decision_reason'] = 'depth_guard_no_yolo'
                d['source_mode'] = self.mode_name
            return sort_and_assign_ids(depth_dets, prioritize_height=True), merged_mask

        # 3. Fuse YOLO detections with Depth Guard
        match_radius_px = float(ctx.config.get('fusion_match_radius_px', 45.0))
        fused: List[Dict[str, Any]] = []
        matched_depth_indices = set()

        for y_det in yolo_dets:
            y_det['source_mode'] = self.mode_name
            best_idx = None
            best_dist = match_radius_px
            for idx, d_det in enumerate(depth_dets):
                dist = math.hypot(y_det['u_sub'] - d_det['u_sub'], y_det['v_sub'] - d_det['v_sub'])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                matched_depth_indices.add(best_idx)
                d_det = depth_dets[best_idx]

                # 3D Depth Cross-Check Veto: catch closed-set YOLO false GOODs
                if y_det['decision'] == 'GOOD':
                    h_peak = float(y_det.get('height_peak_mm', d_det.get('height_peak_mm', 0.0)))
                    flat = float(y_det.get('flatness_ratio', 1.0))
                    h_cv = float(y_det.get('height_cv', 0.0))

                    # Veto 1: Sloped/peaked 3D surface (e.g., Pyramid mistaken for good_cube)
                    if h_peak >= 10.0 and flat < 0.78 and h_cv > 0.14:
                        y_det['class_name'] = f"HITL_{y_det['class_name']}"
                        y_det['decision'] = 'HITL'
                        y_det['decision_reason'] = f"depth_guard_non_flat={flat:.2f}"
                    # Veto 2: Crisp 2D polygon/hexagon contour mistaken for good_cylinder
                    elif 'CYLINDER' in y_det['class_name'].upper() and d_det.get('contour') is not None:
                        d_cnt = d_det['contour']
                        d_perim = cv2.arcLength(d_cnt, True)
                        d_area = cv2.contourArea(d_cnt)
                        if d_perim > 0 and d_area > 0:
                            approx = cv2.approxPolyDP(d_cnt, 0.028 * d_perim, True)
                            pts = d_cnt.reshape(-1, 2).astype(np.float32)
                            dists = np.linalg.norm(
                                pts - np.array([d_det['u_sub'], d_det['v_sub']], dtype=np.float32),
                                axis=1,
                            )
                            p95 = float(np.percentile(dists, 95))
                            p05 = float(np.percentile(dists, 5))
                            r_ratio = (p05 / p95) if p95 > 1e-3 else 1.0
                            if 5 <= len(approx) <= 7 and r_ratio < 0.88:
                                y_det['class_name'] = f"HITL_{y_det['class_name']}"
                                y_det['decision'] = 'HITL'
                                y_det['decision_reason'] = f"depth_guard_poly_v{len(approx)}_r={r_ratio:.2f}"

            fused.append(y_det)

        # 4. Depth Guard Catch: physical object on belt missed by YOLO -> UNKNOWN -> HITL
        for idx, d_det in enumerate(depth_dets):
            if idx not in matched_depth_indices:
                d_det['class_name'] = f"UNKNOWN_{d_det['class_name'][:4]}"
                d_det['decision'] = 'HITL'
                d_det['decision_reason'] = 'depth_guard_yolo_miss'
                d_det['source_mode'] = self.mode_name
                fused.append(d_det)

        return sort_and_assign_ids(fused, prioritize_height=True), merged_mask
