#!/usr/bin/env python3
"""
depth_shape.py — Mode 1: Fast 3D Depth + CIE LAB Shape Detector (No AI)
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .base import (
    BaseDetector,
    DetectorContext,
    classify_geometry,
    compute_3d_from_mask,
    sort_and_assign_ids,
)


class DepthShapeDetector(BaseDetector):
    mode_name = 'depth_shape'

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        super().__init__(config, logger)
        self.model_ready = True
        self.status_note = 'NO_AI_READY'

    def _refine_contour_with_rgb_diff(
        self,
        cv_color: np.ndarray,
        rough_cnt: np.ndarray,
        bg_lab_img: Optional[np.ndarray],
        min_contour_area: float,
    ) -> np.ndarray:
        """Use CIE LAB frame differencing to sharpen object edges."""
        if bg_lab_img is None:
            return rough_cnt

        h, w = cv_color.shape[:2]
        x, y, bw, bh = cv2.boundingRect(rough_cnt)
        pad = 16
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        curr_patch = cv2.GaussianBlur(cv_color[y1:y2, x1:x2], (5, 5), 0)
        curr_lab = cv2.cvtColor(curr_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab_patch = bg_lab_img[y1:y2, x1:x2]

        color_diff = np.linalg.norm(curr_lab - bg_lab_patch, axis=2)
        lab_thresh = float(self.config.get('lab_diff_threshold', 24.0))
        rgb_mask = np.where(color_diff > lab_thresh, 255, 0).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)
        rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(rgb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return rough_cnt

        best_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < min_contour_area * 0.6:
            return rough_cnt

        return best_cnt + np.array([[[x1, y1]]])

    def detect(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        ctx: DetectorContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        if ctx.bg_depth_map is None:
            return [], np.zeros(cv_depth.shape[:2], dtype=np.uint8)

        h, w = cv_depth.shape[:2]
        depth_f = cv_depth.astype(np.float32)

        min_valid_depth = float(ctx.config.get('min_valid_depth_mm', 150.0))
        height_min = float(ctx.config.get('height_min_mm', 3.0))
        height_max = float(ctx.config.get('height_max_mm', 85.0))
        min_area = float(ctx.config.get('min_contour_area', 350.0))
        max_area = float(ctx.config.get('max_contour_area', 35000.0))
        conf_thresh = float(ctx.config.get('conf_threshold', 0.50))

        valid_both = (depth_f > min_valid_depth) & (ctx.bg_depth_map > min_valid_depth)
        height_map = np.where(valid_both, ctx.bg_depth_map - depth_f, 0.0)

        # Use adaptive per-pixel noise floor from BG calibration so empty-belt depth
        # jitter never triggers false positives.
        if ctx.bg_depth_noise_map is not None:
            depth_thresh = np.maximum(height_min, ctx.bg_depth_noise_map)
        else:
            depth_thresh = max(height_min, 4.5)

        depth_fg = np.where(
            (height_map >= depth_thresh) & (height_map <= height_max),
            255,
            0,
        ).astype(np.uint8)

        # RGB-Assisted Foreground Fusion: catches thin (5mm) gray bars even with stereo depth holes.
        fg_mask = depth_fg
        roi_lab_diff_full = None
        if ctx.bg_lab_img is not None:
            rx1, ry1, rx2, ry2 = ctx.roi_rect
            roi_blur = cv2.GaussianBlur(cv_color[ry1:ry2, rx1:rx2], (5, 5), 0)
            roi_lab = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
            bg_roi_lab = ctx.bg_lab_img[ry1:ry2, rx1:rx2]
            roi_diff = np.linalg.norm(roi_lab - bg_roi_lab, axis=2)
            roi_lab_diff_full = np.zeros((h, w), dtype=np.float32)
            roi_lab_diff_full[ry1:ry2, rx1:rx2] = roi_diff
            lab_thresh = float(ctx.config.get('lab_diff_threshold', 28.0))
            rgb_fg = np.zeros((h, w), dtype=np.uint8)
            rgb_fg[ry1:ry2, rx1:rx2] = np.where(roi_diff > lab_thresh, 255, 0).astype(np.uint8)
            fg_mask = cv2.bitwise_or(depth_fg, rgb_fg)

        fg_mask = cv2.bitwise_and(fg_mask, ctx.roi_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        rough_contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Dict[str, Any]] = []
        for r_cnt in rough_contours:
            r_area = cv2.contourArea(r_cnt)
            if r_area < min_area or r_area > max_area:
                continue

            cnt = self._refine_contour_with_rgb_diff(
                cv_color, r_cnt, ctx.bg_lab_img, min_area
            )
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(obj_mask, [cnt], -1, 255, -1)

            geom = compute_3d_from_mask(obj_mask, cnt, cv_depth, ctx)
            if geom is None:
                continue

            # Dual-Confirmation Anti-Noise Gate:
            # - Tall object (e.g., 25mm cube/cyl/pyramid): unmistakable height (>= 7mm peak & >= 5.5mm med)
            # - Thin 5mm object (e.g., 5mm gray bar): confirmed by BOTH depth (>= 2.8mm) AND RGB contrast (>= 22)
            # Empty-belt depth jitter (low RGB diff) & light glare (0mm height) are both rejected!
            med_lab_diff = 0.0
            if roi_lab_diff_full is not None:
                vals = roi_lab_diff_full[obj_mask == 255]
                if vals.size > 0:
                    med_lab_diff = float(np.median(vals))

            is_tall_obj = (geom['height_peak_mm'] >= 8.0 and geom['height_mm'] >= 5.5)
            is_thin_confirmed = (geom['height_mm'] >= 2.8 and med_lab_diff >= 22.0)
            if not (is_tall_obj or is_thin_confirmed):
                continue

            perimeter = cv2.arcLength(cnt, True)
            hitl_thresh = float(ctx.config.get('hitl_conf_threshold', 0.75))
            class_name, conf, decision, reason = classify_geometry(
                geom['area'],
                perimeter,
                geom['rw'],
                geom['rh'],
                cnt=cnt,
                u_sub=geom['u_sub'],
                v_sub=geom['v_sub'],
                short_mm=geom['short_mm'],
                long_mm=geom['long_mm'],
                height_mm=geom['height_mm'],
                height_peak_mm=geom['height_peak_mm'],
                flatness_ratio=geom['flatness_ratio'],
                height_cv=geom['height_cv'],
                plane_rmse_mm=geom['plane_rmse_mm'],
                corner_drop_mm=geom['corner_drop_mm'],
                axis_drop_max_mm=geom['axis_drop_max_mm'],
                axis_drop_min_mm=geom['axis_drop_min_mm'],
                hitl_conf_threshold=hitl_thresh,
            )
            # Do NOT drop low-confidence physical objects — route them to HITL per Proposal!
            if decision != 'HITL' and conf < conf_thresh:
                decision = 'HITL'
                reason = f"below_min_conf_{conf:.2f}"

            detections.append({
                'class_name': class_name,
                'decision': decision,
                'decision_reason': reason,
                'conf': float(conf),
                'anomaly_score': None,
                'is_overlapped': False,
                'u': geom['u'],
                'v': geom['v'],
                'u_sub': geom['u_sub'],
                'v_sub': geom['v_sub'],
                'depth_mm': geom['depth_mm'],
                'height_mm': geom['height_mm'],
                'height_peak_mm': geom['height_peak_mm'],
                'roll_deg': geom['roll_deg'],
                'cam_xyz_m': geom['cam_xyz_m'],
                'box': geom['box'],
                'area': geom['area'],
                'contour': cnt,
                'source_mode': self.mode_name,
            })

        return sort_and_assign_ids(detections), fg_mask
