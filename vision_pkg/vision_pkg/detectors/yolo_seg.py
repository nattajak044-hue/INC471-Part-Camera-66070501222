#!/usr/bin/env python3
"""
yolo_seg.py — Mode 2: High-Speed YOLOv8-seg + 3D Depth Mask Sampling
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .base import (
    BaseDetector,
    DetectorContext,
    compute_3d_from_mask,
    sort_and_assign_ids,
)


def _map_class_to_decision(class_name: str, conf: float, hitl_thresh: float) -> Tuple[str, str]:
    """Map YOLO class name + confidence into GOOD / WASTE / HITL."""
    upper = class_name.upper()
    if conf < hitl_thresh:
        return 'HITL', f'low_conf_{conf:.2f}'
    if any(k in upper for k in ('WASTE', 'DEFECT', 'BAD', 'NG', 'SCRAP', 'REJECT')):
        return 'WASTE', 'yolo_waste'
    if any(k in upper for k in ('GOOD', 'OK', 'PASS', 'NORMAL')):
        return 'GOOD', 'yolo_good'
    return 'HITL', 'yolo_unknown_class'


class YoloSegDetector(BaseDetector):
    mode_name = 'yolo_seg'

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        super().__init__(config, logger)
        self.model = None
        self.model_ready = False
        self.status_note = 'NOT_LOADED'

    def load_model(self):
        model_path = os.path.expanduser(str(self.config.get('yolo_model_path', '')).strip())
        if not model_path or not os.path.isfile(model_path):
            self.model = None
            self.model_ready = False
            self.status_note = 'NO_PT_WEIGHTS'
            self.log_warn(
                f"ไม่พบไฟล์โมเดล YOLO (.pt) ที่ '{model_path or '(ไม่ได้ระบุ)'}' — "
                "โหมด yolo_seg พร้อมเสียบ .pt เมื่อเทรนเสร็จแล้ว"
            )
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_ready = True
            self.status_note = f"LOADED:{os.path.basename(model_path)}"
            self.log_info(f"โหลดโมเดล YOLOv8-seg สำเร็จ: {model_path}")
        except Exception as e:
            self.model = None
            self.model_ready = False
            self.status_note = 'LOAD_ERROR'
            self.log_error(f"โหลดโมเดล YOLO ไม่สำเร็จ: {e}")

    def detect(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        ctx: DetectorContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        self.ensure_loaded()
        h, w = cv_color.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if not self.model_ready or self.model is None:
            return [], combined_mask

        conf_thresh = float(ctx.config.get('conf_threshold', 0.40))
        hitl_thresh = float(ctx.config.get('hitl_conf_threshold', 0.75))
        min_area = float(ctx.config.get('min_contour_area', 350.0))
        max_area = float(ctx.config.get('max_contour_area', 35000.0))
        single_layer_max_mm = float(ctx.config.get('single_layer_max_mm', 45.0))

        full_frame = bool(ctx.config.get('yolo_full_frame', True))
        rx1, ry1, rx2, ry2 = ctx.roi_rect
        if full_frame:
            infer_img = cv_color
            off_x, off_y = 0, 0
            infer_w, infer_h = w, h
        else:
            infer_img = cv_color[ry1:ry2, rx1:rx2]
            if infer_img.size == 0:
                return [], combined_mask
            off_x, off_y = rx1, ry1
            infer_w, infer_h = rx2 - rx1, ry2 - ry1

        results = self.model.predict(infer_img, conf=conf_thresh, imgsz=640, verbose=False)
        if not results:
            return [], combined_mask

        res = results[0]
        boxes = res.boxes
        masks = res.masks
        if boxes is None or len(boxes) == 0:
            return [], combined_mask

        names = res.names or {}
        detections: List[Dict[str, Any]] = []
        raw_masks: List[np.ndarray] = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            class_name = str(names.get(cls_id, f"CLASS_{cls_id}"))

            obj_mask = np.zeros((h, w), dtype=np.uint8)
            if masks is not None and masks.data is not None and i < len(masks.data):
                m_crop = masks.data[i].cpu().numpy()
                m_crop = cv2.resize(
                    m_crop, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST
                )
                obj_mask[off_y:off_y + infer_h, off_x:off_x + infer_w] = np.where(
                    m_crop > 0.5, 255, 0
                ).astype(np.uint8)
            else:
                bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)
                obj_mask[off_y + by1:off_y + by2, off_x + bx1:off_x + bx2] = 255

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
            decision, reason = _map_class_to_decision(class_name, conf, hitl_thresh)
            is_overlapped = geom['height_mm'] > single_layer_max_mm

            detections.append({
                'class_name': class_name,
                'decision': decision,
                'decision_reason': reason,
                'conf': conf,
                'anomaly_score': None,
                'is_overlapped': is_overlapped,
                'u': geom['u'],
                'v': geom['v'],
                'u_sub': geom['u_sub'],
                'v_sub': geom['v_sub'],
                'depth_mm': geom['depth_mm'],
                'height_mm': geom['height_mm'],
                'height_peak_mm': geom['height_peak_mm'],
                'flatness_ratio': geom['flatness_ratio'],
                'height_cv': geom['height_cv'],
                'roll_deg': geom['roll_deg'],
                'cam_xyz_m': geom['cam_xyz_m'],
                'box': geom['box'],
                'area': geom['area'],
                'contour': cnt,
                'source_mode': self.mode_name,
            })
            raw_masks.append(obj_mask)

        # Check mask proximity / overlap between detected instances
        dilate_k = np.ones((7, 7), np.uint8)
        for i in range(len(raw_masks)):
            m_i = cv2.dilate(raw_masks[i], dilate_k, iterations=1)
            for j in range(i + 1, len(raw_masks)):
                if np.any((m_i == 255) & (raw_masks[j] == 255)):
                    detections[i]['is_overlapped'] = True
                    detections[j]['is_overlapped'] = True

        return sort_and_assign_ids(detections, prioritize_height=True), combined_mask
