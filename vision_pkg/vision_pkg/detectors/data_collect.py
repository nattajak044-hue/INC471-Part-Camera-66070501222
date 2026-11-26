#!/usr/bin/env python3
"""
data_collect.py — Mode 5: Auto-Crop Dataset Collector for YOLO & EfficientAD
Uses DepthShapeDetector to detect belt objects and saves RGB, Depth, Crops,
and YOLO-format polygon/bbox labels on trigger or auto-collect interval.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .base import BaseDetector, DetectorContext
from .depth_shape import DepthShapeDetector


class DataCollectDetector(BaseDetector):
    mode_name = 'data_collect'

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        super().__init__(config, logger)
        self.depth_detector = DepthShapeDetector(config, logger)
        self.dataset_dir = os.path.expanduser(
            str(config.get('dataset_dir', '~/ros2_ws/datasets/conveyor_collect'))
        )
        self.saved_count = 0
        self._last_auto_save = 0.0
        self.model_ready = True
        self.status_note = 'COLLECT_READY'

    def _ensure_dirs(self):
        for sub in ('raw_rgb', 'raw_depth', 'crops', 'labels', 'meta'):
            os.makedirs(os.path.join(self.dataset_dir, sub), exist_ok=True)

    def _save_sample(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> int:
        self._ensure_dirs()
        ts = time.strftime('%Y%m%d_%H%M%S') + f"_{int((time.time() % 1) * 1000):03d}"
        h, w = cv_color.shape[:2]

        rgb_path = os.path.join(self.dataset_dir, 'raw_rgb', f"{ts}.png")
        depth_path = os.path.join(self.dataset_dir, 'raw_depth', f"{ts}.png")
        label_path = os.path.join(self.dataset_dir, 'labels', f"{ts}.txt")
        meta_path = os.path.join(self.dataset_dir, 'meta', f"{ts}.json")

        cv2.imwrite(rgb_path, cv_color)
        cv2.imwrite(depth_path, cv_depth)

        yolo_lines = []
        meta_objs = []
        for det in detections:
            cnt = det.get('contour')
            x, y, bw, bh = cv2.boundingRect(cnt if cnt is not None else det['box'])
            pad = 10
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            crop = cv_color[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = os.path.join(self.dataset_dir, 'crops', f"{ts}_obj{det['id']}.png")
                cv2.imwrite(crop_path, crop)

            # YOLO segmentation polygon format (class_id=0 placeholder for labeling tool)
            if cnt is not None and len(cnt) >= 3:
                eps = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
                if len(approx) >= 3:
                    coords = " ".join(f"{pt[0]/w:.5f} {pt[1]/h:.5f}" for pt in approx)
                    yolo_lines.append(f"0 {coords}")

            meta_objs.append({
                'id': det['id'],
                'class_name': det['class_name'],
                'height_mm': round(det['height_mm'], 2),
                'roll_deg': round(det['roll_deg'], 2),
                'cam_xyz_m': det['cam_xyz_m'],
                'bbox_xywh': [int(x), int(y), int(bw), int(bh)],
            })

        with open(label_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': ts, 'objects': meta_objs}, f, indent=2)

        self.saved_count += 1
        self.status_note = f"SAVED:{self.saved_count}"
        self.log_info(f"บันทึก Dataset #{self.saved_count} ({len(detections)} ชิ้น) -> {self.dataset_dir}")
        return self.saved_count

    def detect(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        ctx: DetectorContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        detections, fg_mask = self.depth_detector.detect(cv_color, cv_depth, ctx)
        for det in detections:
            det['source_mode'] = self.mode_name

        auto_interval = float(ctx.config.get('auto_collect_interval_sec', 1.5))
        now = time.time()
        should_save = False
        if ctx.trigger_collect and detections:
            should_save = True
        elif ctx.auto_collect and detections and (now - self._last_auto_save >= auto_interval):
            should_save = True

        if should_save:
            self._last_auto_save = now
            self._save_sample(cv_color, cv_depth, detections)

        return detections, fg_mask
