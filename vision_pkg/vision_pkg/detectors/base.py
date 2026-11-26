#!/usr/bin/env python3
"""
base.py — Base Detector Interface and Shared 3D/Geometry Utilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectorContext:
    """Shared per-frame state passed from ObjectDetectorNode into detectors."""
    fx: float
    fy: float
    ppx: float
    ppy: float
    bg_depth_map: Optional[np.ndarray]
    bg_lab_img: Optional[np.ndarray]
    belt_center_depth_mm: Optional[float]
    roi_mask: np.ndarray
    roi_rect: Tuple[int, int, int, int]
    config: Dict[str, Any] = field(default_factory=dict)
    bg_depth_noise_map: Optional[np.ndarray] = None
    snapshot_locked: bool = False
    trigger_inspect: bool = False
    trigger_collect: bool = False
    auto_collect: bool = False
    logger: Any = None


def wrap_angle_180(angle_deg: float) -> float:
    """Wrap orientation angle into [-90.0, 90.0] degrees."""
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return float(angle_deg)


def _soft_match(val: float, target: float, tol: float) -> float:
    """Return 1.0 when |val - target| <= tol*0.4, decaying smoothly toward 0 outside tol."""
    err = abs(val - target)
    if err <= tol * 0.4:
        return 1.0
    if err >= tol * 1.8:
        return 0.15
    return max(0.15, 1.0 - (err - tol * 0.4) / (tol * 1.4) * 0.85)


def classify_geometry(
    area: float,
    perimeter: float,
    rw: float,
    rh: float,
    cnt: Optional[np.ndarray] = None,
    u_sub: Optional[float] = None,
    v_sub: Optional[float] = None,
    short_mm: float = 0.0,
    long_mm: float = 0.0,
    height_mm: float = 0.0,
    height_peak_mm: float = 0.0,
    flatness_ratio: float = 1.0,
    height_cv: float = 0.0,
    plane_rmse_mm: float = 0.0,
    corner_drop_mm: float = 0.0,
    axis_drop_max_mm: float = 0.0,
    axis_drop_min_mm: float = 0.0,
    hitl_conf_threshold: float = 0.75,
) -> Tuple[str, float, str, str]:
    """
    Proposal-aligned Known-Good + Confidence-Based HITL Classifier.
    Recognizes the 3 Known-Good specs (mm):
      - SQUARE_CUBE    : 25 x 25 x 25 mm (flat top across all 4 corners)
      - ROUND_CYLINDER : Ø25 x 25 mm     (flat top)
      - RECT_BAR       : 25 x 50 x 5 mm  (flat top)

    Uses 3D surface planarity (plane RMSE, 4-corner drop, 2-axis drop) so that:
      - A true flat Cube has uniform height across all 4 corners (corner_drop < 3mm, plane_rmse < 1.8mm)
      - A lying Round/Hex Cylinder (2D square, curved/faceted 3D top) is NOT misclassified as a Cube
      - A Pyramid (sloped on both axes) is NOT misclassified as a Cube
    Any unfamiliar / non-flat / low-confidence object is routed to 'HITL' (Review / Waste Area).
    """
    if perimeter <= 0 or rw <= 0 or rh <= 0:
        return "UNKNOWN", 0.0, "HITL", "invalid_geometry"

    circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
    aspect_ratio = max(rw, rh) / min(rw, rh)
    rectangularity = area / (rw * rh)

    vertices = 0
    solidity = 1.0
    radial_cv = 0.05
    radial_min_max = 0.85

    if cnt is not None and len(cnt) >= 3:
        approx = cv2.approxPolyDP(cnt, 0.028 * perimeter, True)
        vertices = len(approx)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area / hull_area)

        if u_sub is not None and v_sub is not None:
            pts = cnt.reshape(-1, 2).astype(np.float32)
            dists = np.linalg.norm(pts - np.array([u_sub, v_sub], dtype=np.float32), axis=1)
            mean_r = float(np.mean(dists))
            if mean_r > 1e-3:
                radial_cv = float(np.std(dists) / mean_r)
                p95 = float(np.percentile(dists, 95))
                p05 = float(np.percentile(dists, 5))
                if p95 > 1e-3:
                    radial_min_max = p05 / p95

    # 3D Surface Planarity & 4-Corner Uniformity Check:
    # - True flat top (Cube, Upright Cylinder, Rect Bar): low plane RMSE, uniform corners & axes
    # - Pyramid: drops on BOTH axes (axis_drop_min >= 3.0) or low flatness_ratio
    # - Lying Cylinder / Lying Hex: flat on 1 axis, drops on cross-axis (axis_drop_max >= 2.8)
    is_pyramid_like = (
        height_peak_mm >= 10.0
        and (
            (flatness_ratio < 0.78 and height_cv > 0.14)
            or (axis_drop_min_mm >= 3.0 and axis_drop_max_mm >= 4.0)
        )
    )
    is_curved_side = (
        height_peak_mm >= 12.0
        and not is_pyramid_like
        and (plane_rmse_mm >= 2.2 or corner_drop_mm >= 4.0 or axis_drop_max_mm >= 3.8)
    )
    is_non_flat = is_pyramid_like or is_curved_side
    flat_factor = 0.55 if is_non_flat else 1.0

    has_size = (short_mm > 1.0 and long_mm > 1.0)

    # --- Candidate 1: RECT_BAR (25 x 50 x 5 mm) ---
    rect_shape = 0.0
    if aspect_ratio >= 1.35:
        ar_score = _soft_match(aspect_ratio, 2.0, 0.70)
        rect_score = min(1.0, rectangularity / 0.85)
        rect_shape = 0.55 * rect_score + 0.45 * ar_score
    if has_size and rect_shape > 0:
        sz = (
            _soft_match(short_mm, 25.0, 10.0)
            * _soft_match(long_mm, 50.0, 14.0)
            * _soft_match(height_mm, 5.0, 6.0)
        ) ** (1.0 / 3.0)
        rect_conf = rect_shape * (0.45 + 0.55 * sz) * flat_factor
    else:
        rect_conf = rect_shape * flat_factor

    # --- Candidate 2: SQUARE_CUBE (25 x 25 x 25 mm) ---
    cube_shape = 0.0
    if aspect_ratio < 1.35:
        rect_score = min(1.0, rectangularity / 0.88)
        corner_penalty = 1.0 if (rectangularity >= 0.80 and (radial_min_max < 0.81 or vertices == 4)) else 0.60
        cube_shape = rect_score * corner_penalty
    if has_size and cube_shape > 0:
        sz = (
            _soft_match(short_mm, 25.0, 10.0)
            * _soft_match(long_mm, 25.0, 10.0)
            * _soft_match(height_peak_mm, 25.0, 9.0)
        ) ** (1.0 / 3.0)
        cube_conf = cube_shape * (0.45 + 0.55 * sz) * flat_factor
    else:
        cube_conf = cube_shape * flat_factor

    # --- Candidate 3: ROUND_CYLINDER (Ø25 x 25 mm) ---
    cyl_shape = 0.0
    if aspect_ratio < 1.30:
        circ_score = min(1.0, circularity / 0.88)
        if radial_min_max >= 0.87 and radial_cv < 0.045 and circularity >= 0.84 and vertices >= 7:
            rad_score = 1.0
        elif radial_min_max >= 0.84 and radial_cv < 0.050 and vertices >= 7:
            rad_score = 0.80
        else:
            rad_score = 0.50
        cyl_shape = circ_score * rad_score
    if has_size and cyl_shape > 0:
        sz = (
            _soft_match(short_mm, 25.0, 10.0)
            * _soft_match(long_mm, 25.0, 10.0)
            * _soft_match(height_peak_mm, 25.0, 9.0)
        ) ** (1.0 / 3.0)
        cyl_conf = cyl_shape * (0.45 + 0.55 * sz) * flat_factor
    else:
        cyl_conf = cyl_shape * flat_factor

    candidates = [
        ("RECT_BAR", rect_conf),
        ("SQUARE_CUBE", cube_conf),
        ("ROUND_CYLINDER", cyl_conf),
    ]
    best_name, best_conf = max(candidates, key=lambda x: x[1])
    best_conf = float(max(0.0, min(1.0, best_conf)))

    if best_conf >= hitl_conf_threshold and not is_non_flat:
        return best_name, best_conf, "GOOD", f"known_good_conf={best_conf:.2f}"

    # Unfamiliar shape / curved or faceted side / non-flat top -> Route to HITL
    if is_pyramid_like:
        reason = f"non_flat_peak_flat={flatness_ratio:.2f}_rmse={plane_rmse_mm:.1f}"
    elif is_curved_side:
        reason = f"curved_side_drop={axis_drop_max_mm:.1f}mm_rmse={plane_rmse_mm:.1f}"
    elif 5 <= vertices <= 7 and circularity >= 0.74:
        reason = f"unfamiliar_polygon_v{vertices}_r={radial_min_max:.2f}"
    else:
        reason = f"low_conf_{best_conf:.2f}_near_{best_name[:4]}"

    return "UNKNOWN_SHAPE", best_conf, "HITL", reason


def compute_roll_and_box(cnt: np.ndarray) -> Tuple[float, np.ndarray, float, float]:
    """Compute minAreaRect roll angle in [-90, 90], 4-point box, width and height."""
    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), angle = rect
    roll_deg = angle + 90.0 if rw < rh else angle
    roll_deg = wrap_angle_180(roll_deg)
    box = np.int32(cv2.boxPoints(rect))
    return roll_deg, box, float(rw), float(rh)


def compute_3d_from_mask(
    obj_mask: np.ndarray,
    cnt: np.ndarray,
    cv_depth: np.ndarray,
    ctx: DetectorContext,
) -> Optional[Dict[str, Any]]:
    """
    Sample 3D depth, height above belt, centroid (u, v), and camera XYZ (m)
    from a binary object mask and contour. Shared across all detector modes.
    """
    min_valid_depth = float(ctx.config.get('min_valid_depth_mm', 150.0))
    depth_f = cv_depth.astype(np.float32)

    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    u_sub = float(M['m10'] / M['m00'])
    v_sub = float(M['m01'] / M['m00'])
    u, v = int(round(u_sub)), int(round(v_sub))

    eroded_mask = cv2.erode(obj_mask, np.ones((5, 5), np.uint8), iterations=1)
    target_mask = eroded_mask if np.count_nonzero(eroded_mask) > 20 else obj_mask

    valid_depth_pixels = (target_mask == 255) & (depth_f > min_valid_depth)
    obj_depths = depth_f[valid_depth_pixels]
    if obj_depths.size == 0:
        return None

    obj_depth_mm = float(np.median(obj_depths))
    roll_deg, box, rw, rh = compute_roll_and_box(cnt)
    area = float(cv2.contourArea(cnt))

    height_mm = 0.0
    height_peak_mm = 0.0
    flatness_ratio = 1.0
    height_cv = 0.0
    plane_rmse_mm = 0.0
    corner_drop_mm = 0.0
    axis_drop_max_mm = 0.0
    axis_drop_min_mm = 0.0

    if ctx.bg_depth_map is not None:
        valid_both = valid_depth_pixels & (ctx.bg_depth_map > min_valid_depth)
        if np.any(valid_both):
            ys, xs = np.where(valid_both)
            obj_heights = ctx.bg_depth_map[ys, xs] - depth_f[ys, xs]
            pos_mask = obj_heights > 1.0
            pos_heights = obj_heights[pos_mask]
            if pos_heights.size >= 15:
                height_mm = float(np.percentile(pos_heights, 50))
                height_peak_mm = float(np.percentile(pos_heights, 95))
                if height_peak_mm > 1.0:
                    flatness_ratio = float(height_mm / height_peak_mm)
                    height_cv = float(np.std(pos_heights) / height_peak_mm)

                # 3D Surface Planarity on INNER CORE ONLY (erode away ~16% border to ignore
                # camera-parallax vertical side walls on cubes & upright cylinders!)
                core_k = max(5, int(min(rw, rh) * 0.16))
                if core_k % 2 == 0:
                    core_k += 1
                core_mask = cv2.erode(obj_mask, np.ones((core_k, core_k), np.uint8), iterations=1)
                core_valid = (core_mask == 255) & (depth_f > min_valid_depth) & (ctx.bg_depth_map > min_valid_depth)
                if np.count_nonzero(core_valid) >= 25:
                    c_ys, c_xs = np.where(core_valid)
                    c_heights = ctx.bg_depth_map[c_ys, c_xs] - depth_f[c_ys, c_xs]
                    # Keep upper surface band (>= 45% peak) to exclude any remaining side-wall pixels
                    top_m = c_heights >= max(2.0, 0.45 * height_peak_mm)
                    if np.count_nonzero(top_m) >= 20 and rw > 6.0 and rh > 6.0:
                        t_h = c_heights[top_m].astype(np.float32)
                        px_x = c_xs[top_m].astype(np.float32) - u_sub
                        px_y = c_ys[top_m].astype(np.float32) - v_sub

                        # 1) Best-fit plane h = a*x + b*y + c -> RMSE
                        A = np.column_stack([px_x, px_y, np.ones_like(px_x)])
                        coeffs, _, _, _ = np.linalg.lstsq(A, t_h, rcond=None)
                        pred = A @ coeffs
                        plane_rmse_mm = float(np.sqrt(np.mean((t_h - pred) ** 2)))

                        # 2) Rotate into minAreaRect local axes to measure edge/corner drop
                        rect_angle = cv2.minAreaRect(cnt)[2]
                        theta = math.radians(rect_angle)
                        c_th, s_th = math.cos(theta), math.sin(theta)
                        lx = (px_x * c_th + px_y * s_th) / max(1.0, rw * 0.5)
                        ly = (-px_x * s_th + px_y * c_th) / max(1.0, rh * 0.5)

                        cx_m = np.abs(lx) < 0.30
                        ex_m = (np.abs(lx) >= 0.45) & (np.abs(lx) <= 0.85)
                        drop_x = (
                            float(np.median(t_h[cx_m]) - np.median(t_h[ex_m]))
                            if (np.count_nonzero(cx_m) >= 6 and np.count_nonzero(ex_m) >= 6)
                            else 0.0
                        )

                        cy_m = np.abs(ly) < 0.30
                        ey_m = (np.abs(ly) >= 0.45) & (np.abs(ly) <= 0.85)
                        drop_y = (
                            float(np.median(t_h[cy_m]) - np.median(t_h[ey_m]))
                            if (np.count_nonzero(cy_m) >= 6 and np.count_nonzero(ey_m) >= 6)
                            else 0.0
                        )
                        axis_drop_max_mm = max(0.0, max(drop_x, drop_y))
                        axis_drop_min_mm = max(0.0, min(drop_x, drop_y))

                        center_m = (np.abs(lx) < 0.30) & (np.abs(ly) < 0.30)
                        corner_m = (np.abs(lx) >= 0.40) & (np.abs(ly) >= 0.40)
                        if np.count_nonzero(center_m) >= 5 and np.count_nonzero(corner_m) >= 5:
                            corner_drop_mm = max(
                                0.0,
                                float(np.median(t_h[center_m]) - np.median(t_h[corner_m])),
                            )
            elif obj_heights.size > 0:
                height_mm = float(np.median(obj_heights))
                height_peak_mm = float(np.max(obj_heights))
        elif ctx.belt_center_depth_mm is not None:
            height_mm = float(ctx.belt_center_depth_mm - obj_depth_mm)
            height_peak_mm = height_mm
    elif ctx.belt_center_depth_mm is not None:
        height_mm = float(ctx.belt_center_depth_mm - obj_depth_mm)
        height_peak_mm = height_mm

    z_m = obj_depth_mm / 1000.0
    x_m = (u_sub - ctx.ppx) * z_m / ctx.fx
    y_m = (v_sub - ctx.ppy) * z_m / ctx.fy

    avg_f = 0.5 * (ctx.fx + ctx.fy)
    rw_mm = float(rw * obj_depth_mm / avg_f) if avg_f > 0 else 0.0
    rh_mm = float(rh * obj_depth_mm / avg_f) if avg_f > 0 else 0.0

    return {
        'u': u,
        'v': v,
        'u_sub': u_sub,
        'v_sub': v_sub,
        'depth_mm': obj_depth_mm,
        'height_mm': max(0.0, height_mm),
        'height_peak_mm': max(0.0, height_peak_mm),
        'flatness_ratio': flatness_ratio,
        'height_cv': height_cv,
        'plane_rmse_mm': plane_rmse_mm,
        'corner_drop_mm': corner_drop_mm,
        'axis_drop_max_mm': axis_drop_max_mm,
        'axis_drop_min_mm': axis_drop_min_mm,
        'short_mm': min(rw_mm, rh_mm),
        'long_mm': max(rw_mm, rh_mm),
        'roll_deg': roll_deg,
        'cam_xyz_m': [float(x_m), float(y_m), float(z_m)],
        'box': box,
        'rw': rw,
        'rh': rh,
        'area': area,
    }


def sort_and_assign_ids(detections: List[Dict[str, Any]], prioritize_height: bool = False) -> List[Dict[str, Any]]:
    """
    Sort detections and assign 1-based IDs and grasp_priority.
    When prioritize_height is True (overlapped items), taller/top-most objects come first.
    """
    if prioritize_height:
        detections.sort(
            key=lambda d: (
                1 if d.get('is_overlapped', False) else 0,
                round(d.get('height_mm', 0.0), 1),
                d.get('v_sub', 0.0),
            ),
            reverse=True,
        )
    else:
        detections.sort(key=lambda d: d.get('v_sub', 0.0), reverse=True)

    for idx, det in enumerate(detections):
        det['id'] = idx + 1
        det['grasp_priority'] = idx + 1
    return detections


class BaseDetector(ABC):
    """Abstract base class for all vision detector modes."""

    mode_name: str = 'base'

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        self.config = config
        self.logger = logger
        self.model_ready: bool = True
        self.status_note: str = 'OK'
        self._loaded: bool = False

    def log_info(self, msg: str):
        if self.logger is not None:
            self.logger.info(f"[{self.mode_name}] {msg}")

    def log_warn(self, msg: str):
        if self.logger is not None:
            self.logger.warn(f"[{self.mode_name}] {msg}")

    def log_error(self, msg: str):
        if self.logger is not None:
            self.logger.error(f"[{self.mode_name}] {msg}")

    def ensure_loaded(self):
        """Lazy-load weights/resources on first use."""
        if not self._loaded:
            self.load_model()
            self._loaded = True

    def load_model(self):
        """Override in AI detectors to lazy-load model weights."""
        self.model_ready = True
        self.status_note = 'READY'

    @abstractmethod
    def detect(
        self,
        cv_color: np.ndarray,
        cv_depth: np.ndarray,
        ctx: DetectorContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """Run detection and return (detections_list, optional_debug_mask)."""
        raise NotImplementedError
