#!/usr/bin/env python3
"""
object_detector_node.py (v6.0: Modular Multi-Mode Architecture + Dynamic Switch + Bug Fixes)

Supported Modes:
  - depth_shape   : 3D Depth + CIE LAB geometry baseline (No AI, ~13 ms)
  - yolo_seg      : YOLOv8-seg + 3D depth mask sampling (GOOD / WASTE / HITL)
  - sam_anomaly   : FastSAM + EfficientAD fine surface anomaly check (throttled / trigger)
  - hybrid_fusion : Production Depth Guard + YOLOv8-seg + HITL routing
  - data_collect  : Auto-crop + YOLO polygon dataset collector
  - cascade_qc    : Two-stage QC (real-time HybridFusion + on-LOCK/INSPECT EfficientAD)
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    from detectors import (
        AVAILABLE_MODES,
        BaseDetector,
        DetectorContext,
        create_detector,
        wrap_angle_180,
    )
except ImportError:
    from vision_pkg.detectors import (
        AVAILABLE_MODES,
        BaseDetector,
        DetectorContext,
        create_detector,
        wrap_angle_180,
    )

DECISION_COLORS_BGR = {
    'GOOD': (0, 230, 0),
    'WASTE': (0, 40, 255),
    'HITL': (0, 165, 255),
    'UNKNOWN': (0, 165, 255),
}


def yaw_to_quaternion(yaw_rad: float):
    half = yaw_rad * 0.5
    return 0.0, 0.0, float(math.sin(half)), float(math.cos(half))


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')
        self.bridge = CvBridge()

        # ================= ROS 2 PARAMETERS =================
        self.declare_parameter('detector_mode', 'depth_shape')
        self.declare_parameter('height_min_mm', 3.0)
        self.declare_parameter('height_max_mm', 85.0)
        self.declare_parameter('min_valid_depth_mm', 150.0)
        self.declare_parameter('min_contour_area', 600.0)
        self.declare_parameter('max_contour_area', 35000.0)
        self.declare_parameter('conf_threshold', 0.50)
        self.declare_parameter('hitl_conf_threshold', 0.75)
        self.declare_parameter('timeout_sec', 0.5)
        self.declare_parameter('ema_alpha', 0.45)
        self.declare_parameter('bg_calib_frames', 15)
        self.declare_parameter('lab_diff_threshold', 28.0)
        self.declare_parameter('roi_x1_ratio', 0.43)
        self.declare_parameter('roi_x2_ratio', 0.64)
        self.declare_parameter('roi_y1_ratio', 0.02)
        self.declare_parameter('roi_y2_ratio', 0.71)
        self.declare_parameter('single_layer_max_mm', 45.0)
        self.declare_parameter('fusion_match_radius_px', 45.0)
        self.declare_parameter('yolo_model_path', '~/ros2_ws/models/best.pt')
        self.declare_parameter('yolo_full_frame', True)
        self.declare_parameter('fastsam_model_path', '')
        self.declare_parameter('anomaly_model_path', '')
        self.declare_parameter('anomaly_threshold', 0.50)
        self.declare_parameter('anomaly_hitl_low', 0.35)
        self.declare_parameter('sam_interval_sec', 0.5)
        self.declare_parameter('dataset_dir', '~/ros2_ws/datasets/conveyor_collect')
        self.declare_parameter('auto_collect_interval_sec', 1.5)
        # ====================================================

        self.config: Dict[str, Any] = self._read_all_params()
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # Detector Strategy Cache (Lazy-loaded per mode)
        self.detectors_cache: Dict[str, BaseDetector] = {}
        self.current_mode: str = ''
        self.active_detector: Optional[BaseDetector] = None
        self.switch_mode(str(self.config['detector_mode']))

        # Publishers (คงหัวข้อเดิม 100%)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/detected_poses', 10)
        self.info_pub = self.create_publisher(String, '/detected_object_info', 10)
        self.debug_img_pub = self.create_publisher(Image, '/vision/debug_image', 5)
        self.mask_img_pub = self.create_publisher(Image, '/vision/depth_mask', 5)

        # Subscriber รับคำสั่ง LOCK / UNLOCK / TOGGLE / RESET_BG / MODE:<name> / COLLECT / INSPECT
        self.snap_sub = self.create_subscription(
            String,
            '/vision/snapshot_cmd',
            self.snapshot_cmd_callback,
            10,
        )

        # Camera Intrinsics
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.ppx: Optional[float] = None
        self.ppy: Optional[float] = None
        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10,
        )

        # Synchronized RGB + Aligned Depth
        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw'
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # Multi-frame Background Reference State
        self.bg_depth_map: Optional[np.ndarray] = None
        self.bg_depth_noise_map: Optional[np.ndarray] = None
        self.bg_lab_img: Optional[np.ndarray] = None
        self.belt_center_depth_mm: Optional[float] = None
        self.bg_depth_samples: List[np.ndarray] = []
        self.bg_lab_samples: List[np.ndarray] = []
        self.reset_background_flag: bool = True

        # Snapshot Lock State
        self.snapshot_locked: bool = False
        self.lock_request_time: Optional[float] = None
        self.lock_wait_timeout_sec: float = 2.0
        self.locked_payload: Optional[Dict[str, Any]] = None
        self.locked_poses_msg: Optional[PoseArray] = None
        self.locked_primary_pose: Optional[PoseStamped] = None

        # On-demand Triggers
        self.trigger_inspect: bool = False
        self.trigger_collect: bool = False
        self.auto_collect: bool = False

        # Temporal Smoothing สำหรับชิ้นเป้าหมายหลัก
        self.smooth_xyz: Optional[np.ndarray] = None
        self.smooth_roll: Optional[float] = None

        # Latency Tracking
        self.latency_history: List[float] = []
        self.last_detect_time: float = time.time()
        self.watchdog_timer = self.create_timer(0.2, self.check_timeout)

        self.get_logger().info(
            f"Object Detector Node v6.0 Started! Mode='{self.current_mode}' | "
            f"Available={AVAILABLE_MODES}"
        )

    def _read_all_params(self) -> Dict[str, Any]:
        keys = [
            'detector_mode', 'height_min_mm', 'height_max_mm', 'min_valid_depth_mm',
            'min_contour_area', 'max_contour_area', 'conf_threshold', 'hitl_conf_threshold',
            'timeout_sec', 'ema_alpha', 'bg_calib_frames', 'lab_diff_threshold',
            'roi_x1_ratio', 'roi_x2_ratio', 'roi_y1_ratio', 'roi_y2_ratio',
            'single_layer_max_mm', 'fusion_match_radius_px',
            'yolo_model_path', 'yolo_full_frame', 'fastsam_model_path', 'anomaly_model_path',
            'anomaly_threshold', 'anomaly_hitl_low', 'sam_interval_sec',
            'dataset_dir', 'auto_collect_interval_sec',
        ]
        return {k: self.get_parameter(k).value for k in keys}

    def _on_parameters_changed(self, params):
        for p in params:
            self.config[p.name] = p.value
            if p.name == 'detector_mode':
                self.switch_mode(str(p.value))
            elif p.name in ('yolo_model_path', 'fastsam_model_path', 'anomaly_model_path'):
                # Invalidate cached AI detectors so new weights load on next frame
                for m in ('yolo_seg', 'sam_anomaly', 'hybrid_fusion', 'cascade_qc'):
                    self.detectors_cache.pop(m, None)
                self.switch_mode(self.current_mode)
        return SetParametersResult(successful=True)

    def switch_mode(self, mode_name: str):
        req = mode_name.strip().lower()
        if req not in AVAILABLE_MODES:
            self.get_logger().warn(
                f"ไม่รู้จักโหมด '{mode_name}' (รองรับ: {AVAILABLE_MODES}) — คงโหมดเดิมไว้"
            )
            if self.active_detector is None:
                req = 'depth_shape'
            else:
                return

        if req not in self.detectors_cache:
            self.detectors_cache[req] = create_detector(req, self.config, self.get_logger())

        self.active_detector = self.detectors_cache[req]
        self.active_detector.config = self.config
        self.current_mode = self.active_detector.mode_name

        # Reset smoothing & snapshot lock on mode switch
        self.smooth_xyz = None
        self.smooth_roll = None
        self._clear_lock()
        self.get_logger().info(
            f"สลับโหมด Vision -> '{self.current_mode}' (status={self.active_detector.status_note})"
        )

    def _clear_lock(self):
        self.snapshot_locked = False
        self.lock_request_time = None
        self.locked_payload = None
        self.locked_poses_msg = None
        self.locked_primary_pose = None

    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.ppx = msg.k[2]
        self.ppy = msg.k[5]

    def snapshot_cmd_callback(self, msg: String):
        raw = msg.data.strip()
        cmd_upper = raw.upper()

        if cmd_upper.startswith('MODE:'):
            new_mode = raw.split(':', 1)[1].strip()
            self.switch_mode(new_mode)
            return

        if cmd_upper == 'LOCK':
            self.snapshot_locked = True
            self.locked_payload = None
            self.locked_poses_msg = None
            self.locked_primary_pose = None
            self.lock_request_time = time.time()
            self.get_logger().info("Snapshot Command: LOCK (กำลังบันทึกภาพนิ่งชุดวัตถุ...)")
        elif cmd_upper == 'UNLOCK':
            self._clear_lock()
            self.get_logger().info("Snapshot Command: UNLOCK (กลับสู่โหมดตรวจจับต่อเนื่อง)")
        elif cmd_upper == 'TOGGLE':
            if self.snapshot_locked:
                self._clear_lock()
                self.get_logger().info("Snapshot Command: TOGGLE -> UNLOCKED")
            else:
                self.snapshot_locked = True
                self.locked_payload = None
                self.locked_poses_msg = None
                self.locked_primary_pose = None
                self.lock_request_time = time.time()
                self.get_logger().info("Snapshot Command: TOGGLE -> LOCKING...")
        elif cmd_upper == 'RESET_BG':
            self.reset_background_flag = True
            self.bg_depth_samples.clear()
            self.bg_lab_samples.clear()
            self.bg_depth_map = None
            self.bg_depth_noise_map = None
            self.bg_lab_img = None
            self._clear_lock()
            self.get_logger().info("Command: RESET_BG (กำลังเก็บพื้นหลังสายพานแบบ Multi-frame...)")
        elif cmd_upper == 'COLLECT':
            self.trigger_collect = True
            self.get_logger().info("Command: COLLECT (บันทึกภาพ Dataset 1 ชุด)")
        elif cmd_upper == 'AUTO_COLLECT:ON':
            self.auto_collect = True
            self.get_logger().info("Command: AUTO_COLLECT ON")
        elif cmd_upper == 'AUTO_COLLECT:OFF':
            self.auto_collect = False
            self.get_logger().info("Command: AUTO_COLLECT OFF")
        elif cmd_upper == 'INSPECT':
            self.trigger_inspect = True
            self.get_logger().info("Command: INSPECT (สั่งตรวจละเอียด On-Demand)")

    def accumulate_background_reference(self, cv_color: np.ndarray, cv_depth: np.ndarray) -> bool:
        """
        เก็บภาพพื้นหลังหลายเฟรม (Multi-frame median) เพื่อลด noise และกำจัดรู 0mm ของ RealSense
        คืนค่า True เมื่อ Calibrate เสร็จสมบูรณ์
        """
        target_frames = max(1, int(self.config.get('bg_calib_frames', 12)))
        min_valid = float(self.config.get('min_valid_depth_mm', 150.0))

        depth_f = cv_depth.astype(np.float32)
        blurred = cv2.GaussianBlur(cv_color, (5, 5), 0)
        lab_f = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

        self.bg_depth_samples.append(depth_f)
        self.bg_lab_samples.append(lab_f)

        if len(self.bg_depth_samples) < target_frames:
            return False

        depth_stack = np.stack(self.bg_depth_samples, axis=0)
        depth_stack = np.where(depth_stack > min_valid, depth_stack, np.nan)
        with np.errstate(all='ignore'):
            median_depth = np.nanmedian(depth_stack, axis=0)
            std_depth = np.nanstd(depth_stack, axis=0)
        median_depth = np.nan_to_num(median_depth, nan=0.0).astype(np.float32)
        std_depth = np.nan_to_num(std_depth, nan=2.0).astype(np.float32)

        # Adaptive per-pixel noise floor: suppresses empty-belt depth jitter automatically
        noise_floor = np.clip(2.5 * std_depth + 2.0, 4.0, 9.0).astype(np.float32)

        h, w = median_depth.shape[:2]
        belt_crop = median_depth[int(h * 0.15):int(h * 0.65), int(w * 0.45):int(w * 0.62)]
        valid = belt_crop[belt_crop > min_valid]
        if valid.size < 100:
            self.bg_depth_samples.clear()
            self.bg_lab_samples.clear()
            return False

        self.belt_center_depth_mm = float(np.median(valid))
        self.bg_depth_map = median_depth
        self.bg_depth_noise_map = noise_floor
        self.bg_lab_img = np.mean(np.stack(self.bg_lab_samples, axis=0), axis=0).astype(np.float32)

        self.bg_depth_samples.clear()
        self.bg_lab_samples.clear()
        self.reset_background_flag = False

        avg_noise = float(np.median(noise_floor[int(h * 0.15):int(h * 0.65), int(w * 0.45):int(w * 0.62)]))
        self.get_logger().info(
            f"บันทึกพื้นหลังสายพานสำเร็จ ({target_frames} เฟรม Median)! "
            f"(Belt Depth: {self.belt_center_depth_mm:.1f} mm, Noise Floor: {avg_noise:.1f} mm)"
        )
        return True

    def get_roi_mask(self, h: int, w: int):
        """กรอบพื้นที่รางสายพาน (ปรับผ่าน ROS 2 Parameters ได้)"""
        rx1 = int(w * float(self.config.get('roi_x1_ratio', 0.43)))
        rx2 = int(w * float(self.config.get('roi_x2_ratio', 0.64)))
        ry1 = int(h * float(self.config.get('roi_y1_ratio', 0.02)))
        ry2 = int(h * float(self.config.get('roi_y2_ratio', 0.71)))
        rx1, rx2 = max(0, min(rx1, w - 1)), max(1, min(rx2, w))
        ry1, ry2 = max(0, min(ry1, h - 1)), max(1, min(ry2, h))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ry1:ry2, rx1:rx2] = 255
        return mask, (rx1, ry1, rx2, ry2)

    def check_timeout(self):
        # Auto-cancel pending lock if belt stays empty too long
        if (
            self.snapshot_locked
            and self.locked_payload is None
            and self.lock_request_time is not None
            and (time.time() - self.lock_request_time > self.lock_wait_timeout_sec)
        ):
            self.get_logger().warn(
                "ยกเลิกคำสั่ง LOCK อัตโนมัติ: ไม่พบวัตถุบนสายพานภายใน 2.0 วินาที"
            )
            self._clear_lock()

        if self.snapshot_locked and self.locked_payload is not None:
            info_msg = String()
            info_msg.data = json.dumps(self.locked_payload)
            self.info_pub.publish(info_msg)
            if self.locked_primary_pose is not None:
                self.pose_pub.publish(self.locked_primary_pose)
            if self.locked_poses_msg is not None:
                self.poses_pub.publish(self.locked_poses_msg)
            return

        timeout_sec = float(self.config.get('timeout_sec', 0.5))
        if time.time() - self.last_detect_time > timeout_sec:
            self.smooth_xyz = None
            self.smooth_roll = None
            msg = String()
            msg.data = json.dumps({
                'status': 'NO_OBJECT',
                'detector_mode': self.current_mode,
                'available_modes': AVAILABLE_MODES,
                'model_ready': bool(self.active_detector.model_ready) if self.active_detector else False,
                'model_status': self.active_detector.status_note if self.active_detector else 'NONE',
                'object_count': 0,
                'good_count': 0,
                'waste_count': 0,
                'hitl_count': 0,
                'all_objects': [],
                'snapshot_locked': self.snapshot_locked,
                'belt_depth_mm': round(self.belt_center_depth_mm, 1) if self.belt_center_depth_mm else None,
                'timestamp': time.time(),
            })
            self.info_pub.publish(msg)

    def publish_gui_images(self, cv_color: np.ndarray, fg_mask: Optional[np.ndarray]):
        """ส่งภาพเข้า ROS 2 Topic เพื่อให้แอป Node Switchboard ดึงไปแสดงผล"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
            self.debug_img_pub.publish(debug_msg)
            if fg_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(fg_mask, encoding='mono8')
                self.mask_img_pub.publish(mask_msg)
        except Exception:
            pass

    def _draw_hud_and_roi(
        self,
        cv_color: np.ndarray,
        roi_rect,
        det_count: int,
        avg_latency_ms: float,
        fps: float,
    ):
        rx1, ry1, rx2, ry2 = roi_rect
        roi_color = (0, 165, 255) if self.snapshot_locked else (255, 255, 255)
        cv2.rectangle(cv_color, (rx1, ry1), (rx2, ry2), roi_color, 2)

        lock_str = " [LOCKED]" if self.snapshot_locked else ""
        if self.reset_background_flag or self.bg_depth_map is None:
            target_n = max(1, int(self.config.get('bg_calib_frames', 12)))
            bg_text = f"Calibrating BG {len(self.bg_depth_samples)}/{target_n}..."
        else:
            bg_text = (
                f"Mode:{self.current_mode} | Belt:{self.belt_center_depth_mm:.0f}mm | "
                f"Obj:{det_count} | {avg_latency_ms:.1f}ms ({fps:.0f}FPS){lock_str}"
            )
        cv2.putText(cv_color, bg_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

        status_note = self.active_detector.status_note if self.active_detector else ''
        if status_note:
            note_color = (80, 255, 120) if (self.active_detector and self.active_detector.model_ready) else (0, 140, 255)
            cv2.putText(cv_color, f"Status: {status_note}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, note_color, 1)

    def image_callback(self, color_msg: Image, depth_msg: Image):
        if self.fx is None or self.active_detector is None:
            return

        t_start = time.perf_counter()

        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            h, w = cv_color.shape[:2]
            roi_mask, roi_rect = self.get_roi_mask(h, w)

            # Multi-frame background calibration when needed
            if self.bg_depth_map is None or self.reset_background_flag:
                done = self.accumulate_background_reference(cv_color, cv_depth)
                if not done:
                    self._draw_hud_and_roi(cv_color, roi_rect, 0, 0.0, 0.0)
                    self.publish_gui_images(cv_color, np.zeros((h, w), dtype=np.uint8))
                    return

            # If snapshot is already locked with a frozen payload, keep streaming live HUD
            if self.snapshot_locked and self.locked_payload is not None:
                avg_lat = float(np.mean(self.latency_history)) if self.latency_history else 0.0
                fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
                self._draw_hud_and_roi(
                    cv_color, roi_rect, self.locked_payload.get('object_count', 0), avg_lat, fps
                )
                cv2.putText(
                    cv_color, "LOCKED QUEUE ACTIVE", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2,
                )
                self.publish_gui_images(cv_color, None)
                return

            # Build per-frame context & run active detector strategy
            ctx = DetectorContext(
                fx=self.fx,
                fy=self.fy,
                ppx=self.ppx,
                ppy=self.ppy,
                bg_depth_map=self.bg_depth_map,
                bg_depth_noise_map=self.bg_depth_noise_map,
                bg_lab_img=self.bg_lab_img,
                belt_center_depth_mm=self.belt_center_depth_mm,
                roi_mask=roi_mask,
                roi_rect=roi_rect,
                config=self.config,
                snapshot_locked=self.snapshot_locked,
                trigger_inspect=self.trigger_inspect,
                trigger_collect=self.trigger_collect,
                auto_collect=self.auto_collect,
                logger=self.get_logger(),
            )
            # Consume one-shot triggers
            self.trigger_inspect = False
            self.trigger_collect = False

            detections, fg_mask = self.active_detector.detect(cv_color, cv_depth, ctx)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.latency_history.append(latency_ms)
            if len(self.latency_history) > 30:
                self.latency_history.pop(0)
            avg_latency_ms = float(np.mean(self.latency_history))
            fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

            self._draw_hud_and_roi(cv_color, roi_rect, len(detections), avg_latency_ms, fps)

            if detections:
                target = detections[0]
                x_raw, y_raw, z_raw = target['cam_xyz_m']
                roll_raw = float(target['roll_deg'])
                ema_alpha = float(self.config.get('ema_alpha', 0.45))

                curr_xyz = np.array([x_raw, y_raw, z_raw])
                if self.smooth_xyz is None or np.linalg.norm(curr_xyz - self.smooth_xyz) > 0.03:
                    self.smooth_xyz = curr_xyz
                    self.smooth_roll = roll_raw
                else:
                    self.smooth_xyz = ema_alpha * curr_xyz + (1.0 - ema_alpha) * self.smooth_xyz
                    # Shortest-angle wrap across ±90 deg boundary (fixes EMA wrap bug)
                    angle_diff = wrap_angle_180(roll_raw - self.smooth_roll)
                    self.smooth_roll = wrap_angle_180(self.smooth_roll + ema_alpha * angle_diff)

                x_c, y_c, z_c = self.smooth_xyz.tolist()
                roll_final = float(self.smooth_roll)
                self.last_detect_time = time.time()

                pose_msg = PoseStamped()
                pose_msg.header.stamp = color_msg.header.stamp
                pose_msg.header.frame_id = color_msg.header.frame_id
                pose_msg.pose.position.x = float(x_c)
                pose_msg.pose.position.y = float(y_c)
                pose_msg.pose.position.z = float(z_c)
                qx, qy, qz, qw = yaw_to_quaternion(math.radians(roll_final))
                pose_msg.pose.orientation.x = qx
                pose_msg.pose.orientation.y = qy
                pose_msg.pose.orientation.z = qz
                pose_msg.pose.orientation.w = qw
                self.pose_pub.publish(pose_msg)

                poses_msg = PoseArray()
                poses_msg.header = pose_msg.header
                all_objects_list = []

                good_count = 0
                waste_count = 0
                hitl_count = 0

                for idx, det in enumerate(detections):
                    ox, oy, oz = (x_c, y_c, z_c) if idx == 0 else det['cam_xyz_m']
                    oroll = roll_final if idx == 0 else float(det['roll_deg'])
                    oqx, oqy, oqz, oqw = yaw_to_quaternion(math.radians(oroll))

                    p = Pose()
                    p.position.x = float(ox)
                    p.position.y = float(oy)
                    p.position.z = float(oz)
                    p.orientation.x = oqx
                    p.orientation.y = oqy
                    p.orientation.z = oqz
                    p.orientation.w = oqw
                    poses_msg.poses.append(p)

                    decision = det.get('decision', 'UNCLASSIFIED')
                    if decision == 'GOOD':
                        good_count += 1
                    elif decision == 'WASTE':
                        waste_count += 1
                    elif decision in ('HITL', 'UNKNOWN'):
                        hitl_count += 1

                    all_objects_list.append({
                        'id': det['id'],
                        'class_name': det['class_name'],
                        'decision': decision,
                        'decision_reason': det.get('decision_reason', ''),
                        'confidence': round(float(det['conf']), 3),
                        'anomaly_score': det.get('anomaly_score'),
                        'is_overlapped': bool(det.get('is_overlapped', False)),
                        'grasp_priority': int(det.get('grasp_priority', idx + 1)),
                        'source_mode': det.get('source_mode', self.current_mode),
                        'height_mm': round(float(det['height_mm']), 1),
                        'roll_deg': round(oroll, 2),
                        'cam_xyz_m': [round(ox, 4), round(oy, 4), round(oz, 4)],
                        'pixel_uv': [int(det['u']), int(det['v'])],
                    })

                    color = DECISION_COLORS_BGR.get(
                        decision,
                        (0, 255, 0) if idx == 0 else (0, 200, 255),
                    )
                    cv2.drawContours(cv_color, [det['box']], 0, color, 2)
                    cv2.circle(cv_color, (det['u'], det['v']), 4, (0, 0, 255), -1)
                    tag = (
                        f"{det['class_name'][:5]}:{decision}"
                        if decision != 'UNCLASSIFIED'
                        else det['class_name'][:6]
                    )
                    ov_tag = "*" if det.get('is_overlapped') else ""
                    cv2.putText(
                        cv_color,
                        f"#{det['id']}{ov_tag} {tag}",
                        (det['u'] - 38, det['v'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        2,
                    )

                self.poses_pub.publish(poses_msg)

                payload = {
                    'status': 'OK',
                    'detector_mode': self.current_mode,
                    'available_modes': AVAILABLE_MODES,
                    'model_ready': bool(self.active_detector.model_ready),
                    'model_status': self.active_detector.status_note,
                    'object_count': len(all_objects_list),
                    'good_count': good_count,
                    'waste_count': waste_count,
                    'hitl_count': hitl_count,
                    'snapshot_locked': self.snapshot_locked,
                    'latency_ms': round(avg_latency_ms, 2),
                    'class_name': target['class_name'],
                    'decision': target.get('decision', 'UNCLASSIFIED'),
                    'decision_reason': target.get('decision_reason', ''),
                    'anomaly_score': target.get('anomaly_score'),
                    'confidence': round(float(target['conf']), 3),
                    'height_mm': round(float(target['height_mm']), 1),
                    'roll_deg': round(roll_final, 2),
                    'cam_xyz_m': [round(x_c, 4), round(y_c, 4), round(z_c, 4)],
                    'all_objects': all_objects_list,
                }

                info_msg = String()
                info_msg.data = json.dumps(payload)
                self.info_pub.publish(info_msg)

                if self.snapshot_locked and self.locked_payload is None:
                    self.locked_payload = payload
                    self.locked_primary_pose = pose_msg
                    self.locked_poses_msg = poses_msg
                    self.lock_request_time = None
                    self.get_logger().info(
                        f"Snapshot LOCKED! บันทึกวัตถุไว้ในคิวทั้งหมด {len(all_objects_list)} ชิ้น "
                        f"(GOOD={good_count}, WASTE={waste_count}, HITL={hitl_count})"
                    )

            self.publish_gui_images(cv_color, fg_mask)

        except Exception as e:
            self.get_logger().error(f"Error in object_detector_node: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
