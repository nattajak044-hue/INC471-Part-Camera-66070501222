#!/usr/bin/env python3
"""
object_detector_node.py (v5.1: Embedded GUI Camera Stream + Multi-Object Batch Snapshot)
"""

import json
import math
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters

# ================= CONFIGURATION =================
DETECTOR_MODE = 'depth_shape'   # 'depth_shape' หรือ 'yolo'
HEIGHT_MIN_MM = 6.0             # ความสูงขั้นต่ำที่นูนจากพื้นเดิม (mm)
HEIGHT_MAX_MM = 85.0            # ความสูงสูงสุดของชิ้นงาน (mm)
MIN_VALID_DEPTH_MM = 150.0      # ตัดระยะใกล้หน้าเลนส์เกินไป
MIN_CONTOUR_AREA = 350          # พื้นที่วัตถุเล็กสุด (pixel^2)
MAX_CONTOUR_AREA = 35000        # พื้นที่วัตถุใหญ่สุด
CONF_THRESHOLD = 0.50           # เกณฑ์ความมั่นใจขั้นต่ำ
TIMEOUT_SEC = 0.5               # ส่ง NO_OBJECT หากไม่เจอวัตถุเกินเวลานี้
EMA_ALPHA = 0.45                # ค่าความนิ่งของพิกัดชิ้นหลัก (0.1=นิ่งมาก, 1.0=ไวสุด)
# =================================================


def yaw_to_quaternion(yaw_rad: float):
    half = yaw_rad * 0.5
    return 0.0, 0.0, float(math.sin(half)), float(math.cos(half))


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')
        self.bridge = CvBridge()

        # Publishers (คงหัวข้อเดิม + เพิ่ม Debug Image สำหรับแสดงในแอป)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/detected_poses', 10)
        self.info_pub = self.create_publisher(String, '/detected_object_info', 10)
        self.debug_img_pub = self.create_publisher(Image, '/vision/debug_image', 5)
        self.mask_img_pub = self.create_publisher(Image, '/vision/depth_mask', 5)

        # Subscriber สำหรับรับคำสั่ง LOCK / UNLOCK / RESET_BG จาก Coordinator หรือแอป GUI
        self.snap_sub = self.create_subscription(
            String,
            '/vision/snapshot_cmd',
            self.snapshot_cmd_callback,
            10
        )

        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )

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

        # Background Reference (ตอนสายพานโล่ง)
        self.bg_depth_map = None
        self.bg_lab_img = None
        self.belt_center_depth_mm = None
        self.reset_background_flag = True

        # Snapshot Lock State
        self.snapshot_locked = False
        self.locked_payload = None
        self.locked_poses_msg = None
        self.locked_primary_pose = None

        # Temporal Smoothing สำหรับชิ้นเป้าหมายหลัก
        self.smooth_xyz = None
        self.smooth_roll = None

        # Latency Tracking
        self.latency_history = []

        self.last_detect_time = time.time()
        self.watchdog_timer = self.create_timer(0.2, self.check_timeout)

        self.get_logger().info(
            "Object Detector Node v5.1 (Embedded App Stream) Started! "
            "ส่งภาพออกทาง /vision/debug_image เรียบร้อย"
        )

    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.ppx = msg.k[2]
        self.ppy = msg.k[5]

    def snapshot_cmd_callback(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd == 'LOCK':
            self.snapshot_locked = True
            self.locked_payload = None
            self.get_logger().info("Snapshot Command: LOCK (กำลังบันทึกภาพนิ่งชุดวัตถุ...)")
        elif cmd == 'UNLOCK':
            self.snapshot_locked = False
            self.locked_payload = None
            self.locked_poses_msg = None
            self.locked_primary_pose = None
            self.get_logger().info("Snapshot Command: UNLOCK (กลับสู่โหมดตรวจจับต่อเนื่อง)")
        elif cmd == 'TOGGLE':
            self.snapshot_locked = not self.snapshot_locked
            if not self.snapshot_locked:
                self.locked_payload = None
                self.locked_poses_msg = None
                self.locked_primary_pose = None
            self.get_logger().info(f"Snapshot Command: TOGGLE -> Locked = {self.snapshot_locked}")
        elif cmd == 'RESET_BG':
            self.reset_background_flag = True
            self.snapshot_locked = False
            self.locked_payload = None
            self.get_logger().info("Command: RESET_BG (กำลังบันทึกพื้นหลังสายพานใหม่...)")

    def capture_background_reference(self, cv_color: np.ndarray, cv_depth: np.ndarray):
        """บันทึกภาพพื้นหลังทั้งเฟรม (Depth Map + LAB Color Image) ตอนสายพานโล่ง"""
        h, w = cv_depth.shape[:2]
        depth_f = cv_depth.astype(np.float32)

        belt_crop = depth_f[int(h * 0.15):int(h * 0.65), int(w * 0.45):int(w * 0.62)]
        valid = belt_crop[belt_crop > MIN_VALID_DEPTH_MM]
        if valid.size < 100:
            return

        self.belt_center_depth_mm = float(np.median(valid))
        self.bg_depth_map = depth_f.copy()

        blurred = cv2.GaussianBlur(cv_color, (5, 5), 0)
        self.bg_lab_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

        self.reset_background_flag = False
        self.get_logger().info(
            f"บันทึกพื้นหลังสายพานสำเร็จ! (Belt Depth: {self.belt_center_depth_mm:.1f} mm)"
        )

    def get_roi_mask(self, h: int, w: int):
        """กรอบพื้นที่รางสายพาน (ตัดรางอะลูมิเนียม เซนเซอร์ และฐาน Dobot ออก)"""
        rx1, rx2 = int(w * 0.43), int(w * 0.64)
        ry1, ry2 = int(h * 0.02), int(h * 0.71)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ry1:ry2, rx1:rx2] = 255
        return mask, (rx1, ry1, rx2, ry2)

    def refine_contour_with_rgb_diff(self, cv_color: np.ndarray, rough_cnt: np.ndarray):
        """ใช้ CIE LAB Frame Differencing ขัดเกลาขอบวัตถุให้คมกริบทุกสี"""
        if self.bg_lab_img is None:
            return rough_cnt

        h, w = cv_color.shape[:2]
        x, y, bw, bh = cv2.boundingRect(rough_cnt)
        pad = 16
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        curr_patch = cv2.GaussianBlur(cv_color[y1:y2, x1:x2], (5, 5), 0)
        curr_lab = cv2.cvtColor(curr_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab_patch = self.bg_lab_img[y1:y2, x1:x2]

        color_diff = np.linalg.norm(curr_lab - bg_lab_patch, axis=2)
        rgb_mask = np.where(color_diff > 24.0, 255, 0).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)
        rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(rgb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return rough_cnt

        best_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < MIN_CONTOUR_AREA * 0.6:
            return rough_cnt

        return best_cnt + np.array([[[x1, y1]]])

    def classify_geometry(self, area: float, perimeter: float, rw: float, rh: float):
        if perimeter <= 0 or rw <= 0 or rh <= 0:
            return "UNKNOWN", 0.0

        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        aspect_ratio = max(rw, rh) / min(rw, rh)
        rectangularity = area / (rw * rh)

        if circularity >= 0.85 and rectangularity < 0.85 and aspect_ratio < 1.30:
            return "ROUND_CYLINDER", min(1.0, circularity)
        elif rectangularity >= 0.78 and aspect_ratio < 1.30:
            return "SQUARE_BOX", min(1.0, rectangularity)
        elif rectangularity >= 0.70 and aspect_ratio >= 1.30:
            return "RECT_BAR", min(1.0, rectangularity)
        else:
            return "POLYGON_OBJ", min(1.0, max(rectangularity, 0.5))

    def detect_by_depth_shape(self, cv_color: np.ndarray, cv_depth: np.ndarray):
        if self.bg_depth_map is None or self.reset_background_flag:
            self.capture_background_reference(cv_color, cv_depth)
            if self.bg_depth_map is None:
                return [], np.zeros(cv_depth.shape[:2], dtype=np.uint8)

        h, w = cv_depth.shape[:2]
        depth_f = cv_depth.astype(np.float32)

        valid_both = (depth_f > MIN_VALID_DEPTH_MM) & (self.bg_depth_map > MIN_VALID_DEPTH_MM)
        height_map = np.where(valid_both, self.bg_depth_map - depth_f, 0.0)

        fg_mask = np.where(
            (height_map >= HEIGHT_MIN_MM) & (height_map <= HEIGHT_MAX_MM),
            255,
            0
        ).astype(np.uint8)

        roi_mask, _ = self.get_roi_mask(h, w)
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        rough_contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for r_cnt in rough_contours:
            r_area = cv2.contourArea(r_cnt)
            if r_area < MIN_CONTOUR_AREA or r_area > MAX_CONTOUR_AREA:
                continue

            cnt = self.refine_contour_with_rgb_diff(cv_color, r_cnt)
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            u_sub = M['m10'] / M['m00']
            v_sub = M['m01'] / M['m00']
            u, v = int(round(u_sub)), int(round(v_sub))

            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
            eroded_mask = cv2.erode(obj_mask, np.ones((5, 5), np.uint8), iterations=1)
            target_mask = eroded_mask if np.count_nonzero(eroded_mask) > 20 else obj_mask

            obj_depths = depth_f[(target_mask == 255) & (depth_f > MIN_VALID_DEPTH_MM)]
            obj_heights = height_map[(target_mask == 255) & valid_both]
            if obj_depths.size == 0:
                continue

            obj_depth_mm = float(np.median(obj_depths))
            height_mm = float(np.median(obj_heights)) if obj_heights.size > 0 else 0.0

            if height_mm < HEIGHT_MIN_MM * 0.8:
                continue

            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), angle = rect
            if rw < rh:
                roll_deg = angle + 90.0
            else:
                roll_deg = angle
            if roll_deg > 90.0:
                roll_deg -= 180.0
            elif roll_deg < -90.0:
                roll_deg += 180.0

            perimeter = cv2.arcLength(cnt, True)
            class_name, conf = self.classify_geometry(area, perimeter, rw, rh)
            box = np.int32(cv2.boxPoints(rect))

            z_m = obj_depth_mm / 1000.0
            x_m = (u_sub - self.ppx) * z_m / self.fx
            y_m = (v_sub - self.ppy) * z_m / self.fy

            if conf >= CONF_THRESHOLD:
                detections.append({
                    'class_name': class_name,
                    'conf': float(conf),
                    'u': u,
                    'v': v,
                    'u_sub': float(u_sub),
                    'v_sub': float(v_sub),
                    'depth_mm': obj_depth_mm,
                    'height_mm': float(height_mm),
                    'roll_deg': float(roll_deg),
                    'cam_xyz_m': [float(x_m), float(y_m), float(z_m)],
                    'box': box,
                    'area': float(area),
                })

        detections.sort(key=lambda d: d['v_sub'], reverse=True)
        for idx, det in enumerate(detections):
            det['id'] = idx + 1

        return detections, fg_mask

    def detect_by_yolo(self, cv_color: np.ndarray):
        return []

    def check_timeout(self):
        if self.snapshot_locked and self.locked_payload is not None:
            info_msg = String()
            info_msg.data = json.dumps(self.locked_payload)
            self.info_pub.publish(info_msg)
            if self.locked_primary_pose is not None:
                self.pose_pub.publish(self.locked_primary_pose)
            if self.locked_poses_msg is not None:
                self.poses_pub.publish(self.locked_poses_msg)
            return

        if time.time() - self.last_detect_time > TIMEOUT_SEC:
            self.smooth_xyz = None
            self.smooth_roll = None
            msg = String()
            msg.data = json.dumps({
                'status': 'NO_OBJECT',
                'object_count': 0,
                'all_objects': [],
                'snapshot_locked': self.snapshot_locked,
                'belt_depth_mm': round(self.belt_center_depth_mm, 1) if self.belt_center_depth_mm else None,
                'timestamp': time.time()
            })
            self.info_pub.publish(msg)

    def publish_gui_images(self, cv_color: np.ndarray, fg_mask: np.ndarray):
        """ส่งภาพเข้า ROS 2 Topic เพื่อให้แอป Node Switchboard ดึงไปแสดงผล"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
            self.debug_img_pub.publish(debug_msg)
            if fg_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(fg_mask, encoding='mono8')
                self.mask_img_pub.publish(mask_msg)
        except Exception:
            pass

    def image_callback(self, color_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        t_start = time.perf_counter()

        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            h, w = cv_color.shape[:2]

            if DETECTOR_MODE == 'yolo':
                detections = self.detect_by_yolo(cv_color)
                fg_mask = None
            else:
                detections, fg_mask = self.detect_by_depth_shape(cv_color, cv_depth)
                _, (rx1, ry1, rx2, ry2) = self.get_roi_mask(h, w)
                roi_color = (0, 165, 255) if self.snapshot_locked else (255, 255, 255)
                cv2.rectangle(cv_color, (rx1, ry1), (rx2, ry2), roi_color, 2)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.latency_history.append(latency_ms)
            if len(self.latency_history) > 30:
                self.latency_history.pop(0)
            avg_latency_ms = float(np.mean(self.latency_history))
            fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

            lock_str = " [SNAPSHOT LOCKED]" if self.snapshot_locked else ""
            bg_text = (
                f"Belt:{self.belt_center_depth_mm:.0f}mm | Obj:{len(detections)} | "
                f"{avg_latency_ms:.1f}ms ({fps:.0f}FPS){lock_str}"
                if self.belt_center_depth_mm else "Calibrating..."
            )
            cv2.putText(cv_color, bg_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)

            if self.snapshot_locked and self.locked_payload is not None:
                cv2.putText(
                    cv_color, "LOCKED QUEUE ACTIVE", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2
                )
                self.publish_gui_images(cv_color, fg_mask)
                return

            if detections:
                target = detections[0]
                x_raw, y_raw, z_raw = target['cam_xyz_m']
                roll_raw = target['roll_deg']

                curr_xyz = np.array([x_raw, y_raw, z_raw])
                if self.smooth_xyz is None or np.linalg.norm(curr_xyz - self.smooth_xyz) > 0.03:
                    self.smooth_xyz = curr_xyz
                    self.smooth_roll = roll_raw
                else:
                    self.smooth_xyz = EMA_ALPHA * curr_xyz + (1.0 - EMA_ALPHA) * self.smooth_xyz
                    self.smooth_roll = EMA_ALPHA * roll_raw + (1.0 - EMA_ALPHA) * self.smooth_roll

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

                for idx, det in enumerate(detections):
                    ox, oy, oz = (x_c, y_c, z_c) if idx == 0 else det['cam_xyz_m']
                    oroll = roll_final if idx == 0 else det['roll_deg']
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

                    all_objects_list.append({
                        'id': det['id'],
                        'class_name': det['class_name'],
                        'confidence': round(det['conf'], 3),
                        'height_mm': round(det['height_mm'], 1),
                        'roll_deg': round(oroll, 2),
                        'cam_xyz_m': [round(ox, 4), round(oy, 4), round(oz, 4)],
                        'pixel_uv': [det['u'], det['v']]
                    })

                    color = (0, 255, 0) if idx == 0 else (0, 200, 255)
                    cv2.drawContours(cv_color, [det['box']], 0, color, 2)
                    cv2.circle(cv_color, (det['u'], det['v']), 4, (0, 0, 255), -1)
                    cv2.putText(
                        cv_color, f"#{det['id']} {det['class_name'][:6]}",
                        (det['u'] - 25, det['v'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2
                    )

                self.poses_pub.publish(poses_msg)

                payload = {
                    'status': 'OK',
                    'object_count': len(all_objects_list),
                    'snapshot_locked': self.snapshot_locked,
                    'latency_ms': round(avg_latency_ms, 2),
                    'class_name': target['class_name'],
                    'confidence': round(target['conf'], 3),
                    'height_mm': round(target['height_mm'], 1),
                    'roll_deg': round(roll_final, 2),
                    'cam_xyz_m': [round(x_c, 4), round(y_c, 4), round(z_c, 4)],
                    'all_objects': all_objects_list
                }

                info_msg = String()
                info_msg.data = json.dumps(payload)
                self.info_pub.publish(info_msg)

                if self.snapshot_locked and self.locked_payload is None:
                    self.locked_payload = payload
                    self.locked_primary_pose = pose_msg
                    self.locked_poses_msg = poses_msg
                    self.get_logger().info(f"Snapshot LOCKED! บันทึกวัตถุไว้ในคิวทั้งหมด {len(all_objects_list)} ชิ้น")

            # ส่งภาพเข้าแอป Node Switchboard แทนการเปิดหน้าต่าง OpenCV แยก
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
