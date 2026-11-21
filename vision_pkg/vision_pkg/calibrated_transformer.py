#!/usr/bin/env python3
"""
calibrated_transformer.py (v3: Dual-Robot Multi-Object Batch Transformer)

หน้าที่:
  1. โหลดเมทริกซ์ Calibration ของทั้ง Dobot 1 และ Dobot 2 พร้อมกัน
  2. แปลงพิกัดวัตถุทั้งหมดในเฟรม ('all_objects') จาก camera_color_optical_frame (m)
     -> พิกัดฐาน Dobot 1 (dobot1_xyz_mm) และ Dobot 2 (dobot2_xyz_mm) หน่วย mm
  3. คำนวณระยะเอื้อม (Reachability) และแนะนำหุ่นยนต์ที่ใกล้ที่สุด (nearest_robot)
     เพื่อป้อนให้ Dynamic Task Allocator
  4. Broadcast Static TF: world -> dobot1_base -> camera_link -> dobot2_base
  5. คง Topic เดิมทั้งหมด 100% (/dobot1_target_pose, /dobot1_target_objects, /dobot1_target_poses)
     และเพิ่ม /dobot2_target_objects, /dobot2_target_poses, /multi_robot_targets
"""

import json
import math
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, TransformStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster

TRANSFORM_FILE_D1 = os.path.expanduser(
    '~/ros2_ws/src/jog robot/camera_to_robot_transform.npz'
)
TRANSFORM_FILE_D2 = os.path.expanduser(
    '~/ros2_ws/src/jog robot/camera_to_dobot2_transform.npz'
)

# ขอบเขตระยะเอื้อมปลอดภัยของ Dobot Magician (หน่วย mm)
REACH_MIN_MM = 110.0
REACH_MAX_MM = 310.0


def rotation_matrix_to_quaternion(R: np.ndarray):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def yaw_deg_to_quaternion(yaw_deg: float):
    half = math.radians(yaw_deg) * 0.5
    return 0.0, 0.0, float(math.sin(half)), float(math.cos(half))


def is_reachable(xyz_mm):
    r_xy = math.hypot(xyz_mm[0], xyz_mm[1])
    return (REACH_MIN_MM <= r_xy <= REACH_MAX_MM) and (-40.0 <= xyz_mm[2] <= 140.0), round(r_xy, 1)


class CalibratedTransformerNode(Node):
    def __init__(self):
        super().__init__('calibrated_transformer_node')

        if not os.path.isfile(TRANSFORM_FILE_D1):
            self.get_logger().error(f"ไม่เจอไฟล์ transform ของ Dobot 1 ที่ {TRANSFORM_FILE_D1}")
            raise FileNotFoundError(TRANSFORM_FILE_D1)

        d1 = np.load(TRANSFORM_FILE_D1)
        self.R1 = d1['R']
        self.t1 = d1['t']
        self.get_logger().info(f"โหลด transform Dobot 1 สำเร็จ! t1 = {np.round(self.t1, 2)}")

        self.has_dobot2 = os.path.isfile(TRANSFORM_FILE_D2)
        if self.has_dobot2:
            d2 = np.load(TRANSFORM_FILE_D2)
            self.R2 = d2['R']
            self.t2 = d2['t']
            self.get_logger().info(f"โหลด transform Dobot 2 สำเร็จ! t2 = {np.round(self.t2, 2)}")
        else:
            self.R2 = None
            self.t2 = None
            self.get_logger().warn(f"ไม่พบไฟล์ Dobot 2 ({TRANSFORM_FILE_D2}) จะคำนวณเฉพาะ Dobot 1")

        # TF2 Static Broadcaster
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_published = False
        self.tf_timer = self.create_timer(0.5, self.publish_static_tf_once)

        # Single-Pose Legacy Topic
        self.sub_pose = self.create_subscription(
            PoseStamped, '/aruco_pose', self.pose_callback, 10
        )
        self.pub_dobot1_pose = self.create_publisher(
            PoseStamped, '/dobot1_target_pose', 10
        )

        # Multi-Object Batch Topics (Dobot 1, Dobot 2, และ Combined Multi-Robot)
        self.sub_info = self.create_subscription(
            String, '/detected_object_info', self.info_callback, 10
        )
        self.pub_dobot1_objects = self.create_publisher(String, '/dobot1_target_objects', 10)
        self.pub_dobot1_poses = self.create_publisher(PoseArray, '/dobot1_target_poses', 10)

        self.pub_dobot2_objects = self.create_publisher(String, '/dobot2_target_objects', 10)
        self.pub_dobot2_poses = self.create_publisher(PoseArray, '/dobot2_target_poses', 10)

        self.pub_multi_targets = self.create_publisher(String, '/multi_robot_targets', 10)

        self.last_log_time = 0.0
        self.get_logger().info("Calibrated Transformer Node v3 (Dual-Robot Ready) Started!")

    def publish_static_tf_once(self):
        if self.tf_published:
            return

        T_d1_optical = np.eye(4)
        T_d1_optical[:3, :3] = self.R1
        T_d1_optical[:3, 3] = self.t1 / 1000.0

        try:
            tf_cam = self.tf_buffer.lookup_transform(
                'camera_link',
                'camera_color_optical_frame',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            tr = tf_cam.transform.translation
            rot = tf_cam.transform.rotation
            T_link_optical = np.eye(4)
            T_link_optical[:3, :3] = quaternion_to_rotation_matrix(rot.x, rot.y, rot.z, rot.w)
            T_link_optical[:3, 3] = [tr.x, tr.y, tr.z]
        except Exception:
            T_link_optical = np.array([
                [0.0,  0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0,  0.0, 0.0, 1.0],
            ])

        T_d1_link = T_d1_optical @ np.linalg.inv(T_link_optical)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(T_d1_link[:3, :3])
        tx, ty, tz = T_d1_link[:3, 3]

        now = self.get_clock().now().to_msg()
        transforms = []

        # 1. world -> dobot1_base
        t_w_d1 = TransformStamped()
        t_w_d1.header.stamp = now
        t_w_d1.header.frame_id = 'world'
        t_w_d1.child_frame_id = 'dobot1_base'
        t_w_d1.transform.rotation.w = 1.0
        transforms.append(t_w_d1)

        # 2. dobot1_base -> camera_link
        t_d1_cam = TransformStamped()
        t_d1_cam.header.stamp = now
        t_d1_cam.header.frame_id = 'dobot1_base'
        t_d1_cam.child_frame_id = 'camera_link'
        t_d1_cam.transform.translation.x = float(tx)
        t_d1_cam.transform.translation.y = float(ty)
        t_d1_cam.transform.translation.z = float(tz)
        t_d1_cam.transform.rotation.x = qx
        t_d1_cam.transform.rotation.y = qy
        t_d1_cam.transform.rotation.z = qz
        t_d1_cam.transform.rotation.w = qw
        transforms.append(t_d1_cam)

        # 3. world -> dobot2_base (คำนวณผ่าน camera_color_optical_frame)
        if self.has_dobot2:
            T_d2_optical = np.eye(4)
            T_d2_optical[:3, :3] = self.R2
            T_d2_optical[:3, 3] = self.t2 / 1000.0
            # T_w_d2 = T_d1_optical @ inv(T_d2_optical)
            T_w_d2 = T_d1_optical @ np.linalg.inv(T_d2_optical)
            qx2, qy2, qz2, qw2 = rotation_matrix_to_quaternion(T_w_d2[:3, :3])
            tx2, ty2, tz2 = T_w_d2[:3, 3]

            t_w_d2 = TransformStamped()
            t_w_d2.header.stamp = now
            t_w_d2.header.frame_id = 'world'
            t_w_d2.child_frame_id = 'dobot2_base'
            t_w_d2.transform.translation.x = float(tx2)
            t_w_d2.transform.translation.y = float(ty2)
            t_w_d2.transform.translation.z = float(tz2)
            t_w_d2.transform.rotation.x = qx2
            t_w_d2.transform.rotation.y = qy2
            t_w_d2.transform.rotation.z = qz2
            t_w_d2.transform.rotation.w = qw2
            transforms.append(t_w_d2)

        self.static_broadcaster.sendTransform(transforms)
        self.tf_published = True
        self.tf_timer.cancel()
        self.get_logger().info("Broadcast Static TF สำเร็จ: world -> dobot1_base / dobot2_base / camera_link")

    def transform_point(self, R, t, cam_xyz_m):
        cam_mm = np.array(cam_xyz_m, dtype=np.float64) * 1000.0
        robot_mm = R @ cam_mm + t
        return [round(float(robot_mm[0]), 2), round(float(robot_mm[1]), 2), round(float(robot_mm[2]), 2)]

    def pose_callback(self, msg: PoseStamped):
        robot_mm = self.transform_point(
            self.R1, self.t1,
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        out_msg = PoseStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = 'dobot1_base'
        out_msg.pose.position.x = float(robot_mm[0])
        out_msg.pose.position.y = float(robot_mm[1])
        out_msg.pose.position.z = float(robot_mm[2])
        out_msg.pose.orientation = msg.pose.orientation
        self.pub_dobot1_pose.publish(out_msg)

    def info_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        now_msg = self.get_clock().now().to_msg()
        poses_d1 = PoseArray()
        poses_d1.header.stamp = now_msg
        poses_d1.header.frame_id = 'dobot1_base'

        poses_d2 = PoseArray()
        poses_d2.header.stamp = now_msg
        poses_d2.header.frame_id = 'dobot2_base'

        status = data.get('status', 'NO_OBJECT')
        raw_objects = data.get('all_objects', [])

        d1_objects = []
        d2_objects = []
        multi_objects = []

        for obj in raw_objects:
            cam_xyz = obj.get('cam_xyz_m', [0.0, 0.0, 0.0])
            cls_name = obj.get('class_name', 'UNKNOWN')
            raw_roll = float(obj.get('roll_deg', 0.0))
            target_roll = 0.0 if cls_name == 'ROUND_CYLINDER' else raw_roll
            qx, qy, qz, qw = yaw_deg_to_quaternion(target_roll)

            # Dobot 1 Transform
            xyz1 = self.transform_point(self.R1, self.t1, cam_xyz)
            reach1, dist1 = is_reachable(xyz1)
            p1 = Pose()
            p1.position.x, p1.position.y, p1.position.z = float(xyz1[0]), float(xyz1[1]), float(xyz1[2])
            p1.orientation.x, p1.orientation.y, p1.orientation.z, p1.orientation.w = qx, qy, qz, qw
            poses_d1.poses.append(p1)

            d1_objects.append({
                'id': obj.get('id'),
                'class_name': cls_name,
                'confidence': obj.get('confidence'),
                'height_mm': obj.get('height_mm'),
                'dobot1_xyz_mm': xyz1,
                'dobot1_roll_deg': round(target_roll, 2),
                'reachable': reach1,
                'dist_xy_mm': dist1,
            })

            # Dobot 2 Transform
            xyz2 = None
            reach2 = False
            dist2 = 9999.0
            if self.has_dobot2:
                xyz2 = self.transform_point(self.R2, self.t2, cam_xyz)
                reach2, dist2 = is_reachable(xyz2)
                p2 = Pose()
                p2.position.x, p2.position.y, p2.position.z = float(xyz2[0]), float(xyz2[1]), float(xyz2[2])
                p2.orientation.x, p2.orientation.y, p2.orientation.z, p2.orientation.w = qx, qy, qz, qw
                poses_d2.poses.append(p2)

                d2_objects.append({
                    'id': obj.get('id'),
                    'class_name': cls_name,
                    'confidence': obj.get('confidence'),
                    'height_mm': obj.get('height_mm'),
                    'dobot2_xyz_mm': xyz2,
                    'dobot2_roll_deg': round(target_roll, 2),
                    'reachable': reach2,
                    'dist_xy_mm': dist2,
                })

            # ตัดสินใจเบื้องต้นว่าตัวไหนใกล้และเอื้อมถึง
            if reach1 and not reach2:
                nearest = 'dobot1'
            elif reach2 and not reach1:
                nearest = 'dobot2'
            elif reach1 and reach2:
                nearest = 'dobot1' if dist1 <= dist2 else 'dobot2'
            else:
                nearest = 'OUT_OF_RANGE'

            multi_objects.append({
                'id': obj.get('id'),
                'class_name': cls_name,
                'confidence': obj.get('confidence'),
                'height_mm': obj.get('height_mm'),
                'roll_deg': round(target_roll, 2),
                'dobot1_xyz_mm': xyz1,
                'dobot1_dist_mm': dist1,
                'dobot1_reachable': reach1,
                'dobot2_xyz_mm': xyz2,
                'dobot2_dist_mm': dist2 if self.has_dobot2 else None,
                'dobot2_reachable': reach2,
                'nearest_robot': nearest,
            })

        self.pub_dobot1_poses.publish(poses_d1)
        self.pub_dobot1_objects.publish(String(data=json.dumps({
            'status': status,
            'object_count': len(d1_objects),
            'snapshot_locked': data.get('snapshot_locked', False),
            'latency_ms': data.get('latency_ms', 0.0),
            'objects': d1_objects,
        })))

        if self.has_dobot2:
            self.pub_dobot2_poses.publish(poses_d2)
            self.pub_dobot2_objects.publish(String(data=json.dumps({
                'status': status,
                'object_count': len(d2_objects),
                'snapshot_locked': data.get('snapshot_locked', False),
                'latency_ms': data.get('latency_ms', 0.0),
                'objects': d2_objects,
            })))

        self.pub_multi_targets.publish(String(data=json.dumps({
            'status': status,
            'object_count': len(multi_objects),
            'snapshot_locked': data.get('snapshot_locked', False),
            'latency_ms': data.get('latency_ms', 0.0),
            'objects': multi_objects,
        })))

        # Log สรุปทั้ง 2 หุ่นยนต์ทุก 1 วินาที
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if multi_objects and (now_sec - self.last_log_time >= 1.0):
            self.last_log_time = now_sec
            lines = [f"[Dual-Robot Targets: {len(multi_objects)} ชิ้น | Locked={data.get('snapshot_locked', False)}]"]
            for item in multi_objects:
                x1, y1, z1 = item['dobot1_xyz_mm']
                d2_str = (
                    f"D2({item['dobot2_xyz_mm'][0]:5.1f},{item['dobot2_xyz_mm'][1]:5.1f},{item['dobot2_xyz_mm'][2]:4.1f})"
                    if item['dobot2_xyz_mm'] else "D2(N/A)"
                )
                lines.append(
                    f"  #{item['id']} {item['class_name']:<14} | "
                    f"D1({x1:5.1f},{y1:5.1f},{z1:4.1f}) | {d2_str} -> [{item['nearest_robot']}]"
                )
            self.get_logger().info("\n" + "\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = CalibratedTransformerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
