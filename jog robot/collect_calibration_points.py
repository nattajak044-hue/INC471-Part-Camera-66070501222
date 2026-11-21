#!/usr/bin/env python3
"""
collect_calibration_points.py (รองรับทั้ง dobot1 และ dobot2 แยกไฟล์อัตโนมัติ)

วิธีรัน:
  - สำหรับ Dobot 1: python3 collect_calibration_points.py --robot dobot1 --port /dev/ttyUSB0
  - สำหรับ Dobot 2: python3 collect_calibration_points.py --robot dobot2 --port /dev/ttyUSB1
"""

import argparse
import csv
import os
import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pydobot import Dobot

MAX_AGE_SEC = 0.5


class ArucoListener(Node):
    def __init__(self):
        super().__init__('calib_aruco_listener')
        self.latest_pose = None
        self.latest_stamp = None
        self.lock = threading.Lock()
        self.sub = self.create_subscription(
            PoseStamped, '/aruco_pose', self.callback, 10)

    def callback(self, msg):
        with self.lock:
            self.latest_pose = (
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            )
            self.latest_stamp = time.time()

    def get_latest(self, max_age=MAX_AGE_SEC):
        with self.lock:
            if self.latest_pose is None:
                return None
            if (time.time() - self.latest_stamp) > max_age:
                return None
            return self.latest_pose


def main():
    parser = argparse.ArgumentParser(description='Collect Calibration Points for Dobot')
    parser.add_argument('--robot', default='dobot1', choices=['dobot1', 'dobot2'],
                        help='เลือกหุ่นยนต์ (dobot1 หรือ dobot2)')
    parser.add_argument('--port', default='/dev/ttyUSB0',
                        help='Serial port ของ Dobot (เช่น /dev/ttyUSB0 หรือ /dev/ttyUSB1)')
    args = parser.parse_args()

    output_file = 'calibration_points.csv' if args.robot == 'dobot1' else f'calibration_points_{args.robot}.csv'

    rclpy.init()
    node = ArucoListener()

    def spin_loop(n):
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.1)

    spin_thread = threading.Thread(target=spin_loop, args=(node,), daemon=True)
    spin_thread.start()

    print(f"=== โหมดเก็บจุดสำหรับ: {args.robot.upper()} ===")
    print(f"ไฟล์ปลายทาง: {output_file}")
    print(f"กำลังเชื่อมต่อ {args.robot} ที่ {args.port} ...")
    device = Dobot(port=args.port)
    print(f"เชื่อมต่อ {args.robot} สำเร็จ! (อย่าลืมทำ Homing ก่อนเริ่มเก็บจุด)\n")

    file_exists = os.path.isfile(output_file)
    csv_file = open(output_file, 'a', newline='')
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow([
            'label', 'point_id',
            'cam_x', 'cam_y', 'cam_z',
            'robot_x', 'robot_y', 'robot_z', 'robot_r',
        ])

    label = 'calib'
    point_count = 0

    print("ขั้นที่ 1: เอาแขนออกจาก marker ให้กล้องเห็นชัด แล้วกด Enter -> บันทึกพิกัดกล้อง")
    print("ขั้นที่ 2: jog แขนเข้าไปแตะกึ่งกลาง marker แล้วกด Enter -> บันทึกพิกัด Dobot")
    print("- พิมพ์ 'v' แทนการกด Enter ขั้นที่ 1 เพื่อสลับไปเก็บจุด validation")
    print("- พิมพ์ 'q' แทนการกด Enter ขั้นที่ 1 เพื่อจบการเก็บข้อมูล\n")

    try:
        while True:
            user_input = input(
                f"[{args.robot}:{label}] จุดที่ {point_count + 1} - (1) เอาแขนออก ให้กล้องเห็น marker ชัด "
                f"แล้วกด Enter (หรือ v/q): "
            ).strip().lower()

            if user_input == 'q':
                break

            if user_input == 'v':
                label = 'validation'
                point_count = 0
                print(">> เปลี่ยนเป็นเก็บจุด validation แล้ว\n")
                continue

            cam = node.get_latest()
            if cam is None:
                print("!! ไม่เจอ marker ในภาพตอนนี้ (หรือข้อมูลเก่าเกินไป) เช็คว่า aruco_detector.py รันอยู่และแขนไม่บัง\n")
                continue

            print(f"   ได้พิกัดกล้อง -> ({cam[0]:.4f}, {cam[1]:.4f}, {cam[2]:.4f}) m")

            input(f"   (2) ตอนนี้ jog แขน {args.robot} ไปแตะกึ่งกลาง marker จุดเดิม แล้วกด Enter เพื่อบันทึก: ")

            robot_pose = device.pose()
            robot_x, robot_y, robot_z, robot_r = robot_pose[0], robot_pose[1], robot_pose[2], robot_pose[3]

            point_count += 1
            writer.writerow([
                label, point_count,
                cam[0], cam[1], cam[2],
                robot_x, robot_y, robot_z, robot_r,
            ])
            csv_file.flush()

            print(f"   บันทึกจุดที่ {point_count} ({args.robot}) แล้ว -> "
                  f"cam=({cam[0]:.4f}, {cam[1]:.4f}, {cam[2]:.4f}) m  "
                  f"robot=({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f}) mm\n")

    except KeyboardInterrupt:
        print("\n(หยุดกลางคันด้วย Ctrl+C)")
    finally:
        csv_file.close()
        device.close()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        print(f"บันทึกข้อมูลลง {output_file} เรียบร้อย ปิดการเชื่อมต่อ {args.robot} แล้ว")


if __name__ == '__main__':
    main()
