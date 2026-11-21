
#!/usr/bin/env python3
"""
vision_unit_test_logger.py
สคริปต์สำหรับเก็บสถิติ Vision Unit Test (W12)
วัด:
  1. Classification Accuracy (%) เทียบระหว่างสิ่งที่วางจริงกับสิ่งที่กล้องทาย
  2. Coordinate & Roll Stability (Standard Deviation ของ X, Y, Z และ Roll Angle เมื่อวัตถุหยุดนิ่ง)
  3. บันทึกผลลงไฟล์ vision_unit_test_results.csv อัตโนมัติ
"""

import csv
import json
import os
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SAMPLES_PER_TRIAL = 15  # เก็บตัวอย่าง 15 เฟรมต่อการวาง 1 รอบ เพื่อคำนวณความนิ่ง
OUTPUT_CSV = os.path.expanduser('~/ros2_ws/src/vision_pkg/vision_unit_test_results.csv')

CLASS_MAP = {
    '1': 'SQUARE_BOX',
    '2': 'ROUND_CYLINDER',
    '3': 'RECT_BAR',
    '4': 'POLYGON_OBJ'
}


class VisionUnitTestLogger(Node):
    def __init__(self):
        super().__init__('vision_unit_test_logger')
        self.sub = self.create_subscription(
            String,
            '/detected_object_info',
            self.info_callback,
            10
        )
        self.latest_data = None
        self.recording = False
        self.buffer = []
        self.lock = threading.Lock()

    def info_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self.lock:
                self.latest_data = data
                if self.recording and data.get('status') == 'OK':
                    self.buffer.append(data)
        except Exception:
            pass


def main():
    rclpy.init()
    node = VisionUnitTestLogger()

    # รัน ROS2 spin ใน background thread เพื่อรับข้อมูลต่อเนื่องขณะรอ input คีย์บอร์ด
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    trials = []
    print("\n" + "=" * 60)
    print("  VISION UNIT TEST LOGGER (W12) - Ready!")
    print("=" * 60)
    print("คำแนะนำการทดสอบ:")
    print("  1. วางวัตถุ 1 ชิ้นลงในกรอบสีขาวบนสายพาน แล้วปล่อยมือให้วัตถุนิ่ง")
    print("  2. กดหมายเลขชนิดวัตถุที่วางจริง:")
    print("     [1] SQUARE_BOX      (บล็อกสี่เหลี่ยมจัตุรัส)")
    print("     [2] ROUND_CYLINDER  (ฝาขวด / ทรงกลม)")
    print("     [3] RECT_BAR        (แท่งยาว / สี่เหลี่ยมผืนผ้า)")
    print("     [4] POLYGON_OBJ     (รูปทรงอื่นๆ)")
    print("     [q] จบการทดสอบและสรุปผลทั้งหมด")
    print("=" * 60 + "\n")

    try:
        while rclpy.ok():
            choice = input("เลือกชนิดวัตถุที่วางจริง [1/2/3/4] หรือ [q] เพื่อสรุปผล: ").strip().lower()
            if choice == 'q':
                break
            if choice not in CLASS_MAP:
                print("⚠️  กรุณากด 1, 2, 3, 4 หรือ q เท่านั้น\n")
                continue

            gt_class = CLASS_MAP[choice]
            with node.lock:
                node.buffer = []
                node.recording = True

            print(f"⏳ กำลังเก็บข้อมูล {SAMPLES_PER_TRIAL} เฟรมสำหรับ [{gt_class}]... (อย่าเพิ่งขยับวัตถุ)")
            start_wait = time.time()
            while True:
                with node.lock:
                    count = len(node.buffer)
                if count >= SAMPLES_PER_TRIAL:
                    break
                if time.time() - start_wait > 5.0:
                    break
                time.sleep(0.05)

            with node.lock:
                node.recording = False
                samples = list(node.buffer)

            if len(samples) < 5:
                print("❌ ตรวจไม่พบวัตถุในกรอบภาพ (NO_OBJECT) กรุณาเช็กตำแหน่งวัตถุแล้วลองใหม่\n")
                continue

            # วิเคราะห์ผลลัพธ์ของรอบนี้
            pred_classes = [s['class_name'] for s in samples]
            # หาคลาสที่กล้องทายบ่อยที่สุดใน 15 เฟรม (Majority Vote)
            pred_class = max(set(pred_classes), key=pred_classes.count)
            is_correct = (pred_class == gt_class)

            xyz_arr = np.array([s['cam_xyz_m'] for s in samples]) * 1000.0  # แปลงเป็น mm
            roll_arr = np.array([s['roll_deg'] for s in samples])
            height_arr = np.array([s['height_mm'] for s in samples])
            conf_arr = np.array([s['confidence'] for s in samples])

            std_x, std_y, std_z = np.std(xyz_arr, axis=0)
            std_roll = float(np.std(roll_arr))
            mean_h = float(np.mean(height_arr))
            mean_conf = float(np.mean(conf_arr))

            trial_record = {
                'trial': len(trials) + 1,
                'ground_truth': gt_class,
                'predicted': pred_class,
                'correct': is_correct,
                'confidence': round(mean_conf, 3),
                'height_mm': round(mean_h, 1),
                'std_x_mm': round(float(std_x), 3),
                'std_y_mm': round(float(std_y), 3),
                'std_z_mm': round(float(std_z), 3),
                'std_roll_deg': round(std_roll, 3),
            }
            trials.append(trial_record)

            status_icon = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"{status_icon} | รอบที่ {trial_record['trial']}: วางจริง={gt_class} | กล้องทาย={pred_class} (Conf={mean_conf:.2f})")
            print(f"   ↳ ความนิ่ง (Std): X=±{std_x:.2f}mm, Y=±{std_y:.2f}mm, Z=±{std_z:.2f}mm | Roll=±{std_roll:.2f}° | สูง={mean_h:.1f}mm\n")

    except KeyboardInterrupt:
        pass

    # สรุปผลรวมทั้งหมด
    if trials:
        total = len(trials)
        correct_cnt = sum(1 for t in trials if t['correct'])
        acc_pct = (correct_cnt / total) * 100.0

        avg_std_x = np.mean([t['std_x_mm'] for t in trials])
        avg_std_y = np.mean([t['std_y_mm'] for t in trials])
        avg_std_z = np.mean([t['std_z_mm'] for t in trials])
        avg_std_roll = np.mean([t['std_roll_deg'] for t in trials])

        print("\n" + "=" * 60)
        print("  สรุปผล VISION UNIT TEST (W12) สำหรับใส่รายงาน")
        print("=" * 60)
        print(f"จำนวนรอบทดสอบทั้งหมด : {total} รอบ")
        print(f"ทายรูปทรงถูกต้อง       : {correct_cnt}/{total} รอบ")
        print(f"Classification Accuracy: {acc_pct:.2f}%")
        print(f"ความนิ่งพิกัดเฉลี่ย (Jitter Std): X=±{avg_std_x:.2f} mm, Y=±{avg_std_y:.2f} mm, Z=±{avg_std_z:.2f} mm")
        print(f"ความนิ่งมุมหมุนเฉลี่ย (Roll Std) : ±{avg_std_roll:.2f} deg")
        print("=" * 60)

        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=trials[0].keys())
            writer.writeheader()
            writer.writerows(trials)
        print(f"💾 บันทึกไฟล์ผลลัพธ์เรียบร้อยที่: {OUTPUT_CSV}\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
EOF
