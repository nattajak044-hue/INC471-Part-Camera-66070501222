# INC471-Part-Camera-66070501222

โปรเจกต์ ROS 2 (`vision_pkg`) สำหรับระบบกล้องตรวจจับ ArUco / แปลงพิกัด และสคริปต์ควบคุมหุ่นยนต์ Dobot / Calibration (`jog robot`)

---

## โครงสร้างโปรเจกต์และหน้าที่ของแต่ละไฟล์

### 1. `vision_pkg/` (แพ็กเกจ ROS 2 — `ament_python`)
- `vision_pkg/aruco_detector.py` — Node ตรวจจับ ArUco marker จากภาพกล้อง
- `vision_pkg/tf_transformer.py` — Node แปลงพิกัดผ่านระบบ TF ของ ROS 2
- `vision_pkg/calibrated_transformer.py` — Node แปลงพิกัดกล้องเป็นพิกัดหุ่นยนต์โดยใช้ไฟล์ calibration (`.npz`)
- `vision_pkg/dobot_node.py` — Node สำหรับเชื่อมต่อ/ควบคุม Dobot
- `vision_pkg/object_detector_node.py` — Node ตรวจจับวัตถุ
- `vision_pkg/detectors/` — โมดูลย่อยสำหรับตรวจจับ/คัดแยกวัตถุ (`base.py`, `cascade_qc.py`, `data_collect.py`, `depth_shape.py`, `hybrid_fusion.py`, `sam_anomaly.py`, `yolo_seg.py`)
- `vision_pkg/vision_unit_test_logger.py` — สคริปต์บันทึกผลทดสอบลง `vision_unit_test_results.csv`
- `setup.py` / `package.xml` — ไฟล์ตั้งค่าแพ็กเกจและ `entry_points` สำหรับ `ros2 run`

### 2. `jog robot/` (สคริปต์ Python สำหรับ Dobot และ Calibration)
- `collect_calibration_points.py` — เก็บคู่พิกัดระหว่างกล้องกับแขนหุ่นยนต์ Dobot
- `compute_calibration_transform.py` — คำนวณเมทริกซ์แปลงพิกัดจากไฟล์ `.csv` เป็น `.npz`
- `calibration_points.csv` / `calibration_points_dobot2.csv` — ข้อมูลจุด calibration ที่เก็บไว้
- `camera_to_robot_transform.npz` / `camera_to_dobot2_transform.npz` — ไฟล์ผลลัพธ์เมทริกซ์แปลงพิกัด
- `dobot1_bridge_test.py` / `dobot2_bridge_test.py` — สคริปต์ทดสอบเชื่อมต่อ Dobot ตัวที่ 1 และ 2
- `home_check.py` — ตรวจสอบ/สั่งหุ่นยนต์กลับจุด Home
- `test_dobot_connect.py` / `test_dobot_jog.py` — ทดสอบการเชื่อมต่อและการ jog หุ่นยนต์

---

## กรณีไฟล์หายหรือพังทั้งเครื่อง: วิธีกู้คืนและติดตั้งใหม่

### 1. เตรียมเครื่องใหม่ (หลังลง Ubuntu + ROS 2)
```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions git
```

### 2. โหลดโค้ดจาก GitHub ลงมาใน `ros2_ws/src`
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/nattajak044-hue/INC471-Part-Camera-66070501222.git .
```

### 3. Build แพ็กเกจ ROS 2
```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```
*(แนะนำให้เพิ่ม `source ~/ros2_ws/install/setup.bash` ลงใน `~/.bashrc` เพื่อไม่ต้องพิมพ์ซ้ำทุกครั้งที่เปิด Terminal)*

---

## คำสั่งรันใช้งาน

### รัน ROS 2 Nodes (`vision_pkg`)
```bash
source ~/ros2_ws/install/setup.bash

# รันตัวตรวจจับ ArUco
ros2 run vision_pkg aruco_detector

# รันตัวแปลงพิกัด TF
ros2 run vision_pkg tf_transformer

# รันตัวแปลงพิกัดจากไฟล์ Calibration
ros2 run vision_pkg calibrated_transformer
```

### รันสคริปต์ใน `jog robot`
```bash
cd ~/ros2_ws/src/"jog robot"

# ตัวอย่าง: เก็บจุด calibration
python3 collect_calibration_points.py

# ตัวอย่าง: คำนวณ transform
python3 compute_calibration_transform.py

# ตัวอย่าง: ทดสอบเชื่อมต่อ Dobot
python3 test_dobot_connect.py
```

---

## วิธี Backup โค้ดขึ้น GitHub ครั้งถัดไป
```bash
cd ~/ros2_ws/src
git add -A
git commit -m "อัปเดตงาน"
git push backup main
```
