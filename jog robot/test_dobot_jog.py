# test_dobot_jog.py
from pydobot import Dobot
import time

PORT = '/dev/ttyUSB0'  # ใส่ port เดิมที่ใช้ได้แล้ว

device = Dobot(port=PORT)

pose = device.pose()
print(f"ก่อนขยับ: X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}")

# ขยับขึ้นแค่ 10mm ในแกน Z (ปลอดภัยสุด ไม่เสี่ยงชนอะไร)
target_z = pose[2] + 10
device.move_to(pose[0], pose[1], target_z, pose[3], wait=True)

pose2 = device.pose()
print(f"หลังขยับ: X={pose2[0]:.2f}, Y={pose2[1]:.2f}, Z={pose2[2]:.2f}")

device.close()