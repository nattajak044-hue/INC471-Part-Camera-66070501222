# test_dobot_connect.py
from pydobot import Dobot
import time

PORT = '/dev/ttyUSB0'  # แก้ตาม port ที่เจอใน step 2

device = Dobot(port=PORT)

pose = device.pose()
print(f"Current pose: X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}, R={pose[3]:.2f}")

device.close()