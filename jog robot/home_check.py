#!/usr/bin/env python3
import argparse
import time
from pydobot import Dobot
from pydobot.message import Message

def home_command(device):
    msg = Message()
    msg.id = 31          # SetHOMECmd ตาม Dobot protocol
    msg.ctrl = 0x03       # write + queue (ต้องรอ execute)
    msg.params = bytes([0x00, 0x00, 0x00, 0x00])
    return device._send_command(msg, wait=True)

def main():
    parser = argparse.ArgumentParser(description='Home Check for Dobot')
    parser.add_argument('--robot', default='dobot1', help='ชื่อหุ่นยนต์ (dobot1 หรือ dobot2)')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port ของ Dobot')
    args = parser.parse_args()

    print(f"=== กำลังสั่ง Home {args.robot.upper()} ที่พอร์ต {args.port} ===")
    device = Dobot(port=args.port, verbose=True)

    home_command(device)
    time.sleep(1)

    # เช็ค alarm อีกรอบหลัง Home
    msg = Message()
    msg.id = 20
    msg.ctrl = 0x00
    resp = device._send_command(msg)
    print(f"[{args.robot.upper()}] alarm params after homing:", list(resp.params))

    device.close()
    print(f"=== Homing {args.robot.upper()} เสร็จสมบูรณ์ ===")

if __name__ == '__main__':
    main()
