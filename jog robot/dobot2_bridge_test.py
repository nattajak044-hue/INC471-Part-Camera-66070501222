#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydobot import Dobot

class Dobot2BridgeTestNode(Node):
    def __init__(self):
        super().__init__('dobot2_bridge_test_node')
        
        # ล็อกพอร์ตตรงกับบอร์ด CP2102N ของ Dobot 2 (กันสลับกับ Dobot 1)
        self.port = '/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_0ae6da00b9e8ea1192e4c9ac994a5d01-if00-port0'
        self.get_logger().info(f'กำลังเชื่อมต่อ Dobot2 ที่พอร์ต {self.port}...')
        self.dobot = Dobot(port=self.port, verbose=False)
        self.dobot.speed(velocity=50.0, acceleration=50.0)
        self.get_logger().info('เชื่อมต่อ Dobot2 สำเร็จ!')
        
        # Safety Clamp: ขอบเขตปลอดภัยของ Dobot2 (อ้างอิงจากจุด calibration จริง)
        self.X_MIN, self.X_MAX = 130.0, 290.0
        self.Y_MIN, self.Y_MAX = -150.0, 150.0
        self.SAFE_Z_HOVER = 40.0  # ความสูงปลอดภัยขั้นต่ำเหนือสายพาน (mm)
        
        # ล็อกให้เคลื่อนที่ 1 ครั้งต่อการรัน
        self.executed_once = False

        self.sub = self.create_subscription(
            String,
            '/dobot2_target_objects',
            self.target_callback,
            10
        )
        self.get_logger().info('⏳ กำลังรอรับพิกัดจาก /dobot2_target_objects (วางวัตถุในโซน Dobot 2)...')

    def target_callback(self, msg: String):
        if self.executed_once:
            return

        try:
            data = json.loads(msg.data)
        except Exception:
            return

        objects = data.get('objects', [])
        if not objects:
            return

        # เลือกวัตถุเป้าหมายอันดับ #1
        target = objects[0]
        tx, ty, tz = target['dobot2_xyz_mm']
        reachable = target.get('reachable', False)
        cls_name = target.get('class_name', 'UNKNOWN')

        self.get_logger().info(
            f"🎯 พบเป้าหมาย #{target['id']} ({cls_name}) -> "
            f"Dobot2(X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f} mm) | Reachable={reachable}"
        )

        if not reachable or not (self.X_MIN <= tx <= self.X_MAX and self.Y_MIN <= ty <= self.Y_MAX):
            self.get_logger().warn(
                f"⚠️ วัตถุอยู่นอกระยะปลอดภัยของ Dobot2 (X={tx:.2f}, Y={ty:.2f}) — รอวัตถุชิ้นถัดไป"
            )
            return

        target_z = max(tz, self.SAFE_Z_HOVER)
        self.get_logger().info(f'🤖 สั่ง Dobot2 เคลื่อนที่ไป Hover ที่: X={tx:.2f}, Y={ty:.2f}, Z={target_z:.2f}')
        
        self.dobot.move_to(x=tx, y=ty, z=target_z, r=0.0, wait=True)
        
        curr_pose = self.dobot.pose()[:4]
        self.get_logger().info(f'✅ เคลื่อนที่เสร็จสิ้น พิกัดปัจจุบันของ Dobot2: {curr_pose}')
        self.executed_once = True

    def destroy_node(self):
        try:
            self.dobot.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Dobot2BridgeTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
