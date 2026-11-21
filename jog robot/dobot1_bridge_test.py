
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pydobot import Dobot

class Dobot1BridgeNode(Node):
    def __init__(self):
        super().__init__('dobot1_bridge_node')
        
        # ล็อกพอร์ตตรงกับบอร์ด CP2102N ของ Dobot 1 (กันสลับกับ Dobot 2)
        self.port = '/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_32d52bc0104beb11be5b10f21326e067-if00-port0'
        self.get_logger().info(f'กำลังเชื่อมต่อ Dobot1 ที่พอร์ต {self.port}...')
        self.dobot = Dobot(port=self.port, verbose=False)
        self.dobot.speed(velocity=50.0, acceleration=50.0)
        self.get_logger().info('เชื่อมต่อ Dobot1 สำเร็จ!')
        
        # Safety Clamp: จำกัดขอบเขตการทำงานของ Dobot1 (หน่วย mm)
        self.X_MIN, self.X_MAX = 100.0, 300.0
        self.Y_MIN, self.Y_MAX = -180.0, 180.0
        self.SAFE_Z_HOVER = 40.0  # ความสูงปลอดภัยขั้นต่ำเหนือสายพาน (mm)
        
        # ล็อกให้เคลื่อนที่แค่ครั้งเดียวต่อการรัน เพื่อป้องกัน Serial Blocking
        self.executed_once = False

        self.sub = self.create_subscription(
            PoseStamped,
            '/dobot1_target_pose',
            self.target_callback,
            10
        )
        self.get_logger().info('⏳ กำลังรอรับพิกัดจาก /dobot1_target_pose (ลองวาง ArUco ให้กล้องเห็น)...')

    def target_callback(self, msg: PoseStamped):
        if self.executed_once:
            return

        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z

        self.get_logger().info(f'🎯 ได้รับพิกัดเป้าหมายจากกล้องแล้ว! (mm): X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f}')

        # ตรวจสอบว่าอยู่ในระยะเอื้อมที่ปลอดภัยหรือไม่
        if not (self.X_MIN <= tx <= self.X_MAX and self.Y_MIN <= ty <= self.Y_MAX):
            self.get_logger().error(
                f'SAFETY ABORT: พิกัด (X={tx:.2f}, Y={ty:.2f}) อยู่นอกขอบเขตปลอดภัยของ Dobot1!'
            )
            return

        # บังคับใช้ความสูงปลอดภัย (Hover) เพื่อป้องกันข้อผิดพลาดจาก Outlier
        target_z = max(tz, self.SAFE_Z_HOVER)
        self.get_logger().info(f'🤖 กำลังสั่ง Dobot1 เคลื่อนที่ไป Hover ที่: X={tx:.2f}, Y={ty:.2f}, Z={target_z:.2f}')
        
        self.dobot.move_to(x=tx, y=ty, z=target_z, r=0.0, wait=True)
        
        curr_pose = self.dobot.pose()[:4]
        self.get_logger().info(f'✅ เคลื่อนที่เสร็จสิ้น พิกัดปัจจุบันของ Dobot1: {curr_pose}')
        self.executed_once = True

    def destroy_node(self):
        try:
            self.dobot.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Dobot1BridgeNode()
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
EOF
