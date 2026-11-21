import os
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


# ============================================================
# Dobot SDK paths
# ============================================================

DOBOT_ROOT = os.path.expanduser(
    '~/projectp4/dobot-rl'
)

DOBOT_UTILS = os.path.join(
    DOBOT_ROOT,
    'dobot_rl',
    'utils'
)

# ให้ Python หา dobot_rl และ DobotDllType.py
sys.path.insert(0, DOBOT_ROOT)
sys.path.insert(0, DOBOT_UTILS)

import DobotDllType as dType


class DobotNode(Node):

    def __init__(self):
        super().__init__('dobot_node')

        # ----------------------------------------------------
        # ROS2 subscriber
        # ----------------------------------------------------

        self.subscription = self.create_subscription(
            PoseStamped,
            '/dobot_target',
            self.target_callback,
            10
        )

        self.get_logger().info(
            'Dobot Node starting...'
        )

        # ----------------------------------------------------
        # Load Dobot API
        # ----------------------------------------------------

        self.api = dType.load()

        if self.api is None:
            self.get_logger().error(
                'Failed to load Dobot DLL'
            )
            return

        self.get_logger().info(
            'Dobot DLL loaded'
        )

        # ----------------------------------------------------
        # Connect Dobot
        # ----------------------------------------------------

        result = dType.ConnectDobot(
            self.api,
            '',
            115200
        )

        if result[0] != dType.DobotConnect.DobotConnect_NoError:

            self.get_logger().error(
                f'Failed to connect Dobot. Error code: {result[0]}'
            )

            return

        self.connected = True

        self.get_logger().info(
            'Dobot connected successfully!'
        )

        # ----------------------------------------------------
        # Set motion parameters
        # ----------------------------------------------------

        dType.SetPTPJointParams(
            self.api,
            50, 50,
            50, 50,
            50, 50,
            50, 50
        )

        dType.SetPTPCoordinateParams(
            self.api,
            50, 50,
            50, 50
        )

        # Preserve current R for first test
        self.current_r = -59.59

        self.get_logger().info(
            'Dobot Node Ready!'
        )

    # ========================================================
    # Receive target position
    # ========================================================

    def target_callback(self, msg):

        if not hasattr(self, 'connected') or not self.connected:
            self.get_logger().error(
                'Dobot is not connected'
            )
            return

        # ----------------------------------------------------
        # Check frame
        # ----------------------------------------------------

        self.get_logger().info(
            f'Received frame: {msg.header.frame_id}'
        )

        # ----------------------------------------------------
        # ROS2 uses meter
        # Dobot uses millimeter
        # ----------------------------------------------------

        x = msg.pose.position.x * 1000.0
        y = msg.pose.position.y * 1000.0
        z = msg.pose.position.z * 1000.0

        # First test:
        # keep R fixed to current robot orientation
        r = self.current_r

        self.get_logger().info(
            f'Target Dobot position: '
            f'X={x:.2f} mm, '
            f'Y={y:.2f} mm, '
            f'Z={z:.2f} mm, '
            f'R={r:.2f} deg'
        )

        # ----------------------------------------------------
        # Move Dobot
        # ----------------------------------------------------

        try:

            result = dType.SetPTPCmd(
                self.api,
                dType.PTPMode.PTPMOVJXYZMode,
                x,
                y,
                z,
                r,
                0
            )

            self.get_logger().info(
                f'Dobot command sent. Queue index: {result[0]}'
            )

        except Exception as e:

            self.get_logger().error(
                f'Dobot movement error: {e}'
            )

    # ========================================================
    # Shutdown
    # ========================================================

    def destroy_node(self):

        if hasattr(self, 'api') and self.api is not None:

            try:
                dType.DisconnectDobot(self.api)

                self.get_logger().info(
                    'Dobot disconnected'
                )

            except Exception as e:

                self.get_logger().warn(
                    f'Disconnect error: {e}'
                )

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = DobotNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()