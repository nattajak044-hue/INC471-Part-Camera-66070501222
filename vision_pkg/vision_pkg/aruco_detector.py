import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.bridge = CvBridge()
        
        # 1. สร้าง Publisher สำหรับส่งพิกัด 3D และภาพ Debug ออกไปแสดงในแอป
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        self.debug_img_pub = self.create_publisher(Image, '/aruco/debug_image', 5)
        
        # 2. Subscribe ข้อมูล CameraInfo
        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None

        # 3. Synchronize ภาพ Color และ Depth ให้เข้าพร้อมกัน
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.image_callback)
        
        # 4. ตั้งค่า ArUco Dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()

        self.get_logger().info("ArUco 3D Pose Detector Node Started! (ส่งภาพออกทาง /aruco/debug_image)")

    def get_robust_depth(self, cv_depth, u, v, roi_size=9):
        half = roi_size // 2
        h, w = cv_depth.shape[:2]

        u_min = max(0, u - half)
        u_max = min(w, u + half + 1)
        v_min = max(0, v - half)
        v_max = min(h, v + half + 1)

        roi = cv_depth[v_min:v_max, u_min:u_max].astype(np.float32)
        valid = roi[roi > 0]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.ppx = msg.k[2]
        self.ppy = msg.k[5]

    def image_callback(self, color_msg, depth_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            if hasattr(cv2.aruco, 'ArucoDetector'):
                detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                corners, ids, rejected = detector.detectMarkers(cv_color)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_color, self.dictionary, parameters=self.parameters)
            
            if ids is not None and self.fx is not None:
                for i, corner in enumerate(corners):
                    pts = corner[0]

                    u = int(np.mean(pts[:, 0]))
                    v = int(np.mean(pts[:, 1]))

                    depth_mm = self.get_robust_depth(cv_depth, u, v, roi_size=9)
                    if depth_mm is not None:
                        z_c = depth_mm / 1000.0

                        x_c = (u - self.ppx) * z_c / self.fx
                        y_c = (v - self.ppy) * z_c / self.fy

                        # วาดกรอบและแสดงค่า 3D บนภาพ
                        cv2.polylines(cv_color, [np.int32(pts)], True, (0, 255, 0), 2)
                        cv2.circle(cv_color, (u, v), 5, (0, 0, 255), -1)

                        text = f"ID:{ids[i][0]} XYZ:({x_c:.2f}, {y_c:.2f}, {z_c:.2f})m"
                        cv2.putText(cv_color, text, (u - 60, v - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Publish ค่าพิกัดออกทาง ROS 2 Topic
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = color_msg.header.stamp
                        pose_msg.header.frame_id = color_msg.header.frame_id
                        pose_msg.pose.position.x = float(x_c)
                        pose_msg.pose.position.y = float(y_c)
                        pose_msg.pose.position.z = float(z_c)
                        self.pose_pub.publish(pose_msg)
                        
                        self.get_logger().info(f"Marker ID {ids[i][0]} -> X: {x_c:.3f}m, Y: {y_c:.3f}m, Z: {z_c:.3f}m")

            # ส่งภาพเข้าแอป Node Switchboard แทนการเปิดหน้าต่าง OpenCV แยก
            debug_msg = self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8')
            self.debug_img_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
