import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import time

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image, 
            'processed_frames', 
            self.display_callback, 
            qos_profile)
        
        self.bridge = CvBridge()
        
        # FPS Tracking variables
        self.prev_time = 0
        self.fps = 0
        
        # Create the window once during initialization
        cv2.namedWindow("AV Perception Testbed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AV Perception Testbed", 800, 600) 

    def display_callback(self, data):
        # 1. Calculate FPS
        current_time = time.time()
        # Avoid division by zero on first frame
        if self.prev_time != 0:
            time_diff = current_time - self.prev_time
            # Calculate instantaneous FPS
            self.fps = 1 / time_diff
        self.prev_time = current_time

        # 2. Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # 3. Draw FPS on the frame
        fps_text = f"FPS: {self.fps:.1f}"
        # Parameters: (image, text, position, font, scale, color, thickness)
        cv2.putText(cv_image, fps_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 4. Show image
        cv2.imshow("AV Perception Testbed", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode() 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()