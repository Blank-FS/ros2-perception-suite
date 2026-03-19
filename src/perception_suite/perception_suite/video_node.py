import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_node')
        self.publisher_ = self.create_publisher(Image, 'raw_frames', 10)
        self.cap = cv2.VideoCapture('test_video.mp4')
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.033, self.timer_callback) # ~30 FPS

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize to a smaller resolution (e.g., 640x480)
            # This drastically reduces the data sent over DDS
            small_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            
            msg = self.bridge.cv2_to_imgmsg(small_frame, encoding="bgr8")
            self.publisher_.publish(msg)

# Add this to the end of every file (updating the class name accordingly)
def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher() # Use the specific class name for that file
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()