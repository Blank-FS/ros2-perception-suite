import signal

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    """Stage 1: stands in for the BeamNG camera, publishing /raw_frames."""

    def __init__(self):
        super().__init__("video_node")
        video_path = self.declare_parameter("video_path", "test_video.mp4").value
        # Downscaled for DDS bandwidth only; the working size is set by the
        # preprocess stage. Aspect is kept so the camera model stays square-pixel.
        self.width = self.declare_parameter("width", 1280).value
        # The launch file performs its substitutions, so this arrives as a
        # string rather than a bool.
        self.loop = str(
            self.declare_parameter("loop", "true").value).strip().lower() in (
            "1", "true", "yes", "y", "on")
        self.publisher_ = self.create_publisher(Image, "raw_frames", 10)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            # Relative paths resolve against the directory the launch ran in.
            raise RuntimeError(f"Cannot open video '{video_path}'")
        self.get_logger().info(f"Playing '{video_path}'")
        self.bridge = CvBridge()
        # ~30 FPS by default, to stand in for the camera. A throughput
        # benchmark raises this so the pipeline, not the source, is the limit.
        fps = float(self.declare_parameter("video_fps", "30.0").value)
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            if not self.loop:
                self.get_logger().info("End of video", once=True)
                return
            # Rewind. A benchmark run outlasts any single clip, and the
            # alternative is measuring a source that has stopped publishing.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Cannot rewind video", once=True)
                return
        h, w = frame.shape[:2]
        height = round(h * self.width / w)
        small_frame = cv2.resize(
            frame, (self.width, height), interpolation=cv2.INTER_AREA
        )

        msg = self.bridge.cv2_to_imgmsg(small_frame, encoding="bgr8")
        # Every downstream stage copies this header, which is how the
        # visualizer matches a frame to its mask and path.
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(VideoPublisher())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Under launch, Ctrl+C arrives twice (terminal and launch); the second
        # must not interrupt teardown or interpreter exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
