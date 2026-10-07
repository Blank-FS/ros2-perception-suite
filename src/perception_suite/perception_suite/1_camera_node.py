import signal
import threading
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraPublisher(Node):
    """Stage 1: the GMSL dashcam, publishing /raw_frames.

    Interchangeable with 1_video_node: same topic, same header, same downscale,
    so stages 2-6 cannot tell which source they are fed by.
    """

    def __init__(self):
        super().__init__("camera_node")
        # pipeline.launch.py passes every source argument through
        # LaunchConfiguration.perform(), which yields strings, so these are
        # declared as strings and converted here. A float or int default would
        # make the node reject the launch's value at startup.
        index = int(self.declare_parameter("camera_index", "2").value)
        fps = float(self.declare_parameter("camera_fps", "30.0").value)
        # What the sensor is asked to deliver, before the downscale below.
        capture_width = int(self.declare_parameter("capture_width", "1920").value)
        capture_height = int(self.declare_parameter("capture_height", "1080").value)
        # Downscaled for DDS bandwidth only; the working size is set by the
        # preprocess stage. Aspect is kept so the camera model stays square-pixel.
        self.width = int(self.declare_parameter("width", "1280").value)

        self.publisher_ = self.create_publisher(Image, "raw_frames", 10)
        # CAP_V4L2 explicitly: left to choose, OpenCV picks the GStreamer
        # backend on this board and negotiates a format the capture then fails
        # to deliver.
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {index} (/dev/video{index}). "
                "The GMSL camera needs nv_nru2mp loaded and the deserializers "
                "initialised: check 'systemctl status nru-camera' and that "
                "/dev/video* exists."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (capture_width, capture_height):
            # Not fatal: the driver is free to pick the nearest supported mode,
            # but the aspect it lands on drives the camera model, so say so.
            self.get_logger().warn(
                f"Camera delivered {actual_w}x{actual_h}, not the requested "
                f"{capture_width}x{capture_height}"
            )
        self.get_logger().info(
            f"Camera {index} at {actual_w}x{actual_h}, publishing {self.width} wide "
            f"at {fps:g} FPS"
        )

        # INTER_AREA is an area average and costs about 4x INTER_LINEAR on this
        # board (10.9 ms vs 2.8 ms for a 4K source, measured). The source is
        # already being downscaled for bandwidth, not for analysis, and the
        # preprocess stage resizes again anyway, so linear is enough here.
        self.interpolation = cv2.INTER_LINEAR

        self.bridge = CvBridge()
        # cap.read() blocks until the sensor has a frame, so doing it on the
        # timer serialises the wait with the resize and publish: measured
        # 21.3 Hz published from a 30 Hz sensor. Capture runs on its own thread
        # and the timer publishes whatever the latest frame is.
        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._reader = threading.Thread(target=self._capture_loop, daemon=True)
        self._reader.start()
        self.timer = self.create_timer(1.0 / max(fps, 0.1), self.timer_callback)

    def _capture_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                # A live camera can drop a frame without the session being over.
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = frame

    def timer_callback(self):
        with self._lock:
            frame = self._latest
            self._latest = None
        if frame is None:
            # Nothing new since the last tick: either the capture thread is
            # still waiting on the sensor, or the camera has stopped.
            self.get_logger().warn("No new camera frame", throttle_duration_sec=5.0)
            return
        h, w = frame.shape[:2]
        height = round(h * self.width / w)
        small_frame = (frame if (w, h) == (self.width, height)
                       else cv2.resize(frame, (self.width, height),
                                       interpolation=self.interpolation))

        msg = self.bridge.cv2_to_imgmsg(small_frame, encoding="bgr8")
        # Every downstream stage copies this header, which is how the
        # visualizer matches a frame to its mask and path.
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.publisher_.publish(msg)

    def destroy_node(self):
        self._running = False
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.join(timeout=2.0)
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(CameraPublisher())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Under launch, Ctrl+C arrives twice (terminal and launch); the second
        # must not interrupt teardown or interpreter exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
