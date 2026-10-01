import signal
import time
from dataclasses import replace

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.types import FramePacket

from perception_suite.common import LATEST_ONLY, encode_mask, load_pipeline_config


class SegmentationNode(Node):
    """Stage 3: YOLOE-26 road segmentation (RoadSegmenter), publishing /road_mask."""

    def __init__(self):
        super().__init__('segmentation_node')
        config = load_pipeline_config(self)
        # Set by the launch file from the config's perception.scenes.
        prompts = self.declare_parameter('prompts', Parameter.Type.STRING_ARRAY).value
        if prompts:
            config = replace(config, perception_prompts=list(prompts))
        self.get_logger().info(f'Prompts: {config.perception_prompts}')
        self.valid_roi = PerceptionView(config).valid_roi
        self.segmenter = RoadSegmenter(config)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'road_mask', 10)
        self.subscription = self.create_subscription(
            Image, 'preprocessed_frames', self.on_frame, LATEST_ONLY)
        self.get_logger().info('Segmentation online')

    def on_frame(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = image.shape[:2]
        frame = FramePacket(
            raw=image, preprocessed=image, timestamp=time.perf_counter(), height=h, width=w)
        result = self.segmenter.predict(frame, self.valid_roi)

        # Published even when empty, so every segmented frame reaches the
        # planner and the visualizer.
        confidence = max(result.confidences, default=0.0)
        out = self.bridge.cv2_to_imgmsg(encode_mask(result.mask, confidence), '32FC1')
        out.header = msg.header
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(SegmentationNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Under launch, Ctrl+C arrives twice (terminal and launch); the second
        # must not interrupt teardown or interpreter exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
