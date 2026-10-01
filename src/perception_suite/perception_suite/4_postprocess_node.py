import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.types import PerceptionResult

from perception_suite.common import LATEST_ONLY, decode_mask, encode_mask, load_pipeline_config


class PostprocessNode(Node):
    """Stage 4: EMA and morphology over the mask (TemporalStabilizer)."""

    def __init__(self):
        super().__init__('postprocess_node')
        config = load_pipeline_config(self)
        self.valid_roi = PerceptionView(config).valid_roi
        self.stabilizer = TemporalStabilizer(config)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'stabilized_mask', 10)
        self.subscription = self.create_subscription(
            Image, 'road_mask', self.on_mask, LATEST_ONLY)

    def on_mask(self, msg):
        mask, confidence = decode_mask(self.bridge.imgmsg_to_cv2(msg, '32FC1'))
        perception = PerceptionResult(
            mask=mask, confidences=[confidence], valid_roi=self.valid_roi)
        stabilized = self.stabilizer.stabilize(perception)
        self.get_logger().debug(f'stability={stabilized.stability_score:.3f}')

        out = self.bridge.cv2_to_imgmsg(encode_mask(stabilized.mask, confidence), '32FC1')
        out.header = msg.header
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(PostprocessNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Under launch, Ctrl+C arrives twice (terminal and launch); the second
        # must not interrupt teardown or interpreter exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
