import os
import signal
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer
import cv2
import numpy as np
from offroad_autonomy.perception.ego_mask import road_fraction
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.planning.perception_gate import PerceptionGate
from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.types import (
    DEBUG_VIEW_KEYS,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineStepResult,
    StabilizedResult,
)
from offroad_autonomy.visualization import AutonomyDashboard, DashboardTelemetry

from perception_suite.common import decode_mask, load_pipeline_config

# Same title as orfd-lane-detection's dashboard window.
WINDOW = "Off-Road Autonomy Dashboard"


class VisualizationNode(Node):
    """Stage 6: orfd's AutonomyDashboard, fed from the stage topics.

    The dashboard draws from a PipelineStepResult, so one is rebuilt per frame
    and the mask and path overlays are drawn by orfd's own code. Vehicle,
    and control readouts stay at zero, because a video has none of them.
    """

    def __init__(self):
        super().__init__('visualizer_node')
        config = load_pipeline_config(self)
        view = PerceptionView(config)
        self.camera = view.camera
        self.valid_roi = view.valid_roi
        self.ego_coverage = view.ego_coverage
        self.segmentation_mode = view.mode
        self.gate = PerceptionGate(config)
        self.dashboard = AutonomyDashboard(
            width=1600,
            height=900,
            colors=config.dashboard_colors,
            sensor=config.camera.sensor,
            thresholds=config.dashboard_thresholds,
        )
        # Not orfd's DashboardWindow: it shrinks each frame to the window's
        # reported image area, and OpenCV's Qt backend (Linux) reports its
        # tiny pre-show default forever, so the dashboard never grows. Here Qt
        # scales the full canvas itself. GUI_NORMAL drops Qt's toolbar and
        # status bar, so the window is a plain frame like orfd's on Windows.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW, self.dashboard.width, self.dashboard.height)
        self.debug_view = config.ui_debug_view
        self.timing_overlay = config.ui_timing_overlay
        self.stats = RuntimeStats()
        self.bridge = CvBridge()

        # All four share the source frame's stamp and arrive once per segmented
        # frame, so they match however slow segmentation is.
        self.sync = TimeSynchronizer(
            [
                Subscriber(self, Image, 'segmented_frames'),
                Subscriber(self, Image, 'road_mask'),
                Subscriber(self, Image, 'stabilized_mask'),
                Subscriber(self, Path, 'path'),
            ],
            queue_size=10,
        )
        self.sync.registerCallback(self.display_callback)
        self.get_logger().info(
            'Keys: T timing overlay, Q quit, '
            + ' '.join(f'{i}={name}' for i, name in DEBUG_VIEW_KEYS.items()))

    def display_callback(self, frame_msg, raw_mask_msg, mask_msg, path_msg):
        self.stats.tick()
        # Capture to display: the closest ROS analogue of orfd's loop latency.
        captured = Time.from_msg(frame_msg.header.stamp)
        latency_ms = (self.get_clock().now() - captured).nanoseconds / 1e6
        self.stats.record('latency', latency_ms)

        result = self._step_result(frame_msg, raw_mask_msg, mask_msg, path_msg)
        t0 = time.perf_counter()
        canvas = self.dashboard.render(
            result,
            self._telemetry(result),
            plan=result.plan,
            valid_roi=self.valid_roi,
            debug_view=self.debug_view,
            timing_overlay=self.timing_overlay,
        )
        self.stats.record('dashboard_render', (time.perf_counter() - t0) * 1000.0)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        if closed or key in (27, ord('q')):
            self.get_logger().info('Dashboard closed')
            rclpy.try_shutdown()
            return
        self._handle_key(key)

    def _step_result(self, frame_msg, raw_mask_msg, mask_msg, path_msg):
        image = self.bridge.imgmsg_to_cv2(frame_msg, 'bgr8')
        h, w = image.shape[:2]
        raw_mask, confidence = decode_mask(self.bridge.imgmsg_to_cv2(raw_mask_msg, '32FC1'))
        mask, _ = decode_mask(self.bridge.imgmsg_to_cv2(mask_msg, '32FC1'))

        perception = PerceptionResult(
            mask=raw_mask,
            confidences=[confidence],
            num_detections=int(raw_mask.any()),
            valid_roi=self.valid_roi,
            road_fraction=road_fraction(raw_mask, self.valid_roi),
        )
        stabilized = StabilizedResult(
            mask=mask,
            # The postprocess stage does not publish its stability score.
            stability_score=0.0,
            raw_result=perception,
            valid_roi=self.valid_roi,
            road_fraction=road_fraction(mask, self.valid_roi),
        )
        return PipelineStepResult(
            frame=FramePacket(
                raw=image, preprocessed=image, timestamp=time.perf_counter(), height=h, width=w),
            perception=perception,
            stabilized=stabilized,
            plan=self._plan(path_msg, h, w),
            command=ControlCommand(),
        )

    def _plan(self, path_msg, h, w):
        """The planner publishes metres but draws in pixels, so the path is
        projected back onto the image the planner read."""
        roi_top = self.gate.roi_top(h)
        if not path_msg.poses:
            # The planner publishes an empty path only when the gate stops.
            return PathPlan(
                centerline=np.empty((0, 2), dtype=np.float32),
                fallback_active=True,
                fallback_reason='no valid path - stopping',
                speed_scale=0.0,
                roi_top=roi_top,
            )
        forward = np.array([p.pose.position.x for p in path_msg.poses])
        right = -np.array([p.pose.position.y for p in path_msg.poses])
        centerline = self.camera.ground_to_image(forward, right)
        centerline[:, 0] = np.clip(centerline[:, 0], 0.0, w - 1.0)
        return PathPlan(centerline=centerline, roi_top=roi_top)

    def _telemetry(self, result):
        latency = self.stats.stage('latency')
        fallback_state = 'NONE'
        if result.plan.fallback_active:
            fallback_state = 'GATE STOP'
        lines = []
        if self.timing_overlay:
            lines = ['ROS VISUALIZER'] + self.stats.format_lines(('latency', 'dashboard_render'))
        return DashboardTelemetry(
            speed_mph=0.0,
            steering=0.0,
            throttle=0.0,
            brake=0.0,
            perception_confidence=result.perception.confidences[0],
            stability_score=result.stabilized.stability_score,
            kalman_active=False,
            fps=self.stats.fps(),
            latency_ms=latency.mean_ms,
            latency_p95_ms=latency.p95_ms,
            road_fraction=result.stabilized.road_fraction,
            ego_coverage=self.ego_coverage,
            segmentation_mode=self.segmentation_mode,
            fallback_state=fallback_state,
            dashboard_fps=self.stats.fps(),
            timing_lines=lines,
        )

    def _handle_key(self, key):
        """orfd's display keys. E and P start and stop the vehicle, so they
        do nothing here."""
        if key in (ord('t'), ord('T')):
            self.timing_overlay = not self.timing_overlay
        elif ord('0') <= key <= ord('9') and key - ord('0') in DEBUG_VIEW_KEYS:
            self.debug_view = DEBUG_VIEW_KEYS[key - ord('0')]
            self.get_logger().info(f'Debug view: {self.debug_view}')


def main(args=None):
    rclpy.init(args=args)
    # OpenCV's Qt backend only draws through X11, and with no display it
    # aborts the process rather than raising, so check before any window opens.
    if not os.environ.get('DISPLAY'):
        rclpy.logging.get_logger('visualizer_node').error(
            'No display (DISPLAY is unset, e.g. over SSH). Run on the desktop, '
            'export DISPLAY=:0 for a local screen, or launch with visualize:=false.')
        rclpy.try_shutdown()
        sys.exit(1)
    try:
        rclpy.spin(VisualizationNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Under launch, Ctrl+C arrives twice (terminal and launch); the second
        # must not interrupt teardown or interpreter exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
