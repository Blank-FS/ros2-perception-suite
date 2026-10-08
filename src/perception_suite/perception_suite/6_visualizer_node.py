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
        # Placement, which OpenCV otherwise leaves to the window manager: the
        # same code lands bottom-left under Docker and top-left under a venv.
        # -1 keeps that behaviour, so this is opt-in and cannot push the window
        # off a smaller screen than the one it was tuned on.
        win_x = int(self.declare_parameter('window_x', -1).value)
        win_y = int(self.declare_parameter('window_y', -1).value)
        if win_x >= 0 and win_y >= 0:
            cv2.moveWindow(WINDOW, win_x, win_y)
            self.get_logger().info(f'Dashboard window at {win_x},{win_y}')
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
        # Show something straight away: nothing reaches display_callback until
        # the source has waited source_delay and the segmenter has loaded its
        # model, which is 10-20s of blank screen otherwise.
        #
        # WND_PROP_VISIBLE is only honoured once it has reported a visible
        # window at least once (see display_callback). OpenCV's GTK3 build
        # returns -1 for it unconditionally, which is "not implemented", not
        # "closed"; taking it at face value shuts the dashboard down on its
        # first frame on any such build.
        self._visibility_reliable = False
        splash = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(splash, 'waiting for the pipeline...', (40, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        cv2.imshow(WINDOW, splash)
        for _ in range(10):
            cv2.waitKey(30)

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

        # The rate shown ON the dashboard, which is this node's rate and not the
        # /path rate: they diverge when the visualizer falls behind, and only
        # this one is what anyone watching the screen actually sees.
        self._viz_n = getattr(self, '_viz_n', 0) + 1
        if self._viz_n % 60 == 0:
            self.get_logger().info(
                f'render rate {self.stats.fps():.1f} FPS | '
                f'capture-to-display {self.stats.stage("latency").mean_ms:.1f} ms | '
                f'render {self.stats.stage("dashboard_render").mean_ms:.1f} ms')

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        # -1 means the build does not implement this property (OpenCV's GTK3
        # build does exactly that), so it cannot be distinguished from a closed
        # window on its own. Trust it only after it has once reported visible.
        visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
        if visible >= 1:
            self._visibility_reliable = True
        closed = self._visibility_reliable and visible < 1
        if closed or key in (27, ord('q')):
            self.get_logger().info('Dashboard closed')
            rclpy.try_shutdown()
            return
        self._handle_key(key)

    def _step_result(self, frame_msg, raw_mask_msg, mask_msg, path_msg):
        image = self.bridge.imgmsg_to_cv2(frame_msg, 'bgr8')
        h, w = image.shape[:2]
        raw_mask, confidence = decode_mask(self.bridge.imgmsg_to_cv2(raw_mask_msg, 'mono8'))
        mask, _ = decode_mask(self.bridge.imgmsg_to_cv2(mask_msg, 'mono8'))

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
