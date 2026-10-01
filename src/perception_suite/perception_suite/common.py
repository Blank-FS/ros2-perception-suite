"""Helpers shared by the pipeline stage nodes.

Masks travel between stages as 32FC1 images whose road pixels hold the
frame's best detection confidence. That keeps the one number the perception
gate reads attached to the mask it belongs to, without a custom message
package.
"""

from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from offroad_autonomy.types import PipelineConfig
from offroad_autonomy.utils.config import load_config
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# A stage that falls behind skips to the newest message instead of working
# through a backlog.
LATEST_ONLY = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def load_pipeline_config(node: Node) -> PipelineConfig:
    default = Path(get_package_share_directory('perception_suite')) / 'config' / 'perception.yaml'
    path = node.declare_parameter('config', str(default)).value
    node.get_logger().info(f'Loading pipeline config: {path}')
    return load_config(path)


def encode_mask(mask: np.ndarray, confidence: float) -> np.ndarray:
    return mask.astype(np.float32) * np.float32(confidence)


def decode_mask(confidence_mask: np.ndarray) -> tuple[np.ndarray, float]:
    return confidence_mask > 0.0, float(confidence_mask.max(initial=0.0))
