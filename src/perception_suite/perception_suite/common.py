"""Helpers shared by the pipeline stage nodes.

Masks travel between stages as mono8 images whose road pixels hold the
frame's best detection confidence. That keeps the one number the perception
gate reads attached to the mask it belongs to, without a custom message
package.
"""

from dataclasses import replace

import numpy as np
from offroad_autonomy.types import PipelineConfig
from offroad_autonomy.utils.config import load_config
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from perception_suite import settings

# A stage that falls behind skips to the newest message instead of working
# through a backlog.
LATEST_ONLY = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def load_pipeline_config(node: Node) -> PipelineConfig:
    """The config:= file with the model:= entry applied. Every stage applies
    the model, because the planner and visualizer gate on its threshold."""
    path = settings.resolve(node.declare_parameter('config', settings.DEFAULT_CONFIG).value)
    name = node.declare_parameter('model', settings.DEFAULT_MODEL).value
    node.get_logger().info(f'Loading pipeline config: {path} (model: {name})')
    config = load_config(path)
    model = settings.model(path, name)
    if model is None:
        node.get_logger().info('Config has no perception.models; using its model_weights')
        return config
    return replace(config, model_weights=str(model['weights']),
                   gate_min_confidence=float(model['gate_confidence_threshold']))


def encode_mask(mask: np.ndarray, confidence: float) -> np.ndarray:
    """Road mask plus a frame-wide confidence, packed into one mono8 image.

    The confidence is the pixel level, so 0 means "not road". A non-zero
    confidence that rounds to 0 is clamped to 1, otherwise a real but very low
    confidence would be indistinguishable from background and the whole mask
    would decode as empty.
    """
    level = int(round(float(confidence) * 255.0))
    level = min(255, max(0, level))
    if level == 0 and confidence > 0.0:
        level = 1
    return np.asarray(mask, dtype=bool).astype(np.uint8) * np.uint8(level)


def decode_mask(confidence_mask: np.ndarray) -> tuple[np.ndarray, float]:
    return confidence_mask > 0, float(confidence_mask.max(initial=0)) / 255.0
