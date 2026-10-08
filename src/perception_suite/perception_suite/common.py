"""Helpers shared by the pipeline stage nodes.

Masks travel between stages as 32FC1 images whose road pixels hold the
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
    return mask.astype(np.float32) * np.float32(confidence)


def decode_mask(confidence_mask: np.ndarray) -> tuple[np.ndarray, float]:
    return confidence_mask > 0.0, float(confidence_mask.max(initial=0.0))
