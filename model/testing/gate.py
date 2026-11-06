"""The parts of the pipeline's perception gate that gate_sweep.py runs, without its dependencies.

Copied from orfd-lane-detection at commit e59deae, so that testing needs nothing
but this folder: offroad_autonomy's EgoMask, TemporalStabilizer and PerceptionGate,
and the config values they read. If those change there, change them here too, or
the sweep stops matching the vehicle.

The gate's confidence check is left out. A semantic model has no detection score,
so the pipeline reports 1.0 and gates it at 0, and that check always passes.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# The road must start at the vehicle: a blob on the horizon is not somewhere it can drive to.
ANCHOR_FRACTION = 0.25


@dataclass(frozen=True)
class GateConfig:
    """What the gate reads from a pipeline config, with offroad_autonomy's defaults."""

    width: int = 720  # the pipeline's working frame
    height: int = 465
    ego_enabled: bool = False
    ego_polygon: tuple = ()
    ego_margin_px: int = 2
    enable_ema: bool = True
    ema_alpha: float = 0.7
    enable_morphology: bool = True
    morphology_kernel_size: int = 5
    min_mask_area_fraction: float = 0.001
    roi_height: float = 0.50
    min_mask_area: float = 0.05
    min_road_pixels: int = 500


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}

    def section(parent, key):
        value = parent.get(key)
        return value if isinstance(value, dict) else {}

    ego = section(section(section(raw, "beamng"), "camera"), "ego_mask")
    pre, post, plan = section(raw, "preprocessing"), section(raw, "postprocessing"), section(raw, "planning")
    gate = section(plan, "gate")
    default = GateConfig()
    polygon = tuple((float(x), float(y)) for x, y in ego.get("polygon") or ())
    enabled = bool(ego.get("enabled", False))
    if enabled and len(polygon) < 3:
        raise ValueError(f"{path}: beamng.camera.ego_mask is enabled but its polygon has fewer than 3 points")
    return GateConfig(
        width=int(pre.get("target_width", default.width)),
        height=int(pre.get("target_height", default.height)),
        ego_enabled=enabled,
        ego_polygon=polygon,
        ego_margin_px=int(ego.get("margin_px", default.ego_margin_px)),
        enable_ema=bool(post.get("enable_ema", default.enable_ema)),
        ema_alpha=float(post.get("ema_alpha", default.ema_alpha)),
        enable_morphology=bool(post.get("enable_morphology", default.enable_morphology)),
        morphology_kernel_size=int(post.get("morphology_kernel_size", default.morphology_kernel_size)),
        min_mask_area_fraction=float(post.get("min_mask_area_fraction", default.min_mask_area_fraction)),
        roi_height=float(plan.get("roi_height", default.roi_height)),
        min_mask_area=float(gate.get("min_mask_area", default.min_mask_area)),
        min_road_pixels=int(plan.get("min_road_pixels", default.min_road_pixels)),
    )


def valid_roi(config):
    """The working frame's pixels outside the hood polygon, widened by its margin (EgoMask)."""
    height, width = config.height, config.width
    excluded = np.zeros((height, width), dtype=np.uint8)
    if not config.ego_enabled:
        return ~excluded.astype(bool)
    points = np.asarray(config.ego_polygon, dtype=np.float64)
    pixels = np.stack([points[:, 0] * (width - 1), points[:, 1] * (height - 1)], axis=1)
    cv2.fillPoly(excluded, [np.round(pixels).astype(np.int32)], 1)
    margin = config.ego_margin_px
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        excluded = cv2.dilate(excluded, kernel)
    mask = excluded.astype(bool)
    if mask.all():
        raise ValueError("Ego mask covers the entire frame: check beamng.camera.ego_mask.polygon")
    return ~mask


class Stabilizer:
    """Smooths a road mask over time and scores its change from the last frame (TemporalStabilizer)."""

    def __init__(self, config):
        self.config = config
        size = config.morphology_kernel_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        self.accum = None
        self.previous = None

    def stabilize(self, mask, roi):
        """Return the stable mask and its IoU with the previous frame's."""
        config = self.config
        mask_f = mask.astype(np.float32)
        if not config.enable_ema or self.accum is None or self.accum.shape != mask_f.shape:
            self.accum = mask_f.copy()
        else:
            self.accum = config.ema_alpha * mask_f + (1.0 - config.ema_alpha) * self.accum
        smoothed = (self.accum > 0.5).astype(np.uint8)
        if config.enable_morphology:
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, self.kernel)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, self.kernel)
        h, w = smoothed.shape
        if int(smoothed.sum()) < int(h * w * config.min_mask_area_fraction):
            smoothed = np.zeros_like(smoothed)
        stable = smoothed.astype(bool) & roi  # closing a gap must never grow the road onto the hood
        stability = self._temporal_iou(stable)
        self.previous = stable
        return stable, stability

    def _temporal_iou(self, mask):
        if self.previous is None:
            return 1.0
        union = int((mask | self.previous).sum())
        return int((mask & self.previous).sum()) / union if union > 0 else 1.0


@dataclass
class Decision:
    ok: bool
    reason: str
    road_pixels: int
    area_fraction: float


class Gate:
    """Decides whether a frame's road is good enough to plan from (PerceptionGate)."""

    def __init__(self, config):
        self.min_area = config.min_mask_area
        self.min_pixels = config.min_road_pixels
        self.roi_height = float(np.clip(config.roi_height, 0.1, 1.0))

    def roi_top(self, height):
        return round(height * (1.0 - self.roi_height))

    def evaluate(self, mask, valid):
        top = self.roi_top(mask.shape[0])
        band = np.zeros_like(mask)
        band[top:] = mask[top:]
        band &= valid
        roi_pixels = int(valid[top:].sum())
        road_pixels = int(band.sum())
        component = self._vehicle_component(band, top, valid)
        area = float(component.sum()) / max(roi_pixels, 1)

        reason = ""
        if road_pixels < self.min_pixels:
            reason = f"too few road pixels ({road_pixels})"
        elif not component.any():
            reason = "no road at vehicle"
        elif area < self.min_area:
            reason = f"road area {area:.1%}<{self.min_area:.0%}"
        return Decision(not reason, reason, road_pixels, area)

    @staticmethod
    def _vehicle_component(band, top, valid):
        """The biggest road region touching the nearest visible ground of each column."""
        h = band.shape[0]
        if not band.any():
            return band
        _, labels, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
        rows = np.arange(h)[:, None]
        last = np.max(np.where(valid, rows, -1), axis=0)
        span = np.maximum(1, np.rint((last - top + 1) * ANCHOR_FRACTION).astype(int))
        anchors = valid & (rows >= np.maximum(top, last - span + 1)) & (rows <= last)
        anchored = np.unique(labels[anchors])
        anchored = anchored[anchored > 0]
        if len(anchored) == 0:
            return np.zeros_like(band)
        best = int(anchored[np.argmax(stats[anchored, cv2.CC_STAT_AREA])])
        return labels == best
