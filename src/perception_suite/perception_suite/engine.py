"""TensorRT engines for the segmentation stage, built on first use and cached.

An engine has the prompts and input size compiled in, and only runs on the
GPU model and TensorRT version that built it. All of those go into the cached
file's name, so changing any of them builds a new engine instead of loading
one that segments the wrong thing or fails to load.
"""

import hashlib
import json
from pathlib import Path

from offroad_autonomy.types import PipelineConfig


def unsupported_reason() -> str | None:
    """Why TensorRT cannot run here, or None if it can."""
    import torch
    if not torch.cuda.is_available():
        return 'PyTorch sees no CUDA GPU'
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return 'the tensorrt package is not installed (uv sync --extra tensorrt)'
    return None


def engine_path(config: PipelineConfig) -> Path:
    import tensorrt
    import torch
    weights = Path(config.model_weights)
    key = json.dumps({
        'weights': weights.name,
        'prompts': list(config.perception_prompts),
        'input_size': int(config.perception_input_size),
        'gpu': torch.cuda.get_device_name(0),
        'tensorrt': tensorrt.__version__,
    }, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return weights.with_name(f'{weights.stem}.{digest}.engine')


def build_engine(config: PipelineConfig, path: Path) -> None:
    """FP16, which matched the PyTorch model's masks on the test video."""
    from ultralytics import YOLO
    model = YOLO(config.model_weights)
    # Set before export, or the engine segments the checkpoint's own classes.
    model.set_classes(list(config.perception_prompts))
    built = Path(model.export(
        format='engine', imgsz=int(config.perception_input_size), half=True, device=0))
    # Export always writes <weights>.engine; renaming keeps engines for other
    # scenes and sizes from overwriting each other.
    built.replace(path)
    built.with_suffix('.onnx').unlink(missing_ok=True)
