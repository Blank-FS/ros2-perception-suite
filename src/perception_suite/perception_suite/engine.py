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
        # On a Jetson, TensorRT's Python bindings come with JetPack and reach
        # the venv through --system-site-packages.
        return ('the tensorrt package is not installed (uv sync --extra tensorrt; '
                'on a Jetson, sudo apt install python3-libnvinfer)')
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
    import inspect

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.export import engine as ultralytics_export
    # Ultralytics passes dynamo=False to torch.onnx.export on torch 2.4 and
    # later, but NVIDIA's JetPack 6.0 build (2.4.0a0) predates that argument.
    # The flag only decides that argument; newer torch still needs it.
    if 'dynamo' not in inspect.signature(torch.onnx.export).parameters:
        ultralytics_export.TORCH_2_4 = False
    model = YOLO(config.model_weights)
    # Set before export, or the engine segments the checkpoint's own classes.
    model.set_classes(list(config.perception_prompts))
    # Fold the prompts into the head now. Export otherwise does it after
    # fusing the rest of the model, which on a checkpoint saved unfused has
    # already removed the branch this needs, and fails in Ultralytics 8.4.
    model.model.model[-1].fuse(model.model.pe)
    # Unsimplified: simplifying needs onnxruntime-gpu, which has no Jetson
    # build on PyPI, and TensorRT optimises the graph itself.
    built = Path(model.export(
        format='engine', imgsz=int(config.perception_input_size), half=True,
        simplify=False, device=0))
    # Export always writes <weights>.engine; renaming keeps engines for other
    # scenes and sizes from overwriting each other.
    built.replace(path)
    built.with_suffix('.onnx').unlink(missing_ok=True)
