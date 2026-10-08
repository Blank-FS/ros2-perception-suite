"""The pipeline's three choices, resolved the same way by the launch file and
every node: which config (the input), which model, and which scene.

config  A YAML file name, looked up in src/perception_suite/config, so
        config:=demo_route.yaml works from any directory. An absolute path is
        used as it is.
model   A name under perception.models in that config, giving the weights and
        the model's planning.gate.confidence_threshold. A config with no
        perception.models uses its own perception.model_weights instead.
scene   A name under perception.scenes: the prompts for a YOLOE model. A
        semantic model has fixed classes and ignores it.

Kept free of ROS and offroad_autonomy imports so the launch file can use it.
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory

DEFAULT_CONFIG = 'perception.yaml'
DEFAULT_MODEL = 'yoloe'
DEFAULT_SCENE = 'trail'


def config_dir() -> Path:
    """src/perception_suite/config under --symlink-install, else the installed copy.

    A symlink install links each installed YAML back to its source, so following
    one finds the source folder, and new or edited files take effect without a
    rebuild.
    """
    installed = Path(get_package_share_directory('perception_suite')) / 'config'
    return (installed / DEFAULT_CONFIG).resolve().parent


def resolve(name: str) -> Path:
    path = config_dir() / name
    if not path.is_file():
        available = ', '.join(sorted(p.name for p in config_dir().glob('*.yaml')))
        raise FileNotFoundError(f"config '{name}' not found in {config_dir()}; "
                                f'available: {available}')
    return path


def _perception(path: Path) -> dict:
    with open(path, encoding='utf-8') as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get('perception', {}) or {}


def model(path: Path, name: str) -> dict | None:
    """{'weights': ..., 'gate_confidence_threshold': ...}, or None if the
    config predates perception.models and names its weights directly."""
    models = _perception(path).get('models')
    if not models:
        return None
    if name not in models:
        raise ValueError(f"model must be one of {sorted(models)} (perception.models in "
                         f"{path}), got '{name}'")
    return models[name]


def scene_prompts(path: Path, name: str) -> list[str] | None:
    """The scene's prompts, or None if the config defines no scenes."""
    scenes = _perception(path).get('scenes')
    if not scenes:
        return None
    if name not in scenes:
        raise ValueError(f"scene must be one of {sorted(scenes)} (perception.scenes in "
                         f"{path}), got '{name}'")
    return [str(prompt) for prompt in scenes[name]]
