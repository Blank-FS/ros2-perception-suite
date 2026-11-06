"""Check that everything the testing scripts need is in place.

    uv run check_setup.py

Prints what it finds. FAIL means a script won't run; WARN means something is
missing that only some steps need. Exits with 1 if anything failed.
"""

import importlib
import os
import sys
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = TESTING_DIR.parent / "weights"
PIPELINE_CONFIG = TESTING_DIR.parents[1] / "src/perception_suite/config/hatci_vehicle.yaml"
PACKAGES = {  # import name: package name
    "av": "av",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "numpy": "numpy",
    "requests": "requests",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "yaml": "pyyaml",
}

failures = 0


def report(status, what, detail=""):
    global failures
    failures += status == "FAIL"
    print(f"[{status:4}] {what}" + (f": {detail}" if detail else ""))


def check_packages():
    missing = False
    for module, package in PACKAGES.items():
        try:
            version = getattr(importlib.import_module(module), "__version__", "")
            report("OK", package, version)
        except ImportError as error:
            report("FAIL", package, f"{error}. Run `uv sync` in model/testing")
            missing = True
    return not missing


def check_gpu():
    import torch

    if torch.cuda.is_available():
        report("OK", "GPU", torch.cuda.get_device_name(0))
    else:
        report("WARN", "GPU", "no CUDA GPU found, so the scripts will run on the CPU, much more slowly")


def check_weights():
    from ultralytics import YOLO

    weights = sorted(WEIGHTS_DIR.glob("*.pt"))
    if not weights:
        report("FAIL", "Weights", f"no .pt files in {WEIGHTS_DIR}. See SETUP.md")
    for weight in weights:
        try:
            task = YOLO(weight).task
        except Exception as error:  # noqa: BLE001 - any load failure is what this reports
            report("FAIL", weight.name, f"cannot be loaded: {error}")
            continue
        if task == "semantic":
            report("OK", weight.name)
        else:
            report("WARN", weight.name, f"is a {task} model, so the scripts will skip it")


def check_data():
    from dotenv import dotenv_values

    if PIPELINE_CONFIG.exists():
        report("OK", "Pipeline config", PIPELINE_CONFIG.name)
    else:
        report("FAIL", "Pipeline config", f"{PIPELINE_CONFIG} not found, which gate_sweep.py needs")
    key = os.environ.get("ROBOFLOW_API_KEY") or dotenv_values(TESTING_DIR / ".env").get("ROBOFLOW_API_KEY")
    if key:
        report("OK", "Roboflow API key")
    else:
        report("WARN", "Roboflow API key", "not set in .env, which download_dataset.py needs")
    datasets = sorted(p.parent.name for p in (TESTING_DIR / "datasets").glob("*/data.yaml"))
    if datasets:
        report("OK", "Datasets", ", ".join(datasets))
    else:
        report("WARN", "Datasets", "none yet, which evaluate.py needs. Run download_dataset.py")
    videos = sorted(p.name for p in (TESTING_DIR / "recordings").glob("*.mp4"))
    if videos:
        report("OK", "Drive videos", ", ".join(videos))
    else:
        report("WARN", "Drive videos", "none in recordings/, which gate_sweep.py needs")


def main():
    version = ".".join(map(str, sys.version_info[:3]))
    report("OK" if sys.version_info >= (3, 10) else "FAIL", "Python", version)
    if check_packages():
        check_gpu()
        check_weights()
        check_data()
    print("\nSetup has problems: fix each FAIL above." if failures else "\nReady.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
