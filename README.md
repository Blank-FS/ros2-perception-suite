# perception_suite

A ROS 2 perception pipeline for real-time drivable terrain segmentation and path planning. Supports two interchangeable inference backends — **YOLO-E 26n** (fast, open-vocabulary) and **SAM 2.1** (precise, point-prompted) — with a shared downstream path planner that fits a smoothed, horizon-weighted centerline trajectory over the detected road mask.

```
video_node  ──▶  [raw_frames]  ──▶  inference_node  ──▶  [processed_frames]  ──▶  visualizer_node
```

---

## Requirements

| Dependency       | Notes                                                                    |
| ---------------- | ------------------------------------------------------------------------ |
| ROS 2            | Tested on Jazzy                                                          |
| Python           | Tested on v3.12.3; Managed via `uv`                                      |
| CUDA-capable GPU | Required for SAM 2.1; optional for YOLO                                  |
| `uv`             | [Install guide](https://docs.astral.sh/uv/getting-started/installation/) |

---

## Project structure

```
ros2_ws/
├── src/
│   └── perception_suite/
│       ├── perception_suite/
│       │   ├── video_node.py        # Reads video file → publishes /raw_frames
│       │   ├── inference_node.py    # Segmentation (YOLO / SAM) → publishes /processed_frames
│       │   └── visualizer_node.py  # Subscribes to /processed_frames and displays output
│       ├── setup.py
│       └── package.xml
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
uv sync
source .venv/bin/activate
colcon build --symlink-install
```

---

## Running

Each node runs in its own terminal. Source the environment in each one before running.

```bash
source .venv/bin/activate && source install/setup.bash
```

### Terminal 1 — video node

> **Note:** The video node reads from `test_video.mp4` in the workspace root. Place your video file there before running.

```bash
ros2 run perception_suite video_node
```

### Terminal 2 — inference node

```bash
ros2 run perception_suite inference_node --model yolo   # default
```

```bash
ros2 run perception_suite inference_node --model sam    # requires CUDA
```

| `--model` | Backend    | Speed                      | Notes |
| --------- | ---------- | -------------------------- | ----- |
| `yolo`    | YOLO-E 26n | Fast (~30 FPS)             | N/A   |
| `sam`     | SAM 2.1-L  | Slow (20x decrease in FPS) | N/A   |

### Terminal 3 — visualizer node

```bash
ros2 run perception_suite visualizer_node
```
