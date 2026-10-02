# perception_suite

A ROS 2 perception pipeline for real-time drivable terrain segmentation and path planning. Each stage of [orfd-lane-detection](https://github.com/HATCI-MDP/orfd-lane-detection) (Python package `offroad_autonomy`) runs as its own node. The nodes are wrappers that call that package's stage classes, so the segmentation, planning and dashboard logic sync with orfd's code.

```
1_video_node ─[raw_frames]─▶ 2_preprocess_node ─[preprocessed_frames]─▶ 3_segmentation_node ─[road_mask]─▶
4_postprocess_node ─[stabilized_mask]─▶ 5_planner_node ─[path]─▶ 6_visualizer_node
```

| Node                  | Wraps (`offroad_autonomy`) | Publishes                                                          |
| --------------------- | -------------------------- | ------------------------------------------------------------------ |
| `1_video_node`        | BeamNG camera (stand-in)   | `/raw_frames` (bgr8)                                               |
| `2_preprocess_node`   | `ImagePreprocessor`        | `/preprocessed_frames` (bgr8, working size)                        |
| `3_segmentation_node` | `RoadSegmenter` (YOLOE-26) | `/road_mask` (32FC1, road pixels = best detection confidence)      |
| `4_postprocess_node`  | `TemporalStabilizer`       | `/stabilized_mask` (32FC1)                                         |
| `5_planner_node`      | `CenterlinePlanner`        | `/path` (`nav_msgs/Path`, metres, x forward, y left)               |
| `6_visualizer_node`   | `AutonomyDashboard`        | Dashboard window (keys: `0`-`9` debug views, `T` timing, `Q` quit) |

Every stage copies the header of the frame it came from, so the visualizer matches a frame, its mask and its path by timestamp.

Not ported from orfd-lane-detection:

- **Stereo depth and fusion.** The video comes from a single camera, and orfd's baseline planner does not use depth anyway.
- **Control.** There is no vehicle to steer.

As a result, the dashboard's vehicle, control and depth panels read zero.

---

## Requirements

| Dependency                                                              | Notes                                                                    |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| ROS 2                                                                   | Tested on Jazzy (Ubuntu 24.04) and Humble (Ubuntu 22.04, Jetson Orin)    |
| Python                                                                  | Must be the version your ROS distro was built for (see below), managed with `uv` |
| [orfd-lane-detection](https://github.com/HATCI-MDP/orfd-lane-detection) | Cloned next to this workspace (see Setup)                                |
| `uv`                                                                    | [Install guide](https://docs.astral.sh/uv/getting-started/installation/) |
| `git`                                                                   | `uv sync` fetches Ultralytics' CLIP text encoder from GitHub             |
| CUDA-capable GPU                                                        | Optional; YOLOE runs on the CPU, more slowly                             |

---

## Setup

### 1. Clone both repositories side by side

`pyproject.toml` installs orfd-lane-detection from `../orfd-lane-detection` as an editable dependency, so the two folders must share a parent directory:

```
<parent>/
├── orfd-lane-detection/    # offroad_autonomy package
└── ros2-perception-suite/  # this repository
```

```bash
git clone git@github.com:HATCI-MDP/orfd-lane-detection.git
git clone git@github.com:HATCI-MDP/ros2-perception-suite.git
```

Because the dependency is editable, changes made in `orfd-lane-detection` take effect in the ROS nodes without reinstalling.

### 2. Install ROS dependencies

Replace `jazzy` with your distro, e.g. `humble`, here and in every later `source` command.

```bash
cd ros2-perception-suite
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y
```

### 3. Install Python dependencies and build

```bash
uv sync --python /usr/bin/python3
source .venv/bin/activate
colcon build --symlink-install
```

The virtual environment must use the same Python version that your ROS distro was built for. ROS's compiled modules (such as `rclpy`) only load into that version. If the versions differ, every node fails with `No module named 'rclpy._rclpy_pybind11'`. `/usr/bin/python3` is the system Python that ROS is built against, so `--python /usr/bin/python3` always picks the right one:

| ROS 2 distro | Ubuntu | Python |
| ------------ | ------ | ------ |
| Humble       | 22.04  | 3.10   |
| Jazzy        | 24.04  | 3.12   |

If you built with the wrong Python, delete the old environment and build output, then repeat this step:

```bash
rm -rf .venv build install log
```

On a Jetson, the PyTorch that `uv` installs from PyPI may not use the GPU, so YOLOE can fall back to the CPU. For GPU inference, install NVIDIA's PyTorch build for your JetPack version into `.venv`.

### 4. Add a video

Put a video at `test_video.mp4` in the workspace root, or pass another file with `video_path:=`. Videos are gitignored.

On the first run, the model weights (`yoloe-26n-seg.pt`) and the YOLOE text encoder (`mobileclip2_b.ts`) are downloaded into the workspace root automatically.

---

## Running

Run from the workspace root, because the video and model weight paths are relative to it. In each new terminal:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate && source install/setup.bash
ros2 launch perception_suite pipeline.launch.py scene:=snow
```

| Argument       | Default                  | What it does                                                                              |
| -------------- | ------------------------ | ----------------------------------------------------------------------------------------- |
| `source`       | `video`                  | Input node. Only `video` exists so far                                                    |
| `video_path`   | `test_video.mp4`         | Video file for `source:=video`                                                            |
| `source_delay` | `10.0`                   | Seconds before the source starts, so segmentation has loaded its model                    |
| `scene`        | `trail`                  | Segmentation prompt set from `perception.scenes` in the config: `trail`, `snow`, `gravel` |
| `config`       | `config/perception.yaml` | Pipeline config YAML                                                                      |
| `speed_mps`    | `0.0`                    | Fixed speed given to the planner (must be written as a float)                             |
| `visualize`    | `true`                   | Open the dashboard window                                                                 |

The dashboard needs a display. Over SSH without X forwarding, launch with `visualize:=false`, or run `export DISPLAY=:0` first to show it on the machine's own screen. Without a display, the visualizer exits with an error and the other nodes keep running.

Run `ros2 launch perception_suite pipeline.launch.py --show-args` to list the arguments. An invalid `source` or `scene` stops the launch before any node starts.

To run a single stage on its own, use `ros2 run perception_suite <node>`, e.g. `ros2 run perception_suite 3_segmentation_node`. Without a scene, the segmentation node uses orfd's default prompts (the `trail` set).

---

## Configuration

All tuning lives in [config/perception.yaml](src/perception_suite/config/perception.yaml), which `offroad_autonomy`'s `load_config` reads. Any key it leaves out takes orfd's built-in default. To use another file, pass `config:=<path>` to the launch file, or `--ros-args -p config:=<path>` to a single node.

### Scenes

YOLOE is prompted with text, and its confidence measures how well the road matches that text. The planner rejects frames whose best confidence is below `planning.gate.confidence_threshold` (0.20). So prompts that don't describe the scene stall the planner, even when the mask itself is right.

| Scene    | Prompts                                              | Notes                                                                  |
| -------- | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| `trail`  | orfd's 5 defaults ("dirt road", "off-road trail", …) | For BeamNG dirt trails. Median best score 0.19 on `test_video.mp4`     |
| `snow`   | "snow covered road"                                  | Median best score 0.78 on `test_video.mp4`, with a near-identical mask |
| `gravel` | "gravel road", "gravel path", "dirt road"            | Not yet measured on footage                                            |

To add a scene, add a named prompt list under `perception.scenes` in the config. No code changes are needed.

### Camera

The planner fits its path on the ground in metres, using the camera's field of view, height and pitch. The values under `beamng.cameras` in the config (`fov_h: 90`, `height_m: 1.5`, `pitch_deg: 0`) are guesses for `test_video.mp4`. The path's shape in the image is still correct, but its distances in metres are only as accurate as these values. Measure them for your camera.

---

## Extending

**A new input** (camera, BeamNG):

1. Write a node that publishes `bgr8` images on `/raw_frames`, with `header.stamp` set.
2. Register it in `setup.py`.
3. Add one entry to `SOURCES` in [launch/pipeline.launch.py](src/perception_suite/launch/pipeline.launch.py), plus a `DeclareLaunchArgument` for each of its parameters.

**Message format.** Masks travel as 32FC1 images whose road pixels hold the frame's best detection confidence, which is the one number the planner's gate reads. The postprocess node's stability score and the planner's "holding last path" state are not published, so the dashboard shows the stability score as 0 and has no hold status. Carrying them would need a custom message package.

---

## Project structure

```
ros2-perception-suite/
├── src/
│   └── perception_suite/
│       ├── perception_suite/
│       │   ├── common.py                # Config loading, QoS, mask encoding
│       │   ├── 1_video_node.py          # Reads video file → /raw_frames
│       │   ├── 2_preprocess_node.py     # Resize + CLAHE → /preprocessed_frames
│       │   ├── 3_segmentation_node.py   # YOLOE-26 → /road_mask
│       │   ├── 4_postprocess_node.py    # EMA + morphology → /stabilized_mask
│       │   ├── 5_planner_node.py        # Gate + centreline → /path
│       │   └── 6_visualizer_node.py     # orfd AutonomyDashboard window
│       ├── config/perception.yaml       # Pipeline tuning, camera, scene prompts
│       ├── launch/pipeline.launch.py    # Starts the whole pipeline
│       ├── setup.py
│       └── package.xml
├── pyproject.toml                       # uv project; pulls in ../orfd-lane-detection
└── README.md
```
