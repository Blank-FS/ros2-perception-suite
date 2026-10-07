# perception_suite

A ROS 2 perception pipeline for real-time drivable terrain segmentation and path planning. Each stage of [orfd-lane-detection](https://github.com/HATCI-MDP/orfd-lane-detection) (Python package `offroad_autonomy`) runs as its own node. The nodes are wrappers that call that package's stage classes, so the segmentation, planning and dashboard logic sync with orfd's code.

```
1_video_node ─[raw_frames]─▶ 2_preprocess_node ─[preprocessed_frames]─▶ 3_segmentation_node ─[road_mask]─▶
4_postprocess_node ─[stabilized_mask]─▶ 5_planner_node ─[path]─▶ 6_visualizer_node
```

| Node                  | Wraps (`offroad_autonomy`) | Publishes                                                          |
| --------------------- | -------------------------- | ------------------------------------------------------------------ |
| `1_video_node`        | BeamNG camera (stand-in)   | `/raw_frames` (bgr8)                                               |
| `1_camera_node`       | GMSL dashcam (V4L2)        | `/raw_frames` (bgr8), interchangeable with `1_video_node`          |
| `2_preprocess_node`   | `ImagePreprocessor`        | `/preprocessed_frames` (bgr8, working size)                        |
| `3_segmentation_node` | `RoadSegmenter` (YOLOE-26) | `/road_mask` (32FC1, road pixels = best detection confidence), `/segmented_frames` (the frames it segmented, for the visualizer) |
| `4_postprocess_node`  | `TemporalStabilizer`       | `/stabilized_mask` (32FC1)                                         |
| `5_planner_node`      | `CenterlinePlanner`        | `/path` (`nav_msgs/Path`, metres, x forward, y left)               |
| `6_visualizer_node`   | `AutonomyDashboard`        | Dashboard window (keys: `0`-`9` debug views, `T` timing, `Q` quit) |

Every stage copies the header of the frame it came from, so the visualizer matches a frame, its masks and its path by timestamp. It matches against `/segmented_frames` rather than `/preprocessed_frames`, so the dashboard works however slowly segmentation runs.

Not ported from orfd-lane-detection:

- **Control.** There is no vehicle to steer.

As a result, the dashboard's vehicle and control panels read zero.

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

#### Jetson

PyPI's ARM64 PyTorch has no GPU support, and a Jetson's CUDA build has to match its JetPack version. So on ARM64 Linux with Python 3.10 (a Jetson running ROS Humble), `uv sync` does not install torch or torchvision. The environment uses the JetPack-matched build already installed for the system Python instead, the same way other Jetson projects use it.

1. Check that the system Python has a CUDA build of PyTorch:

   ```bash
   /usr/bin/python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
   ```

   If it fails or prints `False`, install NVIDIA's build for your JetPack (`cat /etc/nv_tegra_release` shows the release: R36.3 is JetPack 6.0, R36.4 is 6.1/6.2) with `pip install --user`, plus a torchvision built for it:

   | JetPack | PyTorch |
   | ------- | ------- |
   | 6.0 (CUDA 12.2) | NVIDIA's `torch-2.4.0a0+07cecf4168.nv24.05` wheel from `https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/`. NVIDIA publishes no torchvision for it, so torchvision is built from source |
   | 6.1/6.2 (CUDA 12.6) | `torch==2.8.0 torchvision==0.23.0` from `--index-url https://pypi.jetson-ai-lab.io/jp6/cu126` |

   Either also needs OpenBLAS: `sudo apt install libopenblas0`.

2. Create the environment with access to the system's packages, in place of the plain `uv sync` above. The environment's own packages still take precedence; only torch and torchvision come from outside it:

   ```bash
   rm -rf .venv
   uv venv --system-site-packages --python /usr/bin/python3
   uv sync
   source .venv/bin/activate
   colcon build --symlink-install
   ```

   Later `uv sync` runs keep this setting. Only deleting `.venv` loses it.

3. Confirm the GPU is in use:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   If it prints `False`, segmentation runs on the CPU at a fraction of a frame per second.

### 4. Add a video

Put a video at `test_video.mp4` in the workspace root, or pass another file with `video_path:=`. Videos are gitignored.

On the first run, the model weights (`yoloe-26n-seg.pt`) and the YOLOE text encoder (`mobileclip2_b.ts`) are downloaded into the workspace root automatically.

---

## Running

### Which command

Two inputs, one stack:

```bash
ros2 launch perception_suite pipeline.launch.py                  # test_video.mp4
ros2 launch perception_suite pipeline.launch.py source:=camera   # GMSL camera
```

Add to either:

```bash
config:=$(ros2 pkg prefix perception_suite)/share/perception_suite/config/perception_trt.yaml   # TensorRT
visualize:=false        # no dashboard, worth ~16 ms/frame
video_path:=clip.mp4    # a different clip, with source:=video
camera_index:=0         # a different camera, with source:=camera
scene:=snow             # another prompt set (no effect on a .engine, see Scenes)
```

A single node running stages 2-6 was tried and removed. It did cut the four
image topics between stages, but it was slower: on the demo route it reached
20.8-27.7 FPS against the six-node stack's ~30, which is the source rate. Six
processes pipeline across the Jetson's eight cores, so while segmentation works
on one frame the preprocessor is already on the next; one callback makes that
sequential. The copying it saved was smaller than the parallelism it lost.

Measured per stage on the live camera with a TensorRT `yoloe-26n` engine:

```
segmentation 23 ms   preprocess 7 ms   postprocess 3 ms   planning 1 ms   control 0 ms
```

`planning` is 1 ms only while the perception gate is rejecting; pointed at road
it rises to 11-16 ms. Choosing the grid planner over the baseline costs under
0.3 FPS, so it is a behaviour decision rather than a performance one.

### Running

Run from the workspace root, because the video and model weight paths are relative to it. In each new terminal:

```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch perception_suite pipeline.launch.py scene:=snow
```

Activate the environment before sourcing ROS. If the environment is already active, `activate` resets `PATH` to what it was when it was first activated, which drops ROS from it and fails with `ros2: command not found`.

| Argument       | Default                  | What it does                                                                              |
| -------------- | ------------------------ | ----------------------------------------------------------------------------------------- |
| `source`       | `video`                  | Input node: `video` or `camera`                                                           |
| `video_path`   | `test_video.mp4`         | Video file for `source:=video`                                                            |
| `loop`         | `false`                  | Restart the video at the end instead of stopping, for `source:=video`                     |
| `video_fps`    | `30.0`                   | Publish rate for `source:=video`. Raise it to find the pipeline's ceiling                 |
| `camera_index` | `2`                      | V4L2 index for `source:=camera`. The GMSL dashcam enumerates at `/dev/video2`              |
| `camera_fps`   | `30.0`                   | Capture rate for `source:=camera` (must be written as a float)                             |
| `source_delay` | `10.0`                   | Seconds before the source starts, so segmentation has loaded its model                    |
| `scene`        | `trail`                  | Segmentation prompt set from `perception.scenes` in the config: `trail`, `snow`, `gravel`, `sandy` |
| `config`       | `config/perception.yaml` | Pipeline config YAML                                                                      |
| `speed_mps`    | `0.0`                    | Fixed speed given to the planner (must be written as a float)                             |
| `backend`      | `pytorch`                | Segmentation inference: `pytorch`, or `tensorrt` on an NVIDIA GPU (see [TensorRT](#tensorrt)) |
| `visualize`    | `true`                   | Open the dashboard window                                                                 |

The dashboard needs a display. Over SSH without X forwarding, launch with `visualize:=false`, or run `export DISPLAY=:0` first to show it on the machine's own screen. Without a display, the visualizer exits with an error and the other nodes keep running.

Run `ros2 launch perception_suite pipeline.launch.py --show-args` to list the arguments. An invalid `source` or `scene` stops the launch before any node starts.

`loop` and `video_fps` exist for benchmarking and both default to the previous behaviour.

A short clip stops feeding the pipeline before a measurement window opens: the ORAD-3D sequences are 243-449 frames, which is 8-15 seconds at 30 FPS, so a benchmark run against one measured a source that had already finished.
`loop:=true` rewinds instead.

`video_fps` matters because the rate used to be fixed at ~30, which is right for standing in for the camera but hides what the pipeline can do.
Any model fast enough for 30 FPS measures exactly 30, so `yoloe-26n` and `yoloe-26s` looked identical; uncapped they are 35.7 and 28.9 FPS.
Set it high (`video_fps:=120.0`) to measure the pipeline rather than the source.

### TensorRT

`backend:=tensorrt` runs segmentation through a TensorRT engine instead of PyTorch. On an RTX A2000 Laptop GPU with `yoloe-26n-seg` at input size 640, inference took 9.7 ms per frame against 31.8 ms with PyTorch, and the masks matched closely (mean IoU 0.99).

It needs an NVIDIA GPU that PyTorch can use and the `tensorrt` extra:

```bash
uv sync --python /usr/bin/python3 --extra tensorrt
```

On a Jetson, TensorRT itself comes with JetPack (TensorRT 8.6 on JetPack 6.0), not from the extra, and reaches the environment through `--system-site-packages` (see [Jetson](#jetson)). The extra only adds `onnx`, which building an engine needs. Check that JetPack's TensorRT is visible:

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
```

If that fails, install it with `sudo apt install python3-libnvinfer`.

A plain `uv sync` removes the extra again. If TensorRT is not available, the segmentation node exits with the reason rather than falling back to PyTorch, so a comparison never silently measures the wrong backend.

An engine has the scene's prompts and `perception.input_size` compiled in, and only runs on the GPU model and TensorRT version that built it. The first run with a new combination builds one, which takes several minutes. It is cached in the workspace root as `yoloe-26n-seg.<scene>.<hash>.engine` (for example `yoloe-26n-seg.snow.<hash>.engine`) and loads in seconds after that. Frames sent while it builds are dropped, so let the first run finish building, then launch again, or give it time with `source_delay:=600.0`.

To compare the two backends, launch each in turn on the same video and read the dashboard's FPS and latency. Without the dashboard, run `ros2 topic hz /road_mask` for the segmentation rate. The pipeline cannot run faster than the 30 FPS video, so once TensorRT outruns the source, latency shows the difference better than FPS.

On a Jetson, building an engine takes longer than on a desktop GPU, so give the first run a long `source_delay` or launch twice. The Jetson path has not been tested yet.

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
| `gravel` | "gravel road", "gravel path", "dirt road"            | Median best score 0.20 on `demo_route.mp4`; use `sandy` there          |
| `sandy`  | "sandy gravel path", "sandy path"                    | Light sand and gravel roads. Median best score 0.41 on `demo_route.mp4`, against 0.27 for `trail` |

To add a scene, add a named prompt list under `perception.scenes` in the config. No code changes are needed.

`demo_route.mp4` shows the car's hood, which the planner would otherwise read as part of the scene. Its config, `config/demo_route.yaml`, is `perception.yaml` plus a hood polygon under `beamng.camera.ego_mask`. Keep the two files' other values in step:

```bash
ros2 launch perception_suite pipeline.launch.py video_path:=demo_route.mp4 \
    config:=install/perception_suite/share/perception_suite/config/demo_route.yaml scene:=sandy
```

### Camera

The planner fits its path on the ground in metres, using the camera's field of view, height and pitch. The values under `beamng.camera` in the config (`fov_h: 90`, `height_m: 1.5`, `pitch_deg: 0`) are guesses for `test_video.mp4`. The path's shape in the image is still correct, but its distances in metres are only as accurate as these values. Measure them for your camera.

---

## Live camera (Jetson, GMSL)

`source:=camera` reads a V4L2 device instead of a file:

```bash
ros2 launch perception_suite pipeline.launch.py source:=camera
ros2 launch perception_suite pipeline.launch.py source:=camera camera_index:=0 camera_fps:=20.0
```

The node asks the sensor for 1920x1080, then downscales to `width` (1280) keeping the aspect, exactly as `1_video_node` does, so stages 2-6 cannot tell the two apart. If the driver returns a different mode it logs a warning rather than failing, because the aspect it lands on is what the camera model is built from.

Note that `beamng.camera.sensor` in the config describes the **published** frame, not the sensor's native resolution. With the defaults that is 1280x720.

### Bringing the camera up

The dashcam needs the Neousys deserializers initialised before any `/dev/video*` appears. On this Jetson that is a systemd unit:

```bash
systemctl status nru-camera     # should be "active"
ls /dev/video*                  # video0..video3 once it is up
```

If `/dev/video*` is missing, nothing in userspace can help; check `dmesg | grep nru2mp` first.

### Device tree: UYVY bit depth

The Neousys device tree declares the cameras as `csi_pixel_bit_depth = "8"`, counting bits per *component*. NVIDIA's `tegra-camera.ko` builds a format name of `<mode_type>_<pixel_phase><csi_pixel_bit_depth>` and only ships `yuv_uyvy16`, `yuv_vyuy16`, `yuv_yuyv16` and `yuv_yvyu16`, counting bits per *pixel*. So the stock module rejects the stock device tree:

```
nru2mp 2-0070: Unsupported pixel format
nru2mp 2-0070: Failed to read mode0 image props
nru2mp: probe of 2-0070 failed with error -22
```

A device tree with `16` in those eight properties probes cleanly against the stock module (`Detected NRU2MP sensor`). This Jetson boots such a tree through an `FDT` line in `/boot/extlinux/extlinux.conf`; the original `primary` entry is untouched and still selectable.

This only matters on a board whose vendor BSP has been overwritten, for example by an `apt` upgrade, which replaces vendor-patched modules at package-owned paths without dpkg noticing.

### Running the camera in a container

The Tegra capture path needs far more than `/dev/video*`: `capture-vi-channel*`, `capture-isp-channel*`, `nvhost-ctrl-vi*`, `nvmap` and `tegra_camera_ctrl`. Passing only `--device /dev/video2` opens the device but every read times out with `select() timeout`. Mount the whole node tree:

```bash
docker run --rm --runtime nvidia --network host --ipc=host --privileged \
  -v /dev:/dev ...
```

---

## Extending

**A new input** (BeamNG). `1_camera_node` was added this way and is the worked example:

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
│       │   ├── 1_camera_node.py         # Reads a V4L2 camera → /raw_frames
│       │   ├── 2_preprocess_node.py     # Resize + CLAHE → /preprocessed_frames
│       │   ├── 3_segmentation_node.py   # YOLOE-26 → /road_mask
│       │   ├── 4_postprocess_node.py    # EMA + morphology → /stabilized_mask
│       │   ├── 5_planner_node.py        # Gate + centreline → /path
│       │   └── 6_visualizer_node.py     # orfd AutonomyDashboard window
│       ├── config/perception.yaml       # Pipeline tuning, camera, scene prompts
│       ├── config/perception_trt.yaml   # The same, against a prebuilt TensorRT engine
│       ├── config/demo_route.yaml       # The same, plus demo_route.mp4's hood mask
│       ├── launch/pipeline.launch.py    # Starts the whole pipeline
│       ├── setup.py
│       └── package.xml
├── pyproject.toml                       # uv project; pulls in ../orfd-lane-detection
└── README.md
```
