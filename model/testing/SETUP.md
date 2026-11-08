# Testing setup

Everything here runs from this folder, `model/testing/`. You don't need ROS,
`colcon` or `orfd-lane-detection`.

## 1. Install

You need `git` and [`uv`](https://docs.astral.sh/uv/getting-started/installation/). A CUDA
GPU is optional but much faster. `uv sync` installs about 8 GB, mostly PyTorch.

```bash
git clone git@github.com:HATCI-MDP/ros2-perception-suite.git
cd ros2-perception-suite/model/testing
uv sync
```

## 2. Add the weights

The weights aren't in git. Copy the `.pt` files into `model/weights/`.

## 3. Add the Roboflow key

This step is only needed to download datasets.

```bash
cp .env.example .env
```

Then set `ROBOFLOW_API_KEY` in `.env`. To find the key, open the workspace on
app.roboflow.com and go to **Settings** > **API Keys**.

## 4. Get the data

```bash
uv run download_dataset.py --workspace lucianas-workspace-pqyes --project week-8-vos --version 1
curl -L --create-dirs -o recordings/demo_route_hatci.mp4 \
  "https://www.dropbox.com/scl/fi/2brwgt0hmjszh4x0x8k0e/video.mp4?rlkey=8e3ircva7xzwhfdvcyihjdsgd&dl=1"
```

## 5. Check the setup

```bash
uv run check_setup.py
```

Fix anything marked `FAIL`. A `WARN` only affects the step it names.

## 6. Run the tests

```bash
# Segmentation accuracy on the 900 test images
uv run evaluate.py --data datasets/week-8-vos-v1/data.yaml --split test --static-crop none

# Gate rejections and mask stability on the drive
uv run gate_sweep.py --source recordings/demo_route_hatci.mp4
```

Results go to `runs/`; open `report.html` in each run's folder. [README.md](README.md) explains
the metrics and options.
