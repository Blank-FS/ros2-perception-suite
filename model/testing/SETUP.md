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

The six fine-tuned weights come with the clone, in `model/weights/`. To test other weights,
copy their `.pt` files there too.

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

## Reproducing the R2 results

To reproduce the results in the R2 requirement (Segmentation Performance), run:

```bash
uv run reproduce_r2.py
```

It downloads anything missing from step 4 plus one more dataset, runs the four evaluations
behind R2 (about 25 minutes on a laptop GPU), and prints R2's tables, each with the report it
comes from:

| R2 | Report |
|---|---|
| Table 1, 26n and 26s | `runs/final_demo-vokml-v6/report.html` |
| Table 1, VOS models; Table 2; Current Status and Next Steps numbers | `runs/week-8-vos-v1/report.html` |
| Note under Table 1 (gap per segment) | `runs/week-8-vos-v1/<model>-test-cropnone-input640/segments.csv` |
| Table 3 | `runs/gate-demo_route_hatci/report.html` |

In R2, 26n and 26s are `yolo26n-demo-route` and `yolo26s-demo-route`, and 26n_vos and 26s_vos
are `rf26n-sem-640` and `rf26s-sem-640`. Numbers should match R2 to about the fourth decimal
place. To print the tables again without re-running, use `uv run reproduce_r2.py --tables-only`.
