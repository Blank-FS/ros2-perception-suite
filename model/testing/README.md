# Model evaluation

Scores the semantic segmentation weights in `model/weights/` against ground truth
labels from Roboflow (`evaluate.py`), and measures how often the pipeline's perception gate
would reject frames on a recorded drive (`gate_sweep.py`). All commands run from the
workspace root.

## Weights

The weights are not in git (`*.pt` is gitignored), so copy them into `model/weights/`.
Both scripts use every `.pt` file there by default, or the ones passed with `--weights`. The
current ones are `yolo26n-demo-route.pt` and `yolo26s-demo-route.pt`, YOLO26 semantic
segmentation models trained in Roboflow on the demo route.

## 1. Get the dataset

Export the Roboflow dataset version as **Semantic Segmentation Masks**
(`png-mask-semantic`). Either let the script download it, after putting your API key in
`model/testing/.env`:

```bash
cp model/testing/.env.example model/testing/.env   # fill in ROBOFLOW_API_KEY, optionally the dataset
uv run model/testing/download_dataset.py
uv run model/testing/download_dataset.py --workspace <workspace> --project <project> --version <n>
```

Use the first form if `.env` names the dataset, the second if it doesn't. Flags override
`.env`, and variables already set in the shell take precedence over it.

Or download the zip from the Roboflow UI and pass it in:

```bash
uv run model/testing/download_dataset.py --zip ~/Downloads/export.zip --name demo-route-v<n>
```

Either way, the images and masks land in `model/testing/datasets/<name>/`, in the layout
Ultralytics expects, with a `data.yaml` beside them.

To find the API key, open the workspace on app.roboflow.com, go to **Settings** (bottom of
the left sidebar) > **API Keys**, and copy the **Private API Key**. You can also go
straight to `https://app.roboflow.com/<workspace>/settings/api`. The key belongs to the
workspace, not to you, so anyone with the key can access that workspace's data.

## 2. Evaluate

```bash
uv run model/testing/evaluate.py --data model/testing/datasets/<name>/data.yaml
```

This evaluates every model on each split (train, val and test) on its own, then on all of
them combined (`all`). Each model runs at the image size it was trained at. To score only
some splits, pass them, e.g. `--split all` or `--split val test`.

The results go to `model/testing/runs/<name>/`:

- `summary.csv` and `report.html`, with one row per model and split. The report is a single
  page you can filter by split and model and sort by any column. To open it from WSL, run
  `explorer.exe "$(wslpath -w model/testing/runs/<name>/report.html)"`. Add `--report-only`
  to rebuild just the page from the existing `summary.csv` in under a second, without
  running the models, e.g. while changing the page.
- `<weights>-<split>/`, with Ultralytics' confusion matrix and sample predictions.

The columns, for the road class:

- `IoU/traversable_road`, the headline number: of all pixels that either the labels or the
  model call road, the fraction where both agree.
- `precision/traversable_road`: the fraction of pixels the model calls road that are
  labeled road. Low precision means it paints road onto grass or verges.
- `recall/traversable_road`: the fraction of labeled road the model finds. Low recall means
  it misses road, which shrinks the corridor the planner has to work with.
- `mIoU` averages the road IoU with `IoU/background`, which is high for any model because
  most of the frame isn't road, so it flatters the result.

All of them are computed over all the split's pixels together, not averaged per image.

Before scoring, the script applies the preprocessing the models were trained with in
Roboflow: a static crop to the top 66% of the frame, which removes the car's hood, then a
fit into 1024×1024 with black edges. The black edges don't count toward the scores. The
preprocessed copies go to `datasets/<name>/prepared/`. If the training preprocessing
changes, pass `--static-crop X_MIN X_MAX Y_MIN Y_MAX` (in percent) and `--resize <side>`,
or `none` to skip either step.

To score the models the way the pipeline runs them today, use the whole frame with the
hood masked after inference:

```bash
uv run model/testing/evaluate.py --data ... --static-crop none --resize none \
    --ego-mask src/perception_suite/config/demo_route.yaml
```

In the Roboflow export the road is class 1, but Roboflow names it `Final-Demo` (the project
name). Class 2, `traversable_road`, has no labeled pixels. The results drop classes without
labels and rename `Final-Demo` to `traversable_road`; pass `--rename none` to keep Roboflow's
names.

The dataset's classes must match the model's (currently `background`, `Final-Demo`,
`traversable_road`), because the class ids in the masks are compared directly. The script
stops if they don't match.

Models score higher on images they were trained on, so `all` and `train` are optimistic.
The `val` and `test` rows show how the models handle images they haven't seen. If the
download warns about augmented copies, generate a version without augmentation; otherwise
`all` counts some labeled images several times.

`datasets/`, `runs/` and `recordings/` are gitignored, as is `.env`.

## 3. Gate rejections on a drive

`gate_sweep.py` runs the models over a drive's video and passes every frame through the
pipeline's own perception gate (`PerceptionGate`). It uses the planner ROI, hood mask and
limits from `src/perception_suite/config/hatci_vehicle.yaml`, the vehicle the drives are
recorded on. For each model it reports, at each threshold:

- `reject_rate`: the fraction of frames the gate rejects.
- `longest_reject_streak` and `longest_reject_s`: the longest run of rejected frames in a
  row, in frames and seconds, with `streak_starts_s` saying where it starts.
- `rejects/*`: how many rejections each gate check caused.
- `min_road_pixels` and `min_road_area`: the closest any frame came to the gate's limits
  (1,579 road pixels and 5% of the planning region in `hatci_vehicle.yaml`), with
  `tightest_at_s` saying when. They show how much margin a model has even when nothing is
  rejected.

The threshold `t` is the road probability a pixel needs to count as road. Higher values keep
only road the model is sure of, so the gate rejects more often. `argmax` is the mask the
pipeline uses today. Frames get the same crop and fit as the evaluation.

Drive videos go in `recordings/`, which is gitignored. The demo route is the camera stream
of `bag_20261006_125333`, stitched into a video:

```bash
curl -L -o model/testing/recordings/demo_route_hatci.mp4 \
  "https://www.dropbox.com/scl/fi/2brwgt0hmjszh4x0x8k0e/video.mp4?rlkey=8e3ircva7xzwhfdvcyihjdsgd&dl=1"
uv run model/testing/gate_sweep.py --source model/testing/recordings/demo_route_hatci.mp4
```

It runs at about 30 frames/s on the RTX A2000 laptop, so a drive takes about as long as the
video. Each frame goes through both models and 20 gate checks (2 models × 10 masks), where
the car does one of each. The results go to `model/testing/runs/gate-<video>/`: `summary.csv` and
`report.html` with one row per model and threshold, and `frames.csv` with every frame's
decision, so you can find the frames in a long rejection streak. Use `--thresholds` to change
the sweep, `--stride N` to use every Nth frame, `--max-frames N` for a quick check, and
`--report-only` to rebuild the page.
