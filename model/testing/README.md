# Model evaluation

Scores the semantic segmentation weights in `model/weights/` against ground truth
labels from Roboflow (`evaluate.py`), and measures how often the pipeline's perception gate
would reject frames on a recorded drive (`gate_sweep.py`). All commands run from the
workspace root.

## Weights

The weights are not in git (`*.pt` is gitignored), so copy them into `model/weights/`.
Both scripts use every `.pt` file there by default, or the ones passed with `--weights`.
They're YOLO26 semantic segmentation models trained in Roboflow at 1024:

| Weights | Trained on |
|---|---|
| `yolo26n-demo-route.pt`, `yolo26s-demo-route.pt` | `final_demo-vokml` v4/v6: 236 frames of the demo route |
| `week8yolo26n.pt`, `week8yolo26s.pt` | `week-8-vos` v1: 6,186 frames of the same drive |

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
them combined (`all`). To score only some splits, pass them, e.g. `--split test` or
`--split val test`. The models are compared with the labels pixel by pixel at the labels'
own resolution: the prediction is scaled up to the image rather than the labels down to the
network. (Ultralytics' validator scales the labels down, and that alone costs about 2.5
points of IoU, so its numbers come out lower than these.)

For example, to compare all four models on the 900 test images of `week-8-vos` v1, both as
the new models were trained (see the next section) and on the whole picture:

```bash
uv run model/testing/download_dataset.py --workspace lucianas-workspace-pqyes --project week-8-vos --version 1
uv run model/testing/evaluate.py --data model/testing/datasets/week-8-vos-v1/data.yaml --split test --static-crop 0 100 0 66
uv run model/testing/evaluate.py --data model/testing/datasets/week-8-vos-v1/data.yaml --split test --static-crop none
```

### Preprocessing

Images first get the preprocessing the models were trained with in Roboflow: a static crop,
by default to the top 66% of the frame, which removes the car's hood, then a fit into the
model's square input (1024) with black edges. Change the crop with
`--static-crop X_MIN X_MAX Y_MIN Y_MAX` (in percent), or turn it off with
`--static-crop none`.

A Roboflow version exported with its preprocessing already applied, such as `week-8-vos`
v1 (top 66% of the frame, fitted into 1052×1052 between black bands), is recognised by its
black edges. Those are left out of the scores. Normally such a version wants
`--static-crop none`, and the script warns if it's cropped again.

The week8 models are the exception: their training recipe cropped the top 66% and fitted it
into 1024 *again*, on top of v1's own crop. That second crop cuts off the bottom eighth of
the picture, the road nearest the car, so those models never saw it and call it background.
`--static-crop 0 100 0 66` reproduces what they were trained on, and scores only what they
can see; `--static-crop none` scores the whole picture, which is what they'd get on the
vehicle unless the pipeline crops the same way.

To score the models the way the pipeline runs them today, use the whole frame with the hood
masked after inference:

```bash
uv run model/testing/evaluate.py --data ... --static-crop none \
    --ego-mask src/perception_suite/config/demo_route.yaml
```

### Results

The results go to `model/testing/runs/<name>/`:

- `summary.csv` and `report.html`, with one row per model and split. The report is a single
  page you can filter by split and model and sort by any column. To open it from WSL, run
  `explorer.exe "$(wslpath -w model/testing/runs/<name>/report.html)"`. Add `--report-only`
  to rebuild just the page from the existing `summary.csv` in under a second, without
  running the models.
- `<weights>-<split>-<crop>/images.csv`, with every image's road IoU, precision, recall and
  HD95, to find the frames behind a number.
- `<weights>-<split>-<crop>/worst_*.jpg`, the 8 images with the lowest road IoU: green is
  road the model found, red is road it invented, blue is road it missed, and unscored areas
  are dimmed.

Runs with other models or preprocessing add to the same `summary.csv` and report, so they
can be compared side by side (filter by `preprocessing`); rerunning one replaces only its own
rows.

The columns for the road class, `traversable_road`:

| Column | Meaning |
|---|---|
| `IoU` | The headline number. Of all pixels that either the labels or the model call road, the fraction where both agree. |
| `precision` | The fraction of pixels the model calls road that are labeled road. Low precision means it paints road onto grass, verges or driveways. |
| `recall` | The fraction of labeled road the model finds. Low recall means it misses road, which shrinks the corridor the planner has to work with. |
| `dice` | The F1 of precision and recall (the Dice score). It always ranks models the same way as IoU; it's there because papers often report it. |
| `IoU_near`, `IoU_mid`, `IoU_far` | Road IoU by distance along the road: each image's labeled road, from its far end down to the bottom of the picture, is split into thirds of its height (road predicted above the labeled road's far end counts in `far`). The near band is the road right in front of the car, which the planner's corridor starts from; a model can score well overall and still miss it, as the week8 models do on the full picture. The bands come from the labels only, so every model is scored on the same ones. Each image's near-band IoU is also in `images.csv`. |
| `img_IoU_mean`, `img_IoU_p5`, `img_IoU_min` | The spread of each image's own road IoU: the average, the 5th percentile (the worst 1 in 20 images) and the worst image. The pooled columns above can hide a model that does well overall but fails on a handful of frames. |
| `img_IoU<0.9` | The share of images below 0.9 road IoU. Lower is better. |
| `HD95_px`, `HD95_px_p95` | How far the predicted road edge strays from the labeled one, in pixels at the labels' resolution: the 95th percentile of edge distances in each image, averaged over images, and its 95th percentile across images. Lower is better. IoU barely notices a thin wrong strip along the edge, but the planner's centerline depends on the edges. Edges where the labels stop (image borders, padding, the hood) don't count. |

`IoU`, `precision`, `recall` and `dice` pool every scored pixel of the split. `mIoU` averages
the road IoU with `IoU/background`, which is high for any model because most of the frame
isn't road, so it flatters the result. `inference_ms` is the network's time per image.

Bounding-box metrics (mAP, GIoU, DIoU, CIoU) and panoptic quality aren't included. The
road's bounding box usually spans the whole frame width, so boxes from good and bad masks
look alike, and semantic models give no per-detection confidence for mAP. Panoptic quality
is for scenes with separate object instances; with one road region per frame it reduces to
the per-image IoU, counting images below 0.5 as zero.

### Classes

The dataset's and the model's class IDs must mean the same thing, because the masks' pixel
values are compared directly; the script stops if they don't. Names are compared after
renaming: the demo-route Roboflow export names its road class `Final-Demo` (the project
name), and the models trained on it carry that name, so `Final-Demo` is renamed to
`traversable_road` by default (`--rename none` keeps it). Their extra class 2,
`traversable_road`, never had labels. Classes without labels are left out of the results.

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
- `stability_mean`, `stability_p5` and `unstable_frames`: how much the road mask changes from
  one frame to the next, as the IoU of each frame's mask with the previous one (the
  stabilizer's own stability score), with the mean, the 5th percentile and the share of frames
  below 0.9. Flicker in the mask turns into jitter in the path. The road really does move as
  the car drives, so compare models on the same video rather than reading the number on its
  own; `--stride` makes the gap between compared frames longer.
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
