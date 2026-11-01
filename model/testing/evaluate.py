"""Evaluate the semantic segmentation weights against a ground truth dataset.

Runs each model in model/weights (or the ones given) over one dataset from
download_dataset.py and compares its prediction with the labels, pixel by
pixel, at the labels' own resolution: the model's output is scaled up to the
image, never the labels down to the network, which alone would cost about 2.5
points of IoU.

    python model/testing/evaluate.py --data model/testing/datasets/final_demo-vokml-v6/data.yaml
    python model/testing/evaluate.py --data model/testing/datasets/week-8-vos-v1/data.yaml --split test \
        --static-crop none

By default every split is scored on its own and then all of them together
("all"). Images the models were trained on score higher than new ones, so the
test and val rows are the honest estimate and "all" is an upper bound.

Images first get the preprocessing the models were trained with in Roboflow:
a static crop (default: the top 66% of the frame, which cuts off the car's
hood) and a fit into the model's square input with black edges. A version
Roboflow exported with its preprocessing already applied is recognised by its
black edges, which are left out of the scores; pass --static-crop none for it.
--ego-mask leaves out the polygon from beamng.camera.ego_mask in a pipeline
config, as the pipeline does after inference; with --static-crop none it
scores the whole frame the way the pipeline runs today.

Per class, besides background:
  IoU, precision, recall, dice  pooled over every scored pixel of the split
                                (dice is the F1 of precision and recall)
  img_IoU_mean/_p5/_min         the spread of each image's own IoU, for the
                                frames where a model does badly
  img_IoU<0.9                   the share of images below 0.9 IoU
  HD95_px, HD95_px_p95          how far the predicted edge strays from the
                                labelled one, in pixels: the 95th percentile
                                of edge distances per image (mean over images,
                                and its 95th percentile)
plus mIoU over all classes, pixel accuracy and the network's time per image.

Results go to model/testing/runs/<dataset>/: summary.csv and report.html with
one row per model and split, and <weights>-<split>/ with images.csv (every
image's numbers) and overlays of the worst images (green: road found, red:
road invented, blue: road missed). --report-only rebuilds report.html from the
existing summary.csv without evaluating, for working on the page.
"""

import argparse
import csv
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

TESTING_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = TESTING_DIR.parent / "weights"
RUNS_DIR = TESTING_DIR / "runs"
SPLITS = ("train", "val", "test")
IGNORE = 255  # Ultralytics leaves mask pixels with this value out of every metric
# Roboflow's export names pixel value 1 after the project (Final-Demo), though it
# holds the traversable_road labels, and leaves value 2 (traversable_road) empty.
# The models were trained with the same names, so the results just relabel it.
DEFAULT_RENAME = ["Final-Demo=traversable_road"]
REPORT_FILTERS = ("split", "weights", "preprocessing")  # columns offered as filter chips in report.html
LOWER_IS_BETTER = {"inference_ms"}
# The Roboflow preprocessing the demo route weights were trained with.
DEFAULT_STATIC_CROP = (0, 100, 0, 66)  # x_min, x_max, y_min, y_max, in percent
DEFAULT_RESIZE = 1024  # "Fit (black edges) in" a square of this side


def split_dirs(data, data_file):
    """Return {split: absolute image directory} for the splits the dataset really has.

    download_dataset.py points a missing train or val key at another split, since
    Ultralytics requires both, so keys that only alias another split are dropped.
    """
    root = Path(data.get("path") or data_file.parent)
    dirs = {}
    for split in SPLITS:
        if data.get(split) == f"images/{split}" and (root / data[split]).is_dir():
            dirs[split] = str((root / data[split]).resolve())
    return dirs


def ego_polygon(config_file):
    """Return the enabled ego mask polygon, in normalised (x, y), from a pipeline config, or None."""
    ego = yaml.safe_load(config_file.read_text()).get("beamng", {}).get("camera", {}).get("ego_mask", {})
    return np.array(ego["polygon"], dtype=np.float64) if ego.get("enabled") and ego.get("polygon") else None


def fit_black_edges(image, size, interpolation, fill):
    """Scale image to fit a size x size square, keeping its aspect, and pad the rest with fill, centred."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    return cv2.copyMakeBorder(
        image, top, size - new_h - top, left, size - new_w - left, cv2.BORDER_CONSTANT, value=fill
    )


def black_edges(image, level=2):
    """The picture area of an image letterboxed with pure black bands, as a boolean mask, or None.

    Roboflow's "Fit (black edges)" centres the picture, so it pads two
    opposite sides about equally with black. A real frame can have a black
    band along one edge (under the hood), but not matching ones on both.
    """
    rows = np.flatnonzero(image.max(axis=(1, 2)) > level)
    cols = np.flatnonzero(image.max(axis=(0, 2)) > level)
    h, w = image.shape[:2]
    if len(rows) == 0:
        return None
    top, bottom, left, right = rows[0], h - 1 - rows[-1], cols[0], w - 1 - cols[-1]
    # Nearly equal, not exactly: the export's rounding and JPEG noise at the
    # border move the edges by a few pixels (288 and 299 rows in week-8-vos).
    letterboxed = min(top, bottom) > 0 and abs(top - bottom) <= 0.02 * h
    pillarboxed = min(left, right) > 0 and abs(left - right) <= 0.02 * w
    if not (letterboxed or pillarboxed):
        return None
    inside = np.zeros((h, w), bool)
    inside[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = True
    return inside


def load_sample(image_file, mask_file, crop, polygon):
    """An image and its mask, cropped, with the pixels that must not be scored set to IGNORE.

    Those are the hood (polygon, normalised to the full frame) and the black
    padding of an export that Roboflow already preprocessed.
    """
    image = cv2.imread(str(image_file))
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        return None, None
    h, w = mask.shape[:2]
    edges = black_edges(image)
    if edges is not None:
        if crop and not load_sample.warned:
            # Allowed, since a training recipe can crop a preprocessed version
            # again (week-8's did), but usually a mistake.
            print(
                f"Warning: {image_file.name} already has black edges, so this version was exported with its "
                "preprocessing applied, and --static-crop crops it a second time. Pass --static-crop none "
                "unless the model was trained that way."
            )
            load_sample.warned = True
        mask[~edges] = IGNORE  # the export's padding is labelled background; leave it out
    if polygon is not None:
        cv2.fillPoly(mask, [np.round(polygon * [w, h]).astype(np.int32)], IGNORE)
    if crop:
        x0, x1, y0, y1 = (round(c / 100 * n) for c, n in zip(crop, (w, w, h, h)))
        image, mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    return image, mask


load_sample.warned = False


class Segmenter:
    """Runs a semantic model the way it was trained (fit into a black-edged square) and returns its class
    map at the input image's own resolution.

    The logits, not the class map, are scaled back up, so the prediction is
    compared against the labels as drawn: resampling the labels to the network
    size instead (what Ultralytics' validator does) costs about 2.5 points of
    IoU on its own.
    """

    def __init__(self, weights, size, device):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.yolo = YOLO(weights)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.yolo.model.to(self.device).eval()
        self.size = size or self.yolo.ckpt.get("train_args", {}).get("imgsz", DEFAULT_RESIZE)
        self.seconds = 0.0  # time spent in the network, for inference_ms

    def __call__(self, image):
        torch, F = self.torch, self.torch.nn.functional
        h, w = image.shape[:2]
        square = fit_black_edges(image, self.size, cv2.INTER_AREA, (0, 0, 0))
        scale = self.size / max(h, w)
        new_w, new_h = round(w * scale), round(h * scale)
        top, left = (self.size - new_h) // 2, (self.size - new_w) // 2
        tensor = torch.from_numpy(np.ascontiguousarray(square[..., ::-1].transpose(2, 0, 1)))  # BGR -> RGB
        tensor = tensor.to(self.device).float().div(255)[None]
        with torch.no_grad():
            if self.device != "cpu":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = self.net(tensor)
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            if self.device != "cpu":
                torch.cuda.synchronize()
            self.seconds += time.perf_counter() - started
            logits = F.interpolate(logits.float(), (self.size, self.size), mode="bilinear")
            logits = logits[:, :, top : top + new_h, left : left + new_w]
            logits = F.interpolate(logits, (h, w), mode="bilinear")
            return logits[0].argmax(0).to(torch.uint8).cpu().numpy()


def boundary(mask, scorable):
    """The edge pixels of mask, leaving out those on the image border or next to unscored pixels,
    which are where the labels stop rather than where the road does."""
    kernel = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    edge = m.astype(bool) & ~cv2.erode(m, kernel, borderValue=1).astype(bool)
    inner = cv2.erode(scorable.astype(np.uint8), kernel, borderValue=0).astype(bool)
    return edge & inner


def hd95(truth, pred, scorable):
    """95th-percentile Hausdorff distance between the two masks' edges, in pixels, or None if one has none."""
    a, b = boundary(truth, scorable), boundary(pred, scorable)
    if not a.any() or not b.any():
        return None
    to_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    to_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(np.percentile(np.concatenate([to_b[a], to_a[b]]), 95))


class Scores:
    """Accumulates one model's results over a split: a pixel confusion matrix and per-image numbers."""

    def __init__(self, classes, foreground):
        self.classes, self.foreground = classes, foreground
        self.matrix = np.zeros((classes, classes), np.int64)  # rows: labels, columns: predictions
        # Per class and distance band (far, mid, near): pooled [tp, fp, fn].
        self.bands = {c: np.zeros((len(BANDS), 3), np.int64) for c in foreground}
        self.images = []  # per image: {"image": name, f"{metric}/{class id}": value, ...}
        self.seconds, self.count = 0.0, 0

    def add(self, name, truth, pred):
        scorable = truth != IGNORE
        t, p = truth[scorable].astype(np.int64), np.minimum(pred[scorable], self.classes - 1).astype(np.int64)
        self.matrix += np.bincount(t * self.classes + p, minlength=self.classes**2).reshape(self.classes, -1)
        entry = {"image": name}
        for c in self.foreground:
            g, q = (truth == c) & scorable, (pred == c) & scorable
            tp, fp, fn = int((g & q).sum()), int((q & ~g).sum()), int((g & ~q).sum())
            entry[f"labelled_px/{c}"], entry[f"predicted_px/{c}"] = tp + fn, tp + fp
            entry[f"IoU/{c}"] = tp / (tp + fp + fn) if tp + fp + fn else ""  # no road in either: no score
            entry[f"precision/{c}"] = tp / (tp + fp) if tp + fp else ""
            entry[f"recall/{c}"] = tp / (tp + fn) if tp + fn else ""
            distance = hd95(g, q, scorable)
            entry[f"HD95_px/{c}"] = round(distance, 2) if distance is not None else ""
            for band, rows in enumerate(band_rows(scorable, g)):
                gb, qb = g[rows], q[rows]
                counts = ((gb & qb).sum(), (qb & ~gb).sum(), (gb & ~qb).sum())
                self.bands[c][band] += counts
                if BANDS[band] == "near":
                    hits, union = counts[0], sum(counts)
                    entry[f"IoU_near/{c}"] = hits / union if union else ""
        self.images.append(entry)
        self.count += 1

    def merge(self, other):
        self.matrix += other.matrix
        for c in self.bands:
            self.bands[c] += other.bands[c]
        self.images += other.images
        self.seconds += other.seconds
        self.count += other.count

    def row(self, names):
        """Summary columns: pooled pixel metrics, the per-image IoU distribution and the edge error."""
        m = self.matrix
        labelled, predicted, hits = m.sum(1), m.sum(0), np.diag(m)
        union = labelled + predicted - hits
        present = labelled > 0
        row = {
            "mIoU": float((hits[present] / union[present]).mean()) if present.any() else "",
            "pixel_acc": float(hits.sum() / max(m.sum(), 1)),
        }
        for c, name in names.items():
            if c >= self.classes:
                continue
            row[f"IoU/{name}"] = float(hits[c] / union[c]) if labelled[c] else ""
            if c not in self.foreground:
                continue
            row[f"precision/{name}"] = float(hits[c] / predicted[c]) if labelled[c] and predicted[c] else ""
            row[f"recall/{name}"] = float(hits[c] / labelled[c]) if labelled[c] else ""
            p, r = row[f"precision/{name}"], row[f"recall/{name}"]
            row[f"dice/{name}"] = 2 * p * r / (p + r) if p != "" and r != "" and p + r else ""
            for band, (tp, fp, fn) in zip(BANDS, self.bands[c]):
                row[f"IoU_{band}/{name}"] = float(tp / (tp + fp + fn)) if tp + fp + fn else ""
            ious = np.array([e[f"IoU/{c}"] for e in self.images if e[f"IoU/{c}"] != ""], float)
            row[f"img_IoU_mean/{name}"] = float(ious.mean()) if len(ious) else ""
            row[f"img_IoU_p5/{name}"] = float(np.percentile(ious, 5)) if len(ious) else ""
            row[f"img_IoU_min/{name}"] = float(ious.min()) if len(ious) else ""
            row[f"img_IoU<0.9/{name}"] = float((ious < 0.9).mean()) if len(ious) else ""
            distances = np.array([e[f"HD95_px/{c}"] for e in self.images if e[f"HD95_px/{c}"] != ""], float)
            row[f"HD95_px/{name}"] = float(distances.mean()) if len(distances) else ""
            row[f"HD95_px_p95/{name}"] = float(np.percentile(distances, 95)) if len(distances) else ""
        row["inference_ms"] = self.seconds / max(self.count, 1) * 1000
        return row

    def save_images(self, folder, names, samples, worst=8):
        """images.csv with every image's numbers, and overlays of the worst images by road IoU."""
        folder.mkdir(parents=True, exist_ok=True)
        rename = {f"{k}/{c}": f"{k}/{names[c]}" for c in self.foreground for k in IMAGE_METRICS}
        keys = list(self.images[0]) if self.images else ["image"]
        with open(folder / "images.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[rename.get(k, k) for k in keys])
            writer.writeheader()
            writer.writerows({rename.get(k, k): v for k, v in e.items()} for e in self.images)
        for old in folder.glob("worst_*.jpg"):
            old.unlink()
        c = self.foreground[0] if self.foreground else None
        scored = sorted((e for e in self.images if c is not None and e[f"IoU/{c}"] != ""), key=lambda e: e[f"IoU/{c}"])
        for rank, entry in enumerate(scored[:worst], 1):
            image, truth, pred = samples(entry["image"])
            cv2.imwrite(str(folder / f"worst_{rank}_IoU{entry[f'IoU/{c}']:.3f}_{entry['image']}.jpg"),
                        overlay(image, truth, pred, c))


IMAGE_METRICS = ("labelled_px", "predicted_px", "IoU", "precision", "recall", "HD95_px", "IoU_near")
# Distance bands along the road: the road nearest the car is at the bottom of
# the picture, and it is what the planner's corridor starts from.
BANDS = ("far", "mid", "near")


def band_rows(scorable, road):
    """The rows of each distance band, far to near.

    The labelled road's rows, from its far end down to the bottom of the
    scored area, split into thirds; rows above the road's far end join the far
    band, so road invented above the horizon counts there. The bands come from
    the labels alone, so every model is scored on the same ones. Without
    labelled road, the scored rows are split instead.
    """
    rows = np.flatnonzero(scorable.any(1))
    if not len(rows):
        return [rows] * len(BANDS)
    labelled = np.flatnonzero(road.any(1))
    along = rows[rows >= labelled[0]] if len(labelled) else rows
    bands = np.array_split(along, len(BANDS))
    bands[0] = np.concatenate([rows[rows < along[0]], bands[0]])  # everything above the road is far
    return bands


def overlay(image, truth, pred, c, width=1280):
    """Correct road in green, road the model invented in red, road it missed in blue; unscored dimmed."""
    scorable = truth != IGNORE
    g, q = (truth == c) & scorable, (pred == c) & scorable
    out = image.copy()
    for area, colour in ((g & q, (0, 200, 0)), (q & ~g, (0, 0, 255)), (g & ~q, (255, 80, 0))):
        out[area] = (0.45 * out[area] + 0.55 * np.array(colour)).astype(np.uint8)
    out[~scorable] = (out[~scorable] * 0.35).astype(np.uint8)
    rows, cols = np.flatnonzero(scorable.any(1)), np.flatnonzero(scorable.any(0))
    if len(rows):  # just the scored area, without an export's padding
        out = out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    scale = width / out.shape[1]
    return cv2.resize(out, (width, round(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)


# Per-class columns, in the order they appear.
CLASS_PREFIXES = (
    "IoU/", "precision/", "recall/", "dice/", "IoU_near/", "IoU_mid/", "IoU_far/",
    "img_IoU_mean/", "img_IoU_p5/", "img_IoU_min/", "img_IoU<0.9/",
    "HD95_px/", "HD95_px_p95/",
)


def class_mismatch(model_names, dataset_names, renames):
    """Why the model's and the dataset's class ids disagree, or "" if they agree.

    Ids are compared after the OLD=NEW renames, so a model trained on
    Roboflow's misnamed class still matches a dataset that names it properly,
    and the other way round. Either side may have classes the other lacks:
    they score as wrong or missed rather than stopping the run.
    """
    mapping = dict(rename.partition("=")[::2] for rename in renames)
    named = {i: mapping.get(n, n) for i, n in model_names.items()}
    wrong = {i: (named[i], n) for i, n in dataset_names.items() if i in named and named[i] != mapping.get(n, n)}
    if wrong:
        pairs = ", ".join(f"id {i} is {m!r} in the model but {d!r} in the dataset" for i, (m, d) in wrong.items())
        return f"classes do not match ({pairs})"
    extra = {i: model_names[i] for i in model_names if i not in dataset_names}
    if extra:
        print(f"Model classes {extra} have no labels in this dataset; they still count as wrong predictions.")
    unknown = {i: n for i, n in dataset_names.items() if i not in named}
    if unknown:
        print(f"Dataset classes {unknown} are not in the model; any labels they have count as missed.")
    return ""


def relabel_classes(rows, renames):
    """Drop the columns of classes no row has labels for, then apply OLD=NEW renames to the rest."""
    empty_classes = {c[4:] for c in rows[0] if c.startswith("IoU/") and all(r[c] == "" for r in rows)}
    empty = {f"{p}{name}" for p in CLASS_PREFIXES for name in empty_classes}
    mapping = {}
    for rename in renames:
        old, _, new = rename.partition("=")
        if f"IoU/{old}" not in rows[0]:
            continue  # the dataset already uses the new name, or never had the old one
        if f"IoU/{new}" in rows[0] and new not in empty_classes:
            print(f"Not renaming {old} to {new}: {new} already has its own labels")
            continue
        mapping.update({f"{p}{old}": f"{p}{new}" for p in CLASS_PREFIXES})
    return [{mapping.get(k, k): v for k, v in r.items() if k not in empty} for r in rows]


def order_columns(rows):
    """Put the foreground classes' IoU, precision and recall, the headline numbers, after the row's identity."""
    keys = list(rows[0])
    foreground = [k[4:] for k in keys if k.startswith("IoU/") and k != "IoU/background"]
    per_class = [f"{p}{name}" for name in foreground for p in CLASS_PREFIXES]
    order = ["weights", "split", "images", *per_class, "mIoU", "IoU/background", "pixel_acc", "inference_ms"]
    order = [k for k in order if k in keys]
    order += [k for k in keys if k not in order]
    return [{k: r[k] for k in order} for r in rows]


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f7f7f5; --surface: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --border: #e3e2dd;
    --accent: #2f6fdb; --accent-soft: #e4ecfb; --bar: #dbe7fb; --best: #1f7a3f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161615; --surface: #1f1f1d; --text: #ecebe6; --muted: #9a9993; --border: #34332f;
      --accent: #7aa7f5; --accent-soft: #23314a; --bar: #26354f; --best: #6fd18f;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 20px; }
  .filters { display: flex; flex-wrap: wrap; gap: 16px 28px; margin-bottom: 16px; }
  .group { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  .group-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
                 margin-right: 4px; }
  .chip { border: 1px solid var(--border); background: var(--surface); color: var(--text); border-radius: 999px;
          padding: 4px 12px; font: inherit; font-size: 13px; cursor: pointer; }
  .chip[aria-pressed="true"] { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  .table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { padding: 9px 12px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  th { font-weight: 600; font-size: 12px; color: var(--muted); user-select: none;
       position: sticky; top: 0; background: var(--surface); }
  th:hover { color: var(--text); }
  th.primary { color: var(--accent); }
  th .label { cursor: pointer; }
  .sort { display: inline-flex; gap: 2px; margin-left: 6px; vertical-align: middle; }
  .sort button { border: 1px solid var(--border); background: var(--surface); color: var(--muted);
                 border-radius: 4px; padding: 0 4px; font-size: 10px; line-height: 16px; cursor: pointer; }
  .sort button:hover { color: var(--text); border-color: var(--muted); }
  .sort button.on { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  td.num { text-align: right; position: relative; }
  th.num { text-align: right; }
  td.num .bar { position: absolute; inset: 4px auto 4px 0; background: var(--bar); border-radius: 3px; z-index: 0; }
  td.num span { position: relative; z-index: 1; }
  td.best span { color: var(--best); font-weight: 650; }
  .empty { padding: 24px; color: var(--muted); text-align: center; }
  .note { color: var(--muted); font-size: 12px; margin-top: 10px; }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="sub" id="count"></p>
  <div class="filters" id="filters"></div>
  <div class="table-wrap"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
  <p class="note">__NOTE__ Sort with the ▼ ▲ buttons, or click a column name to flip its order. Bars show
    scores from 0 to 1, and the best value among the visible rows is in green. Scores on train and all
    include images the models were trained on.</p>
</main>
<script>
const COLUMNS = __COLUMNS__;
const ROWS = __ROWS__;
const FILTERS = __FILTERS__;
const LOWER_IS_BETTER = new Set(__LOWER__);
const PRIMARY = __PRIMARY__;
const IDENTITY = new Set(__IDENTITY__);  // numeric columns that describe a row rather than score it

const numeric = c => ROWS.every(r => r[c] === "" || typeof r[c] === "number");
const isNum = Object.fromEntries(COLUMNS.map(c => [c, numeric(c) && !IDENTITY.has(c) ? "metric" : numeric(c)]));
const unit = c => isNum[c] === "metric" && ROWS.every(r => r[c] === "" || (r[c] >= 0 && r[c] <= 1));
const format = (c, v) => Number.isInteger(v) ? String(v) : unit(c) ? v.toFixed(4) : v.toFixed(1);

const active = {};
let sort = { col: PRIMARY || COLUMNS[0], dir: LOWER_IS_BETTER.has(PRIMARY) ? 1 : -1 };  // best first

function buildFilters() {
  const box = document.getElementById("filters");
  for (const col of FILTERS) {
    const values = [...new Set(ROWS.map(r => r[col]))];
    if (values.length < 2) continue;
    active[col] = new Set(values);
    const group = document.createElement("div");
    group.className = "group";
    group.innerHTML = `<span class="group-label">${col}</span>`;
    for (const v of values) {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.textContent = v;
      chip.setAttribute("aria-pressed", "true");
      chip.onclick = () => {
        const on = active[col].has(v);
        on ? active[col].delete(v) : active[col].add(v);
        chip.setAttribute("aria-pressed", String(!on));
        render();
      };
      group.appendChild(chip);
    }
    box.appendChild(group);
  }
}

function buildHead() {
  const head = document.getElementById("head");
  head.innerHTML = "";
  for (const c of COLUMNS) {
    const th = document.createElement("th");
    th.className = [isNum[c] ? "num" : "", c === PRIMARY ? "primary" : ""].join(" ").trim();
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = c.replace("IoU/", "IoU · ");
    label.onclick = () => {
      sort = sort.col === c ? { col: c, dir: -sort.dir } : { col: c, dir: isNum[c] ? -1 : 1 };
      render();
    };
    th.appendChild(label);
    const buttons = document.createElement("span");
    buttons.className = "sort";
    for (const [dir, symbol, name] of [[-1, "▼", "descending"], [1, "▲", "ascending"]]) {
      const b = document.createElement("button");
      b.textContent = symbol;
      b.title = `Sort ${name}`;
      b.className = sort.col === c && sort.dir === dir ? "on" : "";
      b.onclick = () => { sort = { col: c, dir }; render(); };
      buttons.appendChild(b);
    }
    th.appendChild(buttons);
    head.appendChild(th);
  }
}

function render() {
  buildHead();
  const rows = ROWS.filter(r => Object.entries(active).every(([c, set]) => set.has(r[c])));
  rows.sort((a, b) => {
    const x = a[sort.col], y = b[sort.col];
    if (x === "" || y === "") return x === "" ? 1 : -1;
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
  });
  const best = {};
  for (const c of COLUMNS) {
    if (isNum[c] !== "metric") continue;
    const vals = rows.map(r => r[c]).filter(v => v !== "");
    if (vals.length > 1) best[c] = LOWER_IS_BETTER.has(c) ? Math.min(...vals) : Math.max(...vals);
  }
  const body = document.getElementById("body");
  body.innerHTML = rows.length ? "" : `<tr><td class="empty" colspan="${COLUMNS.length}">No rows match the filters.</td></tr>`;
  for (const r of rows) {
    const tr = document.createElement("tr");
    for (const c of COLUMNS) {
      const td = document.createElement("td");
      const v = r[c];
      if (isNum[c] && v !== "") {
        td.className = "num" + (best[c] === v ? " best" : "");
        const text = format(c, v);
        td.innerHTML = (unit(c) ? `<div class="bar" style="width:${(v * 100).toFixed(1)}%"></div>` : "")
          + `<span>${text}</span>`;
      } else {
        if (isNum[c]) td.className = "num";
        td.textContent = v === "" ? "–" : v;
      }
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  document.getElementById("count").textContent = `${rows.length} of ${ROWS.length} results`;
}

buildFilters();
render();
</script>
</body>
</html>
"""


def write_report(
    rows, path, title, primary, note, filters=REPORT_FILTERS, lower_is_better=LOWER_IS_BETTER, identity=("images",)
):
    """Write rows (dicts sharing the same keys) as a self-contained HTML table, sorted by primary first.

    filters are the columns offered as filter chips, lower_is_better the
    numeric columns whose smallest value is the best one, and identity the
    numeric columns that describe a row (counts, settings) rather than score it.
    """

    def as_json(value):  # keeps "</script>" in a value from closing the script tag
        return json.dumps(value).replace("</", "<\\/")

    page = (
        PAGE.replace("__TITLE__", html.escape(title))
        .replace("__NOTE__", html.escape(note))
        .replace("__COLUMNS__", as_json(list(rows[0])))
        .replace("__ROWS__", as_json(rows))
        .replace("__FILTERS__", as_json([c for c in filters if c in rows[0]]))
        .replace("__LOWER__", as_json(sorted(lower_is_better)))
        .replace("__PRIMARY__", as_json(primary))
        .replace("__IDENTITY__", as_json(list(identity)))
    )
    path.write_text(page)
    return path


def save_report(rows, run_dir):
    """Write run_dir/report.html from result rows, headlined by the foreground class's IoU."""
    foreground = [c for c in rows[0] if c.startswith("IoU/") and c != "IoU/background"]
    primary = foreground[0] if len(foreground) == 1 else "mIoU"
    note = (
        "Scored at the labels' own resolution. IoU, precision, recall and dice pool every pixel of the split; "
        "img_IoU_* is the spread of each image's own IoU; HD95_px is how far the predicted road edge strays, "
        "in pixels (lower is better). mIoU averages the IoU of every class, background included."
    )
    lower = LOWER_IS_BETTER | {c for c in rows[0] if c.startswith(("HD95_px", "img_IoU<"))}
    return write_report(
        rows, run_dir / "report.html", f"{run_dir.name} evaluation", primary, note, lower_is_better=lower
    )


def read_summary(csv_path):
    """Read a summary.csv back into result rows, turning numeric cells into numbers."""

    def parse(value):
        for kind in (int, float):
            try:
                return kind(value)
            except ValueError:
                pass
        return value

    with open(csv_path, newline="") as f:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="data.yaml written by download_dataset.py")
    parser.add_argument("--weights", type=Path, nargs="+", help="Defaults to every .pt in model/weights")
    parser.add_argument(
        "--split",
        nargs="+",
        choices=[*SPLITS, "all"],
        help="Splits to evaluate; 'all' combines every split (default: each split, then all)",
    )
    parser.add_argument(
        "--imgsz", type=int, help="Network input size the image is fitted into (default: each model's training size)"
    )
    parser.add_argument("--device", help="e.g. cuda or cpu (default: cuda if available)")
    parser.add_argument(
        "--static-crop",
        nargs="+",
        default=[str(c) for c in DEFAULT_STATIC_CROP],
        metavar="PCT",
        help="Roboflow static crop as X_MIN X_MAX Y_MIN Y_MAX in percent, or 'none' (default: 0 100 0 66)",
    )
    parser.add_argument(
        "--ego-mask",
        default="none",
        help="Pipeline config (e.g. src/perception_suite/config/demo_route.yaml) whose ego mask is left out "
        "of the scores (default: none)",
    )
    parser.add_argument(
        "--rename",
        nargs="*",
        default=DEFAULT_RENAME,
        metavar="OLD=NEW",
        help="Class names to relabel in the results (default: Final-Demo=traversable_road; pass none to keep them)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip evaluation and rebuild report.html from the dataset's existing summary.csv",
    )
    args = parser.parse_args()
    if args.report_only:
        run_dir = RUNS_DIR / args.data.parent.name
        if not (run_dir / "summary.csv").exists():
            sys.exit(f"No {run_dir / 'summary.csv'} yet. Run the evaluation once without --report-only.")
        print(f"Saved {save_report(read_summary(run_dir / 'summary.csv'), run_dir)}")
        return
    crop = None if args.static_crop == ["none"] else tuple(float(c) for c in args.static_crop)
    if crop is not None and len(crop) != 4:
        parser.error("--static-crop takes X_MIN X_MAX Y_MIN Y_MAX, or none")
    renames = [] if args.rename == ["none"] else args.rename

    data = yaml.safe_load(args.data.read_text())
    dataset_names = {int(k): v for k, v in data["names"].items()}
    dirs = split_dirs(data, args.data)
    splits = args.split or [*dirs, "all"]
    missing = [s for s in splits if s != "all" and s not in dirs]
    if missing:
        sys.exit(f"The dataset has no {', '.join(missing)} split. It has: {', '.join(dirs)}")
    scored_splits = list(dirs) if "all" in splits else [s for s in splits if s != "all"]
    weights = args.weights or sorted(WEIGHTS_DIR.glob("*.pt"))
    if not weights:
        sys.exit(f"No weights found in {WEIGHTS_DIR}")
    run_dir = RUNS_DIR / args.data.parent.name
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(data.get("path") or args.data.parent).resolve()

    polygon, ego_name = None, "none"
    if args.ego_mask != "none":
        polygon = ego_polygon(Path(args.ego_mask))
        if polygon is None:
            sys.exit(f"{args.ego_mask} has no enabled beamng.camera.ego_mask polygon.")
        ego_name = Path(args.ego_mask).stem
    crop_name = "none" if crop is None else "-".join(f"{c:g}" for c in crop)
    preprocessing = f"crop {crop_name}, ego mask {ego_name}"
    variant = f"crop{crop_name}" + (f"-ego-{ego_name}" if polygon is not None else "")  # per-image folder suffix

    files = {
        split: [(p, root / "masks" / split / f"{p.stem}.png") for p in sorted(Path(dirs[split]).iterdir()) if p.is_file()]
        for split in scored_splits
    }
    foreground = [c for c, name in dataset_names.items() if name != "background"]

    rows = []
    for weight in weights:
        segmenter = Segmenter(weight, args.imgsz, args.device)
        model_names = segmenter.yolo.names
        if segmenter.yolo.task != "semantic":
            print(f"Skipping {weight.name}: it is a {segmenter.yolo.task} model, not semantic segmentation")
            continue
        # Mask pixel values are class ids, so the dataset and model have to agree
        # on what each id means or the scores are meaningless.
        mismatch = class_mismatch(model_names, dataset_names, renames)
        if mismatch:
            sys.exit(f"{weight.name}: {mismatch}. Re-export the dataset with the classes the model was trained on.")
        classes = max(len(model_names), max(dataset_names) + 1)
        per_split = {}
        for split in scored_splits:
            print(f"=== {weight.name} on {split}: {len(files[split])} images (input {segmenter.size}) ===", flush=True)
            scores = Scores(classes, foreground)
            before = segmenter.seconds
            for image_file, mask_file in files[split]:
                image, truth = load_sample(image_file, mask_file, crop, polygon)
                if image is None:
                    print(f"  skipping {image_file.name}: image or mask unreadable")
                    continue
                scores.add(image_file.stem, truth, segmenter(image))
            scores.seconds = segmenter.seconds - before
            per_split[split] = scores
            by_stem = {p.stem: (p, m) for p, m in files[split]}

            def sample(stem, by_stem=by_stem, segmenter=segmenter):
                image, truth = load_sample(*by_stem[stem], crop, polygon)
                return image, truth, segmenter(image)

            scores.save_images(run_dir / f"{weight.stem}-{split}-{variant}", dataset_names, sample)
        if "all" in splits:
            combined = Scores(classes, foreground)
            for scores in per_split.values():
                combined.merge(scores)
            per_split["all"] = combined
        for split in splits:
            scores = per_split[split]
            row = {"weights": weight.name, "split": split, "preprocessing": preprocessing, "images": scores.count}
            row.update(scores.row(dataset_names))
            rows.append(row)

    if not rows:
        sys.exit("No semantic segmentation models were evaluated.")
    rows = order_columns(relabel_classes(rows, renames))

    columns = list(rows[0])
    widths = [max(len(c), *(len(str(r[c])) if not isinstance(r[c], float) else 6 for r in rows)) for c in columns]
    print(f"\nResults on {args.data.parent.name}:")
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    # Runs with other models or preprocessing build up in the same summary;
    # a rerun replaces only its own rows.
    summary = run_dir / "summary.csv"
    if summary.exists():
        new_keys = {(r["weights"], r["split"], r["preprocessing"]) for r in rows}
        kept = [r for r in read_summary(summary) if (r["weights"], r["split"], r["preprocessing"]) not in new_keys]
        rows = kept + rows
        all_columns = list(dict.fromkeys(c for r in rows for c in r))  # older runs may lack newer columns
        rows = order_columns([{c: r.get(c, "") for c in all_columns} for r in rows])
        columns = list(rows[0])
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {summary} ({len(rows)} rows)")
    print(f"Saved {save_report(rows, run_dir)}")


if __name__ == "__main__":
    main()
