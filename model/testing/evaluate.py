"""Evaluate the semantic segmentation weights against a ground truth dataset.

Runs Ultralytics validation for each model in model/weights (or the ones given)
on one dataset from download_dataset.py, then prints and saves a side-by-side
comparison of mIoU, pixel accuracy, per-class IoU and inference speed.

    python model/testing/evaluate.py --data model/testing/datasets/final_demo-vokml-v6/data.yaml
    python model/testing/evaluate.py --data ... --split all --weights model/weights/yolo26n-demo-route.pt

By default every split is scored on its own and then all of them together
("all"). Images the models were trained on score higher than new ones, so the
test and val rows are the honest estimate and "all" is an upper bound.

Images and masks first go through the preprocessing the models were trained
with in Roboflow: a static crop to the top 66% of the frame, which cuts off the
car's hood, then a fit into 1024x1024 with black edges. The edges are left out
of the scores. --static-crop and --resize change or disable either step, to
match a different Roboflow version.

--ego-mask additionally leaves out the polygon from beamng.camera.ego_mask in
a pipeline config, as the pipeline does after inference. Together with
--static-crop none --resize none it scores the models the way the pipeline
runs them today: the whole frame, with the hood masked afterwards.

The headline number is IoU/traversable_road: of the pixels that are road in
the labels or the prediction, the share both agree on. precision/ and recall/
split its errors into road painted where there is none and road missed. mIoU
averages the IoU with IoU/background, which is high for any model since most
of the frame is not road. All are pooled over every pixel of the split.

Ultralytics' per-model output (confusion matrix, IoU bar chart, example
predictions) goes to model/testing/runs/<dataset>/<weights>-<split>/, and the
comparison to model/testing/runs/<dataset>/summary.csv and report.html, a page
that filters and sorts the results. --report-only rebuilds report.html from the
existing summary.csv without evaluating, for working on the page.
"""

import argparse
import csv
import html
import json
import sys
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


def prepare(root, dirs, out, crop, resize, polygon):
    """Write every split's images and masks to out/{images,masks}/<split>, preprocessed for evaluation.

    The ego mask polygon, normalised to the full frame, is applied before the
    crop; the crop is in percent of the frame, as Roboflow gives it.
    """
    prepared = {}
    for split, image_dir in dirs.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "masks" / split).mkdir(parents=True, exist_ok=True)
        for image_file in sorted(Path(image_dir).iterdir()):
            mask_file = root / "masks" / split / f"{image_file.stem}.png"
            image = cv2.imread(str(image_file))
            mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                continue
            h, w = mask.shape[:2]
            if crop and resize and (h, w) == (resize, resize):
                sys.exit(
                    f"{image_file.name} is already {resize}x{resize}, so this Roboflow version was probably "
                    "exported with its preprocessing. Pass --static-crop none --resize none."
                )
            if polygon is not None:
                cv2.fillPoly(mask, [np.round(polygon * [w, h]).astype(np.int32)], IGNORE)
            if crop:
                x0, x1, y0, y1 = (round(c / 100 * n) for c, n in zip(crop, (w, w, h, h)))
                image, mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            if resize:
                image = fit_black_edges(image, resize, cv2.INTER_AREA, (0, 0, 0))
                mask = fit_black_edges(mask, resize, cv2.INTER_NEAREST, IGNORE)
            cv2.imwrite(str(out / "images" / split / f"{image_file.stem}.png"), image)
            cv2.imwrite(str(out / "masks" / split / f"{image_file.stem}.png"), mask)
        prepared[split] = str(out / "images" / split)
    return prepared


def class_scores(metrics, names):
    """Per-class columns: IoU for every class, and precision and recall for each class but background.

    From the validator's pixel confusion matrix (rows are the labels, columns
    the predictions): precision is the share of pixels predicted as the class
    that are labelled it, recall the share of labelled pixels that are found.
    A class with no labelled pixels gets blanks rather than 0.
    """
    matrix = metrics.matrix.cpu().numpy() if metrics.matrix is not None else None
    scores = {}
    for i, name in names.items():
        present = metrics.nt_per_class[i] > 0
        scores[f"IoU/{name}"] = float(metrics.per_class_iou[i]) if present else ""
        if name == "background" or matrix is None or i >= len(matrix):
            continue
        predicted, labelled = matrix[:, i].sum(), matrix[i].sum()
        scores[f"precision/{name}"] = float(matrix[i, i] / predicted) if present and predicted else ""
        scores[f"recall/{name}"] = float(matrix[i, i] / labelled) if present else ""
    return scores


CLASS_PREFIXES = ("IoU/", "precision/", "recall/")


def relabel_classes(rows, renames):
    """Drop the columns of classes no row has labels for, then apply OLD=NEW renames to the rest."""
    empty_classes = {c[4:] for c in rows[0] if c.startswith("IoU/") and all(r[c] == "" for r in rows)}
    empty = {f"{p}{name}" for p in CLASS_PREFIXES for name in empty_classes}
    mapping = {}
    for rename in renames:
        old, _, new = rename.partition("=")
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
    note = "mIoU averages the IoU of every class, background included."
    return write_report(rows, run_dir / "report.html", f"{run_dir.name} evaluation", primary, note)


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
    parser.add_argument("--imgsz", type=int, help="Inference size (default: the size each model was trained at)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", help="e.g. 0 or cpu (default: Ultralytics picks)")
    parser.add_argument(
        "--static-crop",
        nargs="+",
        default=[str(c) for c in DEFAULT_STATIC_CROP],
        metavar="PCT",
        help="Roboflow static crop as X_MIN X_MAX Y_MIN Y_MAX in percent, or 'none' (default: 0 100 0 66)",
    )
    parser.add_argument(
        "--resize",
        default=str(DEFAULT_RESIZE),
        help="Fit into a square of this side with black edges, as Roboflow does, or 'none' (default: 1024)",
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
    from ultralytics import YOLO  # here, so --report-only skips loading torch
    crop = None if args.static_crop == ["none"] else tuple(float(c) for c in args.static_crop)
    if crop is not None and len(crop) != 4:
        parser.error("--static-crop takes X_MIN X_MAX Y_MIN Y_MAX, or none")
    resize = None if args.resize == "none" else int(args.resize)

    data = yaml.safe_load(args.data.read_text())
    dataset_names = {int(k): v for k, v in data["names"].items()}
    dirs = split_dirs(data, args.data)
    splits = args.split or [*dirs, "all"]
    missing = [s for s in splits if s != "all" and s not in dirs]
    if missing:
        sys.exit(f"The dataset has no {', '.join(missing)} split. It has: {', '.join(dirs)}")
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
    preprocessing = f"crop {crop_name}, resize {resize or 'none'}, ego mask {ego_name}"
    out = root / "prepared" / f"crop{crop_name}_resize{resize or 'none'}_ego-{ego_name}"
    print(f"Preparing images ({preprocessing}) in {out}")
    dirs = prepare(root, dirs, out, crop, resize, polygon)

    # Ultralytics only evaluates its own split names, so each split is handed to
    # it as 'val' in a copy of data.yaml, with absolute paths so the copy can
    # live in the run directory.
    split_data = {}
    for split in splits:
        derived = {**data, "path": str(out), "masks_dir": "masks"}
        derived["val"] = list(dirs.values()) if split == "all" else dirs[split]
        derived_file = run_dir / f"data-{split}.yaml"
        derived_file.write_text(yaml.safe_dump(derived, sort_keys=False))
        split_data[split] = derived_file
    counts = {s: sum(1 for p in Path(d).iterdir() if p.is_file()) for s, d in dirs.items()}
    counts["all"] = sum(counts.values())

    rows = []
    for weight in weights:
        model = YOLO(weight)
        if model.task != "semantic":
            print(f"Skipping {weight.name}: it is a {model.task} model, not semantic segmentation")
            continue
        # Mask pixel values are class ids, so the dataset and model have to agree
        # on what each id means or the scores are meaningless.
        if model.names != dataset_names:
            sys.exit(
                f"{weight.name} classes {model.names} do not match the dataset's {dataset_names}. "
                "Re-export the dataset with the classes the model was trained on."
            )
        imgsz = args.imgsz or model.ckpt.get("train_args", {}).get("imgsz", 640)
        for split in splits:
            print(f"\n=== {weight.name} on {split} (imgsz {imgsz}) ===")
            metrics = model.val(
                data=str(split_data[split]),
                split="val",
                imgsz=imgsz,
                batch=args.batch,
                device=args.device,
                project=str(run_dir),
                name=f"{weight.stem}-{split}",
                exist_ok=True,
                plots=True,
            )
            row = {
                "weights": weight.name,
                "split": split,
                "preprocessing": preprocessing,
                "images": counts[split],
                "mIoU": metrics.miou,
                "pixel_acc": metrics.pixel_accuracy,
            }
            row.update(class_scores(metrics, dataset_names))
            row["inference_ms"] = metrics.speed["inference"]
            rows.append(row)

    if not rows:
        sys.exit("No semantic segmentation models were evaluated.")
    rows = order_columns(relabel_classes(rows, [] if args.rename == ["none"] else args.rename))

    columns = list(rows[0])
    widths = [max(len(c), *(len(str(r[c])) if not isinstance(r[c], float) else 6 for r in rows)) for c in columns]
    print(f"\nResults on {args.data.parent.name}:")
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    summary = run_dir / "summary.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {summary}")
    print(f"Saved {save_report(rows, run_dir)}")


if __name__ == "__main__":
    main()
