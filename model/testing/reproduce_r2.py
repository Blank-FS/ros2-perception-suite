"""Reproduce the results of the R2 requirement (Segmentation Performance).

    uv run reproduce_r2.py               # download what's missing, run everything, print the tables
    uv run reproduce_r2.py --tables-only # print the tables from results already in runs/

Downloads the two datasets and the drive video if they aren't there yet (the
datasets need ROBOFLOW_API_KEY in .env), runs the four evaluations behind R2,
about 25 minutes on a laptop GPU, and prints R2's tables from their results,
saying which report each one comes from.

In R2, 26n and 26s are yolo26n-demo-route and yolo26s-demo-route, and 26n_vos
and 26s_vos are the retrained rf26n-sem-640 and rf26s-sem-640.
"""

import argparse
import csv
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal, getcontext
from pathlib import Path

import requests

getcontext().rounding = ROUND_HALF_UP

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "weights"
RUNS = HERE / "runs"
WORKSPACE = "lucianas-workspace-pqyes"
WEEK8 = HERE / "datasets/week-8-vos-v1/data.yaml"
V6 = HERE / "datasets/final_demo-vokml-v6/data.yaml"
VIDEO = HERE / "recordings/demo_route_hatci.mp4"
VIDEO_URL = (
    "https://www.dropbox.com/scl/fi/2brwgt0hmjszh4x0x8k0e/video.mp4"
    "?rlkey=8e3ircva7xzwhfdvcyihjdsgd&dl=1"
)
MODELS = {  # R2's name: weights file
    "26n": "yolo26n-demo-route.pt",
    "26s": "yolo26s-demo-route.pt",
    "26n_vos": "rf26n-sem-640.pt",
    "26s_vos": "rf26s-sem-640.pt",
}
DOUBLE_CROP = "week8yolo26s.pt"  # the first 26s_vos, before retraining
ROAD = "/traversable_road"


def weights(*files):
    return [str(WEIGHTS / f) for f in files]


def run(description, args):
    print(f"\n### {description}\n", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def get_data():
    for project, version, data in (("week-8-vos", 1, WEEK8), ("final_demo-vokml", 6, V6)):
        if not data.exists():
            run(
                f"Downloading {project} v{version}",
                ["download_dataset.py", "--workspace", WORKSPACE, "--project", project, "--version", str(version)],
            )
    if not VIDEO.exists():
        print(f"\n### Downloading the drive video to {VIDEO.relative_to(HERE)} (2 GB)\n", flush=True)
        VIDEO.parent.mkdir(parents=True, exist_ok=True)
        partial = VIDEO.with_suffix(".part")
        with requests.get(VIDEO_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(partial, "wb") as f:
                f.writelines(response.iter_content(chunk_size=1 << 22))
        partial.rename(VIDEO)


def run_all():
    missing = [f for f in [*MODELS.values(), DOUBLE_CROP] if not (WEIGHTS / f).exists()]
    if missing:
        sys.exit(f"Missing weights in {WEIGHTS}: {', '.join(missing)}")
    get_data()
    run(
        "1/4 Table 1: 26n and 26s on v6",
        ["evaluate.py", "--data", str(V6), "--split", "val", "test",
         "--weights", *weights(MODELS["26n"], MODELS["26s"])],
    )
    run(
        "2/4 Table 1: VOS models, and Table 2: all four on week-8 at 640",
        ["evaluate.py", "--data", str(WEEK8), "--split", "val", "test", "--static-crop", "none", "--imgsz", "640",
         "--weights", *weights(*MODELS.values())],
    )
    run(
        "3/4 Current Status and Next Steps: 26s_vos before retraining, and 26s at 1024",
        ["evaluate.py", "--data", str(WEEK8), "--split", "test", "--static-crop", "none",
         "--weights", *weights(DOUBLE_CROP, MODELS["26s"])],
    )
    run(
        "4/4 Table 3: gate sweep on the drive",
        ["gate_sweep.py", "--source", str(VIDEO), "--weights", *weights(*MODELS.values())],
    )


# --- Printing the tables ---------------------------------------------------


def read(path):
    if not path.exists():
        sys.exit(f"{path.relative_to(HERE)} not found. Run `uv run reproduce_r2.py` first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find(rows, weights_file, split, preprocessing):
    for r in rows:
        if r["weights"] == weights_file and r["split"] == split and r["preprocessing"] == preprocessing:
            return r
    sys.exit(f"No result for {weights_file}, {split}, '{preprocessing}'. Run `uv run reproduce_r2.py` first.")


def table(title, source, header, rows):
    print(f"\n{title}\n  from {source}")
    widths = [max(len(str(x)) for x in column) for column in zip(header, *rows)]
    for line in (header, *rows):
        print("  " + "  ".join(str(x).ljust(w) for x, w in zip(line, widths)))


def f3(value):
    return f"{Decimal(str(value)):.3f}"


def pct(value):
    # Decimal, so a share stored as 0.0045 prints 0.5%, not the 0.4% that float rounding gives
    return f"{Decimal(str(value)) * 100:.1f}%"


def print_tables():
    v6 = read(RUNS / "final_demo-vokml-v6/summary.csv")
    week8 = read(RUNS / "week-8-vos-v1/summary.csv")
    gate = read(RUNS / "gate-demo_route_hatci/summary.csv")
    at_640, full, v6_crop = "crop none, ego mask none, input 640", "crop none, ego mask none", "crop 0-100-0-66, ego mask none"

    rows = []
    for name in ("26n", "26s"):
        val, test = (find(v6, MODELS[name], s, v6_crop) for s in ("val", "test"))
        rows.append([name, "v6 val / test", *(f"{f3(val[c + ROAD])} / {f3(test[c + ROAD])}"
                                              for c in ("IoU", "precision", "recall", "IoU_near"))])
    for name in ("26n_vos", "26s_vos"):
        val, test = (find(week8, MODELS[name], s, at_640) for s in ("val", "test"))
        rows.append([name, "week-8 val / test", *(f"{f3(val[c + ROAD])} / {f3(test[c + ROAD])}"
                                                  for c in ("IoU", "precision", "recall", "IoU_near"))])
    table("Table 1. Held-Out Accuracy",
          "runs/final_demo-vokml-v6/report.html (26n, 26s) and runs/week-8-vos-v1/report.html (VOS, 'input 640')",
          ["Model", "Split", "Road IoU", "Precision", "Recall", "Near IoU"], rows)

    rows = []
    for name, file in MODELS.items():
        r = find(week8, file, "test", at_640)
        rows.append([name, f3(r["IoU" + ROAD]), f3(r["recall" + ROAD]), f3(r["IoU_near" + ROAD]),
                     f3(r["img_IoU_p5" + ROAD]), pct(r["img_IoU<0.9" + ROAD]), f"{float(r['HD95_px' + ROAD]):.1f}"])
    table("Table 2. Week-8 Test Set at 640 Input", "runs/week-8-vos-v1/report.html, preprocessing 'input 640', split test",
          ["Model", "Road IoU", "Recall", "Near IoU", "P5 image IoU", "Images < 0.9 IoU", "HD95 (px)"], rows)

    segments = {}
    for name, file in MODELS.items():
        path = RUNS / f"week-8-vos-v1/{Path(file).stem}-test-cropnone-input640/segments.csv"
        segments[name] = {(r["first_frame"], r["last_frame"]): float(r["IoU" + ROAD]) for r in read(path)}
    rows = [[f"{a} minus {b}", *(f"{segments[a][k] - segments[b][k]:+.3f}" for k in segments[a])]
            for a, b in (("26n", "26n_vos"), ("26s", "26s_vos"))]
    table("Note under Table 1. Road IoU gap in each test segment",
          "runs/week-8-vos-v1/<model>-test-cropnone-input640/segments.csv",
          ["Models", *(f"frames {first}-{last}" for first, last in segments["26n"])], rows)

    rows = []
    for name, file in MODELS.items():
        by_threshold = {r["threshold"]: r for r in gate if r["weights"] == file}
        argmax, strict = by_threshold["argmax"], by_threshold["0.9"]
        rejections = [round(float(r["reject_rate"]) * int(r["frames"])) for r in (argmax, strict)]
        any_threshold = sum(round(float(r["reject_rate"]) * int(r["frames"])) for r in by_threshold.values())
        rows.append([name, f"{rejections[0]} / {rejections[1]}", any_threshold, pct(argmax["min_road_area"]),
                     f3(argmax["stability_mean"]), pct(argmax["unstable_frames"]), argmax["input"]])
    table("Table 3. Perception Gate on the Demo Route", "runs/gate-demo_route_hatci/report.html",
          ["Model", "Rejections (argmax / t = 0.9)", "At any threshold", "Min road area", "Mean stability",
           "Frames < 0.9 stability", "Input"], rows)

    before = find(week8, DOUBLE_CROP, "test", full)
    after = find(week8, MODELS["26s_vos"], "test", at_640)
    at_1024 = find(week8, MODELS["26s"], "test", full)
    at_640_26s = find(week8, MODELS["26s"], "test", at_640)
    print("\nCurrent Status and Next Steps\n  from runs/week-8-vos-v1/report.html, split test")
    print(f"  26s_vos road IoU, before and after retraining:      {f3(before['IoU' + ROAD])} -> {f3(after['IoU' + ROAD])}")
    print(f"  26s_vos near-road IoU, before and after retraining: "
          f"{f3(before['IoU_near' + ROAD])} -> {f3(after['IoU_near' + ROAD])}")
    print(f"  26s road IoU at 640 and at 1024:                    "
          f"{f3(at_640_26s['IoU' + ROAD])} and {f3(at_1024['IoU' + ROAD])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-only", action="store_true", help="Print the tables from existing results")
    args = parser.parse_args()
    if not args.tables_only:
        run_all()
    print_tables()


if __name__ == "__main__":
    main()
