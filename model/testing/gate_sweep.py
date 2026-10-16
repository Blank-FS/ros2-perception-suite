"""Measure how often the perception gate would reject frames, across road thresholds.

Runs each model in model/weights over a drive's video, frame by frame, and passes
every frame through the pipeline's own perception gate (PerceptionGate, with
the pipeline config's ROI, hood mask and limits) at each threshold t:

* a pixel is road when the model's road probability, 1 - P(background), is at
  least t, so a higher t keeps only the road the model is sure of;
* the gate then rejects the frame, as it does on the vehicle, when that road
  has too few pixels in the planner ROI, does not reach the vehicle, or covers
  too little of the ROI.

"argmax" is the mask the pipeline uses today, the model's most likely class.
A semantic model has no detection score, so the gate's confidence check
always passes for it, as it does in the pipeline.

Frames get the preprocessing the models were trained with (the same static
crop and fit as evaluate.py). The road found in the crop is placed back into
the pipeline's working frame, below which the hood mask applies as usual.

    python model/testing/gate_sweep.py --source model/testing/recordings/demo_route_hatci.mp4
    python model/testing/gate_sweep.py --source ... --thresholds 0.3 0.5 0.7 --weights model/weights/yolo26n-demo-route.pt

The results go to model/testing/runs/gate-<source>/: summary.csv and
report.html with one row per model and threshold, and frames.csv with every
frame's decision, to find where the long rejection streaks happen. Add
--report-only to rebuild report.html from summary.csv without running models.
"""

import argparse
import csv
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import cv2
import numpy as np
from evaluate import (
    DEFAULT_RESIZE,
    DEFAULT_STATIC_CROP,
    RUNS_DIR,
    WEIGHTS_DIR,
    fit_black_edges,
    read_summary,
    write_report,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/perception_suite/config"
DEFAULT_CONFIG = CONFIG_DIR / "hatci_vehicle.yaml"  # the vehicle the drives are recorded on
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# CPU budget, tuned on a 16-core laptop (about 32 fps overall): a gate check
# takes ~4 ms, so two processes per model keep up with 30 fps, four threads
# decode 2880x1860 H.264 faster than that, and three prepare frames. More of
# any only starves the thread that feeds the GPU.
GATE_WORKERS_PER_MODEL = 2
DECODE_THREADS = 4
PREPARE_THREADS = 3
INBOX_FRAMES = 64  # how far the gate workers may fall behind the GPU
RING_SLOTS = 2 * INBOX_FRAMES
# Frames segmented together: each GPU call costs a few ms of Python launch
# overhead whatever its size, so a batch shares it.
BATCH = 4
# PerceptionGate's reasons, by how they start, as summary column names.
REJECT_REASONS = {
    "too few road pixels": "too few road pixels",
    "no road at vehicle": "no road at vehicle",
    "road area": "road area too small",
}


# --- Reading frames --------------------------------------------------------


def read_frames(source, stride):
    """Yield (frame index, seconds into the video, BGR image), decoding on several threads."""
    import av

    try:
        container = av.open(str(source))
    except av.error.FFmpegError as error:
        sys.exit(f"Cannot open {source} as a video: {error}")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # 2880x1860 frames decode too slowly on one core
    stream.codec_context.thread_count = DECODE_THREADS
    with container:
        for index, frame in enumerate(container.decode(stream)):
            if index % stride == 0:
                yield index, float(frame.time or 0.0), frame.to_ndarray(format="bgr24")


class Prepared:
    """A frame after the training preprocessing, with where its crop sits in the pipeline's working frame."""

    def __init__(self, image, crop, size, out_w, out_h):
        h, w = image.shape[:2]
        x0, x1, y0, y1 = (round(c / 100 * n) for c, n in zip(crop or (0, 100, 0, 100), (w, w, h, h)))
        cropped = image[y0:y1, x0:x1]
        self.square = fit_black_edges(cropped, size, cv2.INTER_AREA, (0, 0, 0))
        # The fit's black edges, to cut off again, and the crop's box in the working frame.
        scale = size / max(cropped.shape[:2])
        new_w, new_h = round(cropped.shape[1] * scale), round(cropped.shape[0] * scale)
        top, left = (size - new_h) // 2, (size - new_w) // 2
        self.inner = (slice(top, top + new_h), slice(left, left + new_w))
        fx, fy = out_w / w, out_h / h
        self.box = (round(x0 * fx), round(x1 * fx), round(y0 * fy), round(y1 * fy))
        self.out_size = (out_h, out_w)


def prepared_frames(frames, crop, size, out_w, out_h, depth=16):
    """Decode on a background thread and prepare on a few more, in frame order, overlapping the GPU work.

    OpenCV releases the GIL while it resizes, so several frames are prepared
    at once; one thread alone tops out below 30 fps at this frame size.
    """
    queue, done = Queue(maxsize=depth), object()
    failure = []

    def work():
        try:
            with ThreadPoolExecutor(PREPARE_THREADS) as pool:
                window = deque()
                for index, seconds, image in frames:
                    window.append((index, seconds, pool.submit(Prepared, image, crop, size, out_w, out_h)))
                    if len(window) > PREPARE_THREADS:
                        index, seconds, future = window.popleft()
                        queue.put((index, seconds, future.result()))
                for index, seconds, future in window:
                    queue.put((index, seconds, future.result()))
        except BaseException as error:  # noqa: BLE001 - re-raised in the main thread below
            failure.append(error)
        finally:
            queue.put(done)

    Thread(target=work, daemon=True).start()
    while (item := queue.get()) is not done:
        yield item
    if failure:
        raise failure[0]


# --- Model -----------------------------------------------------------------


class RoadProbability:
    """Segments prepared frames into a level map at the pipeline's working size.

    Each pixel of the map holds, in its low 7 bits, how many of the thresholds
    its road probability, 1 - P(background), reaches, and in its top bit
    whether the model's most likely class is road. Every threshold's mask comes
    back out of it exactly (see masks()), in a twentieth of the bytes.
    """

    def __init__(self, weights, thresholds, device):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        yolo = YOLO(weights)
        if yolo.task != "semantic":
            raise ValueError(f"{weights.name} is a {yolo.task} model, not semantic segmentation")
        self.background = next((i for i, n in yolo.names.items() if n == "background"), 0)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = yolo.model.to(self.device).eval()
        self.thresholds = torch.tensor(sorted(thresholds), device=self.device)

    def __call__(self, frames, tensor):
        """Level maps for a batch of prepared frames of one video, whose squares are already in tensor.

        Frames of one video share their size and crop, so the batch is resized
        back in one go, on the GPU; only the small working-size maps come back.
        """
        torch, F = self.torch, self.torch.nn.functional
        frame = frames[0]
        size = tensor.shape[-1]
        x0, x1, y0, y1 = frame.box
        with torch.no_grad():
            logits = self.net(tensor)
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            logits = F.interpolate(logits.float(), (size, size), mode="bilinear")
            inner = logits[:, :, frame.inner[0], frame.inner[1]]
            road = 1.0 - inner.softmax(1)[:, self.background : self.background + 1]
            road = F.interpolate(road, (y1 - y0, x1 - x0), mode="bilinear")[:, 0]
            argmax = (inner.argmax(1, keepdim=True) != self.background).to(torch.uint8)
            argmax = F.interpolate(argmax, (y1 - y0, x1 - x0), mode="nearest")[:, 0]
            reached = torch.bucketize(road, self.thresholds, right=True)  # thresholds <= the probability
            levels = (reached.to(torch.uint8) | (argmax << 7)).cpu().numpy()
        out = np.zeros((len(frames), *frame.out_size), np.uint8)  # outside the crop is not road
        out[:, y0:y1, x0:x1] = levels
        return out


def to_tensor(squares, device):
    """A batch of BGR squares as the network's RGB float input."""
    import torch

    batch = torch.from_numpy(np.ascontiguousarray(np.stack(squares)[..., ::-1].transpose(0, 3, 1, 2)))
    return batch.to(device, non_blocking=True).float().div(255)


def masks(levels, labels):
    """{label: road mask} from a level map, for "argmax" and every threshold among labels."""
    reached = levels & 0x7F
    thresholds = sorted(float(label) for label in labels if label != "argmax")
    out = {f"{t:g}": reached > i for i, t in enumerate(thresholds)}
    if "argmax" in labels:
        out["argmax"] = levels >= 0x80
    return out


# --- Gate ------------------------------------------------------------------


def gate_config(config_path):
    from offroad_autonomy.utils.config import load_config

    # A semantic model's confidence is always 1.0 in the pipeline, so its gate
    # threshold is 0 and only the mask decides.
    return replace(load_config(config_path), gate_min_confidence=0.0)


def gate_worker(config_path, name, labels, own, ring_spec, inbox, outbox):
    """Run the pipeline's stabilizer and gate on one model's level maps, for the labels in own.

    labels are all the model's labels, needed to read the level map; each
    worker process judges only its own share of them, frame by frame in order.
    The maps arrive in a shared-memory ring (ring_spec: its name, shape and this
    model's slot), so the queue only carries which ring slot to read.
    """
    from multiprocessing.shared_memory import SharedMemory

    cv2.setNumThreads(1)  # many workers share the cores; their own thread pools only contend
    ring_name, ring_shape, model_slot = ring_spec
    memory = SharedMemory(name=ring_name)
    ring = np.ndarray(ring_shape, np.uint8, buffer=memory.buf)
    from offroad_autonomy.perception.perception_view import PerceptionView
    from offroad_autonomy.planning.perception_gate import PerceptionGate
    from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
    from offroad_autonomy.types import PerceptionResult

    config = gate_config(config_path)
    valid_roi = PerceptionView(config).valid_roi
    gate = PerceptionGate(config)
    stabilizers = {label: TemporalStabilizer(config) for label in own}
    while (item := inbox.get()) is not None:
        index, seconds, slot = item
        levels = ring[slot, model_slot]
        results = []
        for label, mask in masks(levels, labels).items():
            if label in stabilizers:
                perception = PerceptionResult(mask=mask & valid_roi, confidences=[1.0], valid_roi=valid_roi)
                d = gate.evaluate(stabilizers[label].stabilize(perception))
                results.append(((name, label), index, seconds, d.ok, d.reason, d.road_pixels, d.area_fraction))
        outbox.put(results)
    del ring
    memory.close()
    outbox.put(None)


# --- Results ---------------------------------------------------------------


def longest_run(flags):
    """Length and start index of the longest run of True in flags."""
    best, best_start, run, start = 0, None, 0, 0
    for i, flag in enumerate(flags):
        if flag:
            start = i if run == 0 else start
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    return best, best_start


def summarize(weights, threshold, decisions, stride):
    """One summary row from (frame index, seconds, ok, reason, road pixels, area) for one model and threshold.

    Besides the rejections, it records the margin: the fewest road pixels and
    smallest road area of any frame, which show how close the drive came to
    the gate's limits even when nothing was rejected.
    """
    rejected = [not ok for _, _, ok, *_ in decisions]
    streak, start = longest_run(rejected)
    row = {
        "weights": weights,
        "threshold": threshold,
        "frames": len(decisions),
        "reject_rate": sum(rejected) / max(len(decisions), 1),
        "longest_reject_streak": streak * stride,  # in recorded frames, so --stride does not shrink it
        "longest_reject_s": 0.0,
        "streak_starts_s": "",
        "min_road_pixels": min((d[4] for d in decisions), default=0),
        "min_road_area": round(min((d[5] for d in decisions), default=0.0), 4),
        "tightest_at_s": "",
    }
    if decisions:
        tightest = min(decisions, key=lambda d: d[5])
        row["tightest_at_s"] = round(tightest[1], 2)
    if streak:
        first, last = decisions[start], decisions[start + streak - 1]
        frame_time = (decisions[-1][1] - decisions[0][1]) / max(decisions[-1][0] - decisions[0][0], 1)
        row["longest_reject_s"] = round(last[1] - first[1] + frame_time * stride, 2)
        row["streak_starts_s"] = round(first[1], 2)
    for kind in REJECT_REASONS.values():
        row[f"rejects/{kind}"] = 0
    row["rejects/other"] = 0
    for _, _, ok, reason, *_ in decisions:
        if not ok:
            kind = next((k for prefix, k in REJECT_REASONS.items() if reason.startswith(prefix)), "other")
            row[f"rejects/{kind}"] += 1
    return row


def save_report(rows, run_dir):
    rows = [{**r, "threshold": str(r["threshold"])} for r in rows]  # text, so argmax and numbers sort together
    note = (
        "threshold is the road probability a pixel needs to count as road; argmax is the mask the pipeline "
        "uses today. Streak lengths are in recorded frames, and streak_starts_s is where the longest one begins. "
        "min_road_pixels and min_road_area are the closest any frame came to the gate's limits, at tightest_at_s."
    )
    return write_report(
        rows,
        run_dir / "report.html",
        f"{run_dir.name.removeprefix('gate-')} gate rejections",
        "reject_rate",
        note,
        filters=("weights", "threshold"),
        lower_is_better={c for c in rows[0] if c.startswith(("reject", "longest", "rejects/"))},
        identity=("threshold", "frames", "streak_starts_s", "tightest_at_s"),
    )


# --- Sweep -----------------------------------------------------------------


def sweep(source, weights, thresholds, config_path, crop, size, stride, max_frames, device):
    """Run every model and threshold over the video; return {(model, label): [(index, seconds, ok, reason,
    road_pixels, area)]} in frame order."""
    import multiprocessing

    from offroad_autonomy.perception.perception_view import PerceptionView

    config = gate_config(config_path)
    out_w, out_h = PerceptionView(config).size
    print(f"Gate from {config_path.name}: working frame {out_w}x{out_h}, ROI bottom {config.planner_roi_height:.0%}")
    labels = ["argmax", *[f"{t:g}" for t in sorted(thresholds)]]

    # The gate is mostly Python, so it runs in worker processes rather than
    # threads. Each owns a share of one model's labels and sees every frame in
    # order, so its stabilizers behave as in the pipeline. They start before
    # the models load, so none inherits a CUDA context.
    from multiprocessing.shared_memory import SharedMemory

    context = multiprocessing.get_context("spawn")
    outbox = context.Queue()
    # Level maps go through a ring in shared memory. A slot is reused only
    # RING_SLOTS frames later, and a worker is never more than INBOX_FRAMES
    # behind, so no slot is overwritten while a worker may still read it.
    ring_shape = (RING_SLOTS, len(weights), out_h, out_w)
    memory = SharedMemory(create=True, size=int(np.prod(ring_shape)))
    ring = np.ndarray(ring_shape, np.uint8, buffer=memory.buf)
    per_model = max(1, min(len(labels), GATE_WORKERS_PER_MODEL))
    workers = []  # (model name, inbox, process)
    for slot, weight in enumerate(weights):
        for part in range(per_model):
            inbox = context.Queue(maxsize=INBOX_FRAMES)  # the GPU waits if the gate falls this far behind
            own = labels[part::per_model]
            ring_spec = (memory.name, ring_shape, slot)
            process = context.Process(
                target=gate_worker,
                args=(config_path, weight.name, labels, own, ring_spec, inbox, outbox),
                daemon=True,
            )
            process.start()
            workers.append((weight.name, inbox, process))

    models = {weight.name: RoadProbability(weight, thresholds, device) for weight in weights}
    torch_device = next(iter(models.values())).device
    decisions = {(name, label): [] for name in models for label in labels}
    finished = 0

    def collect(block):
        """Move finished gate results into decisions; with block, wait for at least one batch."""
        nonlocal finished
        while True:
            try:
                results = outbox.get(timeout=600) if block else outbox.get_nowait()
            except Empty:
                if block:
                    sys.exit("The gate workers stopped responding")
                return
            if results is None:
                finished += 1
            else:
                for key, *entry in results:
                    decisions[key].append(tuple(entry))
            if block:
                return

    started, count = time.perf_counter(), 0

    def segment(batch):
        """Run every model on a batch of frames and hand each frame to the gate workers."""
        nonlocal count
        tensor = to_tensor([frame.square for _, _, frame in batch], torch_device)
        slots = [(count + i) % RING_SLOTS for i in range(len(batch))]
        for model_slot, model in enumerate(models.values()):
            ring[slots, model_slot] = model([frame for _, _, frame in batch], tensor)
        for (index, seconds, _), slot in zip(batch, slots):
            for _, inbox, _ in workers:
                inbox.put((index, seconds, slot))
            count += 1
            if count % 300 == 0:
                rate = count / (time.perf_counter() - started)
                print(f"  frame {index} ({seconds:.0f} s of video, {rate:.1f} frames/s)", flush=True)
        collect(block=False)

    try:
        batch = []
        for item in prepared_frames(read_frames(source, stride), crop, size, out_w, out_h):
            if max_frames and count + len(batch) >= max_frames:
                break
            batch.append(item)
            if len(batch) == BATCH:
                segment(batch)
                batch = []
        if batch:
            segment(batch)
    finally:
        for _, inbox, _ in workers:
            inbox.put(None)
    while finished < len(workers):
        collect(block=True)
    for _, _, process in workers:
        process.join()
    ring = None  # releases the view on the shared memory, which must go before it closes
    memory.close()
    memory.unlink()
    elapsed = time.perf_counter() - started
    print(f"  {count} frames in {elapsed:.0f} s ({count / max(elapsed, 1e-9):.1f} frames/s)")
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="The drive's video file")
    parser.add_argument("--weights", type=Path, nargs="+", help="Defaults to every .pt in model/weights")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS, help="Road probability thresholds"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline config for the gate (default: hatci_vehicle.yaml)"
    )
    parser.add_argument(
        "--static-crop",
        nargs="+",
        default=[str(c) for c in DEFAULT_STATIC_CROP],
        metavar="PCT",
        help="Training crop as X_MIN X_MAX Y_MIN Y_MAX in percent, or 'none' (default: 0 100 0 66)",
    )
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE, help="Training fit size (default: 1024)")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame (default: every frame)")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames, for a quick check")
    parser.add_argument("--device", help="e.g. cuda or cpu (default: cuda if available)")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report.html from summary.csv and stop")
    args = parser.parse_args()

    source = args.source.expanduser()
    run_dir = RUNS_DIR / f"gate-{source.stem}"
    if args.report_only:
        if not (run_dir / "summary.csv").exists():
            sys.exit(f"No {run_dir / 'summary.csv'} yet. Run the sweep once without --report-only.")
        print(f"Saved {save_report(read_summary(run_dir / 'summary.csv'), run_dir)}")
        return
    if not source.exists():
        sys.exit(f"{source} does not exist")
    crop = None if args.static_crop == ["none"] else tuple(float(c) for c in args.static_crop)
    if crop is not None and len(crop) != 4:
        parser.error("--static-crop takes X_MIN X_MAX Y_MIN Y_MAX, or none")
    if not 0 < len(args.thresholds) < 0x80 or not all(0 < t <= 1 for t in args.thresholds):
        parser.error("--thresholds takes up to 127 values in (0, 1]")
    weights = args.weights or sorted(WEIGHTS_DIR.glob("*.pt"))
    if not weights:
        sys.exit(f"No weights found in {WEIGHTS_DIR}")

    decisions = sweep(
        source, weights, args.thresholds, args.config, crop, args.resize, args.stride, args.max_frames, args.device
    )
    rows = [summarize(name, label, entries, args.stride) for (name, label), entries in decisions.items()]
    if not rows or not rows[0]["frames"]:
        sys.exit(f"No frames read from {source}")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "frames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weights", "threshold", "frame", "time_s", "ok", "reason", "road_pixels", "area_fraction"])
        for (name, label), entries in decisions.items():
            for index, seconds, ok, reason, road_pixels, area in entries:
                writer.writerow([name, label, index, f"{seconds:.3f}", int(ok), reason, road_pixels, f"{area:.4f}"])
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = ["weights", "threshold", "reject_rate", "longest_reject_streak", "longest_reject_s"]
    print(f"\nGate rejections on {source.name}:")
    print("  ".join(f"{c:>22}" for c in columns))
    for row in rows:
        print("  ".join(f"{row[c]:>22.4f}" if isinstance(row[c], float) else f"{row[c]:>22}" for c in columns))
    print(f"Saved {run_dir / 'summary.csv'}, {run_dir / 'frames.csv'}")
    print(f"Saved {save_report(rows, run_dir)}")


if __name__ == "__main__":
    main()
