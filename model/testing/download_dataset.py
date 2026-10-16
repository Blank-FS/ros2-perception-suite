"""Fetch a Roboflow semantic segmentation export and lay it out for Ultralytics.

Roboflow's "Semantic Segmentation Masks" export (png-mask-semantic) puts each
image next to its mask, as <stem>.jpg and <stem>_mask.png, with the class names
in _classes.csv. Ultralytics' semantic validator wants images/<split>/<stem>.jpg
and masks/<split>/<stem>.png instead, so this rearranges the export and writes
a data.yaml beside it.

    # Download through the API, with the key (and optionally the dataset) in
    # model/testing/.env; see model/testing/.env.example
    python model/testing/download_dataset.py
    python model/testing/download_dataset.py --workspace WS --project PROJ --version 3

    # Or use a zip exported by hand from the Roboflow UI
    python model/testing/download_dataset.py --zip ~/Downloads/export.zip --name demo-route-v3
"""

import argparse
import csv
import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

API_URL = "https://api.roboflow.com"
EXPORT_FORMAT = "png-mask-semantic"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
ENV_FILE = Path(__file__).resolve().parent / ".env"
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # Roboflow name -> Ultralytics name
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fetch_export_zip(workspace, project, version, api_key, timeout=600):
    """Ask Roboflow for the export, wait for it to be generated, and download it."""
    url = f"{API_URL}/{workspace}/{project}/{version}/{EXPORT_FORMAT}"
    deadline = time.monotonic() + timeout
    while True:
        response = requests.get(url, params={"api_key": api_key}, timeout=60)
        if response.status_code in (401, 403):
            sys.exit(f"Roboflow refused the API key ({response.status_code}).")
        if response.status_code == 404:
            sys.exit(f"Roboflow has no {workspace}/{project}/{version}. Check the names in the dataset URL.")
        response.raise_for_status()
        link = response.json().get("export", {}).get("link")
        if link:
            break
        if time.monotonic() > deadline:
            sys.exit(f"Roboflow did not finish generating the export within {timeout}s.")
        print("Waiting for Roboflow to generate the export...")
        time.sleep(5)

    print(f"Downloading {workspace}/{project}/{version} ({EXPORT_FORMAT})...")
    with requests.get(link, stream=True, timeout=60) as download:
        download.raise_for_status()
        buffer = io.BytesIO()
        for chunk in download.iter_content(chunk_size=1 << 20):
            buffer.write(chunk)
    return buffer


def read_classes(csv_path):
    """Return {pixel value: class name} from a Roboflow _classes.csv."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    names = {int(row[0]): row[1].strip() for row in rows[1:] if row}
    if sorted(names) != list(range(len(names))):
        sys.exit(f"{csv_path}: pixel values {sorted(names)} are not 0..N-1, which data.yaml cannot describe directly.")
    return names


def organize(export_dir, out_dir):
    """Move a png-mask-semantic export into images/ and masks/ and return the class names."""
    names, splits = None, {}
    for rf_split, split in SPLITS.items():
        src = export_dir / rf_split
        if not src.is_dir():
            continue
        split_names = read_classes(src / "_classes.csv")
        if names is not None and split_names != names:
            sys.exit(f"{rf_split}/_classes.csv lists different classes from the other splits.")
        names = split_names

        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "masks" / split).mkdir(parents=True, exist_ok=True)
        count = 0
        for image in sorted(src.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES or image.stem.endswith("_mask"):
                continue
            mask = src / f"{image.stem}_mask.png"
            if not mask.exists():
                print(f"  skipping {image.name}: no {mask.name}")
                continue
            shutil.move(image, out_dir / "images" / split / image.name)
            shutil.move(mask, out_dir / "masks" / split / f"{image.stem}.png")
            count += 1
        splits[split] = count
        print(f"  {split}: {count} images")

    if names is None:
        sys.exit(f"No train/valid/test folders in the export. Was it exported as '{EXPORT_FORMAT}'?")
    return names, splits


def warn_augmented_copies(out_dir):
    """Warn if the version holds several augmented copies of the same labeled image.

    Roboflow names every image <original>_<ext>.rf.<hash>.<ext>, and augmentation
    adds copies that share the <original> part. They would count each labeled
    image several times when scoring the whole dataset.
    """
    originals = Counter(p.name.split(".rf.")[0] for p in (out_dir / "images").rglob("*") if ".rf." in p.name)
    copies = sum(n - 1 for n in originals.values() if n > 1)
    if copies:
        print(
            f"WARNING: {copies} images are augmented copies of others. Generate a version without "
            "augmentation to evaluate each labeled image once."
        )


def main():
    load_dotenv(ENV_FILE)  # variables already set in the shell take precedence
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workspace",
        default=os.environ.get("ROBOFLOW_WORKSPACE"),
        help="Roboflow workspace slug from the dataset URL (default: $ROBOFLOW_WORKSPACE)",
    )
    parser.add_argument(
        "--project", default=os.environ.get("ROBOFLOW_PROJECT"), help="Roboflow project slug (default: $ROBOFLOW_PROJECT)"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=os.environ.get("ROBOFLOW_VERSION") or None,
        help="Dataset version number (default: $ROBOFLOW_VERSION)",
    )
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="Default: $ROBOFLOW_API_KEY")
    parser.add_argument("--zip", type=Path, help="A png-mask-semantic export zip downloaded by hand, instead of the API")
    parser.add_argument("--name", help="Folder name under model/testing/datasets (default: <project>-v<version>)")
    parser.add_argument("--force", action="store_true", help="Replace the dataset folder if it already exists")
    args = parser.parse_args()

    if args.zip:
        name = args.name or args.zip.stem
        source = args.zip
    else:
        if not (args.workspace and args.project and args.version):
            parser.error(
                "--workspace, --project and --version (or ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT and "
                "ROBOFLOW_VERSION in .env) are required unless --zip is given"
            )
        if not args.api_key:
            parser.error(f"set ROBOFLOW_API_KEY in {ENV_FILE} (see .env.example next to it) or pass --api-key")
        name = args.name or f"{args.project}-v{args.version}"
        source = None

    out_dir = DATASETS_DIR / name
    if out_dir.exists():
        if not args.force:
            sys.exit(f"{out_dir} already exists. Pass --force to replace it.")
        shutil.rmtree(out_dir)

    if source is None:
        source = fetch_export_zip(args.workspace, args.project, args.version, args.api_key)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(source) as archive:
            archive.extractall(tmp)
        out_dir.mkdir(parents=True)
        names, splits = organize(Path(tmp), out_dir)
    warn_augmented_copies(out_dir)

    data = {"path": str(out_dir), "masks_dir": "masks"}
    data.update({split: f"images/{split}" for split in splits})
    # Ultralytics requires train and val keys even when only evaluating, so an
    # export with only a test split points them at it.
    for key in ("train", "val"):
        data.setdefault(key, data.get("test") or data.get("val") or data.get("train"))
    data["names"] = names
    if args.workspace:
        data["roboflow"] = {"workspace": args.workspace, "project": args.project, "version": args.version}
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Wrote {out_dir / 'data.yaml'}")
    print(f"Classes: {names}")


if __name__ == "__main__":
    main()
