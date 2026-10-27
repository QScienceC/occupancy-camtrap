#!/usr/bin/env python3
"""Download a small, reproducible sample of Caltech Camera Traps (CCT) for M0 tests.

CCT has 243k images from 140 camera locations in the southwestern US, released under
CDLA-Permissive-1.0: https://lila.science/datasets/caltech-camera-traps

The sample uses a fixed seed, so every machine (the Mac, the Linux PC) gets the same images:
animal images that have ground-truth boxes, spread across species and camera locations,
plus empty images for measuring false detections.

    python3 datasets/fetch_cct_sample.py [--data-dir DIR] [--per-category 30] [--empty 200]
        [--manifest sample.json]

Writes to <data-dir>/datasets/cct/: the metadata files, images/, and sample.json (the
manifest the benchmarks read). The data dir defaults to $WILDOBSERVE_DATA_DIR, then
~/data/wildobserve. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://storage.googleapis.com/public-datasets-lila"
LABELS_URL = f"{BASE}/caltechcameratraps/labels/caltech_camera_traps.json.zip"
BOXES_URL = f"{BASE}/caltechcameratraps/labels/caltech_bboxes_20200316.json"
SPLITS_URL = f"{BASE}/caltechcameratraps/CaltechCameraTrapsSplits_v0.json"
IMAGE_URL = f"{BASE}/caltech-unzipped/cct_images/{{file_name}}"
SEED = 20260924

# Categories with too few boxed images to sample fairly, or that aren't animals or vehicles.
SKIP_CATEGORIES = {"empty", "insect", "bat", "badger", "mountain_lion"}


def default_data_dir() -> Path:
    return Path(os.environ.get("WILDOBSERVE_DATA_DIR", Path.home() / "data" / "wildobserve"))


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)
            return dest
        except OSError as error:
            if attempt == 2:
                raise RuntimeError(f"download failed: {url}: {error}") from error
    return dest


def load_metadata(root: Path) -> tuple[dict, dict, dict]:
    labels_zip = download(LABELS_URL, root / "caltech_camera_traps.json.zip")
    with zipfile.ZipFile(labels_zip) as archive:
        labels = json.loads(archive.read(archive.namelist()[0]))
    boxes = json.loads(download(BOXES_URL, root / "caltech_bboxes_20200316.json").read_text())
    splits = json.loads(download(SPLITS_URL, root / "CaltechCameraTrapsSplits_v0.json").read_text())
    return labels, boxes, splits


def choose_sample(
    labels: dict, boxes: dict, splits: dict, per_category: int, empty: int
) -> list[dict]:
    """Pick animal images round-robin across camera locations within each category."""
    rng = random.Random(SEED)
    split_of = {loc: name for name in ("train", "val") for loc in splits["splits"][name]}
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    images = {image["id"]: image for image in labels["images"]}

    boxes_by_image: dict[str, list[dict]] = defaultdict(list)
    for ann in boxes["annotations"]:
        boxes_by_image[ann["image_id"]].append(ann)

    by_category: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for image_id, anns in sorted(boxes_by_image.items()):
        names = {categories[ann["category_id"]] for ann in anns}
        if len(names) == 1 and image_id in images:
            (name,) = names
            if name not in SKIP_CATEGORIES:
                by_category[name][images[image_id]["location"]].append(image_id)

    chosen = []
    for name in sorted(by_category):
        locations = by_category[name]
        for image_ids in locations.values():
            rng.shuffle(image_ids)
        order = sorted(locations)
        rng.shuffle(order)
        picked: list[str] = []
        while len(picked) < per_category and any(locations[loc] for loc in order):
            for loc in order:
                if locations[loc] and len(picked) < per_category:
                    picked.append(locations[loc].pop())
        chosen += [(image_id, name) for image_id in picked]

    labeled = defaultdict(set)
    for ann in labels["annotations"]:
        labeled[ann["image_id"]].add(categories[ann["category_id"]])
    empties = sorted(i for i, names in labeled.items() if names == {"empty"} and i in images)
    chosen += [(image_id, "empty") for image_id in rng.sample(empties, empty)]

    sample = []
    for image_id, name in chosen:
        image = images[image_id]
        sample.append(
            {
                "id": image_id,
                "file": f"images/{image['file_name']}",
                "category": name,
                "location": image["location"],
                "split": split_of.get(image["location"], "other"),
                "width": image["width"],
                "height": image["height"],
                "captured": image.get("date_captured"),
                "sequence": image.get("seq_id"),
                "boxes": [ann["bbox"] for ann in boxes_by_image.get(image_id, [])],
            }
        )
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--per-category", type=int, default=30)
    parser.add_argument("--empty", type=int, default=200)
    parser.add_argument("--manifest", default="sample.json", help="output file name")
    args = parser.parse_args()

    root = args.data_dir / "datasets" / "cct"
    (root / "images").mkdir(parents=True, exist_ok=True)
    labels, boxes, splits = load_metadata(root)
    sample = choose_sample(labels, boxes, splits, args.per_category, args.empty)

    def fetch(item: dict) -> None:
        name = Path(item["file"]).name
        download(IMAGE_URL.format(file_name=name), root / item["file"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, _ in enumerate(pool.map(fetch, sample), start=1):
            if done % 100 == 0 or done == len(sample):
                print(f"images: {done}/{len(sample)}", flush=True)

    manifest = {
        "dataset": "Caltech Camera Traps",
        "license": "CDLA-Permissive-1.0",
        "source": "https://lila.science/datasets/caltech-camera-traps",
        "seed": SEED,
        "boxes": "COCO [x, y, width, height] in pixels",
        "images": sample,
    }
    (root / args.manifest).write_text(json.dumps(manifest, indent=1))
    counts = defaultdict(int)
    for item in sample:
        counts[item["category"]] += 1
    print(f"wrote {root / args.manifest}: {len(sample)} images, {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
