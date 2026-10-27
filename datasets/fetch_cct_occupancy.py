#!/usr/bin/env python3
"""Download a test survey for occupancy (D18) from Caltech Camera Traps: every image that every
camera took in a stretch of weeks, each camera in a folder of its own, with its capture time.

CCT's images carry no capture time of their own (no EXIF), and the CCT sample sits in one folder,
so in the app it's one deployment with file dates: useless for occupancy. Its metadata has each
image's time and camera location, though. This writes each image's time into its EXIF
(DateTimeOriginal) and puts each camera's images in a folder (cct_038, say), so the app sees
each camera as a deployment, and each deployment as a site. Images are shrunk on the way to
--max-width pixels (the detector looks at 640) and saved at JPEG --quality, to save space:
CCT's grainy images compress poorly.

CCT: 243k images from 140 camera locations in the southwestern US, CDLA-Permissive-1.0:
https://lila.science/datasets/caltech-camera-traps. The default window, eight weeks from
2014-08-01, is the one with the most cameras running: 60 cameras, about 15,000 images, about
2.7 GB at the default size.

    uv run --with pillow python datasets/fetch_cct_occupancy.py [--start 2014-08-01]
        [--weeks 8] [--max-width 1024] [--quality 85] [--data-dir DIR]

Writes to <data-dir>/datasets/cct_occupancy/: images/<camera>/, survey.json (each image's
camera, time and CCT labels), truth.csv (each species' cameras by CCT's labels, to check the
app's naive occupancy against), example_covariates.csv (site covariates to import, made up:
CCT has none; cameras where deer were seen are nearer water, on purpose, so the app should find
that effect), and a README.txt saying so. Run again, it only fetches what's missing. Needs
Pillow (the backend's environment has it).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cct_sample import IMAGE_URL, LABELS_URL, default_data_dir, download
from PIL import Image

DATE = "%Y-%m-%d %H:%M:%S"  # CCT's date_captured
NOT_ANIMALS = {"empty", "car"}
SEED = 20261008
README = """\
A test survey for WildObserve's occupancy (datasets/fetch_cct_occupancy.py): every image
that {cameras} Caltech Camera Traps cameras took from {first} to {last}, a folder per camera,
each image's capture time written into its EXIF. CCT: CDLA-Permissive-1.0,
https://lila.science/datasets/caltech-camera-traps.

images/                 add this folder to a project: each camera's folder is a deployment
truth.csv               each species' cameras by CCT's own labels, to check against
example_covariates.csv  site covariates to import (Occupancy > Site covariates...). MADE UP:
                        CCT records none. Cameras where CCT saw deer tend to be nearer water,
                        on purpose: tried one at a time, deer occupancy falls with
                        distance_to_water_m (about -2.3 per standard deviation, from CCT's own
                        labels), while elevation_m, habitat and trail are random and their
                        intervals span 0. All four at once on 60 sites is too many: chance
                        effects can then look real (habitat, here), as they would in a real
                        survey.
survey.json             each image's camera, capture time and CCT labels
"""


def camera(location: str) -> str:
    """A camera's folder: cct_038 for location 38, so they sort in order."""
    return f"cct_{int(location):03d}" if location.isdigit() else f"cct_{location}"


def fetch(image: dict, dest: Path, max_width: int, quality: int) -> str | None:
    """Download an image, shrink it, and save it with its capture time in its EXIF. Returns
    why it couldn't, or None."""
    if dest.exists() and dest.stat().st_size > 0:
        return None
    try:
        save(image, dest, max_width, quality)
    except (OSError, RuntimeError) as error:  # a failed download, or a file Pillow can't read
        return f"{image['file_name']}: {error}"
    return None


def save(image: dict, dest: Path, max_width: int, quality: int) -> None:
    url = IMAGE_URL.format(file_name=image["file_name"])
    taken = datetime.strptime(image["date_captured"], DATE)
    write_image(fetch_bytes(url), dest, taken, max_width, quality)


def fetch_bytes(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except OSError as error:
            if attempt == 2:
                raise RuntimeError(f"download failed: {url}: {error}") from error
    raise AssertionError("unreachable")


def write_image(data: bytes, dest: Path, taken: datetime, max_width: int, quality: int) -> None:
    """Shrink an image to max_width pixels wide and save it as a JPEG with its capture time in
    its EXIF (DateTimeOriginal), via a .part file so an interrupted run leaves no half image."""
    picture = Image.open(io.BytesIO(data))
    picture.load()
    if picture.width > max_width:
        height = round(picture.height * max_width / picture.width)
        picture = picture.resize((max_width, height), Image.Resampling.LANCZOS)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")  # DateTimeOriginal
    part = dest.with_suffix(".part")
    picture.convert("RGB").save(part, "JPEG", quality=quality, exif=exif)
    part.replace(dest)


def write_example_covariates(path: Path, cameras: list[str], deer: set[str]) -> None:
    """Made-up site covariates, the same every time: distance to water depends on whether deer
    were seen (nearer where they were); elevation, habitat and trail are random."""
    rng = random.Random(SEED)
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["place", "distance_to_water_m", "elevation_m", "habitat", "trail"])
        for name in cameras:
            water = rng.uniform(50, 1000) if name in deer else rng.uniform(200, 1500)
            writer.writerow(
                [
                    name,
                    int(round(water, -1)),
                    round(rng.uniform(250, 1400)),
                    rng.choice(["chaparral", "grassland", "oak woodland"]),
                    "on trail" if rng.random() < 0.4 else "off trail",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--start", default="2014-08-01", help="the window's first day")
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    labels_zip = download(
        LABELS_URL, args.data_dir / "datasets" / "cct" / "caltech_camera_traps.json.zip"
    )
    with zipfile.ZipFile(labels_zip) as archive:
        labels = json.loads(archive.read(archive.namelist()[0]))
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    labeled: dict[str, set[str]] = defaultdict(set)
    for annotation in labels["annotations"]:
        labeled[annotation["image_id"]].add(categories[annotation["category_id"]])

    start = datetime.fromisoformat(args.start)
    end = start + timedelta(weeks=args.weeks)

    def in_window(image: dict) -> bool:
        try:
            return start <= datetime.strptime(image["date_captured"], DATE) < end
        except (KeyError, ValueError):
            return False

    chosen = sorted(
        (image for image in labels["images"] if in_window(image)),
        key=lambda image: (camera(image["location"]), image["date_captured"]),
    )
    root = args.data_dir / "datasets" / "cct_occupancy"
    for folder in {camera(image["location"]) for image in chosen}:
        (root / "images" / folder).mkdir(parents=True, exist_ok=True)
    print(
        f"{len(chosen)} images from {len({i['location'] for i in chosen})} cameras, "
        f"{start:%Y-%m-%d} to {end - timedelta(days=1):%Y-%m-%d}",
        flush=True,
    )

    def path(image: dict) -> Path:
        return root / "images" / camera(image["location"]) / Path(image["file_name"]).name

    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        work = pool.map(
            lambda image: fetch(image, path(image), args.max_width, args.quality), chosen
        )
        for done, (image, problem) in enumerate(zip(chosen, work, strict=True), start=1):
            if problem:
                failed[image["id"]] = problem
            if done % 500 == 0 or done == len(chosen):
                print(f"images: {done}/{len(chosen)}, {len(failed)} failed", flush=True)
    for problem in list(failed.values())[:10]:
        print(f"  {problem}")
    chosen = [image for image in chosen if image["id"] not in failed]  # run again to retry

    survey = [
        {
            "id": image["id"],
            "file": str(path(image).relative_to(root)),
            "camera": camera(image["location"]),
            "captured": image["date_captured"],
            "sequence": image.get("seq_id"),
            "labels": sorted(labeled[image["id"]]),
        }
        for image in chosen
    ]
    (root / "survey.json").write_text(
        json.dumps(
            {
                "dataset": "Caltech Camera Traps",
                "license": "CDLA-Permissive-1.0",
                "source": "https://lila.science/datasets/caltech-camera-traps",
                "window": [f"{start:%Y-%m-%d}", f"{end - timedelta(days=1):%Y-%m-%d}"],
                "images": survey,
            },
            indent=1,
        )
    )

    cameras = sorted({item["camera"] for item in survey})
    seen: dict[str, set[str]] = defaultdict(set)
    for item in survey:
        for name in set(item["labels"]) - NOT_ANIMALS:
            seen[name].add(item["camera"])
    with (root / "truth.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["species", "cameras", "of_cameras", "naive_occupancy", "seen_at"])
        for name, where in sorted(seen.items(), key=lambda item: (-len(item[1]), item[0])):
            share = round(len(where) / len(cameras), 3)
            writer.writerow([name, len(where), len(cameras), share, " ".join(sorted(where))])
    write_example_covariates(root / "example_covariates.csv", cameras, seen.get("deer", set()))
    (root / "README.txt").write_text(
        README.format(
            cameras=len(cameras),
            first=f"{start:%Y-%m-%d}",
            last=f"{end - timedelta(days=1):%Y-%m-%d}",
        )
    )
    size = sum(p.stat().st_size for p in (root / "images").rglob("*.jpg"))
    print(
        f"wrote {root}: {len(survey)} images ({size / 1e9:.1f} GB), {len(cameras)} cameras, "
        f"{len(seen)} species; truth.csv has each one's cameras by CCT's labels"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
