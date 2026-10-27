#!/usr/bin/env python3
"""Download a test survey for occupancy (D18) that SpeciesNet hasn't seen, from Nkhotakota Camera
Traps: every image that every camera took in a stretch of weeks, each camera in a folder of its
own, with its capture time in its EXIF.

Caltech Camera Traps (fetch_cct_occupancy.py) is probably in SpeciesNet's training data, so its
suggestions there are a best case. Nkhotakota Camera Traps has 324,635 images from 69 camera
locations in Nkhotakota Wildlife Reserve, Malawi, 2018-2023, labeled per image with counts
(and with people's boxes on about 40,700 images of its train and val splits, September 2019 to
August 2020: none in the default window, which is in its test split; survey.json keeps only the
labels) (Appel et al. 2025, Ecological Applications 35(6): e70096), released under
CDLA-Permissive-1.0:
https://lila.science/datasets/nkhotakota-camera-traps. LILA posted it on 2025-06-24, after
SpeciesNet's weights were last changed (4.0.2 and 4.0.3 changed only metadata), and it was
labeled in Njobvu-AI rather than Wildlife Insights. Unlike Oregon Critters, it keeps every
image the cameras took, empty ones included, each with its camera, so it can stand for a survey.

The default window, eight weeks from 2020-09-22, is the one with the most cameras running: 60
cameras whose memory card spans all of it, and a few more running part of it, about 41,000
images. Images are shrunk to --max-width pixels (the detector looks at 640) and saved at JPEG
--quality, to save space: about 9.4 GB at the defaults (the originals are 1920 x 1440, about
1.1 MB each; quality 80 would be 11.6 GB).

    uv run --with pillow python datasets/fetch_nkhotakota_occupancy.py
        [--start 2020-09-22] [--weeks 8] [--max-width 1024] [--quality 70] [--data-dir DIR]

Writes to <data-dir>/datasets/nkhotakota_occupancy/: images/<camera>/, survey.json (each
image's camera, time, labels and counts; each camera's running time in the window, from its
memory cards' first and last images), truth.csv (each species' cameras by the dataset's labels)
and a README.txt. Run again, it only fetches what's missing. Needs Pillow (the backend's
environment has it).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cct_occupancy import fetch_bytes, write_image
from fetch_cct_sample import default_data_dir, download

LABELS_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/nkhotakota-camera-traps/"
    "nkhotakota_camera_traps.json.zip"
)
IMAGE_URL = "https://storage.googleapis.com/public-datasets-lila/nkhotakota-camera-traps/{path}"
NOT_ANIMALS = {"empty", "human"}
# For the clock check: species active by day, whose images at night suggest a camera's clock is
# hours out. (Occasions are a week long, so hours don't matter to occupancy; days would.)
DAY_ACTIVE = {
    "yellow_baboon", "vervet_monkey", "blue_monkey", "common_warthog", "impala", "sable",
    "plains_zebra", "helmeted_guineafowl", "southern_ground_hornbill",
}  # fmt: skip
NIGHT_HOURS = set(range(19, 24)) | set(range(0, 5))
README = """\
A test survey for WildObserve's occupancy that SpeciesNet hasn't seen
(datasets/fetch_nkhotakota_occupancy.py): every image that {cameras} Nkhotakota Camera
Traps cameras took from {first} to {last}, a folder per camera, each image's capture time
written into its EXIF. Nkhotakota Wildlife Reserve, Malawi. CDLA-Permissive-1.0,
https://lila.science/datasets/nkhotakota-camera-traps. Cite: Appel CL, Subramanian A, Koning
JS, Ngosi M, Sullivan CM, Levi T, Lesmeister DB (2025). Developing custom computer vision models
with Njobvu-AI. Ecological Applications 35(6): e70096.

images/      add this folder to a project: each camera's folder is a deployment
truth.csv    each species' cameras by the dataset's own labels, to check against
survey.json  each image's camera, capture time, labels and counts, and each camera's running
             time in the window (from its memory cards' first and last images)

No site covariates: the dataset records none.

Clock check: day-active species ({day_active}) were photographed at night (19:00-05:00) in
{night} of {day} images. Cameras where that share is over 30% (of 5 or more), whose clocks may
be hours out: {suspect}. Occasions are a week long, so that hardly matters to occupancy.
{failed}"""


def file_url(file_name: str) -> str:
    return IMAGE_URL.format(path=urllib.parse.quote(file_name))


def local_name(file_name: str) -> str:
    """The image's file name, its extension in lower case: T24-B-...__12-15-32(1).jpg."""
    return Path(file_name).stem + ".jpg"


def fetch(image: dict, dest: Path, max_width: int, quality: int) -> str | None:
    """Download an image, shrink it, and save it with its capture time in its EXIF. Returns
    why it couldn't, or None."""
    if dest.exists() and dest.stat().st_size > 0:
        return None
    try:
        taken = datetime.fromisoformat(image["datetime"])
        write_image(fetch_bytes(file_url(image["file_name"])), dest, taken, max_width, quality)
    except (OSError, RuntimeError) as error:  # a failed download, or a file Pillow can't read
        return f"{image['file_name']}: {error}"
    return None


def running(cards: list[tuple[datetime, datetime]], start: datetime, end: datetime) -> dict:
    """A camera's running time in the window: from the earliest first image of the memory cards
    that overlap it to the latest last image, cut to the window. Gaps between cards aren't
    counted out (the app can't take them either)."""
    overlapping = [(first, last) for first, last in cards if first < end and last >= start]
    return {
        "from": max(start, min(first for first, _ in overlapping)).isoformat(sep=" "),
        "to": min(end - timedelta(seconds=1), max(last for _, last in overlapping)).isoformat(
            sep=" "
        ),
        "whole_window": any(first <= start and last >= end for first, last in overlapping),
    }


def clock_check(survey: list[dict]) -> tuple[int, int, list[str]]:
    """How many images of day-active species were taken at night, of how many, and the cameras
    where that share is over 30% (of 5 or more)."""
    day, night = defaultdict(int), defaultdict(int)
    for item in survey:
        if DAY_ACTIVE & set(item["labels"]):
            day[item["camera"]] += 1
            night[item["camera"]] += int(item["captured"][11:13]) in NIGHT_HOURS
    suspect = [
        f"{name} ({night[name]} of {day[name]})"
        for name in sorted(day)
        if day[name] >= 5 and night[name] / day[name] > 0.3
    ]
    return sum(night.values()), sum(day.values()), suspect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--start", default="2020-09-22", help="the window's first day")
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.data_dir / "datasets" / "nkhotakota_occupancy"
    root.mkdir(parents=True, exist_ok=True)
    labels_zip = download(LABELS_URL, root / "nkhotakota_camera_traps.json.zip")
    with zipfile.ZipFile(labels_zip) as archive:
        labels = json.loads(archive.read(archive.namelist()[0]))
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    labeled: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for annotation in labels["annotations"]:
        name = categories[annotation["category_id"]]
        labeled[annotation["image_id"]].add(name)
        if "count" in annotation:
            image = counts[annotation["image_id"]]
            image[name] = max(image.get(name, 0), annotation["count"])

    dated = [image for image in labels["images"] if image.get("datetime")]
    cards: dict[str, dict[str, list[datetime]]] = defaultdict(lambda: defaultdict(list))
    for image in dated:
        card = image["file_name"].split("/")[1]
        cards[image["location"]][card].append(datetime.fromisoformat(image["datetime"]))
    spans = {
        location: [(min(times), max(times)) for times in by_card.values()]
        for location, by_card in cards.items()
    }

    start = datetime.fromisoformat(args.start)
    end = start + timedelta(weeks=args.weeks)
    chosen = sorted(
        (image for image in dated if start <= datetime.fromisoformat(image["datetime"]) < end),
        key=lambda image: (image["location"], image["datetime"], image["file_name"]),
    )
    for folder in {image["location"] for image in chosen}:
        (root / "images" / folder).mkdir(parents=True, exist_ok=True)
    print(
        f"{len(chosen)} images from {len({i['location'] for i in chosen})} cameras, "
        f"{start:%Y-%m-%d} to {end - timedelta(days=1):%Y-%m-%d}",
        flush=True,
    )

    def path(image: dict) -> Path:
        return root / "images" / image["location"] / local_name(image["file_name"])

    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        work = pool.map(
            lambda image: fetch(image, path(image), args.max_width, args.quality), chosen
        )
        for done, (image, problem) in enumerate(zip(chosen, work, strict=True), start=1):
            if problem:
                failed[image["id"]] = problem
            if done % 1000 == 0 or done == len(chosen):
                print(f"images: {done}/{len(chosen)}, {len(failed)} failed", flush=True)
    for problem in list(failed.values())[:10]:
        print(f"  {problem}")
    missing = [image for image in chosen if image["id"] in failed]
    chosen = [image for image in chosen if image["id"] not in failed]  # run again to retry

    survey = [
        {
            "id": image["id"],
            "file": str(path(image).relative_to(root)),
            "camera": image["location"],
            "captured": image["datetime"],
            "labels": sorted(labeled[image["id"]]),
            "counts": counts.get(image["id"], {}),
        }
        for image in chosen
    ]
    cameras = sorted({item["camera"] for item in survey})
    (root / "survey.json").write_text(
        json.dumps(
            {
                "dataset": "Nkhotakota Camera Traps",
                "license": "CDLA-Permissive-1.0",
                "source": "https://lila.science/datasets/nkhotakota-camera-traps",
                "window": [f"{start:%Y-%m-%d}", f"{end - timedelta(days=1):%Y-%m-%d}"],
                "cameras": {name: running(spans[name], start, end) for name in cameras},
                "images": survey,
            },
            indent=1,
        )
    )

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

    night, day, suspect = clock_check(survey)
    why_missing = defaultdict(int)
    for image in missing:
        why_missing[", ".join(sorted(labeled[image["id"]])) or "unlabeled"] += 1
    (root / "README.txt").write_text(
        README.format(
            cameras=len(cameras),
            first=f"{start:%Y-%m-%d}",
            last=f"{end - timedelta(days=1):%Y-%m-%d}",
            day_active=", ".join(sorted(DAY_ACTIVE)),
            night=night,
            day=day,
            suspect=", ".join(suspect) or "none",
            failed=(
                f"\n{len(missing)} images in the labels couldn't be downloaded, by label: "
                + "; ".join(f"{label} {n}" for label, n in sorted(why_missing.items()))
                + ".\n"
                if missing
                else ""
            ),
        )
    )
    size = sum(p.stat().st_size for p in (root / "images").rglob("*.jpg"))
    print(
        f"wrote {root}: {len(survey)} images ({size / 1e9:.1f} GB), {len(cameras)} cameras, "
        f"{len(seen)} species; truth.csv has each one's cameras by the dataset's labels"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
