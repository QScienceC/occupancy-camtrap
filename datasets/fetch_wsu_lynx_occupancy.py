#!/usr/bin/env python3
"""Download a third test survey for occupancy (D18), from WSU Lynx: every image that a random
sample of its cameras took in a stretch of weeks, each camera in a folder of its own, with its
capture time in its EXIF.

WSU Lynx has 1,205,453 images (280,485 sequences) from 637 camera locations in north-central and
northeastern Washington, USA, 2016-2017, snow-off months, in grid cells of 2 to 9 cameras at
least 1 km apart (Cell253Camera5), labeled by sequence in 26 categories, with empty sequences
kept and images of people and vehicles removed (Thornton et al. 2026, Journal of Applied
Ecology, doi:10.1111/1365-2664.70370, whose occupancy models from SpeciesNet with no review
matched the experts' for most species), released under CDLA-Permissive-1.0:
https://lila.science/datasets/wsu-lynx. LILA posted it on 2026-02-14, and SpeciesNet wasn't
trained on it: the last LILA dataset in its training was Orinoquia Camera Traps, of January 2022
(D. Morris, Google, pers. comm., 2026). Its fauna is North America's, which SpeciesNet
knows well: the case between CCT (seen) and Nkhotakota (unseen, and named by related species
from elsewhere).

The default window, eight weeks from 2016-07-10, is the one with the most cameras running all
of it in 2016 (103); --cameras of them (60) are drawn at random with --seed, about 51,000 images.
A camera runs all of the window if that year's first image at its location comes before the
window and the last after it. Images are shrunk to --max-width pixels (the detector looks at
640) and saved at JPEG --quality (the originals are about 0.7 MB each).

    uv run --with pillow python datasets/fetch_wsu_lynx_occupancy.py
        [--start 2016-07-10] [--weeks 8] [--cameras 60] [--seed 1] [--max-width 1024]
        [--quality 70] [--data-dir DIR]

Writes to <data-dir>/datasets/wsu_lynx_occupancy/: images/<camera>/, survey.json (each image's
camera, time and labels, each camera's running time), truth.csv (each species' cameras by the
dataset's labels) and a README.txt. The categories are renamed from scientific names to
lowercase common ones (lynx canadensis is canada_lynx). "unknown" stays: an animal the dataset's
people didn't name. Run again, it only fetches what's missing. Needs Pillow (the backend's
environment has it).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/wsu-lynx/wsu-lynx.26.02.13.1705.zip"
)
IMAGE_URL = "https://storage.googleapis.com/public-datasets-lila/wsu-lynx/{path}"
NAMES = {  # the dataset's categories, as the survey names them
    "alces alces": "moose",
    "aves": "bird",
    "canis latrans": "coyote",
    "canis lupus": "gray_wolf",
    "cervus canadensis": "elk",
    "domestic cattle": "cattle",
    "domestic horse": "horse",
    "erethizon dorsatum": "porcupine",
    "lepus americanus": "snowshoe_hare",
    "lynx canadensis": "canada_lynx",
    "lynx rufus": "bobcat",
    "marmota caligata": "hoary_marmot",
    "marmota flaviventris": "yellow_bellied_marmot",
    "martes americana": "american_marten",
    "mephitis mephitis": "striped_skunk",
    "odocoileus hemionus": "mule_deer",
    "odocoileus virginianus": "white_tailed_deer",
    "oreamnos americanus": "mountain_goat",
    "ovis canadensis": "bighorn_sheep",
    "procyon lotor": "raccoon",
    "puma concolor": "cougar",
    "small mammal": "small_mammal",
    "taxidea taxus": "american_badger",
    "ursus americanus": "black_bear",
    "empty": "empty",
    "unknown": "unknown",
}
NOT_ANIMALS = {"empty", "unknown"}  # for truth.csv
README = """\
A test survey for WildObserve's occupancy (datasets/fetch_wsu_lynx_occupancy.py): every
image that {cameras} WSU Lynx cameras, drawn at random (seed {seed}) from the {running} that ran
from {first} to {last}, took in that time, a folder per camera, each image's capture time
written into its EXIF. North-central and northeastern Washington, USA. CDLA-Permissive-1.0,
https://lila.science/datasets/wsu-lynx. Cite: Thornton D, Morris D, King T, Perera-Romero L,
Anderson A, Garcia-Anleu R, Fitkin S, Vynne C (2026). Identification of camera trap images by
artificial intelligence and human experts produces similar multi-species occupancy models.
Journal of Applied Ecology 63(5). doi:10.1111/1365-2664.70370.

images/      add this folder to a project: each camera's folder is a deployment
truth.csv    each species' cameras by the dataset's own labels, to check against
survey.json  each image's camera, capture time and labels (by sequence: every frame of a
             sequence has its label), and each camera's running time in the window

Labels are by sequence, so a frame the animal had left still has its label. "unknown" is an
animal the dataset's people didn't name. Images of people and vehicles were removed from the
dataset. No site covariates: the dataset records none.
{failed}"""


def file_url(file_name: str) -> str:
    return IMAGE_URL.format(path=urllib.parse.quote(file_name))


def local_name(file_name: str) -> str:
    """The image's file name, its extension in lower case: IMG_0001 (3).jpg."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--start", default="2016-07-10", help="the window's first day")
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--cameras", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.data_dir / "datasets" / "wsu_lynx_occupancy"
    root.mkdir(parents=True, exist_ok=True)
    labels_zip = download(LABELS_URL, root / "wsu-lynx.zip")
    with zipfile.ZipFile(labels_zip) as archive:
        labels = json.loads(archive.read(archive.namelist()[0]))
    categories = {c["id"]: NAMES[c["name"]] for c in labels["categories"]}
    labeled: dict[str, set[str]] = defaultdict(set)
    for annotation in labels["annotations"]:
        labeled[annotation["image_id"]].add(categories[annotation["category_id"]])

    start = datetime.fromisoformat(args.start)
    end = start + timedelta(weeks=args.weeks)
    # Each location's first and last image in the window's year: when its camera ran.
    times: dict[str, list[datetime]] = defaultdict(list)
    for image in labels["images"]:
        taken = datetime.fromisoformat(image["datetime"])
        if taken.year == start.year:
            times[image["location"]].append(taken)
    running = sorted(
        location
        for location, taken in times.items()
        if min(taken) <= start and max(taken) >= end - timedelta(days=1)
    )
    if len(running) < args.cameras:
        print(f"Only {len(running)} cameras ran all of that window.")
        return 1
    drawn = set(random.Random(args.seed).sample(running, args.cameras))
    chosen = sorted(
        (
            image
            for image in labels["images"]
            if image["location"] in drawn
            and start <= datetime.fromisoformat(image["datetime"]) < end
        ),
        key=lambda image: (image["location"], image["datetime"], image["file_name"]),
    )

    def camera(image: dict) -> str:
        return Path(image["location"]).name  # Cell100Camera1

    def path(image: dict) -> Path:
        return root / "images" / camera(image) / local_name(image["file_name"])

    taken_names = defaultdict(list)
    for image in chosen:
        taken_names[path(image)].append(image["file_name"])
    clashes = [names for names in taken_names.values() if len(names) > 1]
    if clashes:
        print(f"{len(clashes)} file names would clash, e.g. {clashes[0]}")
        return 1
    for folder in {camera(image) for image in chosen}:
        (root / "images" / folder).mkdir(parents=True, exist_ok=True)
    print(
        f"{len(chosen)} images from {len(drawn)} cameras (of {len(running)} running), "
        f"{start:%Y-%m-%d} to {end - timedelta(days=1):%Y-%m-%d}",
        flush=True,
    )

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
            "camera": camera(image),
            "captured": image["datetime"],
            "sequence": image["seq_id"],
            "labels": sorted(labeled[image["id"]]),
            "counts": {},
        }
        for image in chosen
    ]
    cameras = sorted({item["camera"] for item in survey})
    whole = {
        "from": f"{start:%Y-%m-%d} 00:00:00",
        "to": f"{end - timedelta(days=1):%Y-%m-%d} 23:59:59",
        "whole_window": True,
    }
    (root / "survey.json").write_text(
        json.dumps(
            {
                "dataset": "WSU Lynx",
                "license": "CDLA-Permissive-1.0",
                "source": "https://lila.science/datasets/wsu-lynx",
                "window": [f"{start:%Y-%m-%d}", f"{end - timedelta(days=1):%Y-%m-%d}"],
                "drawn": {"seed": args.seed, "of_cameras_running": len(running)},
                "cameras": {name: whole for name in cameras},
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

    why_missing = defaultdict(int)
    for image in missing:
        why_missing[", ".join(sorted(labeled[image["id"]])) or "unlabeled"] += 1
    (root / "README.txt").write_text(
        README.format(
            cameras=len(cameras),
            seed=args.seed,
            running=len(running),
            first=f"{start:%Y-%m-%d}",
            last=f"{end - timedelta(days=1):%Y-%m-%d}",
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
