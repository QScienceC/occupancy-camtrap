#!/usr/bin/env python3
"""Download a fourth test survey for occupancy (D18), from one season of Snapshot Safari's 2024
expansion: every image that every camera running throughout a stretch of weeks took, each camera
in a folder of its own, with its capture time in its EXIF.

Snapshot Safari 2024 Expansion has 4,029,374 images from 15 projects in southern and eastern
Africa, empty ones included, labeled by sequence by Zooniverse volunteers in 151 categories
(Pardo et al. 2021, South African Journal of Science 117: 8134), released under
CDLA-Permissive-1.0: https://lila.science/datasets/snapshot-safari-2024-expansion. LILA posted it
in November 2024, and the last LILA dataset in SpeciesNet's training was Orinoquia Camera Traps,
of January 2022 (D. Morris, Google, pers. comm., 2026); its default season, the
Associated Private Nature Reserves (APNR) beside Kruger National Park, South Africa, season 4,
was never on LILA before, and SpeciesNet wasn't trained on it: of the Expansion, only the first
seasons of KRU, MTZ, CDB, ENO, KGA, and KAR, which LILA had also posted on their own, were in its
training (Dan Morris, 2026-10-24). SpeciesNet knows the region's animals, though, from those and
from Snapshot Serengeti (LILA 2019): unseen images of familiar fauna, many of the same kinds as
Nkhotakota's (baboon, common duiker, kudu, elephant), to tell whether SpeciesNet's trouble there
was Africa or particular confusions.

The metadata has no capture times, so each image's comes from its own EXIF (the first --head
bytes of the file, fetched with an HTTP range request, and kept in times.json for a rerun).
Times outside the season's year and the next are taken for a reset clock and left out, as are
cameras with more than 5% of them. The window is the --weeks (8) in which the most cameras ran
throughout (first image before it, last after it), and only those cameras' images are taken.
Images are shrunk to --max-width pixels and saved at JPEG --quality.

    uv run --with pillow python datasets/fetch_snapshot_occupancy.py
        [--season APN_S4] [--weeks 8] [--max-width 1024] [--quality 70] [--data-dir DIR]

Writes to <data-dir>/datasets/snapshot_occupancy/: images/<camera>/, survey.json, truth.csv and a
README.txt, as the other occupancy surveys. Categories are renamed as a reviewer there would
name them (NAMES): lion's sexes become lion, "baboon" chacma_baboon (the only baboon there),
"kudu" greater_kudu, "porcupine" cape_porcupine, any rhinoceros rhinoceros (the season names
none to species), and bird of prey bird; "unknown" stays, an animal the volunteers didn't name.
Run again, it only fetches what's missing. Needs Pillow (the backend's environment has it).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cct_occupancy import fetch_bytes, write_image
from fetch_cct_sample import default_data_dir, download
from PIL import Image

BASE = "https://storage.googleapis.com/public-datasets-lila/snapshot-safari-2024-expansion/"
LABELS_URL = BASE + "snapshot_safari_2024_metadata.zip"
NAMES = {  # Snapshot's categories, as a reviewer in the APNR would name them; others as they are
    "elephant": "african_savanna_elephant",
    "zebraburchells": "plains_zebra",
    "kudu": "greater_kudu",
    "wildebeestblue": "blue_wildebeest",
    "buffalo": "cape_buffalo",
    "warthog": "common_warthog",
    "duikercommongrey": "common_duiker",
    "duikerrednatal": "red_duiker",
    "hyenaspotted": "spotted_hyena",
    "baboon": "chacma_baboon",
    "rhinocerosunknown": "rhinoceros",
    "rhinoceroswhite": "rhinoceros",
    "rhinocerosblack": "rhinoceros",
    "birdother": "bird",
    "birdofprey": "bird",
    "waterbuck": "common_waterbuck",
    "lionfemale": "lion",
    "lionmale": "lion",
    "lioncub": "lion",
    "monkeyvervet": "vervet_monkey",
    "wilddog": "african_wild_dog",
    "mongoosewhitetailed": "white_tailed_mongoose",
    "mongoosebanded": "banded_mongoose",
    "mongoosedwarf": "dwarf_mongoose",
    "reedbuck": "southern_reedbuck",
    "reedbuckmountain": "mountain_reedbuck",
    "hareunknown": "hare",
    "squirreltree": "squirrel",
    "porcupine": "cape_porcupine",
    "civet": "african_civet",
    "jackalblackbacked": "black_backed_jackal",
    "genetcommonsmallspotted": "small_spotted_genet",
    "rabbitredrock": "red_rock_hare",
}
NOT_ANIMALS = {"empty", "unknown"}  # for truth.csv
FIELD = re.compile(r'\s*"(\w+)":\s*(.*?),?\s*$')
KEYS = {
    "id",
    "file_name",
    "location",
    "seq_id",
    "image_id",
    "category_id",
    "name",
}  # the rest skipped
README = """\
A test survey for WildObserve's occupancy (datasets/fetch_snapshot_occupancy.py): every
image that the {cameras} cameras of Snapshot Safari's {season} that ran throughout {first} to
{last} took in that time, a folder per camera, each image's capture time (from its own EXIF)
written into its EXIF. CDLA-Permissive-1.0,
https://lila.science/datasets/snapshot-safari-2024-expansion. Cite: Pardo LE, Bombaci SP, Huebner
S, et al. (2021). Snapshot Safari: a large-scale collaborative to monitor Africa's remarkable
biodiversity. South African Journal of Science 117(1/2): 8134.

images/      add this folder to a project: each camera's folder is a deployment
truth.csv    each species' cameras by the dataset's own labels, to check against
survey.json  each image's camera, capture time, labels and counts (by sequence), and each
             camera's running time in the window

Labels are by sequence (Zooniverse volunteers' consensus), renamed as a reviewer there would
name them: {renamed}. "unknown" is an animal the volunteers didn't name.

Clocks: {bad_times} of the season's {season_images} images had a time outside {years}, taken for a
reset clock; cameras left out for it: {bad_cameras}. {running} of {cameras_all} cameras ran
throughout the window.
{failed}"""


def stream_season(labels_zip: Path, season: str) -> tuple[list[dict], dict[str, list[str]]]:
    """The season's images and each one's labels, read line by line from the 2 GB metadata
    (pretty-printed, a field per line), so it never sits in memory whole."""
    prefix = f"{season.split('_')[0]}/{season}/"
    images: list[dict] = []
    wanted: set[str] = set()
    labels: dict[str, list[str]] = defaultdict(list)
    categories: dict[int, str] = {}
    pending: list[tuple[str, int]] = []
    section, item = None, {}
    with zipfile.ZipFile(labels_zip) as archive, archive.open(archive.namelist()[0]) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            text = line.strip()
            if text.startswith(('"images"', '"annotations"', '"categories"')):
                section = text.split('"')[1]
                continue
            if text.startswith("}"):
                if section == "images" and item.get("file_name", "").startswith(prefix):
                    images.append(item)
                    wanted.add(item["id"])
                elif section == "annotations" and item.get("image_id") in wanted:
                    pending.append((item["image_id"], item["category_id"]))
                elif section == "categories" and "name" in item:
                    categories[item["id"]] = item["name"]
                item = {}
                continue
            found = FIELD.match(line)
            if found and found.group(1) in KEYS:
                key, value = found.groups()
                item[key] = json.loads(value)
    for image_id, category in pending:
        labels[image_id].append(categories[category])
    return images, labels


def head_time(file_name: str, head: int) -> str | None:
    """The image's capture time from its EXIF (DateTimeOriginal, else DateTime), reading only
    its first `head` bytes; None if it has none there."""
    url = BASE + urllib.parse.quote(file_name)
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{head - 1}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            break
        except OSError:
            if attempt == 2:
                return None
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:  # noqa: BLE001 - a header Pillow can't read: no time
        return None
    taken = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    if not taken:
        return None
    try:
        return datetime.strptime(str(taken).strip("\x00 "), "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def fetch(item: dict, dest: Path, max_width: int, quality: int) -> str | None:
    if dest.exists() and dest.stat().st_size > 0:
        return None
    try:
        data = fetch_bytes(BASE + urllib.parse.quote(item["file_name"]))
        write_image(data, dest, datetime.fromisoformat(item["captured"]), max_width, quality)
    except (OSError, RuntimeError) as error:
        return f"{item['file_name']}: {error}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--season", default="APN_S4")
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--head", type=int, default=65536, help="bytes read for the EXIF time")
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.data_dir / "datasets" / "snapshot_occupancy"
    root.mkdir(parents=True, exist_ok=True)
    season_file = root / f"{args.season}.json"
    if season_file.exists():
        season = json.loads(season_file.read_text())
        images, labels = season["images"], season["labels"]
    else:
        labels_zip = download(LABELS_URL, root / "snapshot_safari_2024_metadata.zip")
        print(f"reading {args.season} from the metadata (a few minutes)...", flush=True)
        images, labels = stream_season(labels_zip, args.season)
        season_file.write_text(json.dumps({"images": images, "labels": labels}))
    print(f"{args.season}: {len(images)} images", flush=True)

    times_file = root / f"{args.season}-times.json"
    times: dict[str, str | None] = json.loads(times_file.read_text()) if times_file.exists() else {}
    todo = [item for item in images if item["id"] not in times]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (item, taken) in enumerate(
            zip(todo, pool.map(lambda i: head_time(i["file_name"], args.head), todo), strict=True),
            start=1,
        ):
            times[item["id"]] = taken
            if done % 2000 == 0 or done == len(todo):
                print(f"capture times: {done}/{len(todo)}", flush=True)
                times_file.write_text(json.dumps(times))

    # Reset clocks: times outside the season's year and the next.
    dated = [datetime.fromisoformat(t) for t in times.values() if t]
    year = Counter(t.year for t in dated).most_common(1)[0][0]
    years = {year, year + 1} if sum(t.year == year + 1 for t in dated) else {year - 1, year}
    by_camera: dict[str, list[tuple[datetime, dict]]] = defaultdict(list)
    bad: Counter = Counter()
    for item in images:
        taken = times.get(item["id"])
        when = datetime.fromisoformat(taken) if taken else None
        if when is None or when.year not in years:
            bad[item["location"]] += 1
            continue
        by_camera[item["location"]].append((when, item))
    total = Counter(item["location"] for item in images)
    left_out = sorted(c for c in total if bad[c] > 0.05 * total[c])
    for camera in left_out:
        by_camera.pop(camera, None)

    spans = {c: (min(t for t, _ in v), max(t for t, _ in v)) for c, v in by_camera.items()}
    first_day = min(a for a, _ in spans.values()).replace(hour=0, minute=0, second=0)
    last_day = max(b for _, b in spans.values())
    best = None
    day = first_day
    while day + timedelta(weeks=args.weeks) <= last_day + timedelta(days=1):
        end = day + timedelta(weeks=args.weeks)
        running = [c for c, (a, b) in spans.items() if a <= day and b >= end - timedelta(days=1)]
        if best is None or len(running) > len(best[1]):
            best = (day, running)
        day += timedelta(days=1)
    start, cameras = best
    end = start + timedelta(weeks=args.weeks)
    cameras = sorted(cameras)

    def folder(location: str) -> str:
        return location.split("_")[-1]  # APN_S4_13U is 13U

    chosen = sorted(
        (
            {**item, "captured": when.isoformat()}
            for camera in cameras
            for when, item in by_camera[camera]
            if start <= when < end
        ),
        key=lambda item: (item["location"], item["captured"], item["file_name"]),
    )

    def path(item: dict) -> Path:
        return root / "images" / folder(item["location"]) / Path(item["file_name"]).name

    for camera in cameras:
        (root / "images" / folder(camera)).mkdir(parents=True, exist_ok=True)
    print(
        f"{len(chosen)} images from {len(cameras)} cameras running all of {start:%Y-%m-%d} to "
        f"{end - timedelta(days=1):%Y-%m-%d} (of {len(total)}; left out for their clocks: "
        f"{', '.join(left_out) or 'none'})",
        flush=True,
    )

    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        work = pool.map(lambda i: fetch(i, path(i), args.max_width, args.quality), chosen)
        for done, (item, problem) in enumerate(zip(chosen, work, strict=True), start=1):
            if problem:
                failed[item["id"]] = problem
            if done % 1000 == 0 or done == len(chosen):
                print(f"images: {done}/{len(chosen)}, {len(failed)} failed", flush=True)
    for problem in list(failed.values())[:10]:
        print(f"  {problem}")
    chosen = [item for item in chosen if item["id"] not in failed]

    def renamed(item: dict) -> list[str]:
        return sorted({NAMES.get(name, name) for name in labels.get(item["id"], [])})

    survey = [
        {
            "id": item["id"],
            "file": str(path(item).relative_to(root)),
            "camera": folder(item["location"]),
            "captured": item["captured"],
            "sequence": item["seq_id"],
            "labels": renamed(item),
            "counts": {},
        }
        for item in chosen
    ]
    whole = {
        "from": f"{start:%Y-%m-%d} 00:00:00",
        "to": f"{end - timedelta(days=1):%Y-%m-%d} 23:59:59",
        "whole_window": True,
    }
    (root / "survey.json").write_text(
        json.dumps(
            {
                "dataset": f"Snapshot Safari 2024 Expansion, {args.season}",
                "license": "CDLA-Permissive-1.0",
                "source": "https://lila.science/datasets/snapshot-safari-2024-expansion",
                "window": [f"{start:%Y-%m-%d}", f"{end - timedelta(days=1):%Y-%m-%d}"],
                "cameras": {folder(c): whole for c in cameras},
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
    (root / "README.txt").write_text(
        README.format(
            cameras=len(cameras),
            season=args.season,
            first=f"{start:%Y-%m-%d}",
            last=f"{end - timedelta(days=1):%Y-%m-%d}",
            renamed="; ".join(f"{a} is {b}" for a, b in sorted(NAMES.items())),
            bad_times=sum(bad.values()),
            season_images=len(images),
            years=" and ".join(str(y) for y in sorted(years)),
            bad_cameras=", ".join(left_out) or "none",
            running=len(cameras),
            cameras_all=len(total),
            failed=f"\n{len(failed)} images couldn't be downloaded.\n" if failed else "",
        )
    )
    size = sum(p.stat().st_size for p in (root / "images").rglob("*.JPG"))
    print(
        f"wrote {root}: {len(survey)} images ({size / 1e9:.1f} GB), {len(cameras)} cameras, "
        f"{len(seen)} species"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
