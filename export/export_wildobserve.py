#!/usr/bin/env python3
"""Export one survey's intermediate data from its processed WildObserve project: everything the
open pipeline (src/occreview) needs to repeat the paper's experiments without WildObserve.

This is the only step that needs WildObserve, which detected the animals (MegaDetector v6,
compact) and fingerprinted them (SpeciesNet v4.0.3a). Run it with WildObserve's backend
environment, on a survey processed with WildObserve's tools/benchmark/occupancy_bench.py
--process:

    uv run --project ../wildobserve/backend python export/export_wildobserve.py \
        --survey nkhotakota_occupancy --out data

It works on a scratch copy of the project, cleared of any reviews and project models, and writes
data/<survey>/:

- survey.json: the survey's categories (the Species list), its sites (each camera's running days,
  as WildObserve's occupancy tables take them) and the settings used;
- images.csv: every image (copies of earlier images left out): its id, site, camera (WildObserve's
  deployment id, by which the benchmark ordered cameras before shuffling them), the local date
  and time it was taken, its capture event, and its labels from the dataset;
- boxes.npz: every animal box the detector found at 0.5 or more (none was at a confirmed fixed
  object): its image, confidence, SpeciesNet fingerprint (1,280 numbers, float16), SpeciesNet's
  top animal label with its score and the runner-up's score (as WildObserve's review offers
  them), and whether WildObserve fingerprinted it when processing the image. WildObserve
  fingerprints each image's five surest boxes then, and the rest only when the image is opened,
  so only the former can carry a suggestion on an image not yet reviewed; the export
  fingerprints the rest too, since a review trains the project model on them.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session
from wildobserve import detector, occupancy, spots
from wildobserve.deployments import local_time
from wildobserve.project import DATA_DIR, DATABASE, open_project
from wildobserve.review import add_to_list
from wildobserve.species import BACKBONE, PINNED, SharedSpeciesNet, box_key
from wildobserve.tables import (
    Annotation,
    BoxScore,
    Deployment,
    Detection,
    Embedding,
    ImageRoot,
    Media,
    Setting,
    Species,
    SpeciesModel,
)

PROJECTS = {  # each survey's processed project, in ~/WildObserve/Projects
    "cct_occupancy": "Cct_occupancy",
    "wsu_lynx_occupancy": "Wsu_lynx_occupancy",
    "snapshot_occupancy": "Snapshot_occupancy",
    "nkhotakota_occupancy": "Nkhotakota_occupancy",
}
NOT_ANIMALS = {"empty", "human", "car"}
UNNAMED = {"unknown"}  # an animal the dataset's people didn't name (WSU Lynx, APNR)
THRESHOLD = 0.5  # the review's display threshold: the boxes a reviewer sees and names
OCCASION_DAYS = 7
SPECIES_SCORE = 0.9


class AllowEverything:  # a license store that allows every feature (the project model)
    def allows(self, feature: str) -> bool:
        return True


def scratch_copy(source: Path, folder: Path) -> None:
    """A consistent copy of a project's database, with no WAL."""
    folder.mkdir(parents=True, exist_ok=True)
    reader = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    writer = sqlite3.connect(folder / DATABASE)
    with writer:
        reader.backup(writer)
    reader.close()
    writer.execute("PRAGMA journal_mode=DELETE")
    writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--survey", choices=sorted(PROJECTS), required=True)
    parser.add_argument("--project", type=Path, help="the processed project, if not the usual")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    survey_root = DATA_DIR / "datasets" / args.survey
    survey = json.loads((survey_root / "survey.json").read_text())
    by_file = {tuple(Path(item["file"]).parts[-2:]): item for item in survey["images"]}
    categories = sorted(
        {name for item in survey["images"] for name in item["labels"]} - NOT_ANIMALS - UNNAMED
    )
    project_path = args.project or Path.home() / "WildObserve" / "Projects" / PROJECTS[args.survey]
    out = args.out / args.survey
    out.mkdir(parents=True, exist_ok=True)

    detector.coreml_cache = DATA_DIR / "cache" / "coreml"
    speciesnet = SharedSpeciesNet(DATA_DIR / "models")
    with tempfile.TemporaryDirectory(prefix="occreview-export-") as scratch:
        folder = Path(scratch)
        scratch_copy(project_path / DATABASE, folder)
        models = DATA_DIR / "models"
        project = open_project(
            folder, [], models / detector.DEFAULT_MODEL, AllowEverything(), speciesnet
        )
        try:
            export(project, by_file, categories, out, args.survey, survey)
        finally:
            project.close()
    return 0


def export(project, by_file: dict, categories: list[str], out: Path, name: str, survey: dict):
    species = project.species
    with Session(project.engine) as session:
        # As the benchmark's base: no reviews, no project model, the categories as the list.
        session.execute(delete(Annotation))
        session.execute(
            update(Media).values(
                review_status="unreviewed", reviewed_at=None, uncertain=False,
                saved_by_rules=False, notes=None,
            )
        )  # fmt: skip
        session.execute(delete(Setting).where(Setting.key == PINNED))
        session.execute(delete(SpeciesModel))
        session.execute(delete(Species))
        add_to_list(session, set(categories))
        session.commit()
    species.rescore()  # SpeciesNet's own scores on every fingerprinted box

    with Session(project.engine) as session:
        tables = occupancy.build(session, occupancy.Settings())
        at = {info.deployment.id: i for i, s in enumerate(tables.sites) for info in s.deployments}
        correction = {d.id: d.clock_correction or 0 for d in session.scalars(select(Deployment))}
        roots = {root.id: Path(root.path) for root in session.scalars(select(ImageRoot))}
        media = session.scalars(select(Media).where(Media.duplicate_of.is_(None))).all()
        images = []
        for m in media:
            key = tuple((roots[m.root_id] / m.rel_path).parts[-2:])
            item = by_file.get(key)
            if item is None or m.deployment_id not in at:
                continue
            taken = local_time(m, correction[m.deployment_id])
            images.append(
                {
                    "image": m.id,
                    "site": at[m.deployment_id],
                    "camera": m.deployment_id,  # WildObserve's deployment id: orders the cameras
                    "date": taken.date().isoformat(),
                    "time": taken.isoformat(timespec="seconds"),
                    "event": m.event_id if m.event_id is not None else -m.id,
                    "labels": ";".join(sorted(item["labels"])),
                    "file": "/".join(key),
                }
            )
        kept = {row["image"] for row in images}

        boxes = session.execute(
            select(Detection.id, Detection.media_id, Detection.confidence,
                   Detection.x1, Detection.y1, Detection.x2, Detection.y2)
            .where(
                Detection.label == "animal",
                Detection.confidence >= THRESHOLD,
                spots.counted(Detection),
            )
            .order_by(Detection.media_id, Detection.confidence.desc())
        ).all()  # fmt: skip
        boxes = [b for b in boxes if b.media_id in kept]
        scored = set(session.scalars(select(BoxScore.detection_id)))
        fingerprinted_at_processing = np.array([b.id in scored for b in boxes])

        by_media: dict[int, list] = defaultdict(list)
        for b in boxes:
            by_media[b.media_id].append(b)
        vectors: dict[int, np.ndarray] = {}
        missing = 0
        for media_id, found in by_media.items():
            stored = {
                (e.x1, e.y1, e.x2, e.y2): e.vector
                for e in session.scalars(
                    select(Embedding).where(
                        Embedding.media_id == media_id, Embedding.embedder == BACKBONE
                    )
                )
            }
            if any(box_key((b.x1, b.y1, b.x2, b.y2)) not in stored for b in found):
                missing += 1
                features = species.features(
                    session,
                    session.get(Media, media_id),
                    [(b.x1, b.y1, b.x2, b.y2) for b in found],
                    wait=True,
                )
                for b, vector in zip(found, features, strict=True):
                    vectors[b.id] = np.asarray(vector, dtype=np.float16)
            else:
                for b in found:
                    key = box_key((b.x1, b.y1, b.x2, b.y2))
                    vectors[b.id] = np.frombuffer(stored[key], dtype=np.float16)
        fingerprints = np.stack([vectors[b.id] for b in boxes])

        # SpeciesNet's top animal label, its score and the runner-up's, as the review offers
        # them (ZeroShot.suggest: softmax over every label, then the animal labels; runner-ups
        # under 0.01 dropped).
        suggestions = species.zero_shot.suggest(fingerprints.astype(np.float32))
        labels = sorted({s[0]["species"] for s in suggestions})
        top_label = np.array([labels.index(s[0]["species"]) for s in suggestions])
        top_score = np.array([s[0]["score"] for s in suggestions], dtype=np.float32)
        runner_up = np.array(
            [s[1]["score"] if len(s) > 1 else 0.0 for s in suggestions], dtype=np.float32
        )

    with (out / "images.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(images[0]))
        writer.writeheader()
        writer.writerows(sorted(images, key=lambda r: (r["site"], r["time"], r["image"])))
    np.savez_compressed(
        out / "boxes.npz",
        box=np.array([b.id for b in boxes]),
        image=np.array([b.media_id for b in boxes]),
        confidence=np.array([b.confidence for b in boxes], dtype=np.float32),
        fingerprint=fingerprints,
        speciesnet_label=top_label,
        speciesnet_score=top_score,
        speciesnet_runner_up=runner_up,
        fingerprinted_at_processing=fingerprinted_at_processing,
        speciesnet_labels=np.array(labels),
    )
    (out / "survey.json").write_text(
        json.dumps(
            {
                "survey": name,
                "dataset": survey.get("dataset"),
                "source": survey.get("source"),
                "license": survey.get("license"),
                "window": survey.get("window"),
                "exported": datetime.date.today().isoformat(),
                "categories": categories,
                "not_animals": sorted(NOT_ANIMALS),
                "unnamed": sorted(UNNAMED),
                "settings": {
                    "threshold": THRESHOLD,
                    "occasion_days": OCCASION_DAYS,
                    "species_score": SPECIES_SCORE,
                    "detector": "MegaDetector v6 (MDV6-apa-rtdetr-c)",
                    "fingerprints": "SpeciesNet v4.0.3a crop classifier, final pooling layer",
                },
                "sites": [
                    {
                        "site": i,
                        "name": s.name,
                        "first_day": s.first_day.isoformat(),
                        "days": sorted(d.isoformat() for d in s.days),
                    }
                    for i, s in enumerate(tables.sites)
                ],
            },
            indent=1,
        )
    )
    print(
        f"{name}: {len(images):,} images, {len(tables.sites)} sites, {len(boxes):,} boxes "
        f"({int((~fingerprinted_at_processing).sum()):,} fingerprinted for the export, on "
        f"{missing:,} images), {len(categories)} categories -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
