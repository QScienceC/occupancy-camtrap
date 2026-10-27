"""A survey's exported data (export/export_wildobserve.py), and the detection histories that a
review and suggestions make of it.

Sites, occasions and effort follow WildObserve's occupancy tables: a site is a camera, its
survey divided into seven-day occasions from its first running day, its effort in an occasion the
days a camera ran. A species is detected at a site in an occasion when a reviewed image of it was
taken then (a confirmed detection), or, with suggestions counted, when an animal box on an image
not yet reviewed has a top suggestion naming it with a score of at least 0.9 (only species on the
survey's list count).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from pathlib import Path

import numpy as np

NAMES = {  # each survey's map from SpeciesNet's labels to its categories
    "cct_occupancy": "cct_occupancy_names.csv",
    "wsu_lynx_occupancy": "wsu_lynx_names.csv",
    "snapshot_occupancy": "snapshot_names.csv",
    "nkhotakota_occupancy": "nkhotakota_names.csv",
}


def load_names(survey: str) -> dict[str, str]:
    """SpeciesNet's labels (lowercase) and the category each one's taxon falls in."""
    text = resources.files("occreview.names").joinpath(NAMES[survey]).read_text()
    found = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, category = line.rsplit(",", 1)
            found[name.lower()] = category
    return found


@dataclass
class Survey:
    name: str
    categories: list[str]
    occasion_days: int
    species_score: float
    first_day: list[date]  # each site's
    effort: np.ndarray  # sites by occasions: camera-days
    # Images, by row: site, occasion (-1 outside the occasions), capture event, labels
    image_id: np.ndarray
    site: np.ndarray
    camera: np.ndarray  # WildObserve's deployment id (the order cameras are shuffled from)
    occasion: np.ndarray
    event: np.ndarray
    labels: list[frozenset]
    # Boxes, by row: their image's row, fingerprint, SpeciesNet's top animal label (a category
    # or None, by the name map), its score, the runner-up's score, and whether WildObserve
    # fingerprinted it when processing (only those carry suggestions on unreviewed images)
    box_image: np.ndarray
    fingerprint: np.ndarray
    speciesnet_label: np.ndarray  # SpeciesNet's own label, as text
    speciesnet_category: list[str | None]
    speciesnet_score: np.ndarray
    speciesnet_runner_up: np.ndarray
    suggestable: np.ndarray
    # Derived
    animals: list[frozenset] = field(default_factory=list)  # each image's named species
    shows_animal: np.ndarray | None = None  # reviewed, it would have an animal box
    has_box: np.ndarray | None = None  # the detector boxed an animal on it

    @property
    def sites(self) -> int:
        return self.effort.shape[0]

    @property
    def occasions(self) -> int:
        return self.effort.shape[1]

    @property
    def images(self) -> int:
        return len(self.image_id)


def load(folder: Path) -> Survey:
    """A survey exported to `folder` (data/<survey>)."""
    meta = json.loads((folder / "survey.json").read_text())
    settings = meta["settings"]
    length = settings["occasion_days"]
    first = [date.fromisoformat(s["first_day"]) for s in meta["sites"]]
    days = [[date.fromisoformat(d) for d in s["days"]] for s in meta["sites"]]
    occasions = max((max(d) - f).days // length + 1 for d, f in zip(days, first, strict=True))
    effort = np.zeros((len(first), occasions), dtype=int)
    for i, (d, f) in enumerate(zip(days, first, strict=True)):
        for day in d:
            effort[i, (day - f).days // length] += 1

    with (folder / "images.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    image_id = np.array([int(r["image"]) for r in rows])
    site = np.array([int(r["site"]) for r in rows])
    camera = np.array([int(r.get("camera", r["site"])) for r in rows])
    occasion = np.array(
        [(date.fromisoformat(r["date"]) - first[int(r["site"])]).days // length for r in rows]
    )
    occasion[(occasion < 0) | (occasion >= occasions)] = -1
    event = np.array([int(r["event"]) for r in rows])
    labels = [frozenset(r["labels"].split(";")) if r["labels"] else frozenset() for r in rows]

    boxes = np.load(folder / "boxes.npz")
    row_of = {image: i for i, image in enumerate(image_id)}
    box_image = np.array([row_of[i] for i in boxes["image"]])
    names = load_names(meta["survey"])
    text = boxes["speciesnet_labels"][boxes["speciesnet_label"]]
    category = [names.get(str(label).lower()) for label in text]
    categories = meta["categories"]
    survey = Survey(
        name=meta["survey"],
        categories=categories,
        occasion_days=length,
        species_score=settings["species_score"],
        first_day=first,
        effort=effort,
        image_id=image_id,
        site=site,
        camera=camera,
        occasion=occasion,
        event=event,
        labels=labels,
        box_image=box_image,
        fingerprint=boxes["fingerprint"],
        speciesnet_label=text,
        speciesnet_category=[c if c in categories else None for c in category],
        speciesnet_score=boxes["speciesnet_score"].astype(float),
        speciesnet_runner_up=boxes["speciesnet_runner_up"].astype(float),
        suggestable=boxes["fingerprinted_at_processing"].astype(bool),
    )
    not_animals = set(meta["not_animals"])
    unnamed = set(meta["unnamed"])
    survey.animals = [frozenset(set(found) - not_animals - unnamed) for found in labels]
    survey.shows_animal = np.array([bool(set(found) - not_animals) for found in labels])
    survey.has_box = np.zeros(survey.images, dtype=bool)
    survey.has_box[box_image] = True
    return survey


@dataclass
class Tables:
    """Detection histories: for each species, where and when it was detected (sites by
    occasions), and where a reviewed image confirms it; the effort; and, for the false-positive
    model, the share of each site-occasion's animal images reviewed."""

    seen: dict[str, np.ndarray]
    confirmed: dict[str, np.ndarray]
    effort: np.ndarray
    reviewed_share: np.ndarray | None

    @property
    def species(self) -> list[str]:
        return sorted(self.seen, key=str.lower)

    def cells(self) -> set[tuple[int, int, str]]:
        """The confirmed species-site-occasions."""
        return {
            (int(i), int(j), name)
            for name, found in self.confirmed.items()
            for i, j in zip(*np.nonzero(found), strict=True)
        }


def tables(
    survey: Survey,
    reviewed: np.ndarray,
    suggestion: list[str | None] | None = None,
    score: np.ndarray | None = None,
) -> Tables:
    """The histories from the reviewed images (a boolean per image), and, given a suggestion
    (a category or None) and its score for every box, from the suggestions on the rest."""
    shape = survey.effort.shape
    seen: dict[str, np.ndarray] = {}
    confirmed: dict[str, np.ndarray] = {}
    valid = survey.occasion >= 0
    for row in np.flatnonzero(reviewed & valid):
        i, j = survey.site[row], survey.occasion[row]
        for name in survey.animals[row]:
            seen.setdefault(name, np.zeros(shape, dtype=bool))[i, j] = True
            confirmed.setdefault(name, np.zeros(shape, dtype=bool))[i, j] = True
    share = None
    if suggestion is not None:
        listed = set(survey.categories)
        for b in np.flatnonzero(survey.suggestable & (score >= survey.species_score)):
            row = survey.box_image[b]
            name = suggestion[b]
            if reviewed[row] or not valid[row] or name not in listed:
                continue
            seen.setdefault(name, np.zeros(shape, dtype=bool))[
                survey.site[row], survey.occasion[row]
            ] = True
        # Images with an animal: reviewed, one the review gave an animal box; not yet, one the
        # detector boxed an animal on.
        animal = np.where(reviewed, survey.shows_animal, survey.has_box) & valid
        done, total = np.zeros(shape), np.zeros(shape)
        np.add.at(total, (survey.site[animal], survey.occasion[animal]), 1)
        np.add.at(done, (survey.site[animal & reviewed], survey.occasion[animal & reviewed]), 1)
        share = np.divide(done, total, out=np.zeros(shape), where=total > 0)
    for name in seen:
        confirmed.setdefault(name, np.zeros(shape, dtype=bool))
    return Tables(seen, confirmed, survey.effort, share)
