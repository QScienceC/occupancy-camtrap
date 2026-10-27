"""The paper's experiments on one survey: a full review, partial reviews chosen five ways at fixed
shares of the images, and the review procedure run round by round.

Every review is simulated: reviewing an image saves its labels from the dataset. A partial review
takes whole capture events. Each estimate of a species' occupancy is compared with the full
review's, over the species the full review estimates well (scored()).
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from occreview import classifier, occupancy, scores
from occreview.survey import Survey, Tables, load_names, tables

SEED = 20261009
FEW_SITES = 3
WIDEST_REFERENCE = 0.5
ROUNDS = 5  # close calls and confirm review in this many equal rounds
STRATEGIES = ["random", "animals", "by_camera", "close_calls", "confirm"]


# ---- Reviewing, and the suggestions on the rest --------------------------------------------


@dataclass
class State:
    """A survey part-way through its review: the images reviewed, and the project model trained
    on them (None until one can be)."""

    survey: Survey
    reviewed: np.ndarray
    model: classifier.ProjectModel | None = None
    trained_on: int = -1  # how many labeled boxes the model learned from
    judged: tuple | None = field(default=None, repr=False)  # the model's verdicts on every box

    @classmethod
    def start(cls, survey: Survey) -> State:
        return cls(survey, np.zeros(survey.images, dtype=bool))

    def review(self, rows) -> dict[str, int]:
        """Review these images; what kinds of review were saved, as WildObserve's benchmark
        counts them."""
        s = self.survey
        kinds: dict[str, int] = defaultdict(int)
        for row in rows:
            self.reviewed[row] = True
            if s.animals[row]:
                kind = "one species" if len(s.animals[row]) == 1 else "several species"
            elif s.shows_animal[row]:
                kind = "animal, not named"
            else:
                kind = "empty"
            kinds[kind] += 1
        return dict(kinds)

    def labeled_boxes(self) -> np.ndarray:
        """The boxes a review names: every detector box on a reviewed image of one species."""
        s = self.survey
        rows = s.box_image
        one = np.array([len(s.animals[r]) == 1 for r in range(s.images)])
        return np.flatnonzero(self.reviewed[rows] & one[rows])

    def train(self) -> bool:
        """Retrain the project model if the labels changed; whether there is one."""
        s = self.survey
        boxes = self.labeled_boxes()
        if len(boxes) != self.trained_on:
            names = [next(iter(s.animals[s.box_image[b]])) for b in boxes]
            model = classifier.train(s.fingerprint[boxes], names, s.speciesnet_label[boxes])
            self.trained_on = len(boxes)
            if model is not None:
                self.model, self.judged = model, None
        return self.model is not None

    def forget(self) -> None:
        self.model, self.judged, self.trained_on = None, None, -1

    def verdicts(self):
        if self.judged is None:
            s = self.survey
            self.judged = self.model.judge(s.fingerprint, s.speciesnet_label)
        return self.judged

    def suggestions(self, kind: str) -> tuple[list, np.ndarray]:
        """Every box's suggestion (a category or None) and score: SpeciesNet's, the project
        model's, or the project model's where SpeciesNet agrees ("agreed")."""
        s = self.survey
        if kind == "speciesnet" or self.model is None:
            return s.speciesnet_category, s.speciesnet_score
        names, score, margin, _agrees = self.verdicts()
        if kind == "agreed":  # as WildObserve's benchmark: a lead of 0 (a disagreement, or a tie)
            score = np.where(margin == 0, 0.0, score)
        return names, score

    def margins(self) -> np.ndarray:
        """Every box's lead of its top suggestion over the runner-up (0 when the two models
        disagree), as the Close calls list orders them."""
        s = self.survey
        if self.model is None:
            return np.round(s.speciesnet_score - s.speciesnet_runner_up, 4)
        return self.verdicts()[2]


# ---- Estimates -------------------------------------------------------------------------------


def estimate(state: State, suggestions: str | None, reference: Tables | None = None) -> dict:
    """Every species' occupancy (and, with suggestions, allowing for false positives), as
    WildObserve's Occupancy panel gives them, and the occasions it detected the species in that
    the full review didn't (false) and the other way round (missed)."""
    s = state.survey
    if suggestions is None:
        found = tables(s, state.reviewed)
    else:
        found = tables(s, state.reviewed, *state.suggestions(suggestions))
    observed = s.effort > 0
    rows = {}
    for name in found.species:
        y = found.seen[name]
        h = occupancy.History(y, observed, s.effort)
        fit = occupancy.occupancy(h)
        notes = []
        if not fit.converged:
            notes.append("The occupancy model didn't settle on an answer.")
        if fit.at_limit:
            notes.append("Occupancy is at its limit.")
        row = {
            "sites": int(y.any(axis=1).sum()),
            "occasions": int(y.sum()),
            "occupancy": interval(fit),
            "false_positive_occupancy": None,
            "notes": notes,
        }
        if suggestions is not None and found.confirmed[name].any():
            h.certain, h.reviewed = found.confirmed[name], found.reviewed_share
            row["false_positive_occupancy"] = interval(occupancy.false_positives(h))
        if reference is not None:
            truth = reference.seen.get(name, np.zeros_like(y))
            row["false_occasions"] = int((y & ~truth & observed).sum())
            row["missed_occasions"] = int((~y & truth & observed).sum())
        rows[name] = row
    return {"species": rows, "tables": found}


def add_scores(state: State, run: dict) -> None:
    """Score-based occupancy (scores.py) beside a run's reviewed-only estimates: for each species
    with enough reviewed detections to calibrate its scores; None for the rest."""
    s = state.survey
    found = scores.evidence(state)
    for name, row in run["species"].items():
        row["score_occupancy"] = None
        if name in found:
            seen = run["tables"].seen[name]
            h = occupancy.History(seen, s.effort > 0, s.effort, evidence=found[name])
            row["score_occupancy"] = interval(occupancy.score_occupancy(h))


def interval(fit: occupancy.Estimate) -> dict:
    return {"estimate": fit.value, "lower": fit.lower, "upper": fit.upper}


def strip(run: dict) -> dict:
    return {k: v for k, v in run.items() if k != "tables"}


def scored(reference: dict, widest: float | None = WIDEST_REFERENCE) -> list[str]:
    """The species the full review estimates well: detected at FEW_SITES or more sites, its fit
    settled and not at a limit, and (unless `widest` is None) its interval no wider than that."""
    return sorted(
        name
        for name, row in reference["species"].items()
        if row["sites"] >= FEW_SITES
        and row["occupancy"]["lower"] is not None
        and not any("settle" in n or "limit" in n for n in row["notes"])
        and (widest is None or row["occupancy"]["upper"] - row["occupancy"]["lower"] <= widest)
    )


def compare(run: dict, reference: dict, names: list[str], key: str) -> dict:
    """A run's occupancy (`key`) against the full review's, over `names`. A species the run never
    detected counts as occupancy 0; where the false-positive model gives no estimate (no reviewed
    image shows the species), or score-based occupancy none (too few reviewed detections), the
    standard estimate stands in."""
    errors, covered, false, missed, fallbacks = [], [], 0, 0, 0
    for name in names:
        truth = reference["species"][name]
        row = run["species"].get(name)
        found = row and row.get(key)
        if row and not found and key in ("false_positive_occupancy", "score_occupancy"):
            found, fallbacks = row["occupancy"], fallbacks + 1
        value = found["estimate"] if found else 0.0
        errors.append(value - truth["occupancy"]["estimate"])
        if found and found["lower"] is not None:
            covered.append(found["lower"] <= truth["occupancy"]["estimate"] <= found["upper"])
        if row:
            false += row.get("false_occasions", 0)
            missed += row.get("missed_occasions", 0)
        else:
            missed += truth["occasions"]
    return {
        "species": len(names),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "mean_error": float(np.mean(errors)),
        "largest_error": float(np.max(np.abs(errors))),
        "intervals_holding_reference": float(np.mean(covered)) if covered else None,
        "with_interval": len(covered),
        "abundance_ratio_median": None,  # not part of this paper
        "false_occasions": false,
        "missed_occasions": missed,
        "fallbacks": fallbacks,
    }


def reference_of(survey: Survey) -> tuple[dict, Tables]:
    state = State.start(survey)
    state.reviewed[:] = True
    full = estimate(state, None)
    found = full.pop("tables")
    return full, found


# ---- Choosing the images to review ------------------------------------------------------


def capture_events(survey: Survey):
    """Each capture event's images, its camera, and the events the detector found an animal in."""
    events: dict[int, list[int]] = defaultdict(list)
    camera: dict[int, int] = {}
    for row, key in enumerate(survey.event):
        events[int(key)].append(row)
        camera[int(key)] = int(survey.camera[row])
    animals = {k for k, rows in events.items() if survey.has_box[rows].any()}
    return events, camera, animals


def take(order, events, chosen: set, goal: float) -> list:
    taken = []
    for key in order:
        if len(chosen) >= goal:
            break
        chosen.update(events[key])
        taken.append(key)
    return taken


def closest_first(state: State, events, keys: set, rng) -> list:
    s = state.survey
    margin = state.margins()
    per_image: dict[int, float] = {}
    for b in np.flatnonzero(s.suggestable):
        row = int(s.box_image[b])
        per_image[row] = min(per_image.get(row, 2.0), float(margin[b]))
    closest = {
        k: min((per_image[r] for r in events[k] if r in per_image), default=2.0) for k in keys
    }
    order = sorted(keys)
    rng.shuffle(order)
    return sorted(order, key=closest.__getitem__)


def to_confirm(state: State, events, keys: set, rng, then_closest: bool = True) -> list:
    """For each species-site-occasion that the suggestions show but no review confirms, in
    random order, the capture event holding its most confident suggestion; then (unless not
    `then_closest`) the rest, closest call first."""
    s = state.survey
    confirmed = tables(s, state.reviewed).cells()
    names, score = state.suggestions("project")
    listed = set(s.categories)
    event_of = s.event
    surest: dict[tuple, tuple[float, int]] = {}
    for b in np.flatnonzero(s.suggestable & (score >= s.species_score)):
        row = int(s.box_image[b])
        key = int(event_of[row])
        name = names[b]
        if name not in listed or key not in keys or s.occasion[row] < 0:
            continue
        cell = (int(s.site[row]), int(s.occasion[row]), name.lower())
        if cell not in confirmed and (cell not in surest or score[b] > surest[cell][0]):
            surest[cell] = (float(score[b]), key)
    cells = sorted(surest)
    rng.shuffle(cells)
    first = list(dict.fromkeys(surest[c][1] for c in cells))
    if not then_closest:
        return first
    return first + closest_first(state, events, keys - set(first), rng)


def choose(state: State, strategy: str, share: float, rng) -> dict:
    """Review `share` of the images, choosing capture events by `strategy`; the kinds of review."""
    events, camera, animals = capture_events(state.survey)
    goal = share * state.survey.images
    chosen: set[int] = set()
    kinds: dict[str, int] = defaultdict(int)

    def review(rows):
        for kind, n in state.review(sorted(rows)).items():
            kinds[kind] += n

    if strategy == "random":
        order = sorted(events)
        rng.shuffle(order)
        take(order, events, chosen, goal)
        review(chosen)
        return dict(kinds)
    if strategy in ("animals", "by_camera"):
        if strategy == "animals":
            order = sorted(animals)
            rng.shuffle(order)
        else:
            order = by_camera(camera, animals, rng)
        take(order, events, chosen, goal)
        review(chosen)
        return dict(kinds)
    done: set[int] = set()
    for n in range(ROUNDS):
        left = animals - done
        if n == 0:
            order = sorted(left)
            rng.shuffle(order)
        else:
            state.train()
            if strategy == "close_calls":
                order = closest_first(state, events, left, rng)
            else:
                order = to_confirm(state, events, left, rng)
        before = set(chosen)
        done.update(take(order, events, chosen, goal * (n + 1) / ROUNDS))
        review(chosen - before)
    return dict(kinds)


def by_camera(camera: dict, animals: set, rng, per_camera: int | None = None) -> list:
    """Capture events with animals, one camera's at a time in turn (cameras and events in random
    order); with `per_camera`, at most that many from each."""
    queues: dict[int, list[int]] = defaultdict(list)
    for key in sorted(animals):
        queues[camera[key]].append(key)
    cameras = sorted(queues)
    rng.shuffle(cameras)
    for c in cameras:
        rng.shuffle(queues[c])
    longest = per_camera or max(map(len, queues.values()))
    return [queues[c][n] for n in range(longest) for c in cameras if n < len(queues[c])]


# ---- Fixed shares -----------------------------------------------------------------------------


def fixed_shares(survey: Survey, strategy: str, shares: list[float], draws: int, log=print) -> dict:
    reference, reference_tables = reference_of(survey)
    runs = []
    for share in [0.0, *shares]:
        rng = random.Random(f"{SEED}:{share}")
        for draw in range(1 if share == 0 else draws):
            state = State.start(survey)
            kinds = choose(state, strategy, share, rng) if share else {}
            state.forget()
            run = {"share": share, "draw": draw, "images_reviewed": int(state.reviewed.sum())}
            if share:
                run["reviewed_only"] = strip(estimate(state, None, reference_tables))
            run["speciesnet"] = strip(estimate(state, "speciesnet", reference_tables))
            if share and state.train():
                run["project_model"] = strip(estimate(state, "project", reference_tables))
                run["project_model"]["model"] = {
                    "species": len(state.model.species),
                    "examples": state.model.examples,
                }
                run["project_model_agreeing"] = strip(estimate(state, "agreed", reference_tables))
            run["reviews"] = kinds
            runs.append(run)
            log(f"{survey.name} {strategy} {share:g} draw {draw + 1}: {run['images_reviewed']:,}")
    results = {
        "survey": survey.name,
        "settings": {
            "strategy": strategy,
            "shares": shares,
            "draws": draws,
            "seed": SEED,
            "species_score": survey.species_score,
            "rounds": ROUNDS if strategy in ("close_calls", "confirm") else 1,
        },
        "images": survey.images,
        "sites": survey.sites,
        "occasions": survey.occasions,
        "reference": reference,
        "runs": runs,
    }
    summarize(results)
    return results


ARMS = [
    ("reviewed only", "reviewed_only", "occupancy"),
    ("SpeciesNet", "speciesnet", "occupancy"),
    ("SpeciesNet, false positives", "speciesnet", "false_positive_occupancy"),
    ("Project Model", "project_model", "occupancy"),
    ("Project Model, false positives", "project_model", "false_positive_occupancy"),
    ("Project Model, agreed", "project_model_agreeing", "occupancy"),
    (
        "Project Model, agreed, false positives",
        "project_model_agreeing",
        "false_positive_occupancy",
    ),
]


def summarize(results: dict) -> None:
    """Each arm at each share against the full review, averaged over the draws ("summary"), and
    the same over every species with no limit on the reference's interval ("summary_all")."""
    reference = results["reference"]
    category_of = set(load_names(results["survey"]).values())
    for field_name, widest in (("summary", WIDEST_REFERENCE), ("summary_all", None)):
        names = scored(reference, widest)
        with_speciesnet = [n for n in names if n in category_of]
        results["scored_species" if widest else "scored_species_all"] = names
        if widest:
            results["scored_species_with_speciesnet_names"] = with_speciesnet
        summary = []
        for share in [0.0, *results["settings"]["shares"]]:
            for label, arm, key in ARMS:
                if share == 0 and key == "false_positive_occupancy":
                    continue
                chosen = with_speciesnet if arm == "speciesnet" else names
                found = [
                    compare(run[arm], reference, chosen, key)
                    for run in results["runs"]
                    if run["share"] == share and arm in run
                ]
                if not found:
                    continue
                row = {"share": share, "arm": label, "draws": len(found)}
                for name in found[0]:
                    values = [f[name] for f in found if f[name] is not None]
                    row[name] = statistics.mean(values) if values else None
                    if name == "mean_absolute_error" and len(values) > 1:
                        row["mean_absolute_error_sd"] = statistics.stdev(values)
                summary.append(row)
        results[field_name] = summary


# ---- The review procedure ----------------------------------------------------------------


def procedure(
    survey: Survey,
    per_camera: int,
    rng,
    reference: dict,
    reference_tables: Tables,
    names: list[str],
    round_share: float = 0.02,
    max_share: float = 0.4,
    floor: float = 0.02,
) -> list[dict]:
    """The paper's procedure, a round at a time: up to `per_camera` capture events with animals
    from each camera in turn; then rounds, each of at most `round_share` of the images, confirming
    the most confident suggestion of each species-site-occasion not yet confirmed, the project
    model retrained after each. It stops when no unconfirmed suggestion is left, at `max_share` of
    the images, or when a round confirms fewer than `floor` new species-site-occasions per capture
    event (below any stopping rule worth trying). Returns a record of every round."""
    events, camera, animals = capture_events(survey)
    state = State.start(survey)
    truth = reference_tables.cells()
    chosen: set[int] = set()
    done: set[int] = set()
    rounds: list[dict] = []
    step, order, queue = "seed", by_camera(camera, animals, rng, per_camera), None
    while True:
        before = set(chosen)
        goal = len(chosen) + (1e18 if step == "seed" else round_share * survey.images)
        new = take(
            [k for k in order if k not in done],
            events,
            chosen,
            min(goal, max_share * survey.images),
        )
        done.update(new)
        kinds = state.review(sorted(chosen - before)) if new else {}
        model = None
        if state.train():
            model = {"species": len(state.model.species), "examples": state.model.examples}
        alone = estimate(state, None, reference_tables)
        add_scores(state, alone)
        with_model = estimate(state, "project", reference_tables)
        confirmed = alone["tables"].cells()
        previous = rounds[-1]["confirmed_cells"] if rounds else 0
        rounds.append(
            {
                "step": step,
                "queue": queue,
                "events_reviewed": len(done),
                "images_reviewed": len(chosen),
                "new_events": len(new),
                "new_images": len(chosen) - len(before),
                "new_cells": len(confirmed) - previous,
                "reviews": kinds,
                "model": model,
                "reviewed_only": compare(alone, reference, names, "occupancy"),
                "project_model": compare(with_model, reference, names, "occupancy"),
                "project_model_fp": compare(
                    with_model, reference, names, "false_positive_occupancy"
                ),
                "score_based": compare(alone, reference, names, "score_occupancy"),
                "confirmed_cells": len(confirmed),
                "confirmed_true_cells": len(confirmed & truth),
                "species_reviewed_only": {
                    n: (r["occupancy"] or {}).get("estimate")
                    for n, r in alone["species"].items()
                    if n in names
                },
                "species_project_model": {
                    n: (r["occupancy"] or {}).get("estimate")
                    for n, r in with_model["species"].items()
                    if n in names
                },
                "species_score_based": {
                    n: (r["score_occupancy"] or r["occupancy"] or {}).get("estimate")
                    for n, r in alone["species"].items()
                    if n in names
                },
            }
        )
        if len(chosen) >= max_share * survey.images:
            break
        last = rounds[-1]
        if last["step"] == "confirm" and last["new_cells"] < floor * max(last["new_events"], 1):
            break
        step = "confirm"
        order = to_confirm(state, events, animals - done, rng, then_closest=False)
        queue = len(order)
        if not order:
            break
    return rounds


def procedures(survey: Survey, seeds: list[int], draws: int, log=print, **options) -> dict:
    reference, reference_tables = reference_of(survey)
    names = scored(reference)
    runs = []
    for per_camera in seeds:
        rng = random.Random(f"{SEED}:procedure:{per_camera}")
        for draw in range(draws):
            rounds = procedure(
                survey, per_camera, rng, reference, reference_tables, names, **options
            )
            runs.append({"seed_events": per_camera, "draw": draw, "rounds": rounds})
            last = rounds[-1]
            log(
                f"{survey.name} procedure seed {per_camera} draw {draw + 1}: {len(rounds)} "
                f"rounds, {last['images_reviewed'] / survey.images:.1%} of images; reviews alone "
                f"{last['reviewed_only']['mean_absolute_error']:.3f}, score-based "
                f"{last['score_based']['mean_absolute_error']:.3f}"
            )
    return {
        "survey": survey.name,
        "settings": {"procedure": True, "seed_events": seeds, "draws": draws, "seed": SEED,
                     "species_score": survey.species_score, **options},
        "images": survey.images,
        "sites": survey.sites,
        "occasions": survey.occasions,
        "reference": reference,
        "scored_species": names,
        "reference_cells": len(reference_tables.cells()),
        "runs": runs,
    }  # fmt: skip
