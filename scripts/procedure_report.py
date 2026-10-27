#!/usr/bin/env python3
"""The review procedure's results (occreview procedure), for the occupancy review
paper: for each survey and seed size, where the estimates stood
after the seed, and where each stopping rule would have stopped (the review it took and how far
occupancy stood from the full review's), over the draws; and the estimates at fixed shares of the
images, to set beside the fixed-share strategies.

    python3 scripts/procedure_report.py RESULTS.json... [--yields 0.1 0.2 0.3 0.5]
        [--table OUT.tex] [--trajectories OUT.tex]

Files of one survey are pooled (a survey's seed sizes may be split between runs). A stopping
rule "yield < y" stops after the first confirm round that confirmed fewer than y new
species-site-weeks per capture event it reviewed, that round included; a run that ended first
(its queue empty, or the floor) stops at its end. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SURVEYS = {  # the paper's order and names
    "cct_occupancy": "CCT",
    "wsu_lynx_occupancy": "WSU Lynx",
    "snapshot_occupancy": "APNR",
    "nkhotakota_occupancy": "Nkhotakota",
}
ARMS = [
    ("reviewed_only", "alone"),
    ("project_model", "model"),
    ("project_model_fp", "model FP"),
    ("score_based", "scores"),  # score-based occupancy, WildObserve's option (scores.py)
]


def load(paths: list[Path]) -> dict:
    """{survey: {"results": the first file's, "runs": every run, "files": names}}."""
    found: dict[str, dict] = {}
    for path in paths:
        results = json.loads(path.read_text())
        if results.get("partial"):
            print(
                f"{path.name}: unfinished (partial); its finished draws are used", file=sys.stderr
            )
        entry = found.setdefault(results["survey"], {"results": results, "runs": [], "files": []})
        entry["runs"] += results["runs"]
        entry["files"].append(path.name)
    return found


def yield_of(row: dict) -> float:
    return row["new_cells"] / max(row["new_events"], 1)


def stop(rounds: list[dict], threshold: float) -> dict:
    """The round at which the rule "yield < threshold" stops (that round included)."""
    for row in rounds:
        if row["step"] == "confirm" and yield_of(row) < threshold:
            return row
    return rounds[-1]


def at_share(rounds: list[dict], share: float, images: int) -> dict | None:
    """The first record with at least `share` of the images reviewed, if the run got there."""
    for row in rounds:
        if row["images_reviewed"] >= share * images:
            return row
    return None


def mean_sd(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f}±{statistics.stdev(values):.3f}"


def describe(rows: list[dict], images: int, cells: int, sites: int) -> dict:
    """Means over draws of a set of records (one per draw)."""
    return {
        "draws": len(rows),
        "share": statistics.mean(r["images_reviewed"] / images for r in rows),
        "events": statistics.mean(r["events_reviewed"] for r in rows),
        "events_per_cell": statistics.mean(r["events_reviewed"] / cells for r in rows),
        "events_per_site": statistics.mean(r["events_reviewed"] / sites for r in rows),
        "found": statistics.mean(r["confirmed_cells"] / cells for r in rows),
        **{name: [r[arm]["mean_absolute_error"] for r in rows] for arm, name in ARMS},
    }


def line(label: str, d: dict) -> str:
    return (
        f"  {label:16} {d['share']:6.1%} of images, {d['events']:6.0f} events "
        f"({d['events_per_cell']:.2f} per species-site-week, {d['events_per_site']:.0f} per site), "
        f"{d['found']:5.1%} found; alone {mean_sd(d['alone'])}, model {mean_sd(d['model'])}, "
        f"model FP {mean_sd(d['model FP'])}, scores {mean_sd(d['scores'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--yields", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--shares", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--table", type=Path, help="write the stopping table (LaTeX) here")
    parser.add_argument(
        "--table-yields", type=float, nargs="+", default=[0.2], help="the table's rules"
    )
    args = parser.parse_args()
    found = load(args.results)
    table: dict[tuple[str, int], dict] = {}
    for survey in [s for s in SURVEYS if s in found]:
        entry = found[survey]
        results = entry["results"]
        images, cells, sites = results["images"], results["reference_cells"], results["sites"]
        print(
            f"\n{SURVEYS[survey]}: {images:,} images, {sites} sites, {cells:,} species-site-weeks; "
            f"{len(results['scored_species'])} species scored; files {', '.join(entry['files'])}"
        )
        by_seed: dict[int, list[list[dict]]] = defaultdict(list)
        for run in entry["runs"]:
            by_seed[run["seed_events"]].append(run["rounds"])
        for seed in sorted(by_seed):
            runs = by_seed[seed]
            print(f" seed {seed} per camera ({len(runs)} draws):")
            print(line("after the seed", describe([r[0] for r in runs], images, cells, sites)))
            for threshold in args.yields:
                d = describe([stop(r, threshold) for r in runs], images, cells, sites)
                print(line(f"yield < {threshold:g}", d))
                table[survey, seed, threshold] = d
            d = describe([r[-1] for r in runs], images, cells, sites)
            print(line("end of run", d))
            for share in args.shares:
                rows = [row for r in runs if (row := at_share(r, share, images))]
                if len(rows) == len(runs):
                    print(line(f"at {share:.0%}", describe(rows, images, cells, sites)))
    if args.table:
        args.table.write_text(stopping_table(found, table, args.table_yields))
        print(f"\nwrote {args.table}")
    return 0


def stopping_table(found: dict, table: dict, yields: list[float]) -> str:
    """For each survey and seed size, the review the procedure took and how close occupancy came,
    at each stopping rule: share of images, capture events per species-site-week of the full
    review, the share of those found, and the mean absolute difference from the full review
    (reviews alone; with the project model's suggestions; with its scores, score-based)."""
    files = ", ".join(name for entry in found.values() for name in entry["files"])
    lines = [
        f"% Made by scripts/procedure_report.py --table from: {files}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & First step & Images & Events per & Found & \multicolumn{3}{c}{MAE} \\",
        r"\cmidrule(lr){6-8}",
        r"Survey & per camera & (\%) & species-site-week & (\%) & alone & with model "
        r"& with scores \\",
        r"\midrule",
    ]
    for number, threshold in enumerate(yields):
        if len(yields) > 1:
            if number:
                lines.append(r"\addlinespace")
            lines.append(
                rf"\multicolumn{{8}}{{l}}{{\emph{{Stop when a round confirms fewer than "
                rf"{threshold:g} new species-site-weeks per capture event}}}} \\"
            )
        for survey, name in SURVEYS.items():
            seeds = sorted(s for (v, s, y) in table if v == survey and y == threshold)
            for n, seed in enumerate(seeds):
                d = table[survey, seed, threshold]
                lines.append(
                    f"{name if n == 0 else ''} & {seed} & {d['share'] * 100:.0f} & "
                    f"{d['events_per_cell']:.1f} & {d['found'] * 100:.0f} & "
                    f"{statistics.mean(d['alone']):.2f} & {statistics.mean(d['model']):.2f} & "
                    f"{statistics.mean(d['scores']):.2f} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
