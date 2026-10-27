#!/usr/bin/env python3
"""Tables and figure data for the paper from occreview's saved runs: for each survey, review
strategy, share reviewed and arm, the mean absolute error of
occupancy against the full review (the mean, and sd, over the draws), its bias, and how often the
95% intervals held the full review's value; the reviews' make-up (how many reviewed images showed
an animal); the smallest share from which each arm's error stayed below --good (0.1); and the
pgfplots coordinates of each line.

    python3 scripts/strategy_report.py RESULTS.json... [--good 0.1] [--coordinates]

Each file is one survey and strategy (occreview strategy; files from before it are
random). Files of the same survey and strategy are pooled by share, so a run split by shares
(--merge's inputs) can be given as its parts. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SURVEYS = {  # as the paper names them, in its order
    "nkhotakota_occupancy": "Nkhotakota",
    "wsu_lynx_occupancy": "WSU Lynx",
    "snapshot_occupancy": "APNR",
    "cct_occupancy": "CCT",
}
STRATEGIES = ["random", "animals", "by_camera", "close_calls", "confirm"]
ARMS = [  # summary's arm, and its short name here
    ("reviewed only", "reviewed"),
    ("SpeciesNet", "speciesnet"),
    ("SpeciesNet, false positives", "speciesnet_fp"),
    ("Project Model", "project"),
    ("Project Model, false positives", "project_fp"),
    ("Project Model, agreed", "agreed"),
]


def load(paths: list[Path]) -> dict:
    """{(survey, strategy): {"summary": rows by (share, arm), "runs": [...], "reference": ...}},
    pooling files of one survey and strategy: each share's rows from the file that has it."""
    found: dict[tuple[str, str], dict] = {}
    for path in paths:
        results = json.loads(path.read_text())
        if results.get("partial"):
            print(f"{path.name}: unfinished (partial), skipped", file=sys.stderr)
            continue
        key = (results["survey"], results["settings"].get("strategy", "random"))
        entry = found.setdefault(key, {"summary": {}, "runs": [], "files": [], "results": results})
        entry["files"].append(path.name)
        for row in results["summary"]:
            if row["share"] == 0 and (0.0, row["arm"]) in entry["summary"]:
                continue
            entry["summary"][row["share"], row["arm"]] = row
        entry["runs"] += [run for run in results["runs"] if run["share"] > 0]
    return found


def animal_share(runs: list[dict], share: float) -> float | None:
    """Of the images a share's draws reviewed, the part that showed an animal (by the reviews
    saved: anything but empty)."""
    shown, total = 0, 0
    for run in runs:
        if run["share"] == share:
            kinds = run.get("reviews", {})
            total += sum(kinds.values())
            shown += sum(n for kind, n in kinds.items() if kind != "empty")
    return shown / total if total else None


def first_good(summary: dict, arm: str, good: float) -> float | None:
    """The smallest share from which the arm's mean absolute error stayed below `good` at every
    larger share run (so a lucky dip doesn't count), or None."""
    found = None
    for share in sorted((s for s, name in summary if name == arm and s > 0), reverse=True):
        if summary[share, arm]["mean_absolute_error"] >= good:
            break
        found = share
    return found


TABLE_SURVEYS = [
    "cct_occupancy",
    "wsu_lynx_occupancy",
    "snapshot_occupancy",
    "nkhotakota_occupancy",
]
TABLE_STRATEGIES = {
    "random": "At random",
    "animals": "With animals",
    "by_camera": "By camera",
    "close_calls": "Close calls",
    "confirm": "Confirm",
}
TABLE_ARMS = [
    ("reviewed only", "Reviews alone"),
    ("SpeciesNet", "With SpeciesNet"),
    ("Project Model", "With the project model"),
]


def effort_table(found: dict, good: float) -> str:
    """The manuscript's effort table: for each arm and strategy (rows) and survey (columns), the
    smallest share reviewed (%) from which occupancy stayed within `good` of the full review's on
    average (first_good), or more than the largest share run."""
    surveys = [s for s in TABLE_SURVEYS if any(k[0] == s for k in found)]
    lines = [
        "% Made by scripts/strategy_report.py --effort-table from: "
        + ", ".join(name for entry in found.values() for name in entry["files"]),
        r"\begin{tabular}{ll" + "c" * len(surveys) + "}",
        r"\toprule",
        " & & " + " & ".join(SURVEYS[s] for s in surveys) + r" \\",
        r"\midrule",
    ]
    for number, (arm, arm_label) in enumerate(TABLE_ARMS):
        if number:
            lines.append(r"\addlinespace")
        for n, (strategy, label) in enumerate(TABLE_STRATEGIES.items()):
            cells = []
            for survey in surveys:
                entry = found.get((survey, strategy))
                if entry is None:
                    cells.append("")
                    continue
                share = first_good(entry["summary"], arm, good)
                largest = max(s for s, a in entry["summary"] if a == arm)
                cells.append(f"{share * 100:g}" if share else f"$>${largest * 100:g}")
            head = arm_label if n == 0 else ""
            lines.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--good", type=float, default=0.1)
    parser.add_argument("--coordinates", action="store_true", help="print pgfplots data")
    parser.add_argument("--effort-table", type=Path, help="write the effort table (LaTeX) here")
    args = parser.parse_args()
    found = load(args.results)
    order = sorted(found, key=lambda k: (list(SURVEYS).index(k[0]), STRATEGIES.index(k[1])))
    if args.effort_table:
        args.effort_table.write_text(effort_table(found, args.good))
        print(f"wrote {args.effort_table}")

    print("==== Occupancy against the full review: mean absolute error (sd over draws) ====")
    for survey, strategy in order:
        entry = found[survey, strategy]
        summary = entry["summary"]
        results = entry["results"]
        print(
            f"\n{SURVEYS[survey]}, {strategy}: {results['images']:,} images, "
            f"{results['sites']} sites; {len(results['scored_species'])} species scored "
            f"({', '.join(results['scored_species'])}); files {', '.join(entry['files'])}"
        )
        alone = summary.get((0.0, "SpeciesNet"))
        if alone:
            print(
                f"  SpeciesNet alone: {alone['mean_absolute_error']:.3f} "
                f"(bias {alone['mean_error']:+.3f}, "
                f"intervals {alone['intervals_holding_reference']:.2f})"
            )
        shares = sorted({share for share, _ in summary if share > 0})
        print(f"  {'share':>5} {'animals':>7}  " + "  ".join(f"{s:>16}" for _, s in ARMS))
        for share in shares:
            cells = []
            for arm, _ in ARMS:
                row = summary.get((share, arm))
                if row is None:
                    cells.append(f"{'':>16}")
                    continue
                sd = row.get("mean_absolute_error_sd")
                cells.append(
                    f"{row['mean_absolute_error']:.3f}"
                    + (f"±{sd:.3f}" if sd is not None else "      ")
                    + f" {row['mean_error']:+.2f}"
                )
            shown = animal_share(entry["runs"], share)
            print(
                f"  {share:>5g} {'' if shown is None else f'{shown:.2f}':>7}  " + "  ".join(cells)
            )
        print(
            f"  below {args.good:g} from: "
            + ", ".join(
                f"{name} {first_good(summary, arm, args.good) or '-'}" for arm, name in ARMS
            )
        )

    if args.coordinates:
        print("\n==== pgfplots coordinates: (percent reviewed, mean) +- (0, sd) ====")
        for survey, strategy in order:
            summary = found[survey, strategy]["summary"]
            for arm, name in ARMS:
                points = [
                    (share, row)
                    for (share, a), row in sorted(summary.items())
                    if a == arm and share > 0
                ]
                if points:
                    data = " ".join(
                        f"({share * 100:g},{row['mean_absolute_error']:.3f})"
                        f" +- (0,{row.get('mean_absolute_error_sd') or 0:.3f})"
                        for share, row in points
                    )
                    print(f"% {SURVEYS[survey]}, {strategy}, {name}\n{data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
