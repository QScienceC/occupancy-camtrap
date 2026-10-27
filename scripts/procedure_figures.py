#!/usr/bin/env python3
"""The paper's procedure figure, as pgfplots, from
occreview procedure runs: how far occupancy stood from the full review's as the
procedure went on, for each survey (one seed size), against (left) the share of the images
reviewed and (right) the capture events reviewed per species-site-week of the full review; a row
per estimate (the reviews alone; with the project model's suggestions). Each line is the mean over
the draws, each draw's record interpolated between its rounds; an open circle marks where the
stopping rule (a round confirming fewer than --stop new species-site-weeks per capture event)
stopped, on average.

    python3 scripts/procedure_figures.py OUT.tex RESULTS.json... [--seed 2] [--stop 0.2]

Colors: CCT, the survey SpeciesNet was probably trained on, neutral grey; the others the first
three categorical slots, which hold apart for every pair (dataviz validator, --pairs all); each
also has its own marker. Standard library only.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

from procedure_report import load, stop
from strategy_figures import y_scale

LINES = [  # survey, name, style
    ("cct_occupancy", "CCT", "wmgrey, mark=square*"),
    ("wsu_lynx_occupancy", "WSU Lynx", "wmblue, mark=*"),
    ("snapshot_occupancy", "APNR", "wmaqua, mark=diamond*"),
    ("nkhotakota_occupancy", "Nkhotakota", "wmorange, mark=triangle*"),
]
ROWS = [("reviewed_only", "Reviews alone"), ("project_model", "With the project model")]
STEPS = 40  # interpolation points per line


def interpolate(points: list[tuple[float, float]], x: float) -> float:
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def mean_line(runs: list[list[dict]], x_of, arm: str) -> list[tuple[float, float]]:
    """The mean over draws of `arm`'s error along x (x_of(record)), where every draw has data."""
    tracks = [[(x_of(r), r[arm]["mean_absolute_error"]) for r in rounds] for rounds in runs]
    start = max(track[0][0] for track in tracks)
    end = min(track[-1][0] for track in tracks)
    grid = [start + (end - start) * n / STEPS for n in range(STEPS + 1)]
    return [(x, statistics.mean(interpolate(t, x) for t in tracks)) for x in grid]


def figure(found: dict, seed: int, threshold: float) -> str:
    files = ", ".join(name for entry in found.values() for name in entry["files"])
    out = [
        f"% Made by scripts/procedure_figures.py --seed {seed} --stop {threshold:g} from: {files}",
        r"\definecolor{wmaqua}{HTML}{1BAF7A}",
        r"\pgfplotsset{",
        r"  procedure axis/.style={",
        r"    width=0.48\linewidth, height=5.4cm, ymin=0,",
        r"    yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1,",
        r"                      /pgf/number format/fixed zerofill},",
        r"    axis lines=left, axis line style={black!55}, tick style={black!55},",
        r"    ymajorgrids, grid style={black!12},",
        r"    tick label style={font=\footnotesize}, label style={font=\small},",
        r"    title style={font=\small\bfseries},",
        r"    legend style={draw=none, fill=none, font=\footnotesize,",
        r"                  /tikz/every even column/.append style={column sep=0.5cm}},",
        r"    legend columns=4, legend cell align=left,",
        r"    every axis plot/.append style={line width=1.1pt, mark size=1.6pt, mark repeat=8},",
        r"  },",
        r"  stop mark/.style={only marks, mark=o, mark size=4pt, line width=1pt, forget plot},",
        r"}",
        r"\begin{tikzpicture}",
    ]
    panels = []
    for row, (arm, label) in enumerate(ROWS):
        for column, scale in enumerate(["share", "per_cell"]):
            panels.append((row, column, arm, label, scale))
    drawn: dict[int, list[tuple[str, list, tuple]]] = {}  # each panel's lines, then its scale
    for n, (_row, _column, arm, _label, scale) in enumerate(panels):
        drawn[n] = []
        for survey, _name, style in LINES:
            if survey not in found:
                continue
            results = found[survey]["results"]
            images, cells = results["images"], results["reference_cells"]
            runs = [r["rounds"] for r in found[survey]["runs"] if r["seed_events"] == seed]
            if not runs:
                continue

            def x_of(record, images=images, cells=cells, scale=scale):
                if scale == "share":
                    return 100 * record["images_reviewed"] / images
                return record["events_reviewed"] / cells

            stops = [stop(rounds, threshold) for rounds in runs]
            point = (
                statistics.mean(x_of(s) for s in stops),
                statistics.mean(s[arm]["mean_absolute_error"] for s in stops),
            )
            drawn[n].append((style, mean_line(runs, x_of, arm), point))
    tops = {  # each row its own y scale
        row: max(
            max(max(y for _, y in line), point[1])
            for n, (r, *_rest) in enumerate(panels)
            if r == row
            for _style, line, point in drawn[n]
        )
        for row in range(len(ROWS))
    }
    right = {  # each x scale ends at the first tick past its data
        scale: max(
            max(max(x for x, _ in line), point[0])
            for n, (*_rest, s) in enumerate(panels)
            if s == scale
            for _style, line, point in drawn[n]
        )
        for scale in ("share", "per_cell")
    }
    share_top = 5 * math.ceil(right["share"] / 5)
    cell_top = math.ceil(right["per_cell"] * 2) / 2
    for n, (row, _column, _arm, label, scale) in enumerate(panels):
        place = (
            ""
            if n == 0
            else ", at={(p0.south east)}, anchor=south west, xshift=0.6cm"
            if n == 1
            else f", at={{(p{n - 2}.south west)}}, anchor=north west, yshift=-1.7cm"
        )
        options = [f"procedure axis, name=p{n}{place}", f"title={{({'abcd'[n]})}}"]
        options.append(y_scale(tops[row]))
        if scale == "share":
            ticks = ",".join(str(t) for t in range(0, share_top + 1, 5))
            options += [f"xmin=0, xmax={share_top}", f"xtick={{{ticks}}}"]
            options.append(f"ylabel={{{label}}}")
        else:
            ticks = ",".join(f"{t / 2:g}" for t in range(round(cell_top * 2) + 1))
            options += [f"xmin=0, xmax={cell_top:g}", f"xtick={{{ticks}}}", "yticklabels={}"]
        if row == len(ROWS) - 1:
            options.append(
                r"xlabel={Images reviewed (\%)}"
                if scale == "share"
                else "xlabel={Capture events reviewed per species-site-week}"
            )
        if n == len(panels) - 1:
            options.append("legend to name=procedure legend")
        out.append(f"  \\begin{{axis}}[{', '.join(options)}]")
        for style, line, (sx, sy) in drawn[n]:
            out.append(f"    \\addplot[{style}, forget plot] coordinates {{")
            out.append("      " + " ".join(f"({x:.3f},{y:.3f})" for x, y in line) + "};")
            color = style.split(",")[0]
            out.append(f"    \\addplot[stop mark, {color}] coordinates {{({sx:.3f},{sy:.3f})}};")
        if n == len(panels) - 1:
            for _, name, style in LINES:
                out.append(f"    \\addlegendimage{{{style}}}")
                out.append(f"    \\addlegendentry{{{name}}}")
        out.append(r"  \end{axis}")
    out.append(
        r"  \node[anchor=north] at ([yshift=-1.2cm]$(p2.south west)!0.5!(p3.south east)$) "
        r"{\ref{procedure legend}};"
    )
    # One y label for both rows, left of each row's own label and centered on the plots
    out.append(
        r"  \path let \p1 = (p0.outer west), \p2 = (p2.outer west), \p3 = (p0.north), "
        r"\p4 = (p2.south) in node[rotate=90, anchor=south, font=\small] "
        r"at ({min(\x1,\x2)}, {(\y3+\y4)/2}) {Mean absolute error (MAE)};"
    )
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--stop", type=float, default=0.2)
    args = parser.parse_args()
    args.out.write_text(figure(load(args.results), args.seed, args.stop))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
