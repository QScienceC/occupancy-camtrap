#!/usr/bin/env python3
"""The paper's figures of the fixed-share strategies, as pgfplots, from occreview's
saved runs, so they can be redrawn when a run is repeated:

- arms: for each survey, how far occupancy from a partial review (drawn at random) falls from
  the full review's, by the share reviewed: from the reviews alone, with SpeciesNet's
  suggestions, and with the project model's, each with and without allowing for false
  positives, and SpeciesNet's suggestions with no review as a dotted line;
- strategies: the same for some arms (--arms: the reviews alone and the project model's, by
  default), a row of panels for each, with a line for each way of choosing the images to
  review.

    python3 scripts/strategy_figures.py arms OUT.tex RESULTS.json...
    python3 scripts/strategy_figures.py strategies OUT.tex RESULTS.json...
        [--arms reviewed project]

RESULTS are the runs (strategy_report.load pools them by survey and strategy). Panels follow
the paper's order of surveys (strategy_report.SURVEYS order: PANELS). Points are nudged left
and right so their error bars don't overlap. Standard library only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from strategy_report import load

PANELS = [  # the paper's order, and each panel's title
    ("cct_occupancy", "CCT"),
    ("wsu_lynx_occupancy", "WSU Lynx"),
    ("snapshot_occupancy", "APNR"),
    ("nkhotakota_occupancy", "Nkhotakota"),
]
ARMS = [  # summary arm, style, legend, nudge
    ("reviewed only", "wmgrey, mark=square*", "Reviews alone", 1.0),
    ("SpeciesNet", "wmblue, mark=*", "With SpeciesNet's suggestions", 0.96),
    (
        "SpeciesNet, false positives",
        "wmblue, dashed, mark=o, mark options={solid, fill=white}",
        "the same, allowing for false positives",
        0.96,
    ),
    ("Project Model", "wmorange, mark=triangle*", "With the project model's suggestions", 1.04),
    (
        "Project Model, false positives",
        "wmorange, dashed, mark=triangle, mark options={solid, fill=white}",
        "the same, allowing for false positives",
        1.04,
    ),
]
STRATEGIES = [  # strategy, style, legend, nudge
    ("random", "wmgrey, mark=square*", "At random", 0.92),
    ("animals", "wmaqua, mark=*", "With animals", 0.96),
    ("by_camera", "wmviolet, densely dashed, mark=diamond*", "By camera", 1.0),
    ("close_calls", "wmyellow, dashdotted, mark=triangle*", "Close calls", 1.04),
    ("confirm", "wmmagenta, loosely dashed, mark=pentagon*", "Confirm", 1.08),
]
ARM_NAMES = {"reviewed": "reviewed only", "speciesnet": "SpeciesNet", "project": "Project Model"}
ARM_LABELS = {  # a strategies row's y label
    "reviewed": "Reviews alone",
    "speciesnet": "With SpeciesNet",
    "project": "With the project model",
}
COLORS = r"""\definecolor{wmaqua}{HTML}{1BAF7A}
\definecolor{wmyellow}{HTML}{EDA100}
\definecolor{wmmagenta}{HTML}{E87BA4}
\definecolor{wmviolet}{HTML}{6250D6}
"""
STYLE = r"""\pgfplotsset{
  occupancy axis/.style={
    width=%(width)s, height=%(height)s,
    xmin=1.6, xmax=%(xmax)s,
    xtick={%(ticks)s}, xticklabels={%(ticks)s},
    minor xtick={},
    ymin=0, ymax=%(ymax)s, ytick={%(yticks)s},
    yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1,
                      /pgf/number format/fixed zerofill},
    axis lines=left, axis line style={black!55}, tick style={black!55},
    ymajorgrids, grid style={black!12},
    tick label style={font=\footnotesize}, label style={font=\small},
    title style={font=\small\bfseries, align=left},
    legend style={draw=none, fill=none, font=\footnotesize, row sep=1pt,
                  /tikz/every even column/.append style={column sep=0.5cm}},
    legend cell align=left, legend columns=%(legend_columns)s,
    error bars/y dir=both, error bars/y explicit,
    error bars/error bar style={line width=0.6pt, solid},
    every axis plot/.append style={line width=1.2pt, mark size=2.4pt},
  },
  no review line/.style={black!55, densely dotted, line width=1pt, forget plot,
                         error bars/y dir=none},
}
"""


def y_scale(top: float) -> str:
    """A y axis from 0 to just above `top`: ticks every 0.1, or every 0.05 (two decimals) when
    the axis ends at 0.3 or below, so a row of small values gets a scale of its own."""
    step, digits = (0.1, 1) if top > 0.3 else (0.05, 2)
    count = math.ceil(top / step - 1e-9)
    ticks = ",".join(f"{n * step:.2f}".rstrip("0").rstrip(".") for n in range(count + 1))
    return (
        f"ymax={count * step:.2f}, ytick={{{ticks}}}, "
        f"yticklabel style={{/pgf/number format/precision={digits}}}"
    )


def points(summary: dict, arm: str, nudge: float, shares: set) -> list[str]:
    rows = sorted(
        (share, row) for (share, name), row in summary.items() if name == arm and share in shares
    )
    return [
        f"({share * 100 * nudge:.3g},{row['mean_absolute_error']:.3f})"
        f" +- (0,{row.get('mean_absolute_error_sd') or 0:.3f})"
        for share, row in rows
    ]


def figure(kind: str, found: dict, arms: list[str]) -> str:
    """The figure's pgfplots. arms: a 2 x 2 grid, a panel per survey. strategies: a row per arm
    in `arms`, a column per survey."""
    surveys = [(s, title) for s, title in PANELS if any(k[0] == s for k in found)]
    if kind == "arms":
        columns, width, height = 2, r"0.5\linewidth", "6.2cm"
        cells = [(s, None, f"{title}", True) for s, title in surveys]
        plotted = [a for a, *_ in ARMS]
    else:
        columns, width, height = len(surveys), rf"{0.99 / len(surveys) + 0.02:.3f}\linewidth", "5cm"
        cells = [(s, a, title, row == 0) for row, a in enumerate(arms) for s, title in surveys]
        plotted = [ARM_NAMES[a] for a in arms]
    tops: dict[str, list[float]] = {name: [] for name in plotted}  # by arm: each row's scale
    shares = set()
    for (_, strategy), entry in found.items():
        if (kind == "arms") != (strategy == "random"):
            continue  # arms: the random draws; strategies: the shares the others drew
        for (share, name), row in entry["summary"].items():
            if name in plotted and share:
                tops[name].append(
                    row["mean_absolute_error"] + (row.get("mean_absolute_error_sd") or 0)
                )
            if share:
                shares.add(share)
    ymax = math.ceil(max(max(v) for v in tops.values() if v) * 10) / 10
    ticks = ",".join(f"{s * 100:g}" for s in sorted(shares))
    xmax = max(shares) * 100 * 1.24
    out = [
        f"% Made by scripts/strategy_figures.py {kind}"
        + (f" --arms {' '.join(arms)}" if kind == "strategies" else "")
        + " from:",
        *[f"%   {name}" for entry in found.values() for name in entry["files"]],
        COLORS
        + STYLE
        % {
            "width": width,
            "height": height,
            "legend_columns": 2 if kind == "arms" else len(STRATEGIES),
            "xmax": f"{xmax:.3g}",
            "ticks": ticks,
            "ymax": f"{ymax:g}",
            "yticks": ",".join(f"{y / 10:g}" for y in range(round(ymax * 10) + 1)),
        },
        r"\begin{tikzpicture}",
    ]
    rows = (len(cells) + columns - 1) // columns
    for n, (survey, arm, title, titled) in enumerate(cells):
        row, column = divmod(n, columns)
        letter = "abcdefghijklmnop"[n]
        place = (
            ""
            if n == 0
            else f", at={{(p{n - 1}.south east)}}, anchor=south west, xshift=0.25cm"
            if column
            else f", at={{(p{n - columns}.south west)}}, anchor=north west, yshift=-1.5cm"
        )
        if kind == "arms":
            heading = f"({letter}) {title}"
        else:
            heading = f"({letter}) {title}" if titled else f"({letter})"
        options = [f"xmode=log, occupancy axis, name=p{n}{place}", f"title={{{heading}}}"]
        if kind == "strategies":  # each row (arm) its own y scale
            options.append(y_scale(max(tops[ARM_NAMES[arm]])))
        if column == 0:
            label = "Mean absolute error (MAE)"
            if kind == "strategies":
                label = ARM_LABELS[arm]
            options.append(f"ylabel={{{label}}}")
        else:
            options.append("yticklabels={}")
        if row == rows - 1 and kind == "arms":  # strategies: one x label for the row, below
            options.append(r"xlabel={Images reviewed (\%, log scale)}")
        if n == len(cells) - 1:
            options.append(f"legend to name={kind} legend")
        out.append(f"  \\begin{{axis}}[{', '.join(options)}]")
        if kind == "arms":
            entry = found.get((survey, "random"))
            lines = []
            if entry is not None:
                alone = entry["summary"].get((0.0, "SpeciesNet"))
                if alone:
                    value = f"{alone['mean_absolute_error']:.3f}"
                    ends = f"(1.6,{value}) ({xmax:.3g},{value})"
                    out.append(f"    \\addplot[no review line] coordinates {{{ends}}};")
                lines = [(a, style, nudge, entry["summary"]) for a, style, _, nudge in ARMS]
        else:
            lines = [
                (ARM_NAMES[arm], style, nudge, found[survey, strategy]["summary"])
                for strategy, style, _, nudge in STRATEGIES
                if (survey, strategy) in found
            ]
        for a, style, nudge, summary in lines:
            data = points(summary, a, nudge, shares)
            if data:
                out.append(f"    \\addplot[{style}, forget plot] coordinates {{")
                out.append("      " + " ".join(data) + "};")
        if n == len(cells) - 1:  # the legend: every line's style, whichever panels have it
            keys = ARMS if kind == "arms" else STRATEGIES
            for number, (_, style, legend, _) in enumerate(keys):
                out.append(f"    \\addlegendimage{{{style}}}")
                out.append(f"    \\addlegendentry{{{legend}}}")
                if kind == "arms" and number == 0:
                    out.append(r"    \addlegendimage{no review line}")
                    out.append(r"    \addlegendentry{SpeciesNet's suggestions, no review}")
        out.append(r"  \end{axis}")
    corner = f"p{(rows - 1) * columns}.south west"
    if kind == "strategies":  # the panels share their x axis: label it once, centered
        middle = f"$(p{(rows - 1) * columns}.south west)!0.5!(p{len(cells) - 1}.south east)$"
        out.append(
            f"  \\node[anchor=north, font=\\small] (xlabel) at ([yshift=-0.55cm]{middle}) "
            r"{Images reviewed (\%, log scale)};"
        )
        out.append(
            r"  \node[anchor=north] at ([yshift=-0.15cm]xlabel.south) {\ref{" + kind + " legend}};"
        )
        # One y label for both rows, left of each row's own label and centered on the plots
        bottom = f"p{(rows - 1) * columns}"
        out.append(
            rf"  \path let \p1 = (p0.outer west), \p2 = ({bottom}.outer west), \p3 = (p0.north), "
            rf"\p4 = ({bottom}.south) in node[rotate=90, anchor=south, font=\small] "
            r"at ({min(\x1,\x2)}, {(\y3+\y4)/2}) {Mean absolute error (MAE)};"
        )
    else:
        out.append(
            f"  \\node[anchor=north west] at ([yshift=-1.1cm]{corner}) "
            r"{\ref{" + kind + " legend}};"
        )
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=["arms", "strategies"])
    parser.add_argument("out", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--arms", choices=list(ARM_NAMES), nargs="+", default=["reviewed", "project"]
    )
    args = parser.parse_args()
    args.out.write_text(figure(args.kind, load(args.results), args.arms))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
