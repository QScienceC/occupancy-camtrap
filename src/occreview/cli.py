"""Run the paper's experiments on exported surveys (data/<survey>), writing results/:

    uv run occreview strategy --survey nkhotakota_occupancy --strategy confirm
    uv run occreview procedure --survey nkhotakota_occupancy --seeds 2 5 10

strategy: partial reviews at fixed shares of the images (2, 5, 10 and 25%; and 50% at random),
five draws each, chosen by one of five strategies (random, animals, by_camera, close_calls,
confirm). procedure: the review procedure, round by round, for each first-step size (capture
events with animals per camera). See README.md for the whole set and the reports and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from occreview import experiments, survey

SURVEYS = ["cct_occupancy", "wsu_lynx_occupancy", "snapshot_occupancy", "nkhotakota_occupancy"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="occreview", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("strategy", "procedure"):
        p = sub.add_parser(name)
        p.add_argument("--survey", choices=SURVEYS, required=True)
        p.add_argument("--data", type=Path, default=Path("data"))
        p.add_argument("--out", type=Path, default=Path("results"))
        p.add_argument("--draws", type=int, default=5)
        if name == "strategy":
            p.add_argument("--strategy", choices=experiments.STRATEGIES, required=True)
            p.add_argument("--shares", type=float, nargs="+")
        else:
            p.add_argument("--seeds", type=int, nargs="+", default=[2, 5, 10])
    args = parser.parse_args(argv)
    started = time.monotonic()

    def log(text: str) -> None:
        print(f"[{time.monotonic() - started:6.0f} s] {text}", flush=True)

    found = survey.load(args.data / args.survey)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "strategy":
        shares = args.shares or (
            [0.02, 0.05, 0.1, 0.25, 0.5] if args.strategy == "random" else [0.02, 0.05, 0.1, 0.25]
        )
        results = experiments.fixed_shares(found, args.strategy, shares, args.draws, log)
        path = args.out / f"strategy-{args.survey}-{args.strategy}.json"
    else:
        results = experiments.procedures(found, args.seeds, args.draws, log)
        seeds = "-".join(map(str, args.seeds))
        path = args.out / f"procedure-{args.survey}-seed-{seeds}.json"
    path.write_text(json.dumps(results, indent=1))
    log(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
