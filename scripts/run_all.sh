#!/bin/bash
# Runs every experiment in the paper on the exported surveys in data/, writing results/ and a log
# per run in results/logs/. Each survey runs in its own process, in parallel.
#   bash scripts/run_all.sh [SURVEY...]
set -u
cd "$(dirname "$0")/.."
surveys=("$@")
[ ${#surveys[@]} -eq 0 ] && surveys=(cct_occupancy wsu_lynx_occupancy snapshot_occupancy nkhotakota_occupancy)
mkdir -p results/logs
for survey in "${surveys[@]}"; do
  (
    for strategy in random animals by_camera close_calls confirm; do
      uv run occreview strategy --survey "$survey" --strategy "$strategy" \
        > "results/logs/$survey-$strategy.log" 2>&1 || echo "FAILED $survey $strategy"
    done
    uv run occreview procedure --survey "$survey" \
      > "results/logs/$survey-procedure.log" 2>&1 || echo "FAILED $survey procedure"
    echo "done $survey"
  ) &
done
wait
echo "all done"
