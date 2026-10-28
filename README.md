# occupancy-camtrap

Code for the paper *Which camera-trap images to review? Estimating occupancy from a small part of a survey* (Tabak, in preparation). It reproduces every experiment in the paper:

- partial reviews of four camera-trap surveys, at fixed shares of their images (2, 5, 10, and 25%, and 50% at random), choosing the images five ways: at random, from capture events with animals, spread across cameras, least certain first (close calls), and by confirming the suggestions an occupancy model counts;
- the review procedure the paper proposes (review a few capture events with animals from each camera; then, in rounds, confirm the most confident suggestion of each species-site-week not yet confirmed, retraining a model of the project's species after each round), with the stopping rules tested on its record;
- the occupancy estimates from each review, compared with those of a full review, including, in
  the procedure's runs, score-based estimates that weigh the project model's scores as evidence
  instead of counting suggestions over a cut-off.

Every review is simulated: reviewing an image means saving its label from the published dataset.

## What runs where

| Step | Needs | Code |
|---|---|---|
| 1. Fetch the four surveys from LILA BC | Python, Pillow | `datasets/` |
| 2. Detect the animals (MegaDetector v6) and fingerprint them (SpeciesNet v4.0.3a), then export | [WildObserve](https://www.qsc.earth/WildObserve) | `export/export_wildobserve.py` |
| 3. Run the experiments | Python, NumPy, SciPy, scikit-learn | `src/occreview/` |
| 4. Tables and figures | Python | `scripts/` |


Step 2 is the only one that needs WildObserve, which is commercial software. Its exported data, the input to step 3, are archived on Zenodo (10.5281/zenodo.23734806), so steps 3 and 4 can be repeated with open code alone. Reviewers can request a WildObserve license with every feature from the author.

## The data (step 2's output)

`data/<survey>/`, one folder per survey:

- `survey.json`: the categories, each site's running days, and the settings (detector threshold 0.5, seven-day occasions, suggestions counted from a score of 0.9).
- `images.csv`: every image, with its site, camera, date and time, capture event (images taken no more than 60 s apart at one camera), and labels from the dataset.
- `boxes.npz`: every animal box the detector found at 0.5 or more, with its image, its SpeciesNet fingerprint (the 1,280 numbers of the crop classifier's final pooling layer), SpeciesNet's top animal label with its score and the runner-up's score, and whether WildObserve fingerprinted it when processing the image (WildObserve fingerprints each image's five surest boxes then; only those carry suggestions on images not yet reviewed).

The surveys are Caltech Camera Traps (`cct_occupancy`), WSU Lynx (`wsu_lynx_occupancy`), Snapshot Safari's Associated Private Nature Reserves, season 4 (`snapshot_occupancy`), and Nkhotakota Camera Traps (`nkhotakota_occupancy`). All are distributed by [LILA BC](https://lila.science) under CDLA-Permissive-1.0; cite their papers (see the paper).

## Running the experiments

With [uv](https://docs.astral.sh/uv/), and `data.zip` from Zenodo unzipped in the repository's root (it makes `data/`):

```sh
uv sync
bash scripts/run_all.sh                 # every experiment on every survey, in parallel: results/
uv run pytest                           # the occupancy models' checks
```

Or one at a time:

```sh
uv run occreview strategy --survey nkhotakota_occupancy --strategy confirm
uv run occreview procedure --survey nkhotakota_occupancy --seeds 2 5 10
```

The whole set takes about 8 minutes on a laptop. Then the paper's tables and figures, from `results/` (run from the repository's root):
`results.zip` on Zenodo holds the results the paper reports; unzip it in the root to make the tables and figures without rerunning step 3.


```sh
uv run python scripts/strategy_report.py results/strategy-*.json --effort-table effort_table.tex
uv run python scripts/strategy_figures.py arms occupancy_arms.tex results/strategy-*.json
uv run python scripts/strategy_figures.py strategies occupancy_strategies.tex results/strategy-*.json
uv run python scripts/procedure_report.py results/procedure-*.json --yields 0.1 0.2 0.3 --table stopping_table.tex
uv run python scripts/procedure_figures.py procedure_progress.tex results/procedure-*.json
```

## How the code matches WildObserve

The experiments were first run inside WildObserve. This code repeats them from the exported data with independent, open implementations of the same models:

- **Occupancy** (`occupancy.py`): single-season occupancy (MacKenzie et al. 2002), detection depending on the camera-days in an occasion when they vary, and occupancy allowing for false positives among suggestions (Miller et al. 2011, unmarked's `occuFP` with type 3 data), fitted by maximum likelihood with SciPy's L-BFGS; 95% intervals on the logit scale from the inverse Hessian, as unmarked's `predict()` gives them.
- **The project model** (`classifier.py`): standardized fingerprints and a multinomial logistic regression with balanced class weights and C = 0.5 (scikit-learn), retrained as the review grows.
- **Reviews, suggestions, and detection histories** (`survey.py`, `experiments.py`): as WildObserve's occupancy tables and review lists make them.
- **Score-based occupancy** (`scores.py`, `occupancy.py`): WildObserve's score-based option, a continuous-score occupancy model (Rhinehart et al. 2022) with the scores' calibration (two Normal distributions, by EM) fitted apart from it, as ACORN fits it (Haucke et al. 2026), and the occupancy likelihood fitted by maximum likelihood. An occasion that no review settles weighs the highest log-odds that any box with a suggestion on an image not yet reviewed has for the species. Because the project model learned from the reviewed boxes, its calibration is fitted to the unsettled occasions while they still hold three or more likely detections, as WildObserve fits it.

The draws use the same random seeds and the same ordering of capture events as WildObserve's own runs, so they choose the same images. Compared with those runs:

- **The full review:** on all four surveys, the same species are scored, their occupancy agrees to within 0.0001, and their intervals agree.
- **Random review, review of capture events with animals, and review by camera:** the same images are reviewed; the mean differences from the full review agree to within 0.009 (0.001 on average).
- **Close calls, confirm, and the procedure:** these choose images by the project model's scores, which scikit-learn's solver gives very slightly differently from WildObserve's, so after the first round they review somewhat different images, and their means differ by 0.01 to 0.02 on average. The paper's conclusions are the same either way.
- **Score-based occupancy:** WildObserve's own runs of it (2026-10-25, two capture events per camera) drew their procedures from other seeds, so they reviewed other images; their means agree with these to within 0.02 after the first step, at 10% of the images, and at the stop.

The occupancy fits stop, as WildObserve's do, when no gradient component exceeds 1e-4. Where a species' detections let detection probability run toward a step in camera-days, a fit carried further reaches a slightly higher likelihood with parameters running off toward infinity and no interval; and for a species detected at only one or two sites, the likelihood can have a second, interior optimum that this starting point doesn't find. Both affect few species and draws.

The whole set runs in about 8 minutes on a 2021 laptop (Apple M1 Pro).

## License

The code is under the MIT License (LICENSE). The datasets keep their own licenses (CDLA-Permissive-1.0), and so do the data derived from them.
