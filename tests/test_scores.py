"""Score-based occupancy's calibration (scores.py)."""

import numpy as np
import pytest

from occreview import scores


def test_calibration_recovers_the_two_distributions():
    rng = np.random.default_rng(3)
    absent, present = rng.normal(-6, 2, 4000), rng.normal(3, 1.5, 600)
    unknown = np.concatenate([absent[:3600], present[:300]])
    found = scores.calibrate(present[300:], absent[3600:], unknown)
    assert found.absent == pytest.approx((-6, 2), abs=0.15)
    assert found.present == pytest.approx((3, 1.5), abs=0.2)


def test_the_models_own_boxes_only_start_the_calibration_while_detections_are_left():
    rng = np.random.default_rng(6)
    learned = rng.normal(9, 1, 60)  # the project model's scores on the boxes it learned from
    left = np.concatenate([rng.normal(-6, 2, 500), rng.normal(3, 2, 300)])
    assert scores.calibration_for(learned, np.array([]), left).present[0] == pytest.approx(
        3, abs=0.4
    )
    none_left = rng.normal(-6, 2, 500)
    found = scores.calibration_for(learned, np.array([]), none_left)
    assert found.present[0] == pytest.approx(9, abs=0.5)  # the reviewed ones joined the fit


def test_far_scores_never_turn_around():
    found = scores.Calibration(absent=(-6.0, 1.0), present=(3.0, 3.0), share=0.1)
    evidence = found.evidence(np.array([-40.0, -6.0, 0.0, 3.0, 40.0]))
    assert np.all(np.diff(evidence) >= 0) and evidence[0] < 0 < evidence[-1]
