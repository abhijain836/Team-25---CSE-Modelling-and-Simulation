import json
from pathlib import Path

import numpy as np

from fp_calibration import loss_and_gradients, make_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_true_state_has_zero_residual_without_noise():
    config = json.loads((ROOT / "experiments/configs/m3_baseline.json").read_text())
    config["noise_fraction_of_peak"] = 0.0
    experiment = make_experiment(config)
    loss = loss_and_gradients(
        experiment.object_true, experiment.true_pose, experiment, need_object_gradient=False
    )[0]
    assert loss < 1e-20
