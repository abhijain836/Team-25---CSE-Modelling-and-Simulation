import json
from pathlib import Path

import numpy as np

from fp_calibration import finite_difference_pose_gradient, loss_and_gradients, make_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_pose_gradient_matches_centered_finite_difference():
    config = json.loads((ROOT / "experiments/configs/m3_baseline.json").read_text())
    config.update({"grid_size": 32, "led_side": 3, "noise_fraction_of_peak": 0.0})
    experiment = make_experiment(config)
    rng = np.random.default_rng(17)
    obj = experiment.object_true * (1 + 0.01 * rng.normal(size=experiment.object_true.shape))
    pose = np.array([0.08, -0.06, np.deg2rad(0.4)])
    analytic = loss_and_gradients(obj, pose, experiment, need_object_gradient=False)[2]
    records = []
    for step in (1e-2, 1e-3, 1e-4):
        finite_difference = finite_difference_pose_gradient(obj, pose, experiment, step=step)
        relative_error = np.abs(analytic - finite_difference) / np.maximum(
            1e-12, np.maximum(np.abs(analytic), np.abs(finite_difference))
        )
        records.append(
            {
                "step": step,
                "finite_difference_gradient": finite_difference.tolist(),
                "relative_error": relative_error.tolist(),
                "max_relative_error": float(relative_error.max()),
            }
        )
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "gradient_check.json").write_text(
        json.dumps(
            {
                "analytic_gradient": analytic.tolist(),
                "records": records,
                "best_max_relative_error": min(r["max_relative_error"] for r in records),
                "fail_threshold": 1e-4,
                "passed": bool(
                    records[-1]["max_relative_error"] < 1e-4
                    and all(
                        records[i + 1]["max_relative_error"] < records[i]["max_relative_error"]
                        for i in range(len(records) - 1)
                    )
                ),
            },
            indent=2,
        )
        + "\n"
    )
    assert records[-1]["max_relative_error"] < 1e-4
    assert all(
        records[i + 1]["max_relative_error"] < records[i]["max_relative_error"]
        for i in range(len(records) - 1)
    )
