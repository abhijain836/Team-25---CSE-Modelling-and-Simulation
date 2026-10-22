"""Regenerate the M3 fixed-geometry versus joint-calibration result."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fp_calibration import aligned_nrmse, make_experiment, reconstruct  # noqa: E402


def main() -> None:
    config_path = ROOT / "experiments/configs/m3_baseline.json"
    config = json.loads(config_path.read_text())
    start = time.perf_counter()
    experiment = make_experiment(config)
    fixed = reconstruct(experiment, config, estimate_pose=False)
    joint = reconstruct(experiment, config, estimate_pose=True)

    fixed_nrmse = aligned_nrmse(fixed["object"], experiment.object_true)
    joint_nrmse = aligned_nrmse(joint["object"], experiment.object_true)
    improvement = 100 * (fixed_nrmse - joint_nrmse) / fixed_nrmse
    fixed_amp_nrmse = float(
        np.linalg.norm(np.abs(fixed["object"]) - np.abs(experiment.object_true))
        / np.linalg.norm(np.abs(experiment.object_true))
    )
    joint_amp_nrmse = float(
        np.linalg.norm(np.abs(joint["object"]) - np.abs(experiment.object_true))
        / np.linalg.norm(np.abs(experiment.object_true))
    )
    pose_true = experiment.true_pose.copy()
    pose_joint = joint["pose"].copy()

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    summary = {
        "seed": experiment.seed,
        "grid_size": config["grid_size"],
        "led_frames": int(config["led_side"] ** 2),
        "iterations": config["iterations"],
        "fixed_aligned_nrmse": fixed_nrmse,
        "joint_aligned_nrmse": joint_nrmse,
        "nrmse_improvement_percent": improvement,
        "fixed_amplitude_nrmse": fixed_amp_nrmse,
        "joint_amplitude_nrmse": joint_amp_nrmse,
        "amplitude_nrmse_improvement_percent": 100
        * (fixed_amp_nrmse - joint_amp_nrmse)
        / fixed_amp_nrmse,
        "true_pose": {
            "dx_led_pitch": float(pose_true[0]),
            "dy_led_pitch": float(pose_true[1]),
            "theta_deg": float(np.rad2deg(pose_true[2])),
        },
        "estimated_pose": {
            "dx_led_pitch": float(pose_joint[0]),
            "dy_led_pitch": float(pose_joint[1]),
            "theta_deg": float(np.rad2deg(pose_joint[2])),
        },
        "final_fixed_loss": float(fixed["losses"][-1]),
        "final_joint_loss": float(joint["losses"][-1]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(9.1, 2.8))
    metric_x = np.arange(2)
    axes[0].bar(metric_x - 0.18, [fixed_nrmse, fixed_amp_nrmse], 0.36, label="Fixed", color="#7A8793")
    axes[0].bar(metric_x + 0.18, [joint_nrmse, joint_amp_nrmse], 0.36, label="Joint", color="#2F6675")
    axes[0].set_xticks(metric_x, ["Complex", "Amplitude"])
    axes[0].set_ylabel("NRMSE")
    axes[0].set_title(f"Complex NRMSE: {improvement:.1f}% lower")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].semilogy(fixed["losses"], label="Fixed", color="#7A8793")
    axes[1].semilogy(joint["losses"], label="Joint", color="#2F6675")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Amplitude loss")
    axes[1].set_title("Data-fit convergence")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    labels = ["dx", "dy", "theta"]
    truth = [pose_true[0], pose_true[1], np.rad2deg(pose_true[2])]
    estimate = [pose_joint[0], pose_joint[1], np.rad2deg(pose_joint[2])]
    pos = np.arange(3)
    width = 0.36
    axes[2].bar(pos - width / 2, truth, width, label="True", color="#B7CDD4")
    axes[2].bar(pos + width / 2, estimate, width, label="Estimated", color="#2F6675")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_xticks(pos, labels)
    axes[2].set_title("Pose (pitch, pitch, degree)")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_path = results / "m3_first_result.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary["runtime_seconds"] = time.perf_counter() - start
    (results / "m3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {figure_path.relative_to(ROOT)}")
    print(f"wrote {(results / 'm3_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
