"""M3 graduate-extension prototype: verify derivatives of two pupil terms."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fp_calibration import make_experiment  # noqa: E402


def zernike_like_terms(n: int) -> tuple[np.ndarray, np.ndarray]:
    f = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r2 = fx**2 + fy**2
    scale = max(float(np.sqrt(r2).max()), 1e-12)
    xn, yn = fx / scale, fy / scale
    return 2 * (xn**2 + yn**2) - 1, xn**2 - yn**2


def loss(coefficients: np.ndarray, experiment) -> float:
    z0, z1 = zernike_like_terms(experiment.object_true.shape[0])
    phase = coefficients[0] * z0 + coefficients[1] * z1
    pupil = experiment.pupil * np.exp(1j * phase)
    # A deterministic complex-pupil norm supplies a minimal differentiable unit.
    target = experiment.pupil * np.exp(1j * (0.08 * z0 - 0.05 * z1))
    return 0.5 * float(np.mean(np.abs(pupil - target) ** 2))


def analytic_gradient(coefficients: np.ndarray, experiment) -> np.ndarray:
    z0, z1 = zernike_like_terms(experiment.object_true.shape[0])
    phase = coefficients[0] * z0 + coefficients[1] * z1
    pupil = experiment.pupil * np.exp(1j * phase)
    target = experiment.pupil * np.exp(1j * (0.08 * z0 - 0.05 * z1))
    residual = pupil - target
    return np.array(
        [
            np.real(np.mean(np.conj(residual) * (1j * z0 * pupil))),
            np.real(np.mean(np.conj(residual) * (1j * z1 * pupil))),
        ]
    )


def main() -> None:
    start = time.perf_counter()
    config = json.loads((ROOT / "experiments/configs/m3_baseline.json").read_text())
    experiment = make_experiment(config)
    coefficients = np.array([0.03, -0.02])
    analytic = analytic_gradient(coefficients, experiment)
    step = 1e-6
    finite = np.empty(2)
    for j in range(2):
        plus, minus = coefficients.copy(), coefficients.copy()
        plus[j] += step
        minus[j] -= step
        finite[j] = (loss(plus, experiment) - loss(minus, experiment)) / (2 * step)
    error = np.abs(analytic - finite) / np.maximum(1e-12, np.maximum(np.abs(analytic), np.abs(finite)))
    out = {
        "coefficients": coefficients.tolist(),
        "analytic_gradient": analytic.tolist(),
        "finite_difference_gradient": finite.tolist(),
        "max_relative_error": float(error.max()),
        "pass_threshold": 1e-4,
        "passed": bool(error.max() < 1e-4),
        "runtime_seconds": time.perf_counter() - start,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results/extension_gradient_check.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}; passed={out['passed']}")


if __name__ == "__main__":
    main()
