"""Small differentiable Fourier-ptychography model implemented with NumPy.

The model uses a coherent thin-object approximation. Each LED produces a
plane-wave tilt; a circular pupil filters the modulated object's spectrum; the
camera records intensity. Analytic directional derivatives are used for the
three rigid LED-array pose variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Experiment:
    object_true: np.ndarray
    measured_amplitude: np.ndarray
    led_indices: np.ndarray
    pupil: np.ndarray
    x: np.ndarray
    y: np.ndarray
    q_step: float
    true_pose: np.ndarray
    seed: int


def _synthetic_object(n: int) -> np.ndarray:
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    amp = 0.52
    amp += 0.28 * np.exp(-((xx + 0.32) ** 2 + (yy - 0.18) ** 2) / 0.08)
    amp += 0.22 * np.exp(-((xx - 0.30) ** 2 + (yy + 0.28) ** 2) / 0.04)
    amp += 0.10 * (np.cos(5 * np.pi * xx) * np.cos(4 * np.pi * yy) + 1) / 2
    amp = np.clip(amp, 0.2, 1.0)
    phase = 0.55 * np.sin(2.5 * np.pi * xx) * np.cos(2 * np.pi * yy)
    phase += 0.35 * np.exp(-((xx - 0.05) ** 2 + (yy + 0.05) ** 2) / 0.12)
    return amp * np.exp(1j * phase)


def _pupil(n: int, radius: float) -> np.ndarray:
    f = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    return (fx**2 + fy**2 <= radius**2).astype(float)


def _led_grid(side: int) -> np.ndarray:
    half = side // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(float)


def _wavevectors(indices: np.ndarray, pose: np.ndarray, q_step: float) -> tuple[np.ndarray, np.ndarray]:
    dx, dy, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    x0, y0 = indices[:, 0], indices[:, 1]
    xr = c * x0 - s * y0 + dx
    yr = s * x0 + c * y0 + dy
    q = q_step * np.column_stack([xr, yr])

    dq = np.empty((len(indices), 3, 2), dtype=float)
    dq[:, 0, :] = (q_step, 0.0)
    dq[:, 1, :] = (0.0, q_step)
    dq[:, 2, 0] = q_step * (-s * x0 - c * y0)
    dq[:, 2, 1] = q_step * (c * x0 - s * y0)
    return q, dq


def _forward_fields(
    obj: np.ndarray,
    pose: np.ndarray,
    led_indices: np.ndarray,
    pupil: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    q_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, dq = _wavevectors(led_indices, pose, q_step)
    fields = []
    illuminations = []
    for qx, qy in q:
        illum = np.exp(2j * np.pi * (qx * x + qy * y))
        field = np.fft.ifft2(pupil * np.fft.fft2(obj * illum, norm="ortho"), norm="ortho")
        fields.append(field)
        illuminations.append(illum)
    return np.asarray(fields), np.asarray(illuminations), dq


def make_experiment(config: dict[str, Any]) -> Experiment:
    n = int(config["grid_size"])
    side = int(config["led_side"])
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    coord = np.arange(n, dtype=float) - n / 2
    y, x = np.meshgrid(coord, coord, indexing="ij")
    obj = _synthetic_object(n)
    pupil = _pupil(n, float(config["pupil_radius_cycles_per_pixel"]))
    led_indices = _led_grid(side)
    true_pose = np.array(
        [
            float(config["true_pose"]["dx_led_pitch"]),
            float(config["true_pose"]["dy_led_pitch"]),
            np.deg2rad(float(config["true_pose"]["theta_deg"])),
        ]
    )
    fields, _, _ = _forward_fields(
        obj,
        true_pose,
        led_indices,
        pupil,
        x,
        y,
        float(config["q_step_cycles_per_pixel"]),
    )
    intensity = np.abs(fields) ** 2
    sigma = float(config["noise_fraction_of_peak"]) * float(intensity.max())
    if sigma:
        intensity = np.maximum(intensity + rng.normal(0.0, sigma, intensity.shape), 0.0)
    return Experiment(
        object_true=obj,
        measured_amplitude=np.sqrt(intensity),
        led_indices=led_indices,
        pupil=pupil,
        x=x,
        y=y,
        q_step=float(config["q_step_cycles_per_pixel"]),
        true_pose=true_pose,
        seed=seed,
    )


def loss_and_gradients(
    obj: np.ndarray,
    pose: np.ndarray,
    experiment: Experiment,
    need_object_gradient: bool = True,
) -> tuple[float, np.ndarray | None, np.ndarray]:
    fields, illuminations, dq = _forward_fields(
        obj,
        pose,
        experiment.led_indices,
        experiment.pupil,
        experiment.x,
        experiment.y,
        experiment.q_step,
    )
    predicted = np.sqrt(np.abs(fields) ** 2 + 1e-12)
    residual = predicted - experiment.measured_amplitude
    scale = float(np.prod(residual.shape))
    loss = 0.5 * float(np.sum(residual**2)) / scale
    field_grad = residual * fields / predicted / scale

    obj_grad = np.zeros_like(obj) if need_object_gradient else None
    pose_grad = np.zeros(3, dtype=float)
    for m, (field, illum, gfield) in enumerate(zip(fields, illuminations, field_grad)):
        if need_object_gradient:
            back = np.fft.ifft2(
                experiment.pupil * np.fft.fft2(gfield, norm="ortho"), norm="ortho"
            )
            obj_grad += np.conj(illum) * back
        for j in range(3):
            phase_derivative = 2j * np.pi * (
                dq[m, j, 0] * experiment.x + dq[m, j, 1] * experiment.y
            )
            dfield = np.fft.ifft2(
                experiment.pupil
                * np.fft.fft2(obj * illum * phase_derivative, norm="ortho"),
                norm="ortho",
            )
            pose_grad[j] += float(np.real(np.vdot(gfield, dfield)))
    return loss, obj_grad, pose_grad


def finite_difference_pose_gradient(
    obj: np.ndarray,
    pose: np.ndarray,
    experiment: Experiment,
    step: float,
) -> np.ndarray:
    gradient = np.zeros(3, dtype=float)
    for j in range(3):
        plus = pose.copy()
        minus = pose.copy()
        plus[j] += step
        minus[j] -= step
        lp = loss_and_gradients(obj, plus, experiment, need_object_gradient=False)[0]
        lm = loss_and_gradients(obj, minus, experiment, need_object_gradient=False)[0]
        gradient[j] = (lp - lm) / (2 * step)
    return gradient


def _adam_step(value, gradient, m, v, step, learning_rate):
    beta1, beta2 = 0.9, 0.999
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (np.abs(gradient) ** 2)
    mhat = m / (1 - beta1**step)
    vhat = v / (1 - beta2**step)
    value = value - learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
    return value, m, v


def reconstruct(experiment: Experiment, config: dict[str, Any], estimate_pose: bool) -> dict[str, Any]:
    center = len(experiment.led_indices) // 2
    obj = experiment.measured_amplitude[center].astype(complex)
    pose = np.zeros(3, dtype=float)
    mo = np.zeros_like(obj)
    vo = np.zeros(obj.shape, dtype=float)
    mp = np.zeros(3, dtype=float)
    vp = np.zeros(3, dtype=float)
    losses = []
    pose_history = []
    iterations = int(config["iterations"])
    warmup = int(config["pose_warmup_iterations"])

    for step in range(1, iterations + 1):
        loss, obj_grad, pose_grad = loss_and_gradients(obj, pose, experiment)
        obj, mo, vo = _adam_step(
            obj, obj_grad, mo, vo, step, float(config["object_learning_rate"])
        )
        amplitude = np.clip(np.abs(obj), 0.05, 1.25)
        obj = amplitude * np.exp(1j * np.angle(obj))
        if estimate_pose and step > warmup:
            pose, mp, vp = _adam_step(
                pose, pose_grad, mp, vp, step - warmup, float(config["pose_learning_rate"])
            )
            pose[:2] = np.clip(pose[:2], -1.5, 1.5)
            pose[2] = np.clip(pose[2], np.deg2rad(-5.0), np.deg2rad(5.0))
        losses.append(loss)
        pose_history.append(pose.copy())

    return {
        "object": obj,
        "pose": pose,
        "losses": np.asarray(losses),
        "pose_history": np.asarray(pose_history),
    }


def aligned_nrmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    denom = np.vdot(estimate, estimate)
    scale = np.vdot(estimate, truth) / denom if abs(denom) > 0 else 0.0
    return float(np.linalg.norm(scale * estimate - truth) / np.linalg.norm(truth))
