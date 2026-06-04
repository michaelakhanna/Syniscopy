"""Dynamic Bayesian Fisher accumulation for per-frame CRLB estimation.

This module is estimator-focused and intentionally independent from rendering,
optics, or detector code. It provides:

- Brownian-process prior construction
- Static cumulative Fisher accumulation
- Bayesian information-filter recursion
- Optional PCRLB-style RTS smoothing
- Diffusion-prior sensitivity sweeps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "build_brownian_process_covariance",
    "build_velocity_state_transition_matrix",
    "compute_dynamic_bayesian_crlb",
    "compute_brownian_prior_sensitivity_sweep",
    "sequence_sum_fisher_to_crlb",
    "compute_dynamic_bayesian_crlb_from_fisher_sequence",
    "summarize_fisher_sequence",
]


@dataclass(frozen=True)
class DynamicBayesianCRLBResult:
    """Structured dynamic-fidelity diagnostic for a frame sequence."""

    state_axes: tuple[str, ...]
    state_transition_matrix: np.ndarray
    process_noise_covariance: np.ndarray
    initial_covariance: np.ndarray
    initial_precision: np.ndarray
    per_frame_fisher_matrices: list[np.ndarray]
    static_fisher_matrices: list[np.ndarray]
    static_covariance_matrices: list[np.ndarray]
    static_crlb: list[np.ndarray]
    dynamic_fisher_matrices: list[np.ndarray]
    dynamic_covariance_matrices: list[np.ndarray]
    dynamic_crlb: list[np.ndarray]
    dynamic_improvement_vs_static: list[np.ndarray]
    dynamic_predicted_covariance_matrices: list[np.ndarray]
    dynamic_ranks: list[int]
    static_ranks: list[int]
    per_frame_fisher_shape: tuple[int, int]
    smoothed_fisher_matrices: list[np.ndarray] | None
    smoothed_covariance_matrices: list[np.ndarray] | None
    smoothed_crlb: list[np.ndarray] | None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "state_axes": list(self.state_axes),
            "state_transition_matrix": self.state_transition_matrix.tolist(),
            "process_noise_covariance": self.process_noise_covariance.tolist(),
            "initial_covariance": self.initial_covariance.tolist(),
            "initial_precision": self.initial_precision.tolist(),
            "per_frame_fisher_shape": list(self.per_frame_fisher_shape),
            "per_frame_fisher_matrices": [m.tolist() for m in self.per_frame_fisher_matrices],
            "static_fisher_matrices": [m.tolist() for m in self.static_fisher_matrices],
            "static_covariance_matrices": [m.tolist() for m in self.static_covariance_matrices],
            "static_crlb": [c.tolist() for c in self.static_crlb],
            "dynamic_fisher_matrices": [m.tolist() for m in self.dynamic_fisher_matrices],
            "dynamic_covariance_matrices": [m.tolist() for m in self.dynamic_covariance_matrices],
            "dynamic_crlb": [c.tolist() for c in self.dynamic_crlb],
            "dynamic_improvement_vs_static": [c.tolist() for c in self.dynamic_improvement_vs_static],
            "dynamic_predicted_covariance_matrices": [m.tolist() for m in self.dynamic_predicted_covariance_matrices],
            "dynamic_ranks": list(self.dynamic_ranks),
            "static_ranks": list(self.static_ranks),
        }
        if self.smoothed_fisher_matrices is not None:
            out["smoothed_fisher_matrices"] = [m.tolist() for m in self.smoothed_fisher_matrices]
            out["smoothed_covariance_matrices"] = [m.tolist() for m in self.smoothed_covariance_matrices or []]
            out["smoothed_crlb"] = [c.tolist() for c in self.smoothed_crlb or []]
        return out


def _as_float(value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"all numeric coefficients must be finite; got {value!r}")
    return out


def _as_axis_list(state_axes: Sequence[str] | None, dim: int) -> tuple[str, ...]:
    if state_axes is None:
        if dim == 2:
            return ("x", "y")
        if dim == 3:
            return ("x", "y", "z")
        if dim == 6:
            return ("x", "y", "z", "omega_x", "omega_y", "omega_z")
        return tuple(f"s{i}" for i in range(dim))

    axes = tuple(str(axis).strip() for axis in state_axes)
    if len(axes) != dim:
        raise ValueError(f"len(state_axes) must match matrix dimension {dim}; got {len(axes)}")
    if len(axes) != len(set(axes)):
        raise ValueError("state_axes must be unique")
    return axes


def _to_square_matrix(value: Any, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (dim, dim):
        raise ValueError(f"{name} must have shape {(dim, dim)}; got {arr.shape!r}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values only")
    return arr


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _safe_inverse_psd(matrix: np.ndarray, *, floor_ratio: float = 1e-14) -> np.ndarray:
    """Robust inverse of symmetric positive semidefinite matrices."""
    mat = _symmetrize(np.asarray(matrix, dtype=float))
    if mat.shape == (0, 0):
        return mat.copy()
    eigvals, eigvecs = np.linalg.eigh(mat)
    if not np.all(np.isfinite(eigvals)):
        raise ValueError("Cannot invert a non-finite matrix")
    eps = np.finfo(float).eps
    max_eval = float(np.max(np.abs(eigvals)))
    tol = max(eps * max(1.0, max_eval), floor_ratio * eps)
    inv_vals = np.where(eigvals > tol, 1.0 / eigvals, 0.0)
    out = (eigvecs * inv_vals) @ eigvecs.T
    return _symmetrize(out)


def _matrix_rank(matrix: np.ndarray, tol: float = 1e-12) -> int:
    if matrix.size == 0:
        return 0
    sym = _symmetrize(np.asarray(matrix, dtype=float))
    eigvals = np.linalg.eigvalsh(sym)
    if not np.all(np.isfinite(eigvals)):
        return 0
    threshold = float(tol * max(1.0, np.max(np.abs(eigvals))))
    return int(np.count_nonzero(eigvals > threshold))


def _diag_crlb(covariance: np.ndarray) -> np.ndarray:
    return np.maximum(np.diag(covariance), 0.0)


def _diag_crlb_for_fisher(fisher: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return axis variances, using inf for axes outside the Fisher range."""
    fisher_sym = _symmetrize(np.asarray(fisher, dtype=float))
    cov = _symmetrize(np.asarray(covariance, dtype=float))
    variances = _diag_crlb(cov)
    if fisher_sym.shape == (0, 0):
        return variances
    eigvals, eigvecs = np.linalg.eigh(fisher_sym)
    if not np.all(np.isfinite(eigvals)):
        return np.full(fisher_sym.shape[0], float("inf"), dtype=float)
    eps = np.finfo(float).eps
    max_eval = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    tol = max(eps * max(1.0, max_eval), 1e-14 * eps)
    observable = eigvals > tol
    if not np.any(observable):
        return np.full(fisher_sym.shape[0], float("inf"), dtype=float)
    axis_observable_fraction = np.sum(eigvecs[:, observable] ** 2, axis=1)
    variances = np.asarray(variances, dtype=float)
    variances[axis_observable_fraction < 1.0 - 1e-10] = float("inf")
    return variances


def _crlb_covariance_for_fisher(fisher: np.ndarray) -> np.ndarray:
    """Pseudoinverse covariance with CRLB diagonal marked inf for null axes."""
    covariance = _safe_inverse_psd(fisher)
    out = covariance.copy()
    np.fill_diagonal(out, _diag_crlb_for_fisher(fisher, covariance))
    return out


def _axis_to_values(
    coeff: float | Sequence[float] | Mapping[str, float],
    target_axes: Sequence[str],
) -> dict[str, float]:
    """Normalize diffusion-like coefficients to an axis dictionary."""
    if isinstance(coeff, Mapping):
        result: dict[str, float] = {}
        for axis, value in coeff.items():
            axis_key = str(axis).strip()
            result[axis_key] = _as_float(value)
        return result

    arr = np.asarray(coeff, dtype=float)
    if arr.ndim == 0:
        scalar = _as_float(arr)
        return {axis: scalar for axis in target_axes}
    if arr.ndim != 1:
        raise ValueError("coefficients must be scalar-like or a one-dimensional sequence")
    if arr.size == 1:
        scalar = _as_float(arr.reshape(()))
        return {axis: scalar for axis in target_axes}
    if arr.size != len(target_axes):
        raise ValueError(
            "sequence-valued coefficients must match either one value or "
            f"one value per target axis; expected 1 or {len(target_axes)}, got {arr.size}."
        )
    return {axis: _as_float(arr[i]) for i, axis in enumerate(target_axes)}


def build_brownian_process_covariance(
    state_axes: Sequence[str],
    *,
    fps: float,
    translational_diffusion_coeff_m2_s: float | Sequence[float] | Mapping[str, float] = 0.0,
    rotational_diffusion_coeff_rad2_s: float | Sequence[float] | Mapping[str, float] = 0.0,
    translational_axes: Sequence[str] = ("x", "y", "z"),
    rotational_axes: Sequence[str] = ("omega_x", "omega_y", "omega_z"),
    extra_axis_process_variance_per_step: float | Sequence[float] | Mapping[str, float] | None = None,
) -> np.ndarray:
    """Build diagonal one-step process covariance from diffusion coefficients.

    Translational diffusion follows :math:`Q = 2 D \\Delta t` with nm units.
    Rotational coefficients are in rad^2/s and do not need unit conversion.
    """
    axes = _as_axis_list(state_axes, len(state_axes))
    fps_f = _as_float(fps)
    if fps_f <= 0.0:
        raise ValueError("fps must be > 0")
    dt = 1.0 / fps_f

    trans_axes = tuple(str(a).strip() for a in translational_axes)
    rot_axes = tuple(str(a).strip() for a in rotational_axes)

    D_trans = _axis_to_values(translational_diffusion_coeff_m2_s, trans_axes)
    D_rot = _axis_to_values(rotational_diffusion_coeff_rad2_s, rot_axes)

    q_values = {axis: 0.0 for axis in axes}
    for axis in axes:
        if axis in D_trans:
            q_values[axis] += 2.0 * D_trans[axis] * dt * 1.0e18
        if axis in D_rot:
            q_values[axis] += 2.0 * D_rot[axis] * dt

    if extra_axis_process_variance_per_step is not None:
        extra = _axis_to_values(extra_axis_process_variance_per_step, axes)
        for axis, value in extra.items():
            if axis in q_values:
                q_values[axis] = q_values[axis] + _as_float(value)

    return np.diag([q_values[axis] for axis in axes])


def build_velocity_state_transition_matrix(
    state_axes: Sequence[str],
    *,
    fps: float,
    velocity_pairs: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Build a diagonal plus position/velocity first-order kinematic block transition."""
    axes = _as_axis_list(state_axes, len(state_axes))
    fps_f = _as_float(fps)
    if fps_f <= 0.0:
        raise ValueError("fps must be > 0")
    dt = 1.0 / fps_f
    axis_to_idx = {axis: idx for idx, axis in enumerate(axes)}

    if velocity_pairs is None:
        velocity_pairs = {
            "x": "vx",
            "y": "vy",
            "z": "vz",
            "omega_x": "omega_vx",
            "omega_y": "omega_vy",
            "omega_z": "omega_vz",
        }

    F = np.eye(len(axes), dtype=float)
    for position_axis, velocity_axis in velocity_pairs.items():
        pos = axis_to_idx.get(str(position_axis).strip())
        vel = axis_to_idx.get(str(velocity_axis).strip())
        if pos is None or vel is None:
            continue
        F[pos, vel] = dt

    return F


def sequence_sum_fisher_to_crlb(
    per_frame_fisher: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Compute static cumulative Fisher and CRLB for a sequence."""
    matrices = _validate_fisher_sequence(per_frame_fisher)
    dim = matrices[0].shape[0]

    cumulative_fisher: list[np.ndarray] = []
    cumulative_covariance: list[np.ndarray] = []
    cumulative_ranks: list[int] = []

    running = np.zeros((dim, dim), dtype=float)
    for frame_fisher in matrices:
        running = _symmetrize(running + frame_fisher)
        cumulative_fisher.append(running.copy())
        cumulative_ranks.append(_matrix_rank(running))
        cumulative_covariance.append(_crlb_covariance_for_fisher(running))

    return cumulative_fisher, cumulative_covariance, cumulative_ranks


def _validate_fisher_sequence(per_frame_fisher: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(per_frame_fisher) == 0:
        raise ValueError("per_frame_fisher must contain at least one frame")

    matrices = [_symmetrize(np.asarray(item, dtype=float)) for item in per_frame_fisher]
    first_shape = matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError(f"each per-frame Fisher matrix must be square; got {first_shape!r}")
    dim = int(first_shape[0])
    if dim <= 0:
        raise ValueError("state dimension must be positive")

    for idx, matrix in enumerate(matrices):
        if matrix.shape != (dim, dim):
            raise ValueError(
                f"per_frame_fisher[{idx}] has shape {matrix.shape!r}; expected {(dim, dim)}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"per_frame_fisher[{idx}] must contain only finite values")
    return matrices


def compute_dynamic_bayesian_crlb(
    per_frame_fisher: Sequence[np.ndarray],
    process_noise_covariance: np.ndarray,
    *,
    state_axes: Sequence[str] | None = None,
    state_transition_matrix: np.ndarray | None = None,
    state_transition_fps: float | None = None,
    initial_covariance: np.ndarray | None = None,
    initial_precision: np.ndarray | None = None,
    initial_fisher: np.ndarray | None = None,
    initial_variance_fallback: float | None = None,
    include_smoothing: bool = False,
) -> DynamicBayesianCRLBResult:
    """Compute dynamic Bayesian CRLB over a sequence of per-frame Fisher matrices.

    Measurement information is accumulated by covariance prediction followed by
    an information-form measurement update:

    ``P^-_t = A P_{t-1} A^T + Q``
    ``J^-_t = (P^-_t)^{-1}``
    ``J_t = J^-_t + H_t``
    ``P_t = J_t^{-1}``

    where ``H_t`` is the per-frame measurement Fisher matrix, ``A`` the state
    transition and ``Q`` the process covariance for one step.
    """
    fisher = _validate_fisher_sequence(per_frame_fisher)
    dim = fisher[0].shape[0]
    axes = _as_axis_list(state_axes, dim)
    q = _symmetrize(_to_square_matrix(process_noise_covariance, dim, "process_noise_covariance"))
    if state_transition_matrix is None:
        if state_transition_fps is None:
            transition = np.eye(dim, dtype=float)
        else:
            transition = build_velocity_state_transition_matrix(
                axes,
                fps=state_transition_fps,
            )
    else:
        transition = _to_square_matrix(state_transition_matrix, dim, "state_transition_matrix")

    count_inputs = sum(v is not None for v in (initial_covariance, initial_precision, initial_fisher))
    if count_inputs > 1:
        raise ValueError("Only one of initial_covariance, initial_precision, initial_fisher may be supplied.")

    if initial_covariance is not None:
        p = _symmetrize(_to_square_matrix(initial_covariance, dim, "initial_covariance"))
    elif initial_fisher is not None:
        p = _safe_inverse_psd(
            _symmetrize(_to_square_matrix(initial_fisher, dim, "initial_fisher"))
        )
    elif initial_precision is not None:
        p = _safe_inverse_psd(
            _symmetrize(_to_square_matrix(initial_precision, dim, "initial_precision"))
        )
    else:
        if initial_variance_fallback is None:
            raise ValueError(
                "initial_variance_fallback must be supplied when no initial "
                "covariance, precision, or Fisher matrix is supplied."
            )
        fallback = _as_float(initial_variance_fallback)
        if not np.isfinite(fallback) or fallback <= 0.0:
            raise ValueError("initial_variance_fallback must be finite and positive")
        p = np.eye(dim, dtype=float) * fallback

    initial_covariance_used = p.copy()
    initial_precision_used = _safe_inverse_psd(initial_covariance_used)

    static_fisher: list[np.ndarray] = []
    static_covariance: list[np.ndarray] = []
    static_crlb: list[np.ndarray] = []
    static_ranks: list[int] = []

    dynamic_fisher: list[np.ndarray] = []
    dynamic_covariance: list[np.ndarray] = []
    dynamic_crlb: list[np.ndarray] = []
    dynamic_ranks: list[int] = []
    predicted_covariances: list[np.ndarray] = []

    running_fisher = np.zeros((dim, dim), dtype=float)
    for frame_fisher in fisher:
        # Static, frame-wise accumulation for comparison.
        running_fisher = _symmetrize(running_fisher + frame_fisher)
        static_fisher.append(running_fisher.copy())
        static_ranks.append(_matrix_rank(running_fisher))
        static_cov = _safe_inverse_psd(running_fisher)
        static_covariance.append(static_cov)
        static_crlb.append(_diag_crlb_for_fisher(running_fisher, static_cov))

        # Bayesian prediction/update in information form.
        predicted_cov = _symmetrize(transition @ p @ transition.T + q)
        predicted_fisher = _safe_inverse_psd(predicted_cov)
        posterior_fisher = _symmetrize(predicted_fisher + frame_fisher)
        posterior_cov = _safe_inverse_psd(posterior_fisher)

        predicted_covariances.append(predicted_cov)
        dynamic_fisher.append(posterior_fisher)
        dynamic_covariance.append(posterior_cov)
        dynamic_crlb.append(_diag_crlb_for_fisher(posterior_fisher, posterior_cov))
        dynamic_ranks.append(_matrix_rank(posterior_fisher))

        p = posterior_cov

    dynamic_improvement_vs_static: list[np.ndarray] = [
        np.maximum(static - dynamic, 0.0)
        for static, dynamic in zip(static_crlb, dynamic_crlb)
    ]

    smoothed_fisher: list[np.ndarray] | None = None
    smoothed_covariance: list[np.ndarray] | None = None
    smoothed_crlb: list[np.ndarray] | None = None

    if include_smoothing and fisher:
        n = len(fisher)
        smoothed_covariance = [None] * n  # type: ignore[assignment]
        smoothed_fisher = [None] * n  # type: ignore[assignment]
        smoothed_crlb = [None] * n  # type: ignore[assignment]

        smoothed_covariance[-1] = dynamic_covariance[-1]
        for k in range(n - 2, -1, -1):
            inv_pred_next = _safe_inverse_psd(predicted_covariances[k + 1])
            gain = _symmetrize(dynamic_covariance[k] @ transition.T @ inv_pred_next)
            candidate = smoothed_covariance[k + 1]
            if candidate is None:
                raise RuntimeError("PCRLB smoothing failed due to an internal state error.")
            smoothed_cov = _symmetrize(
                dynamic_covariance[k] + gain @ (candidate - predicted_covariances[k + 1]) @ gain.T
            )
            smoothed_covariance[k] = smoothed_cov

        smoothed_fisher = []
        smoothed_crlb = []
        for k in range(n):
            cov_k = smoothed_covariance[k]
            if cov_k is None:
                raise RuntimeError("PCRLB smoothing produced a missing covariance state.")
            cov_k = _symmetrize(np.asarray(cov_k, dtype=float))
            fisher_k = _safe_inverse_psd(cov_k)
            smoothed_fisher.append(fisher_k)
            smoothed_crlb.append(_diag_crlb_for_fisher(fisher_k, cov_k))

    return DynamicBayesianCRLBResult(
        state_axes=axes,
        state_transition_matrix=transition,
        process_noise_covariance=q,
        initial_covariance=initial_covariance_used,
        initial_precision=initial_precision_used,
        per_frame_fisher_matrices=list(fisher),
        static_fisher_matrices=static_fisher,
        static_covariance_matrices=static_covariance,
        static_crlb=static_crlb,
        dynamic_fisher_matrices=dynamic_fisher,
        dynamic_covariance_matrices=dynamic_covariance,
        dynamic_crlb=dynamic_crlb,
        dynamic_improvement_vs_static=dynamic_improvement_vs_static,
        dynamic_predicted_covariance_matrices=predicted_covariances,
        dynamic_ranks=dynamic_ranks,
        static_ranks=static_ranks,
        per_frame_fisher_shape=(dim, dim),
        smoothed_fisher_matrices=smoothed_fisher,
        smoothed_covariance_matrices=smoothed_covariance,
        smoothed_crlb=smoothed_crlb,
    )


def compute_brownian_prior_sensitivity_sweep(
    per_frame_fisher: Sequence[np.ndarray],
    process_noise_covariance: np.ndarray,
    *,
    scale_factors: Sequence[float],
    state_axes: Sequence[str] | None = None,
    state_transition_matrix: np.ndarray | None = None,
    state_transition_fps: float | None = None,
    initial_covariance: np.ndarray | None = None,
    initial_precision: np.ndarray | None = None,
    initial_fisher: np.ndarray | None = None,
    include_smoothing: bool = False,
    initial_variance_fallback: float | None = None,
) -> list[dict[str, Any]]:
    """Run repeated dynamic CRLB passes for scaled process priors.

    Used to measure diffusion-prior sensitivity.
    """
    if len(scale_factors) == 0:
        raise ValueError("scale_factors must contain at least one scale value")

    q_base = _to_square_matrix(process_noise_covariance, len(_as_axis_list(state_axes, len(process_noise_covariance))), "process_noise_covariance")
    results: list[dict[str, Any]] = []

    for raw_scale in scale_factors:
        scale = _as_float(raw_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(f"scale_factors values must be finite and non-negative; got {scale}")

        dyn = compute_dynamic_bayesian_crlb(
            per_frame_fisher,
            q_base * scale,
            state_axes=state_axes,
            state_transition_matrix=state_transition_matrix,
            state_transition_fps=state_transition_fps,
            initial_covariance=initial_covariance,
            initial_precision=initial_precision,
            initial_fisher=initial_fisher,
            include_smoothing=include_smoothing,
            initial_variance_fallback=initial_variance_fallback,
        )

        dynamic_last = dyn.dynamic_crlb[-1] if dyn.dynamic_crlb else np.array([])
        static_last = dyn.static_crlb[-1] if dyn.static_crlb else np.array([])
        improvement = np.maximum(static_last - dynamic_last, 0.0)
        results.append(
            {
                "scale": scale,
                "dynamic_crlb_last": dynamic_last.tolist(),
                "static_crlb_last": static_last.tolist(),
                "improvement_last": improvement.tolist(),
                "dynamic_rank_last": int(dyn.dynamic_ranks[-1]) if dyn.dynamic_ranks else 0,
                "static_rank_last": int(dyn.static_ranks[-1]) if dyn.static_ranks else 0,
            }
        )

    return results

def compute_dynamic_bayesian_crlb_from_fisher_sequence(
    per_frame_fisher_matrices,
    process_noise_covariance,
    *,
    state_axes: Sequence[str] | None = None,
    state_transition_matrix: np.ndarray | None = None,
    state_transition_fps: float | None = None,
    initial_covariance: np.ndarray | None = None,
    initial_precision: np.ndarray | None = None,
    initial_fisher: np.ndarray | None = None,
    initial_variance_fallback: float | None = None,
    include_smoothing: bool = False,
    include_fisher_matrices: bool = False,
    measurement_domain: str | None = None,
    signal_units: str | None = None,
    noise_variance_units: str | None = None,
    state_axis_units: Mapping[str, str] | None = None,
    sequence_crlb_model: str = "dynamic_bayesian_information_filter",
    fps: float | None = None,
    process_model: str | None = None,
    dynamic_validation_status: str = "implemented_estimator_layer",
) -> dict[str, Any]:
    """Estimator-layer wrapper around :func:`compute_dynamic_bayesian_crlb`."""
    dyn = compute_dynamic_bayesian_crlb(
        per_frame_fisher_matrices,
        process_noise_covariance,
        state_axes=state_axes,
        state_transition_matrix=state_transition_matrix,
        state_transition_fps=state_transition_fps,
        initial_covariance=initial_covariance,
        initial_precision=initial_precision,
        initial_fisher=initial_fisher,
        initial_variance_fallback=initial_variance_fallback,
        include_smoothing=include_smoothing,
    )

    axes = list(dyn.state_axes)
    if state_axis_units is None:
        state_axis_units = {
            axis: ("radian" if str(axis).startswith("omega") else "nm")
            for axis in axes
        }
    q_units = [
        f"{state_axis_units.get(axis, 'state_unit')}^2/frame"
        for axis in axes
    ]
    state_covariance_units = [
        f"{state_axis_units.get(axis, 'state_unit')}^2"
        for axis in axes
    ]
    out: dict[str, Any] = {
        "sequence_crlb_model": str(sequence_crlb_model),
        "sequence_enabled": len(dyn.per_frame_fisher_matrices) > 1,
        "dynamic_bayesian_enabled": True,
        "dynamic_validation_status": str(dynamic_validation_status),
        "frame_count": int(len(dyn.per_frame_fisher_matrices)),
        "fps": None if fps is None else float(fps),
        "dt_seconds": None if fps is None else float(1.0 / float(fps)),
        "measurement_domain": measurement_domain,
        "signal_units": signal_units,
        "noise_variance_units": noise_variance_units,
        "state_axis_units": dict(state_axis_units),
        "process_model": process_model or "configured_linear_gaussian_process",
        "process_noise_covariance_units": q_units,
        "initial_covariance_units": state_covariance_units,
        "state_axes": list(dyn.state_axes),
        "state_transition_matrix": dyn.state_transition_matrix.tolist(),
        "process_noise_covariance": dyn.process_noise_covariance.tolist(),
        "initial_covariance": dyn.initial_covariance.tolist(),
        "initial_precision": dyn.initial_precision.tolist(),
        "static_crlb_final": dyn.static_crlb[-1].tolist() if dyn.static_crlb else [],
        "dynamic_crlb_final": dyn.dynamic_crlb[-1].tolist() if dyn.dynamic_crlb else [],
        "dynamic_improvement_final": dyn.dynamic_improvement_vs_static[-1].tolist()
        if dyn.dynamic_improvement_vs_static
        else [],
        "dynamic_ranks": list(dyn.dynamic_ranks),
        "static_ranks": list(dyn.static_ranks),
        "per_frame_fisher_shape": list(dyn.per_frame_fisher_shape),
    }

    if include_fisher_matrices:
        out.update(
            {
                "static_crlb": [row.tolist() for row in dyn.static_crlb],
                "dynamic_crlb": [row.tolist() for row in dyn.dynamic_crlb],
                "dynamic_improvement_vs_static": [
                    row.tolist() for row in dyn.dynamic_improvement_vs_static
                ],
            }
        )
    if include_smoothing and dyn.smoothed_crlb is not None:
        out.update(
            {
                "smoothed_crlb": [row.tolist() for row in dyn.smoothed_crlb],
                "smoothed_covariance": [row.tolist() for row in (dyn.smoothed_covariance_matrices or [])],
            }
        )

    return out

def summarize_fisher_sequence(
    per_frame_fisher_matrices,
    *,
    state_axes: Sequence[str] | None = None,
    measurement_domain: str | None = None,
    signal_units: str | None = None,
    noise_variance_units: str | None = None,
    state_axis_units: Mapping[str, str] | None = None,
    dynamic_process_noise_covariance: np.ndarray | None = None,
    dynamic_bayesian_enabled: bool = False,
    fps: float | None = None,
    initial_covariance: np.ndarray | None = None,
    include_smoothing: bool = False,
) -> dict[str, Any]:
    """Return static-cumulative and optional dynamic sequence Fisher metadata."""
    matrices = [np.asarray(item, dtype=float) for item in per_frame_fisher_matrices]
    if not matrices:
        raise ValueError("per_frame_fisher_matrices must contain at least one matrix.")
    cumulative_fisher, cumulative_covariance, cumulative_ranks = sequence_sum_fisher_to_crlb(matrices)
    axes = list(state_axes) if state_axes is not None else [f"s{i}" for i in range(matrices[0].shape[0])]
    static_final_crlb = np.maximum(np.diag(cumulative_covariance[-1]), 0.0)
    out: dict[str, Any] = {
        "sequence_crlb_model": "static_same_state_cumulative",
        "same_state_assumption": True,
        "safe_for_dynamic_sequence_claim": False,
        "sequence_enabled": len(matrices) > 1,
        "dynamic_bayesian_enabled": bool(dynamic_bayesian_enabled),
        "frame_count": int(len(matrices)),
        "state_axes": axes,
        "measurement_domain": measurement_domain,
        "signal_units": signal_units,
        "noise_variance_units": noise_variance_units,
        "state_axis_units": dict(state_axis_units or {}),
        "static_fisher_final": cumulative_fisher[-1].tolist(),
        "static_crlb_final": static_final_crlb.tolist(),
        "static_ranks": list(cumulative_ranks),
    }
    if dynamic_bayesian_enabled:
        if dynamic_process_noise_covariance is None:
            raise ValueError("dynamic_process_noise_covariance is required when dynamic_bayesian_enabled=True.")
        out["dynamic_bayesian_crlb"] = compute_dynamic_bayesian_crlb_from_fisher_sequence(
            matrices,
            dynamic_process_noise_covariance,
            state_axes=state_axes,
            initial_covariance=initial_covariance,
            state_transition_fps=fps,
            include_smoothing=include_smoothing,
            measurement_domain=measurement_domain,
            signal_units=signal_units,
            noise_variance_units=noise_variance_units,
            state_axis_units=state_axis_units,
            fps=fps,
            process_model="brownian_or_configured_linear_gaussian",
        )
    return out
