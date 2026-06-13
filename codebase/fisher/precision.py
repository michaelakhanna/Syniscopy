"""Shared Fisher precision operators for analysis-noise likelihoods."""

from __future__ import annotations

from typing import Any

import numpy as np

from noise_contracts import AnalysisNoiseModel, IndependentPixelNoiseModel

from ._constants import _FISHER_VARIANCE_FLOOR


def _validate_gradient_stack(grads: tuple[np.ndarray, ...], *, context: str) -> tuple[tuple[np.ndarray, ...], tuple[int, int]]:
    if not grads:
        raise ValueError(f"{context} requires at least one derivative image.")
    arrays = tuple(np.asarray(grad, dtype=float) for grad in grads)
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError(f"{context} derivative images must be 2D; got {shape}.")
    for grad in arrays:
        if grad.shape != shape:
            raise ValueError(f"{context} derivative images must share shape {shape}; got {grad.shape}.")
        if np.any(~np.isfinite(grad)):
            raise ValueError(f"{context} derivative images must contain only finite values.")
    return arrays, shape


def _validate_complex_gradient_stack(
    grads: tuple[np.ndarray, ...],
    *,
    context: str,
) -> tuple[tuple[np.ndarray, ...], tuple[int, int]]:
    if not grads:
        raise ValueError(f"{context} requires at least one derivative image.")
    arrays = tuple(np.asarray(grad, dtype=np.complex128) for grad in grads)
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError(f"{context} derivative images must be 2D; got {shape}.")
    for grad in arrays:
        if grad.shape != shape:
            raise ValueError(f"{context} derivative images must share shape {shape}; got {grad.shape}.")
        if np.any(~np.isfinite(grad.real)) or np.any(~np.isfinite(grad.imag)):
            raise ValueError(f"{context} derivative images must contain only finite values.")
    return arrays, shape


def _total_variance_array(noise_variance_map: np.ndarray | float, shape: tuple[int, int], *, context: str) -> np.ndarray:
    if np.isscalar(noise_variance_map):
        value = float(noise_variance_map)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{context} noise variance scalar must be positive; got {noise_variance_map!r}.")
        return np.full(shape, value, dtype=float)
    var = np.asarray(noise_variance_map, dtype=float)
    if var.shape != shape:
        raise ValueError(f"{context} noise variance shape {var.shape} does not match derivative shape {shape}.")
    if np.any(~np.isfinite(var)):
        raise ValueError(f"{context} noise variance must contain only finite values.")
    if np.any(var <= 0.0):
        raise ValueError(f"{context} noise variance must contain only positive values.")
    return var


def _variance_floor_support_mask(
    total_var: np.ndarray,
    *,
    variance_floor: float,
    context: str,
) -> np.ndarray:
    """Return the Fisher-active support for a diagonal variance array.

    The detector/noise layer uses ``_FISHER_VARIANCE_FLOOR`` as a numerical
    regularizer for pixels with no physical count/readout support.  That floor
    is not a calibrated variance.  Treating it as physical would make
    ``g**2 / floor`` dominate Fisher maps and scalar CRLBs.
    """

    floor = float(variance_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"{context} variance_floor must be positive and finite; got {variance_floor!r}.")
    return np.asarray(total_var, dtype=float) > floor


def _independent_precision_weighted_gradients(
    grads: tuple[np.ndarray, ...],
    total_var: np.ndarray,
    *,
    variance_floor: float = _FISHER_VARIANCE_FLOOR,
    context: str,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Return diagonal ``Sigma^-1 g`` with the Syniscopy floor contract applied."""

    active = _variance_floor_support_mask(
        total_var,
        variance_floor=variance_floor,
        context=context,
    )
    # This is the central Fisher likelihood contract: the numerical variance
    # floor is an inactive-support sentinel, not a physical precision source.
    # Scalar CRLBs, explicit-derivative CRLBs, and saved density maps must all
    # pass through this mask so they cannot assign different information to
    # the same detector/noise model.
    inv = np.divide(
        1.0,
        total_var,
        out=np.zeros_like(total_var, dtype=float),
        where=active,
    )
    inactive_count = int(total_var.size - np.count_nonzero(active))
    return tuple(grad * inv for grad in grads), {
        "precision_covariance_kind": "independent_pixels",
        "density_basis": "independent_pixel_variance",
        "density_may_be_signed": False,
        "precision_variance_floor": float(variance_floor),
        "precision_floor_policy": "zero_information_at_or_below_numerical_floor",
        "precision_inactive_floor_pixel_count": inactive_count,
        "precision_active_pixel_count": int(np.count_nonzero(active)),
    }


def _project_baseband_sideband(image: np.ndarray, baseband_mask: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(np.asarray(image, dtype=np.complex128)) * baseband_mask)


def _sideband_phase_correction_array(
    noise_model: AnalysisNoiseModel,
    shape: tuple[int, int],
    *,
    context: str,
) -> np.ndarray:
    phase_raw = noise_model.fourier_sideband_phase_correction
    if phase_raw is None:
        return np.ones(shape, dtype=np.complex128)
    phase = np.asarray(phase_raw, dtype=np.complex128)
    if phase.shape != shape:
        raise ValueError(
            f"{context} sideband phase-correction shape {phase.shape} "
            f"does not match gradient shape {shape}."
        )
    if np.any(~np.isfinite(phase.real)) or np.any(~np.isfinite(phase.imag)):
        raise ValueError(f"{context} sideband phase correction must be finite.")
    if not np.allclose(np.abs(phase), 1.0, rtol=1.0e-6, atol=1.0e-6):
        raise ValueError(f"{context} sideband phase correction must be unit magnitude.")
    return phase


def _sideband_output_normalization_array(
    noise_model: AnalysisNoiseModel,
    shape: tuple[int, int],
    *,
    context: str,
) -> np.ndarray:
    norm_raw = noise_model.fourier_sideband_output_normalization
    if norm_raw is None:
        return np.ones(shape, dtype=np.complex128)
    norm = np.asarray(norm_raw, dtype=np.complex128)
    if norm.shape != shape:
        raise ValueError(
            f"{context} sideband output-normalization shape {norm.shape} "
            f"does not match gradient shape {shape}."
        )
    if np.any(~np.isfinite(norm.real)) or np.any(~np.isfinite(norm.imag)):
        raise ValueError(f"{context} sideband output normalization must be finite.")
    if np.any(np.abs(norm) <= 0.0):
        raise ValueError(f"{context} sideband output normalization must be nonzero.")
    return norm


def _apply_sideband_output_transform(
    baseband_image: np.ndarray,
    output_normalization: np.ndarray,
    *,
    conjugate_output: bool,
) -> np.ndarray:
    z = np.asarray(baseband_image, dtype=np.complex128)
    if bool(conjugate_output):
        z = np.conj(z)
    return np.asarray(output_normalization, dtype=np.complex128) * z


def _apply_sideband_output_adjoint(
    observation_image: np.ndarray,
    output_normalization: np.ndarray,
    *,
    conjugate_output: bool,
) -> np.ndarray:
    """Real-adjoint of z -> n*z or z -> n*conj(z)."""

    z = np.asarray(observation_image, dtype=np.complex128)
    norm = np.asarray(output_normalization, dtype=np.complex128)
    if bool(conjugate_output):
        return norm * np.conj(z)
    return np.conj(norm) * z


def _invert_sideband_output_transform(
    observation_image: np.ndarray,
    output_normalization: np.ndarray,
    *,
    conjugate_output: bool,
) -> np.ndarray:
    z = np.asarray(observation_image, dtype=np.complex128) / np.asarray(
        output_normalization,
        dtype=np.complex128,
    )
    if bool(conjugate_output):
        return np.conj(z)
    return z


def _project_demodulated_sideband(
    image: np.ndarray,
    baseband_mask: np.ndarray,
    phase_correction: np.ndarray,
) -> np.ndarray:
    z = np.asarray(image, dtype=np.complex128)
    demodulated = np.asarray(phase_correction, dtype=np.complex128)
    return demodulated * _project_baseband_sideband(np.conj(demodulated) * z, baseband_mask)


def _project_sideband_observation(
    image: np.ndarray,
    baseband_mask: np.ndarray,
    phase_correction: np.ndarray,
    output_normalization: np.ndarray,
    *,
    conjugate_output: bool,
) -> np.ndarray:
    baseband = _invert_sideband_output_transform(
        image,
        output_normalization,
        conjugate_output=conjugate_output,
    )
    projected = _project_demodulated_sideband(
        baseband,
        baseband_mask,
        phase_correction,
    )
    return _apply_sideband_output_transform(
        projected,
        output_normalization,
        conjugate_output=conjugate_output,
    )


def _sideband_forward_from_raw(
    raw_image: np.ndarray,
    *,
    carrier_mask: np.ndarray,
    sideband_shift: tuple[int, int],
    phase_correction: np.ndarray | None = None,
    output_normalization: np.ndarray | None = None,
    conjugate_output: bool = False,
) -> np.ndarray:
    spectrum = np.fft.fft2(np.asarray(raw_image, dtype=np.complex128))
    baseband = np.fft.ifft2(np.roll(spectrum * carrier_mask, sideband_shift, axis=(0, 1)))
    if phase_correction is not None:
        baseband = np.asarray(phase_correction, dtype=np.complex128) * baseband
    if output_normalization is None:
        return baseband
    return _apply_sideband_output_transform(
        baseband,
        output_normalization,
        conjugate_output=conjugate_output,
    )


def _sideband_adjoint_to_raw(
    baseband_image: np.ndarray,
    *,
    carrier_mask: np.ndarray,
    sideband_shift: tuple[int, int],
    phase_correction: np.ndarray | None = None,
    output_normalization: np.ndarray | None = None,
    conjugate_output: bool = False,
) -> np.ndarray:
    z = np.asarray(baseband_image, dtype=np.complex128)
    if output_normalization is not None:
        z = _apply_sideband_output_adjoint(
            z,
            output_normalization,
            conjugate_output=conjugate_output,
        )
    if phase_correction is not None:
        z = np.conj(np.asarray(phase_correction, dtype=np.complex128)) * z
    spectrum = np.fft.fft2(z)
    return np.fft.ifft2(carrier_mask * np.roll(spectrum, (-sideband_shift[0], -sideband_shift[1]), axis=(0, 1)))


def _conjugate_gradient_complex(
    operator,
    rhs: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    context: str,
    solver_name: str = "complex_conjugate_gradient",
) -> tuple[np.ndarray, dict[str, Any]]:
    b = np.asarray(rhs, dtype=np.complex128)
    x = np.zeros_like(b, dtype=np.complex128)
    r = b - operator(x)
    p = r.copy()
    rsold = float(np.real(np.vdot(r, r)))
    initial = max(float(np.sqrt(max(rsold, 0.0))), 1e-300)
    if initial <= tolerance:
        return x, {
            "precision_solver": solver_name,
            "precision_solver_iterations": 0,
            "precision_solver_relative_residual": 0.0,
        }
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        Ap = operator(p)
        denom = np.vdot(p, Ap)
        denom_real = float(np.real(denom))
        if (not np.isfinite(denom_real)) or denom_real <= 0.0:
            raise RuntimeError(
                f"{context} sideband covariance precision solve encountered a "
                "non-positive conjugate-gradient denominator."
            )
        alpha = rsold / denom_real
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = float(np.real(np.vdot(r, r)))
        if not np.isfinite(rsnew) or rsnew < 0.0:
            raise RuntimeError(f"{context} sideband covariance precision solve diverged.")
        relative = float(np.sqrt(rsnew) / initial)
        if relative <= float(tolerance):
            return x, {
                "precision_solver": solver_name,
                "precision_solver_iterations": iterations,
                "precision_solver_relative_residual": relative,
            }
        beta = rsnew / max(rsold, 1e-300)
        p = r + beta * p
        rsold = rsnew
    return x, {
        "precision_solver": solver_name,
        "precision_solver_iterations": iterations,
        "precision_solver_relative_residual": float(np.sqrt(max(rsold, 0.0)) / initial),
        "precision_solver_status": "max_iterations_reached",
    }


def compute_fisher_from_complex_fourier_sideband_gradients(
    grads: tuple[np.ndarray, ...],
    noise_model: AnalysisNoiseModel,
    *,
    variance_floor: float = _FISHER_VARIANCE_FLOOR,
    tolerance: float = 1e-8,
    max_iterations: int = 256,
    context: str = "complex Fourier-sideband Fisher information",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute Fisher for a demodulated complex sideband with propagated noise.

    Off-axis DHM reconstructs a complex field by applying a Fourier sideband
    projection to a raw interferogram.  The raw detector-count noise is diagonal
    before this linear demodulation and spatially correlated afterward.  This
    helper applies the covariance inverse through the sideband operator rather
    than treating the reconstructed field as independent pixels.
    """

    if not isinstance(noise_model, AnalysisNoiseModel):
        raise TypeError(f"{context} requires an AnalysisNoiseModel.")
    noise_model.require_safe_for_fisher(context=context)
    if str(noise_model.covariance_kind) != "fourier_sideband_demodulated_complex_field":
        raise ValueError(
            f"{context} requires covariance_kind='fourier_sideband_demodulated_complex_field'; "
            f"got {noise_model.covariance_kind!r}."
        )
    grads, shape = _validate_complex_gradient_stack(grads, context=context)
    raw_variance = _total_variance_array(
        noise_model.fourier_sideband_raw_variance,
        shape,
        context=context,
    )
    mask = np.asarray(noise_model.fourier_sideband_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{context} sideband mask shape {mask.shape} does not match gradient shape {shape}.")
    if not np.any(mask):
        raise ValueError(f"{context} sideband mask is empty.")
    if np.any(raw_variance <= float(variance_floor)):
        raise ValueError(
            f"{context} raw demodulation variance contains numerical-floor or "
            "inactive pixels; a Fourier sideband covariance cannot assign "
            "physical precision to those samples."
        )
    raw_row_variance = float(noise_model.row_correlated_variance)
    if not np.isfinite(raw_row_variance) or raw_row_variance < 0.0:
        raise ValueError(
            f"{context} raw row-correlated variance must be finite and non-negative; "
            f"got {raw_row_variance!r}."
        )
    independent_raw_variance = raw_variance
    if raw_row_variance > 0.0:
        independent_raw_variance = raw_variance - raw_row_variance
        if np.any(~np.isfinite(independent_raw_variance)) or np.any(independent_raw_variance <= float(variance_floor)):
            raise ValueError(
                f"{context} requires raw signal independent variance above the "
                "numerical floor after removing scan-line row covariance."
            )
    sideband_shift = tuple(int(v) for v in noise_model.fourier_sideband_shift)
    baseband_mask = np.roll(mask, sideband_shift, axis=(0, 1)).astype(float)
    carrier_mask = mask.astype(float)
    phase_correction = _sideband_phase_correction_array(noise_model, shape, context=context)
    has_fractional_phase = not np.allclose(phase_correction, 1.0 + 0.0j, rtol=0.0, atol=1.0e-12)
    output_normalization = _sideband_output_normalization_array(
        noise_model,
        shape,
        context=context,
    )
    has_output_normalization = not np.allclose(
        output_normalization,
        1.0 + 0.0j,
        rtol=0.0,
        atol=1.0e-12,
    )
    conjugate_output = bool(noise_model.fourier_sideband_output_conjugate)

    def covariance_action(baseband_image: np.ndarray) -> np.ndarray:
        # The sideband field is complex, but it is a linear transform of one
        # real raw detector frame. The likelihood therefore lives on the real
        # vector [Re(field), Im(field)], with covariance B Sigma_raw B^T.  The
        # real adjoint is Re(A^* u), not the complex adjoint A^* u.
        supported = _project_sideband_observation(
            baseband_image,
            baseband_mask,
            phase_correction,
            output_normalization,
            conjugate_output=conjugate_output,
        )
        raw_adjoint = np.real(_sideband_adjoint_to_raw(
            supported,
            carrier_mask=carrier_mask,
            sideband_shift=sideband_shift,
            phase_correction=phase_correction,
            output_normalization=output_normalization,
            conjugate_output=conjugate_output,
        ))
        weighted_raw = independent_raw_variance * raw_adjoint
        if raw_row_variance > 0.0:
            # Raw scan-line noise is one independent additive offset per
            # detector row.  Propagate that raw-domain low-rank covariance
            # through the same sideband operator instead of folding it into a
            # fake independent-pixel diagonal on the demodulated field.
            weighted_raw = weighted_raw + raw_row_variance * np.sum(raw_adjoint, axis=1, keepdims=True)
        return _project_sideband_observation(
            _sideband_forward_from_raw(
                weighted_raw,
                carrier_mask=carrier_mask,
                sideband_shift=sideband_shift,
                phase_correction=phase_correction,
                output_normalization=output_normalization,
                conjugate_output=conjugate_output,
            ),
            baseband_mask,
            phase_correction,
            output_normalization,
            conjugate_output=conjugate_output,
        )

    supported_grads = tuple(
        _project_sideband_observation(
            grad,
            baseband_mask,
            phase_correction,
            output_normalization,
            conjugate_output=conjugate_output,
        )
        for grad in grads
    )
    weighted: list[np.ndarray] = []
    solver_records: list[dict[str, Any]] = []
    for grad in supported_grads:
        solved, record = _conjugate_gradient_complex(
            covariance_action,
            grad,
            tolerance=float(tolerance),
            max_iterations=int(max_iterations),
            context=context,
            solver_name="real_augmented_conjugate_gradient",
        )
        weighted.append(
            _project_sideband_observation(
                solved,
                baseband_mask,
                phase_correction,
                output_normalization,
                conjugate_output=conjugate_output,
            )
        )
        solver_records.append(record)

    F = np.zeros((len(supported_grads), len(supported_grads)), dtype=float)
    for i, gi in enumerate(supported_grads):
        for j, wgj in enumerate(weighted[i:], start=i):
            value = float(np.real(np.vdot(gi, wgj)))
            F[i, j] = value
            if j != i:
                F[j, i] = value
    max_relative = max(
        (float(record.get("precision_solver_relative_residual", 0.0)) for record in solver_records),
        default=0.0,
    )
    metadata = {
        "precision_covariance_kind": "fourier_sideband_demodulated_complex_field",
        "precision_complex_observation_model": "real_augmented_re_im_covariance",
        "density_basis": "fourier_sideband_real_augmented_covariance_weighted_quadratic_form",
        "density_may_be_signed": True,
        "precision_variance_floor": float(variance_floor),
        "precision_floor_policy": "require_raw_variance_above_numerical_floor",
        "precision_sideband_selected_coefficient_count": int(np.count_nonzero(mask)),
        "precision_sideband_shift": [int(sideband_shift[0]), int(sideband_shift[1])],
        "precision_sideband_fractional_phase_correction": bool(has_fractional_phase),
        "precision_sideband_output_normalization": bool(has_output_normalization),
        "precision_sideband_output_conjugate": bool(conjugate_output),
        "precision_raw_row_correlated_variance": raw_row_variance,
        "precision_solver": "real_augmented_conjugate_gradient",
        "precision_solver_max_relative_residual": max_relative,
        "precision_solver_iterations_by_axis": [
            int(record.get("precision_solver_iterations", 0)) for record in solver_records
        ],
    }
    return F, metadata


def _component_precision_weighted_gradients(
    grads: tuple[np.ndarray, ...],
    total_var: np.ndarray,
    component_variances: np.ndarray,
    couplings: np.ndarray,
    *,
    context: str,
    variance_floor: float = _FISHER_VARIANCE_FLOOR,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    if couplings.ndim != 3 or couplings.shape[1:] != total_var.shape:
        raise ValueError(
            f"{context} row-correlated couplings must have shape (components, H, W); "
            f"got {couplings.shape!r} for variance {total_var.shape!r}."
        )
    if component_variances.shape != (couplings.shape[0],):
        raise ValueError(f"{context} component variances must have one value per coupling component.")
    if np.any(~np.isfinite(component_variances)) or np.any(component_variances < 0.0):
        raise ValueError(f"{context} component variances must be finite and non-negative.")

    row_diag = np.sum(component_variances[:, None, None] * np.square(couplings), axis=0)
    independent_var = total_var - row_diag
    if np.any(~np.isfinite(independent_var)) or np.any(independent_var <= float(variance_floor)):
        raise ValueError(
            f"{context} requires a physical independent per-pixel variance above "
            "the Fisher numerical floor after removing row-correlated covariance "
            "diagonals. A singular row covariance must be represented by an "
            "explicit covariance model, not by inverting the numerical floor."
        )

    weighted = [np.zeros_like(grad, dtype=float) for grad in grads]
    for row in range(total_var.shape[0]):
        inv_diag = 1.0 / independent_var[row]
        active = component_variances > 0.0
        if np.any(active):
            u = couplings[active, row, :].T * np.sqrt(component_variances[active])[None, :]
            middle = np.eye(u.shape[1]) + (u.T * inv_diag[None, :]) @ u
            middle_inv = np.linalg.pinv(middle)
        else:
            u = np.zeros((total_var.shape[1], 0), dtype=float)
            middle_inv = np.zeros((0, 0), dtype=float)
        for out, grad in zip(weighted, grads):
            g_row = np.asarray(grad[row], dtype=float)
            g_dinv = g_row * inv_diag
            if u.shape[1]:
                alpha = middle_inv @ (u.T @ g_dinv)
                out[row] = g_dinv - (inv_diag[:, None] * u) @ alpha
            else:
                out[row] = g_dinv
    return tuple(weighted), {
        "precision_covariance_kind": "row_correlated_scan_lines",
        "density_basis": "covariance_weighted_quadratic_form",
        "density_may_be_signed": True,
        "precision_variance_floor": float(variance_floor),
        "precision_floor_policy": "require_independent_variance_above_numerical_floor",
    }


def apply_analysis_noise_precision(
    grads: tuple[np.ndarray, ...],
    noise_variance_map: AnalysisNoiseModel | IndependentPixelNoiseModel,
    *,
    row_correlated_line_variance: float = 0.0,
    variance_floor: float = _FISHER_VARIANCE_FLOOR,
    context: str = "Fisher precision application",
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Return ``Sigma^-1 g`` for derivative images in a typed likelihood basis.

    This helper is the single contract seam between detector-noise likelihoods
    and Fisher/CRLB diagnostics. Fisher precision requires a typed Fisher
    likelihood: ``AnalysisNoiseModel`` for structured covariance or
    ``IndependentPixelNoiseModel`` for an explicitly complete diagonal
    covariance. Raw arrays are rejected because diagonal report summaries and
    complete independent covariance have identical shapes but different physics.
    """
    grads, shape = _validate_gradient_stack(grads, context=context)

    if row_correlated_line_variance != 0.0:
        raise TypeError(
            f"{context} row_correlated_line_variance is no longer accepted as a "
            "side-channel covariance. Pass an AnalysisNoiseModel carrying the "
            "row-correlated likelihood, or use IndependentPixelNoiseModel when "
            "the diagonal variance is the complete independent-pixel covariance."
        )

    if isinstance(noise_variance_map, AnalysisNoiseModel):
        noise_variance_map.require_safe_for_fisher(context=context)
        total_var = _total_variance_array(noise_variance_map.diagonal_variance, shape, context=context)
        if noise_variance_map.row_correlated_couplings is not None:
            component_variances = np.asarray(
                noise_variance_map.row_correlated_component_variances,
                dtype=float,
            )
            couplings = np.asarray(noise_variance_map.row_correlated_couplings, dtype=float)
            return _component_precision_weighted_gradients(
                grads,
                total_var,
                component_variances,
                couplings,
                context=context,
                variance_floor=variance_floor,
            )
        line_variance = float(noise_variance_map.row_correlated_variance)
        covariance_kind = str(noise_variance_map.covariance_kind)
    elif isinstance(noise_variance_map, IndependentPixelNoiseModel):
        noise_variance_map.require_safe_for_fisher(context=context)
        total_var = _total_variance_array(noise_variance_map.diagonal_variance, shape, context=context)
        line_variance = 0.0
        covariance_kind = "independent_pixels"
    else:
        raise TypeError(
            f"{context} requires a typed Fisher likelihood. Pass AnalysisNoiseModel "
            "for structured covariance, or wrap diagonal independent-pixel variance "
            "with independent_pixel_noise_model(...). Raw arrays are rejected because "
            "a diagonal summary is not a complete covariance contract."
        )

    if not np.isfinite(line_variance) or line_variance < 0.0:
        raise ValueError(f"{context} row-correlated variance must be finite and non-negative; got {line_variance!r}.")
    if line_variance > 0.0:
        couplings = np.ones((1, *shape), dtype=float)
        return _component_precision_weighted_gradients(
            grads,
            total_var,
            np.asarray((line_variance,), dtype=float),
            couplings,
            context=context,
            variance_floor=variance_floor,
        )

    if covariance_kind not in {"independent_pixels", "row_correlated_scan_lines"}:
        raise ValueError(f"{context} does not support covariance_kind={covariance_kind!r}.")
    return _independent_precision_weighted_gradients(
        grads,
        total_var,
        variance_floor=variance_floor,
        context=context,
    )


def compute_fisher_from_gradients_with_noise(
    grads: tuple[np.ndarray, ...],
    noise_variance_map: AnalysisNoiseModel | IndependentPixelNoiseModel,
    *,
    row_correlated_line_variance: float = 0.0,
    variance_floor: float = _FISHER_VARIANCE_FLOOR,
    context: str = "Fisher information",
) -> np.ndarray:
    """Compute ``g_i.T @ Sigma^-1 @ g_j`` using the shared precision operator."""
    grads, _shape = _validate_gradient_stack(grads, context=context)
    weighted, _metadata = apply_analysis_noise_precision(
        grads,
        noise_variance_map,
        row_correlated_line_variance=row_correlated_line_variance,
        variance_floor=variance_floor,
        context=context,
    )
    F = np.zeros((len(grads), len(grads)), dtype=float)
    for i, gi in enumerate(grads):
        for j, wgj in enumerate(weighted[i:], start=i):
            value = float(np.sum(gi * wgj))
            F[i, j] = value
            if j != i:
                F[j, i] = value
    return F


def compute_fisher_density_maps_from_gradients(
    named_grads: dict[str, np.ndarray],
    noise_variance_map: AnalysisNoiseModel | IndependentPixelNoiseModel,
    *,
    row_correlated_line_variance: float = 0.0,
    variance_floor: float = _FISHER_VARIANCE_FLOOR,
    context: str = "Fisher density maps",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return per-parameter maps ``g * (Sigma^-1 g)`` in the CRLB likelihood basis."""
    names = list(named_grads)
    grads = tuple(np.asarray(named_grads[name], dtype=float) for name in names)
    weighted, metadata = apply_analysis_noise_precision(
        grads,
        noise_variance_map,
        row_correlated_line_variance=row_correlated_line_variance,
        variance_floor=variance_floor,
        context=context,
    )
    maps = {
        name: (grad * weighted_grad).astype(float)
        for name, grad, weighted_grad in zip(names, grads, weighted)
    }
    return maps, metadata


__all__ = [
    "apply_analysis_noise_precision",
    "compute_fisher_from_complex_fourier_sideband_gradients",
    "compute_fisher_density_maps_from_gradients",
    "compute_fisher_from_gradients_with_noise",
]
