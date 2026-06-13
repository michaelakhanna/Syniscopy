"""Lateral Fisher-information and localization CRLB implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiment_contracts import (
    FisherMode,
)
from measurement_units import normalize_poisson_mean_basis
from noise_contracts import (
    AnalysisNoiseModel,
    IndependentPixelNoiseModel,
)
from direct_signal_contracts import reject_direct_particle_signal_product

from ._constants import _FISHER_VARIANCE_FLOOR
from ._metadata_helpers import (
    _derivative_unit,
    _diagnostic_metadata_aliases,
    _fisher_rank_metadata,
    _localization_derivative_metadata,
    _variance_units,
)
from .precision import compute_fisher_from_gradients_with_noise
from .spectral_fisher import (
    boundary_energy_fraction,
    lateral_fisher_continuous,
    nyquist_band_fraction,
    spectral_gradient,
)
from unit_contracts import assert_compatible


def _reject_correlated_noise_for_diagonal_fisher(
    params: dict[str, Any] | None,
    *,
    context: str,
    measurement_domain: str = "contrast",
    signal_units: str = "contrast",
) -> None:
    """Reject typed independent likelihoods when params declare covariance.

    The typed Fisher-noise object is now the sole owner of covariance. ``params``
    may still be supplied for unit compatibility, but it must not silently add a
    scan-line covariance side channel beside an independent-pixel likelihood.
    """
    del measurement_domain, signal_units
    if params is None:
        return
    from camera_noise import camera_noise_metadata

    metadata = camera_noise_metadata(params)
    covariance_kind = str(metadata.get("noise_covariance_kind", "independent_pixels"))
    if covariance_kind == "independent_pixels":
        return
    raise ValueError(
        f"{context} received an independent-pixel Fisher likelihood, but "
        f"camera noise declares noise_covariance_kind={covariance_kind!r}. "
        "Typed Fisher likelihoods own covariance; params are only a consistency "
        "guard. Pass an AnalysisNoiseModel that carries the transformed "
        "row/reference covariance instead of relying on a diagonal variance plus "
        "params side channel."
    )


def _row_correlated_count_variance(
    params: dict[str, Any] | None,
    measurement_domain: str,
    signal_units: str,
) -> float:
    if params is None:
        return 0.0
    from camera_noise import CameraNoiseConfig

    cfg = CameraNoiseConfig.from_params(params)
    line_std = float(cfg.scan_line_noise_counts)
    if line_std <= 0.0:
        return 0.0
    domain = str(measurement_domain or "").strip().lower()
    units = str(signal_units or "").strip().lower()
    count_like = (
        domain == "count"
        or units in {
            "detector_count",
            "detector_count_difference",
            "camera_count",
            "camera_counts",
            "count",
            "counts",
        }
    )
    return float(line_std * line_std) if count_like else 0.0


def _build_row_correlated_fisher_from_gradients(
    grads: tuple[np.ndarray, ...],
    noise_variance_map: np.ndarray | float,
    *,
    line_variance: float,
) -> np.ndarray:
    if not np.isfinite(line_variance) or line_variance <= 0.0:
        raise ValueError(f"line_variance must be positive; got {line_variance!r}.")
    shape = grads[0].shape
    for grad in grads:
        if grad.shape != shape:
            raise ValueError("all derivative images must have the same shape.")
    if np.isscalar(noise_variance_map):
        total_var = np.full(shape, float(noise_variance_map), dtype=float)
    else:
        total_var = np.asarray(noise_variance_map, dtype=float)
        if total_var.shape != shape:
            raise ValueError(
                f"noise_variance_map shape {total_var.shape} does not match "
                f"gradient shape {shape}."
            )
    if np.any(~np.isfinite(total_var)):
        raise ValueError("noise_variance_map must contain only finite values.")
    independent_var = total_var - float(line_variance)
    if np.any(independent_var <= 0.0):
        raise ValueError(
            "row-correlated scan-line Fisher requires total variance to include "
            "a positive independent per-pixel component in addition to the row "
            "line covariance."
        )
    # Build a full likelihood object before entering the precision seam. The
    # row variance side channel was the architectural loophole that let a raw
    # diagonal array carry different meanings in different Fisher consumers.
    noise_model = AnalysisNoiseModel(
        diagonal_variance=total_var,
        measurement_domain="detector_count",
        signal_units="detector_count",
        noise_variance_units="detector_count_squared",
        covariance_kind="row_correlated_scan_lines",
        row_correlated_variance=float(line_variance),
        status_reason="count-domain row-correlated scan-line covariance",
    )
    return compute_fisher_from_gradients_with_noise(
        grads,
        noise_model,
        context="_build_row_correlated_fisher_from_gradients",
    )


def _build_analysis_noise_model_fisher_from_gradients(
    grads: tuple[np.ndarray, ...],
    noise_model: AnalysisNoiseModel,
) -> np.ndarray:
    # Delegate through the shared precision seam. This makes the scalar
    # Fisher matrix and any returned Fisher-density maps use the same
    # covariance inverse for AnalysisNoiseModel inputs.
    return compute_fisher_from_gradients_with_noise(
        grads,
        noise_model,
        context="_build_analysis_noise_model_fisher_from_gradients",
    )

def _spatial_gradients(contrast: np.ndarray, pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Spectral spatial gradients of a 2D contrast image.

    Returns (dC_dx, dC_dy) both in units of [contrast] / nm.

    The derivative is taken on the continuous band-limited interpolant
    represented by the sampled image, so lateral Fisher no longer
    depends on a finite-difference step.
    """
    contrast = np.asarray(contrast, dtype=float)
    if contrast.ndim != 2:
        raise ValueError(
            f"compute_*_crlb expects a 2D contrast image; got shape {contrast.shape}."
        )
    if min(contrast.shape) < 2:
        raise ValueError(
            "compute_*_crlb requires at least two pixels along each image axis; "
            f"got shape {contrast.shape}."
        )
    if not np.all(np.isfinite(contrast)):
        raise ValueError("contrast image must contain only finite values.")
    if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0.0:
        raise ValueError(f"pixel_size_nm must be positive; got {pixel_size_nm}.")

    return spectral_gradient(contrast, pixel_size_nm)

def _lateral_coordinate_derivatives(
    contrast: np.ndarray,
    pixel_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate derivatives ``dI/dx0`` and ``dI/dy0`` for in-plane shifts."""
    dC_dx, dC_dy = _spatial_gradients(contrast, pixel_size_nm)
    return -dC_dx, -dC_dy


def _resolve_analysis_noise_input(
    noise_variance_map: np.ndarray | float | AnalysisNoiseModel | IndependentPixelNoiseModel,
    *,
    context: str,
    measurement_domain: str,
    signal_units: str,
    noise_variance_units: str | None,
) -> tuple[np.ndarray | float, str, str, str | None, float, AnalysisNoiseModel | IndependentPixelNoiseModel | None]:
    if not isinstance(noise_variance_map, (AnalysisNoiseModel, IndependentPixelNoiseModel)):
        raise TypeError(
            f"{context} requires a typed Fisher noise likelihood. Pass "
            "AnalysisNoiseModel for structured covariance, or wrap diagonal "
            "independent-pixel variance with independent_pixel_noise_model(...). "
            "Raw arrays are rejected because diagonal report summaries and complete "
            "independent covariance have identical shapes but different physics."
        )
    noise_variance_map.require_safe_for_fisher(context=context)
    return (
        noise_variance_map.diagonal_variance,
        noise_variance_map.measurement_domain,
        noise_variance_map.signal_units,
        noise_variance_map.noise_variance_units,
        float(noise_variance_map.row_correlated_variance)
        if isinstance(noise_variance_map, AnalysisNoiseModel)
        else 0.0,
        noise_variance_map,
    )

def compute_fisher_information(
    per_particle_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float | AnalysisNoiseModel,
    pixel_size_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Build the 2x2 Fisher information matrix F for localization from a
    per-particle contrast image.

    Parameters
    ----------
    per_particle_contrast : 2D array
        Per-particle intensity image C(r) (any real units). Typically the
        output of ``ImagingModel.compute_per_particle_contrast`` for a single
        particle, evaluated on the native detector grid.
    noise_variance_map : 2D array or scalar float
        Pixel-wise variance of the observed image, var(r). For a Gaussian
        detector-noise model this is the variance of the readout; for a
        shot-noise-limited image this is approximately the background
        reference intensity |E_ref|^2. A scalar broadcasts to every pixel.
    pixel_size_nm : float
        Detector pixel pitch in nanometres (used to convert index-space
        gradients to per-nm gradients).

    Returns
    -------
    F : (2, 2) array
        Fisher information matrix with
            F[0, 0] = sum_r (dI/dx0)^2 / var(r),  [units: 1 / nm^2]
            F[1, 1] = sum_r (dI/dy0)^2 / var(r),
            F[0, 1] = F[1, 0] = sum_r (dI/dx0)(dI/dy0) / var(r).

    Notes
    -----
    The Fisher information as computed here is appropriate for a *Gaussian*
    pixel-noise model with known variance. For a pure-Poisson noise model the
    caller should pass ``noise_variance_map = I_total(r)`` (the *observed*
    intensity) rather than the background-only variance; the formula is
    otherwise identical in the high-photon / small-contrast limit.
    """
    # DirectParticleSignalProduct is a typed API boundary, not an ndarray.
    # Fisher must only see the explicitly requested Fisher-safe array, otherwise
    # source-density/yield/phase products can bypass the detector-transfer
    # contract and reproduce the direct-source-map CRLB bug.
    reject_direct_particle_signal_product(per_particle_contrast, context="compute_fisher_information")
    noise_variance_map, measurement_domain, signal_units, noise_variance_units, model_line_variance, noise_model = (
        _resolve_analysis_noise_input(
            noise_variance_map,
            context="compute_fisher_information",
            measurement_domain=measurement_domain,
            signal_units=signal_units,
            noise_variance_units=noise_variance_units,
        )
    )
    assert_compatible(
        context="compute_fisher_information",
        measurement_domain=measurement_domain,
        signal_units=signal_units,
        noise_variance_units=noise_variance_units or _variance_units(signal_units),
        params=params,
    )
    if noise_model is None or noise_model.covariance_kind == "independent_pixels":
        _reject_correlated_noise_for_diagonal_fisher(
            params,
            context="compute_fisher_information",
            measurement_domain=measurement_domain,
            signal_units=signal_units,
        )
    contrast = np.asarray(per_particle_contrast, dtype=float)
    if noise_model is not None and noise_model.covariance_kind == "independent_pixels":
        return np.asarray(
            lateral_fisher_continuous(
                contrast,
                noise_model.diagonal_variance,
                pixel_size_nm,
            )["fisher_matrix"],
            dtype=float,
        )

    dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(contrast, pixel_size_nm)
    if noise_model is not None and noise_model.covariance_kind == "row_correlated_scan_lines":
        return _build_analysis_noise_model_fisher_from_gradients((dI_dx0, dI_dy0), noise_model)
    line_variance = float(model_line_variance)
    if line_variance > 0.0:
        return _build_row_correlated_fisher_from_gradients(
            (dI_dx0, dI_dy0),
            noise_variance_map,
            line_variance=line_variance,
        )

    # Independent-pixel Fisher must use the same precision seam as density
    # maps.  Keeping a local inverse-variance path here lets scalar CRLBs and
    # saved Fisher-density artifacts disagree on numerical variance-floor
    # pixels.
    return compute_fisher_from_gradients_with_noise(
        (dI_dx0, dI_dy0),
        noise_model,
        context="compute_fisher_information",
    )

def _localization_crlb_from_fisher(
    F: np.ndarray,
    derivative_metadata: dict[str, Any],
    *,
    return_density: bool = False,
    density_contrast: np.ndarray | None = None,
    density_noise_variance: np.ndarray | float | AnalysisNoiseModel | None = None,
    density_pixel_size_nm: float | None = None,
    density_row_correlated_line_variance: float = 0.0,
) -> dict[str, Any]:
    from .density import compute_information_density_maps
    from .fusion import _axis_sigmas_from_fisher

    F = np.asarray(F, dtype=float)
    det_F = float(F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
    rank_metadata = _fisher_rank_metadata(F)

    if not np.all(np.isfinite(F)) or not np.isfinite(det_F):
        axes_singular = ["x", "y"]
        result = {
            "sigma_x_nm": float("inf"),
            "sigma_y_nm": float("inf"),
            "sigma_xy_nm": float("inf"),
            "fisher_matrix": F,
            "fisher_det": det_F,
            "singular": True,
            "rank": 0,
            "axes_singular": axes_singular,
            "derivative_metadata": derivative_metadata,
            **rank_metadata,
            **_diagnostic_metadata_aliases(
                derivative_metadata,
                rank_metadata,
                axes_singular=axes_singular,
                sigma_units=["nm", "nm"],
            ),
        }
        if return_density and density_contrast is not None and density_noise_variance is not None:
            result["information_density_maps"] = compute_information_density_maps(
                density_contrast,
                density_noise_variance,
                float(density_pixel_size_nm),
                row_correlated_line_variance=density_row_correlated_line_variance,
            )
        return result

    axis_sigmas, axis_singular = _axis_sigmas_from_fisher(F)
    sigma_x = axis_sigmas[0]
    sigma_y = axis_sigmas[1]
    singular = axis_singular[0] or axis_singular[1]
    axes_singular = [
        axis for axis, is_singular in zip(("x", "y"), axis_singular)
        if is_singular
    ]
    sigma_xy = (
        float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))
        if not singular and np.isfinite(sigma_x) and np.isfinite(sigma_y)
        else float("inf")
    )

    result = {
        "sigma_x_nm": sigma_x,
        "sigma_y_nm": sigma_y,
        "sigma_xy_nm": sigma_xy,
        "fisher_matrix": F,
        "fisher_det": det_F,
        "singular": singular,
        "rank": int(2 - sum(bool(flag) for flag in axis_singular[:2])),
        "axes_singular": axes_singular,
        "derivative_metadata": derivative_metadata,
        **rank_metadata,
        **_diagnostic_metadata_aliases(
            derivative_metadata,
            rank_metadata,
            axes_singular=axes_singular,
            sigma_units=["nm", "nm"],
        ),
    }
    if return_density and density_contrast is not None and density_noise_variance is not None:
        result["information_density_maps"] = compute_information_density_maps(
            density_contrast,
            density_noise_variance,
            float(density_pixel_size_nm),
            row_correlated_line_variance=density_row_correlated_line_variance,
        )
        correlated_density = (
            isinstance(density_noise_variance, AnalysisNoiseModel)
            and density_noise_variance.covariance_kind == "row_correlated_scan_lines"
        ) or float(density_row_correlated_line_variance) > 0.0
        result["information_density_basis"] = (
            "covariance_weighted_quadratic_form"
            if correlated_density
            else "independent_pixel_variance"
        )
        result["information_density_sums_match_fisher_diagonal"] = True
        result["information_density_may_be_signed"] = bool(correlated_density)
    return result

def compute_localization_crlb(
    per_particle_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float | AnalysisNoiseModel,
    pixel_size_nm: float,
    *,
    return_density: bool = False,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Per-particle Cramér-Rao lower bound on (x, y) localization error.

    Parameters
    ----------
    per_particle_contrast, noise_variance_map, pixel_size_nm :
        See :func:`compute_fisher_information`.
    return_density : bool, default False
        If True, include per-pixel Fisher information density maps in the result.

    Returns
    -------
    result : dict
        Keys:
          - ``sigma_x_nm``   : CRLB on x-localization, nanometres.
          - ``sigma_y_nm``   : CRLB on y-localization, nanometres.
          - ``sigma_xy_nm``  : Total 2D bound, sqrt(sigma_x^2 + sigma_y^2).
          - ``fisher_matrix`` : The 2x2 Fisher information matrix (array).
          - ``fisher_det``   : Determinant of F (pre-inversion).
          - ``singular``     : True if F was (effectively) singular and the
                               bounds were set to +inf.
          - ``information_density_maps`` : Present only when ``return_density``
                               is True.

    A singular Fisher matrix arises for an image with no spatial gradient
    (constant across the field, e.g. a particle with zero contrast), in
    which case localization is information-theoretically impossible and
    the CRLB is +inf — a useful signal to upstream trackability code.
    """
    # DirectParticleSignalProduct is intentionally not array-like.  Rejecting it
    # here gives compute_localization_crlb a clearer error than a later np.asarray
    # failure and keeps source-basis products out of Fisher by construction.
    reject_direct_particle_signal_product(per_particle_contrast, context="compute_localization_crlb")
    original_noise_model = (
        noise_variance_map
        if isinstance(noise_variance_map, (AnalysisNoiseModel, IndependentPixelNoiseModel))
        else None
    )
    noise_variance_map, measurement_domain, signal_units, noise_variance_units, model_line_variance, _noise_model = (
        _resolve_analysis_noise_input(
            noise_variance_map,
            context="compute_localization_crlb",
            measurement_domain=measurement_domain,
            signal_units=signal_units,
            noise_variance_units=noise_variance_units,
        )
    )
    assert_compatible(
        context="compute_localization_crlb",
        measurement_domain=measurement_domain,
        signal_units=signal_units,
        noise_variance_units=noise_variance_units or _variance_units(signal_units),
        params=params,
    )
    F = compute_fisher_information(
        per_particle_contrast,
        original_noise_model if original_noise_model is not None else noise_variance_map,
        pixel_size_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
        params=params,
    )
    derivative_metadata = _localization_derivative_metadata(
        pixel_size_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
    )
    contrast_arr = np.asarray(per_particle_contrast, dtype=float)
    derivative_metadata.update(
        {
            "derivative_basis": "spectral_band_limited",
            "step_size_free": True,
            "boundary_energy_fraction": boundary_energy_fraction(contrast_arr),
            "nyquist_band_fraction": nyquist_band_fraction(contrast_arr),
            "bandlimit_validity_basis": "sampling_and_fft_periodicity_diagnostics",
        }
    )
    line_variance = float(model_line_variance)
    density_noise_input = original_noise_model if original_noise_model is not None else noise_variance_map
    result = _localization_crlb_from_fisher(
        F,
        derivative_metadata,
        return_density=return_density,
        density_contrast=contrast_arr,
        density_noise_variance=density_noise_input,
        density_pixel_size_nm=pixel_size_nm,
        density_row_correlated_line_variance=0.0 if original_noise_model is not None else float(line_variance),
    )
    if line_variance > 0.0:
        result["fisher_noise_covariance_model"] = "row_correlated_scan_line_covariance"
        result["scan_line_noise_variance_counts2"] = float(line_variance)
        result["safe_for_covariance_fisher_variance"] = True
    if original_noise_model is not None:
        result["analysis_noise_covariance_kind"] = original_noise_model.covariance_kind
        result["analysis_noise_status_reason"] = original_noise_model.status_reason
        if original_noise_model.covariance_kind == "row_correlated_scan_lines":
            result["fisher_noise_covariance_model"] = (
                "row_correlated_scan_line_covariance"
                if float(original_noise_model.row_correlated_variance) > 0.0
                else "transformed_row_correlated_scan_line_covariance"
            )
            result["safe_for_covariance_fisher_variance"] = True
    return result

def crlb_efficiency_ratio(
    measured_sigma_nm: float,
    crlb_sigma_nm: float,
) -> float:
    """
    Fisher efficiency of an estimator: CRLB / measured.

    Values in (0, 1] are valid; 1.0 means the estimator saturates the bound
    (optimal unbiased estimator). Values > 1 are not physically impossible —
    they indicate a *biased* estimator, and the CRLB assumes unbiasedness.
    Values close to 0 indicate a poor estimator.

    Special cases:
      - If ``crlb_sigma_nm`` is 0 or inf, returns 0.0 (undefined efficiency
        under a degenerate bound).
      - If ``measured_sigma_nm`` is 0 (or negative), returns +inf.
    """
    if crlb_sigma_nm <= 0.0 or not np.isfinite(crlb_sigma_nm):
        return 0.0
    if measured_sigma_nm <= 0.0:
        return float("inf")
    return float(crlb_sigma_nm / measured_sigma_nm)

def compute_likelihood_fisher_information(
    mean_image: np.ndarray,
    derivative_images: dict[str, np.ndarray],
    *,
    variance_image: np.ndarray | float | None = None,
    variance_derivative_images: dict[str, np.ndarray] | None = None,
    fisher_mode: str = FisherMode.GAUSSIAN_FIXED_VARIANCE.value,
    variance_floor: float = _FISHER_VARIANCE_FLOOR,
    signal_units: str = "model_signal",
    measurement_domain: str = "model_signal",
    noise_variance_units: str | None = None,
    parameter_units_by_axis: dict[str, str] | None = None,
    poisson_mean_units: str | None = None,
    camera_gain_e_per_count: float | None = None,
) -> dict[str, Any]:
    from .convergence import _matrix_rank_condition

    mean = np.asarray(mean_image, dtype=float)
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean_image must contain only finite values.")
    if not np.isfinite(variance_floor) or float(variance_floor) <= 0.0:
        raise ValueError(f"variance_floor must be positive and finite; got {variance_floor!r}.")
    keys = list(derivative_images.keys())
    derivs = [np.asarray(derivative_images[k], dtype=float) for k in keys]
    if not keys:
        raise ValueError("derivative_images must contain at least one axis")
    for key, arr in zip(keys, derivs):
        if arr.shape != mean.shape:
            raise ValueError(f"derivative image {key!r} shape {arr.shape} does not match mean image {mean.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"derivative image {key!r} must contain only finite values.")
    requested_mode = str(fisher_mode).strip().lower()
    mode = requested_mode
    warnings = []
    exact_or_diagnostic = "exact"
    poisson_modes = {
        FisherMode.POISSON_EXACT.value,
        FisherMode.POISSON_GAUSSIAN_APPROX.value,
    }
    if poisson_mean_units is not None:
        poisson_mean_units_resolved = str(poisson_mean_units).strip()
    elif str(signal_units or "").strip() not in {"", "model_signal"}:
        poisson_mean_units_resolved = str(signal_units).strip()
    else:
        poisson_mean_units_resolved = str(measurement_domain or signal_units or "").strip()
    poisson_mean_basis = (
        normalize_poisson_mean_basis(poisson_mean_units_resolved)
        if mode in poisson_modes
        else None
    )
    poisson_variance_units = str(noise_variance_units or _variance_units(str(signal_units)))
    poisson_variance_source = ""

    def _poisson_variance_from_declared_mean(mean_arr: np.ndarray) -> np.ndarray:
        nonlocal poisson_variance_source, poisson_variance_units
        if np.any(~np.isfinite(mean_arr)):
            raise ValueError("Poisson Fisher mean_image must contain only finite values.")
        if np.any(mean_arr < 0.0):
            raise ValueError("Poisson Fisher mean_image must be non-negative.")
        if poisson_mean_basis == "camera_count":
            if camera_gain_e_per_count is None:
                raise ValueError(
                    "Poisson Fisher on camera/detector counts requires "
                    "camera_gain_e_per_count so count-domain variance is not "
                    "silently treated as detected-quanta variance."
                )
            gain = float(camera_gain_e_per_count)
            if not np.isfinite(gain) or gain <= 0.0:
                raise ValueError(
                    "camera_gain_e_per_count must be positive and finite when "
                    "Poisson mean units are camera/detector counts."
                )
            poisson_variance_source = "camera_count_poisson_variance_converted_by_gain"
            poisson_variance_units = _variance_units(str(signal_units))
            return mean_arr / gain
        if poisson_mean_basis in {"detected_quanta", "electron_count"}:
            poisson_variance_source = f"{poisson_mean_basis}_poisson_variance"
            poisson_variance_units = str(
                noise_variance_units or _variance_units(str(signal_units))
            )
            return mean_arr
        raise ValueError(
            "Poisson Fisher requires poisson_mean_units/signal_units to declare "
            "detected quanta or camera counts; got "
            f"{poisson_mean_units_resolved!r}. Use a Gaussian Fisher mode for "
            "phase, contrast, or other signed/non-count observables."
        )

    def _variance_array(
        value: np.ndarray | float,
        name: str,
        *,
        positive: bool,
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")
        if arr.shape == ():
            scalar = float(arr)
            if positive and scalar <= 0.0:
                raise ValueError(f"{name} must be positive; got {scalar!r}.")
            if not positive and scalar < 0.0:
                raise ValueError(f"{name} must be non-negative; got {scalar!r}.")
            return np.full(mean.shape, scalar, dtype=float)
        if arr.shape != mean.shape:
            raise ValueError(f"{name} shape {arr.shape} does not match mean image {mean.shape}")
        if positive and np.any(arr <= 0.0):
            raise ValueError(f"{name} must contain only positive values.")
        if not positive and np.any(arr < 0.0):
            raise ValueError(f"{name} must contain only non-negative values.")
        return arr

    if mode == FisherMode.POISSON_EXACT.value:
        var = np.maximum(_poisson_variance_from_declared_mean(mean), variance_floor)
    elif mode == FisherMode.GAUSSIAN_FIXED_VARIANCE.value:
        if variance_image is None:
            raise ValueError("variance_image is required for gaussian_fixed_variance Fisher mode")
        var = _variance_array(variance_image, "variance_image", positive=True)
    elif mode == FisherMode.POISSON_GAUSSIAN_APPROX.value:
        additive = (
            0.0
            if variance_image is None
            else _variance_array(variance_image, "variance_image", positive=False)
        )
        var = np.maximum(_poisson_variance_from_declared_mean(mean) + additive, variance_floor)
        exact_or_diagnostic = "diagnostic"
        warnings.append("Poisson-Gaussian plug-in Fisher ignores variance derivatives")
    elif mode == FisherMode.GAUSSIAN_PARAMETER_DEPENDENT_VARIANCE.value:
        if variance_image is None or variance_derivative_images is None:
            raise ValueError("variance_image and variance_derivative_images are required")
        var = _variance_array(variance_image, "variance_image", positive=True)
    elif mode == FisherMode.POISSON_GAUSSIAN_NUMERICAL.value:
        raise NotImplementedError("poisson_gaussian_numerical score/Hessian Fisher is not implemented; use explicit diagnostic status")
    else:
        raise ValueError(f"unknown fisher_mode {fisher_mode!r}")
    F = np.zeros((len(keys), len(keys)), dtype=float)
    active = np.asarray(var, dtype=float) > float(variance_floor)
    inv = np.divide(
        1.0,
        var,
        out=np.zeros_like(var, dtype=float),
        where=active,
    )
    for i, di in enumerate(derivs):
        for j, dj in enumerate(derivs[i:], start=i):
            F[i, j] = F[j, i] = float(np.sum(di * dj * inv))
    if mode == FisherMode.GAUSSIAN_PARAMETER_DEPENDENT_VARIANCE.value:
        variance_derivs = {}
        for key in keys:
            if key not in variance_derivative_images:
                raise ValueError(f"variance_derivative_images is missing derivative for axis {key!r}")
            dv = np.asarray(variance_derivative_images[key], dtype=float)
            if dv.shape != mean.shape:
                raise ValueError(
                    f"variance derivative image {key!r} shape {dv.shape} "
                    f"does not match mean image {mean.shape}"
                )
            if not np.all(np.isfinite(dv)):
                raise ValueError(f"variance derivative image {key!r} must contain only finite values.")
            variance_derivs[key] = dv
        for i, ki in enumerate(keys):
            dvi = variance_derivs[ki]
            for j, kj in enumerate(keys[i:], start=i):
                dvj = variance_derivs[kj]
                inv_var2 = np.square(inv)
                value = float(0.5 * np.sum(dvi * dvj * inv_var2))
                F[i, j] += value
                if j != i:
                    F[j, i] += value
    rank, condition, axes = _matrix_rank_condition(F)
    parameter_units = {axis: str((parameter_units_by_axis or {}).get(axis, "parameter")) for axis in keys}
    derivative_units = {
        axis: _derivative_unit(str(signal_units), parameter_units[axis])
        for axis in keys
    }
    variance_units = str(
        poisson_variance_units
        if mode in poisson_modes
        else noise_variance_units or _variance_units(str(signal_units))
    )
    return {
        "fisher_matrix": F,
        "axis_order": keys,
        "fisher_mode": requested_mode,
        "resolved_fisher_mode": mode,
        "exact_or_diagnostic": exact_or_diagnostic,
        "variance_model": (
            "parameter_dependent"
            if mode == FisherMode.GAUSSIAN_PARAMETER_DEPENDENT_VARIANCE.value
            else "fixed_or_mean"
        ),
        "variance_parameter_dependent": mode == FisherMode.GAUSSIAN_PARAMETER_DEPENDENT_VARIANCE.value,
        "readout_noise_included": variance_image is not None,
        "precision_variance_floor": float(variance_floor),
        "precision_floor_policy": "zero_information_at_or_below_numerical_floor",
        "precision_inactive_floor_pixel_count": int(var.size - np.count_nonzero(active)),
        "precision_active_pixel_count": int(np.count_nonzero(active)),
        "unit_metadata": {
            "measurement_domain": str(measurement_domain),
            "mean_units": str(signal_units),
            "signal_units": str(signal_units),
            "poisson_mean_units": poisson_mean_units_resolved if mode in poisson_modes else None,
            "poisson_mean_basis": poisson_mean_basis if mode in poisson_modes else None,
            "poisson_variance_source": poisson_variance_source if mode in poisson_modes else None,
            "camera_gain_e_per_count": (
                None if camera_gain_e_per_count is None else float(camera_gain_e_per_count)
            ),
            "noise_variance_units": variance_units,
            "parameter_units_by_axis": parameter_units,
            "derivative_units_by_axis": derivative_units,
            "fisher_units": {
                axis_i: {
                    axis_j: f"{derivative_units[axis_i]}*{derivative_units[axis_j]}/{variance_units}"
                    for axis_j in keys
                }
                for axis_i in keys
            },
        },
        "warnings": warnings,
        "rank": rank,
        "condition_number": condition,
        "singular_axes": axes,
    }

__all__ = [
    "compute_fisher_information",
    "compute_localization_crlb",
    "crlb_efficiency_ratio",
    "compute_likelihood_fisher_information",
]
