"""Lateral Fisher-information and localization CRLB implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from common_utils import init_infinite_dict
from experiment_contracts import (
    ConvergenceStatus,
    FisherMode,
    ValidationStatus,
    combine_parent_statuses,
    normalize_convergence_status,
)

from ._constants import _FISHER_VARIANCE_FLOOR
from ._metadata_helpers import (
    _derivative_unit,
    _diagnostic_metadata_aliases,
    _fisher_rank_metadata,
    _localization_derivative_metadata,
    _variance_units,
)

def _spatial_gradients(contrast: np.ndarray, pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Central-difference spatial gradients of a 2D contrast image.

    Returns (dC_dx, dC_dy) both in units of [contrast] / nm.

    The gradient is taken with `np.gradient`, which uses second-order-accurate
    central differences at interior points and first-order one-sided differences
    at the boundary. This matches the standard Fisher-info treatment where the
    estimator considers all pixels in the support.
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

    # np.gradient returns (d/dy, d/dx) when given a 2D array with indexing=(i, j).
    dC_dy, dC_dx = np.gradient(contrast, pixel_size_nm)
    return dC_dx, dC_dy

def _lateral_coordinate_derivatives(
    contrast: np.ndarray,
    pixel_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate derivatives ``dI/dx0`` and ``dI/dy0`` for stationary shifts."""
    dC_dx, dC_dy = _spatial_gradients(contrast, pixel_size_nm)
    return -dC_dx, -dC_dy

def _sort_key_finite_then_value(pair: tuple[str, float]) -> tuple[int, float]:
    v = pair[1]
    if not np.isfinite(v) or v <= 0.0:
        return (1, 0.0)
    return (0, v)

def _positive_finite_or_inf(value: Any) -> float:
    v = float(value)
    return v if np.isfinite(v) and v > 0.0 else float("inf")

def _build_symmetric_fisher_from_gradients(
    grads: tuple[np.ndarray, ...],
    inverse_variance: np.ndarray | float,
) -> np.ndarray:
    n = len(grads)
    F = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            F[i, j] = float(np.sum(grads[i] * grads[j] * inverse_variance))
            F[j, i] = F[i, j]
    return F

def compute_fisher_information(
    per_particle_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
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
    contrast = np.asarray(per_particle_contrast, dtype=float)
    dI_dx0, dI_dy0 = _lateral_coordinate_derivatives(contrast, pixel_size_nm)

    if np.isscalar(noise_variance_map):
        if not np.isfinite(noise_variance_map) or noise_variance_map <= 0.0:
            raise ValueError(
                f"noise_variance_map scalar must be positive; got {noise_variance_map}."
            )
        inv_var = 1.0 / float(noise_variance_map)
    else:
        var = np.asarray(noise_variance_map, dtype=float)
        if var.shape != contrast.shape:
            raise ValueError(
                f"noise_variance_map shape {var.shape} does not match contrast shape "
                f"{contrast.shape}."
            )
        if np.any(~np.isfinite(var)):
            raise ValueError("noise_variance_map must contain only finite values.")
        if np.any(var <= 0.0):
            raise ValueError("noise_variance_map must contain only positive values.")
        inv_var = 1.0 / var
    return _build_symmetric_fisher_from_gradients((dI_dx0, dI_dy0), inv_var)

def _fisher_information_from_lateral_derivatives(
    dI_dx0: np.ndarray,
    dI_dy0: np.ndarray,
    noise_variance_map: np.ndarray | float,
) -> np.ndarray:
    dI_dx0 = np.asarray(dI_dx0, dtype=float)
    dI_dy0 = np.asarray(dI_dy0, dtype=float)
    if dI_dx0.ndim != 2 or dI_dy0.ndim != 2 or dI_dx0.shape != dI_dy0.shape:
        raise ValueError(
            "lateral derivative maps must be 2D arrays with matching shape; "
            f"got {dI_dx0.shape} and {dI_dy0.shape}."
        )
    if np.any(~np.isfinite(dI_dx0)) or np.any(~np.isfinite(dI_dy0)):
        raise ValueError("lateral derivative maps must contain only finite values.")

    if np.isscalar(noise_variance_map):
        if not np.isfinite(noise_variance_map) or noise_variance_map <= 0.0:
            raise ValueError(
                f"noise_variance_map scalar must be positive; got {noise_variance_map}."
            )
        inv_var = 1.0 / float(noise_variance_map)
    else:
        var = np.asarray(noise_variance_map, dtype=float)
        if var.shape != dI_dx0.shape:
            raise ValueError(
                f"noise_variance_map shape {var.shape} does not match derivative map "
                f"shape {dI_dx0.shape}."
            )
        if np.any(~np.isfinite(var)):
            raise ValueError("noise_variance_map must contain only finite values.")
        if np.any(var <= 0.0):
            raise ValueError("noise_variance_map must contain only positive values.")
        inv_var = 1.0 / var
    return _build_symmetric_fisher_from_gradients((dI_dx0, dI_dy0), inv_var)

def _localization_crlb_from_fisher(
    F: np.ndarray,
    derivative_metadata: dict[str, Any],
    *,
    return_density: bool = False,
    density_contrast: np.ndarray | None = None,
    density_noise_variance: np.ndarray | float | None = None,
    density_pixel_size_nm: float | None = None,
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
        )
    return result

def compute_localization_crlb(
    per_particle_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    *,
    return_density: bool = False,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
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
    F = compute_fisher_information(
        per_particle_contrast, noise_variance_map, pixel_size_nm,
    )
    derivative_metadata = _localization_derivative_metadata(
        pixel_size_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
    )
    return _localization_crlb_from_fisher(
        F,
        derivative_metadata,
        return_density=return_density,
        density_contrast=np.asarray(per_particle_contrast, dtype=float),
        density_noise_variance=noise_variance_map,
        density_pixel_size_nm=pixel_size_nm,
    )

def compute_localization_crlb_from_lateral_rerenders(
    x_minus_contrast: np.ndarray,
    x_plus_contrast: np.ndarray,
    y_minus_contrast: np.ndarray,
    y_plus_contrast: np.ndarray,
    noise_variance_map: np.ndarray | float,
    pixel_size_nm: float,
    lateral_step_nm: float,
    *,
    signal_units: str = "contrast",
    measurement_domain: str = "contrast",
    noise_variance_units: str | None = None,
) -> dict[str, Any]:
    """Compute lateral CRLB from explicit symmetric x/y scene rerenders."""
    if not np.isfinite(lateral_step_nm) or lateral_step_nm <= 0.0:
        raise ValueError(f"lateral_step_nm must be positive; got {lateral_step_nm}.")
    x_minus = np.asarray(x_minus_contrast, dtype=float)
    x_plus = np.asarray(x_plus_contrast, dtype=float)
    y_minus = np.asarray(y_minus_contrast, dtype=float)
    y_plus = np.asarray(y_plus_contrast, dtype=float)
    if x_minus.shape != x_plus.shape or x_minus.shape != y_minus.shape or x_minus.shape != y_plus.shape:
        raise ValueError(
            "rerendered lateral derivative images must share one shape; got "
            f"{x_minus.shape}, {x_plus.shape}, {y_minus.shape}, {y_plus.shape}."
        )
    for name, arr in {
        "x_minus_contrast": x_minus,
        "x_plus_contrast": x_plus,
        "y_minus_contrast": y_minus,
        "y_plus_contrast": y_plus,
    }.items():
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D contrast image; got {arr.shape}.")
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")

    dI_dx0 = (x_plus - x_minus) / (2.0 * float(lateral_step_nm))
    dI_dy0 = (y_plus - y_minus) / (2.0 * float(lateral_step_nm))
    F = _fisher_information_from_lateral_derivatives(dI_dx0, dI_dy0, noise_variance_map)
    derivative_metadata = _localization_derivative_metadata(
        pixel_size_nm,
        signal_units=signal_units,
        measurement_domain=measurement_domain,
        noise_variance_units=noise_variance_units,
        lateral_derivative_mode="rerendered_xy",
        lateral_step_nm=float(lateral_step_nm),
    )
    return _localization_crlb_from_fisher(F, derivative_metadata)

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

def _resolve_modality_scalar(
    value: float | dict[str, float],
    modality: str,
    name: str,
    *,
    positive: bool = True,
) -> float:
    """Resolve a scalar or per-modality scalar mapping for one modality."""
    raw = value[modality] if isinstance(value, dict) else value
    out = float(raw)
    if not np.isfinite(out) or (positive and out <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name}[{modality!r}] must be {qualifier}; got {raw!r}.")
    return out

def _resolve_modality_scalar_map(
    value: float | dict[str, float],
    modalities,
    name: str,
    override: dict[str, float] | None = None,
    *,
    positive: bool = True,
) -> dict[str, float]:
    """Resolve one scalar per modality, accepting a scalar, mapping, or override."""
    source = override if override is not None else value
    if isinstance(source, dict):
        missing = set(modalities) - set(source)
        if missing:
            raise ValueError(
                f"{name} mapping is missing modality key(s): {sorted(missing)!r}."
            )
    return {
        modality: _resolve_modality_scalar(
            source,
            modality,
            name,
            positive=positive,
        )
        for modality in modalities
    }

def _resolve_modality_string_map(
    value: str | dict[str, str] | None,
    modalities,
    default: str,
) -> dict[str, str]:
    if isinstance(value, dict):
        return {modality: str(value.get(modality, default)) for modality in modalities}
    if value is None:
        return {modality: str(default) for modality in modalities}
    return {modality: str(value) for modality in modalities}

def compare_modality_information_content(
    contrast_by_modality: dict[str, np.ndarray],
    noise_variance_by_modality: dict[str, np.ndarray | float],
    pixel_size_nm: float | dict[str, float],
    z_step_nm: float | None = None,
    *,
    pixel_size_nm_by_modality: dict[str, float] | None = None,
    parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None = None,
    measurement_domain_by_modality: str | dict[str, str] | None = None,
    signal_units_by_modality: str | dict[str, str] | None = None,
    noise_variance_units_by_modality: str | dict[str, str] | None = None,
) -> dict[str, Any]:
    r"""
    Order imaging modalities by the physical information they deliver about the
    position of the same underlying particle.

    For every modality the caller supplies a per-particle contrast image (or a
    three-plane axial stack if ``z_step_nm`` is given) and its noise variance,
    this function computes the modality's CRLB on ``(x, y)`` [and ``z``] and
    returns a single comparison table plus the lowest-bound modality.

    Parameters
    ----------
    contrast_by_modality : dict[str, ndarray]
        Mapping ``modality_name -> per-particle contrast image``. For the 2D
        comparison each value is a ``(H, W)`` array; for the 3D comparison
        (``z_step_nm`` supplied) each value is a ``(3, H, W)`` three-plane stack
        in the same convention as :func:`compute_localization_crlb_3d`.
    noise_variance_by_modality : dict[str, ndarray | float]
        Mapping ``modality_name -> pixel-wise variance`` (or scalar) for the
        same modalities. Must have the same keys as ``contrast_by_modality``.
    pixel_size_nm : float or dict
        Detector pixel pitch in nanometres. A scalar keeps the historical
        shared-pitch behavior; a mapping supplies per-modality detector
        pitches for configured/native-profile comparisons.
    z_step_nm : float or None, default None
        If None: 2D comparison. If a positive float: 3D comparison using the
        3x3 Fisher machinery of :func:`compute_localization_crlb_3d`.

    Returns
    -------
    result : dict with keys
        - ``per_modality`` : dict ``modality -> sub-dict`` containing
          ``sigma_xy_nm``, ``fisher_det``, ``singular``, and (if 3D)
          ``sigma_z_nm``, ``sigma_xyz_nm``, ``axially_singular``. These are the
          outputs of the underlying 2D/3D CRLB routines, preserved verbatim.
        - ``ordering_xy`` : list of ``(modality, sigma_xy_nm)`` sorted ascending
          (lowest-bound first). Singular modalities end up last with ``+inf``.
          ``ranking_xy`` is retained as an alias for existing callers.
        - ``best_modality_xy`` : the key of the modality with the smallest
          lateral CRLB. ``None`` if every modality is singular.
        - ``relative_sigma_xy`` : dict ``modality -> sigma_xy_nm / best_sigma_xy``.
          A value of 1.0 marks the lowest-bound profile; larger values are
          relative to that profile.
        - ``frames_to_match_best_xy`` : dict ``modality -> (sigma / sigma_best)^2``.
          Because localization variance scales as 1/F and Fisher information
          adds linearly across independent frames, this is the number of frames
          of ``modality`` that would be needed to match the single-frame CRLB
          of the lowest-bound modality under the supplied profile.
        - ``ordering_xyz`` (only 3D): list of ``(modality, sigma_xyz_nm)``
          sorted ascending. ``ranking_xyz`` is retained as an alias.
        - ``best_modality_xyz`` (only 3D): argmin modality for sigma_xyz.
        - ``relative_sigma_xyz`` (only 3D): dict ``modality -> sigma_xyz / best``.

    Notes on comparability
    ----------------------
    The reported bound is conditional on the supplied contrast images and noise
    variances. Cross-modality orderings are comparable when each modality is
    rendered under matched profile/noise assumptions and the same sample truth.
    The Syniscopy rendering pipeline arranges this when the same particle
    configuration is routed through multiple ``ImagingModel`` instances.

    Ties are broken by the order in which modalities appear in the input dict.
    A singular modality (zero contrast gradient) sorts last and is assigned
    an infinite ``frames_to_match_best`` — it cannot catch up by averaging.
    """
    from .axial import compute_localization_crlb_3d


    if set(contrast_by_modality.keys()) != set(noise_variance_by_modality.keys()):
        raise ValueError(
            "contrast_by_modality and noise_variance_by_modality must share keys; "
            f"missing from contrast: "
            f"{set(noise_variance_by_modality) - set(contrast_by_modality)}; "
            f"missing from noise: "
            f"{set(contrast_by_modality) - set(noise_variance_by_modality)}."
        )
    if not contrast_by_modality:
        raise ValueError("contrast_by_modality is empty; nothing to compare.")
    if z_step_nm is not None and (not np.isfinite(z_step_nm) or z_step_nm <= 0.0):
        raise ValueError(f"z_step_nm must be positive when supplied; got {z_step_nm}.")

    pixel_sizes = _resolve_modality_scalar_map(
        pixel_size_nm,
        contrast_by_modality.keys(),
        "pixel_size_nm",
        override=pixel_size_nm_by_modality,
    )
    measurement_domains = _resolve_modality_string_map(
        measurement_domain_by_modality,
        contrast_by_modality.keys(),
        "contrast",
    )
    signal_units = _resolve_modality_string_map(
        signal_units_by_modality,
        contrast_by_modality.keys(),
        "contrast",
    )
    noise_variance_units = _resolve_modality_string_map(
        noise_variance_units_by_modality,
        contrast_by_modality.keys(),
        "",
    )
    per_modality: dict[str, dict[str, Any]] = {}
    for modality, contrast in contrast_by_modality.items():
        noise = noise_variance_by_modality[modality]
        px = pixel_sizes[modality]
        if z_step_nm is None:
            res = compute_localization_crlb(
                contrast,
                noise,
                px,
                signal_units=signal_units[modality],
                measurement_domain=measurement_domains[modality],
                noise_variance_units=noise_variance_units[modality] or None,
            )
        else:
            res = compute_localization_crlb_3d(
                contrast,
                noise,
                px,
                z_step_nm,
                signal_units=signal_units[modality],
                measurement_domain=measurement_domains[modality],
                noise_variance_units=noise_variance_units[modality] or None,
            )
        if parent_result_metadata_by_modality is not None:
            res["parent_convergence_status"] = normalize_convergence_status(
                dict(parent_result_metadata_by_modality).get(
                    modality, {},
                ).get("convergence_status", "unchecked")
            )
        per_modality[modality] = res

    # Lateral ordering (always valid: 2D and 3D both report sigma_xy_nm).
    def _xy_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, int]:
        modality, res = item
        sigma = float(res["sigma_xy_nm"])
        # sort order stable in input-dict order for ties
        idx = list(contrast_by_modality.keys()).index(modality)
        invalid, value = _sort_key_finite_then_value((modality, sigma))
        return (invalid, value, idx)

    ordered_xy = sorted(per_modality.items(), key=_xy_key)
    ranking_xy = [(m, _positive_finite_or_inf(r["sigma_xy_nm"])) for m, r in ordered_xy]

    best_modality_xy: str | None
    best_sigma_xy = ordered_xy[0][1]["sigma_xy_nm"] if ordered_xy else float("inf")
    if np.isfinite(best_sigma_xy) and best_sigma_xy > 0.0:
        best_modality_xy = ordered_xy[0][0]
        relative_sigma_xy = {
            m: (
                float(r["sigma_xy_nm"]) / float(best_sigma_xy)
                if np.isfinite(float(r["sigma_xy_nm"])) and float(r["sigma_xy_nm"]) > 0.0
                else float("inf")
            )
            for m, r in per_modality.items()
        }
        frames_to_match_best_xy = {
            m: (
                float("inf")
                if not np.isfinite(float(r["sigma_xy_nm"])) or float(r["sigma_xy_nm"]) <= 0.0
                else (float(r["sigma_xy_nm"]) / float(best_sigma_xy)) ** 2
            )
            for m, r in per_modality.items()
        }
    else:
        # Every modality singular: no meaningful ordering.
        best_modality_xy = None
        relative_sigma_xy = init_infinite_dict(per_modality)
        frames_to_match_best_xy = init_infinite_dict(per_modality)

    out: dict[str, Any] = {
        "per_modality": per_modality,
        "ordering_xy": ranking_xy,
        "ranking_xy": ranking_xy,
        "best_modality_xy": best_modality_xy,
        "relative_sigma_xy": relative_sigma_xy,
        "frames_to_match_best_xy": frames_to_match_best_xy,
        "pixel_size_nm_by_modality": pixel_sizes,
        "measurement_domain_by_modality": measurement_domains,
        "signal_units_by_modality": signal_units,
        "noise_variance_units_by_modality": {
            modality: per_modality[modality].get("noise_variance_units")
            for modality in per_modality
        },
    }

    if z_step_nm is not None:
        # Full-3D ordering.
        def _xyz_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, int]:
            modality, res = item
            sigma = float(res.get("sigma_xyz_nm", float("inf")))
            idx = list(contrast_by_modality.keys()).index(modality)
            invalid, value = _sort_key_finite_then_value((modality, sigma))
            return (invalid, value, idx)

        ordered_xyz = sorted(per_modality.items(), key=_xyz_key)
        ranking_xyz = [
            (m, _positive_finite_or_inf(r.get("sigma_xyz_nm", float("inf"))))
            for m, r in ordered_xyz
        ]
        best_sigma_xyz = ordered_xyz[0][1].get("sigma_xyz_nm", float("inf"))
        if np.isfinite(best_sigma_xyz) and best_sigma_xyz > 0.0:
            best_modality_xyz = ordered_xyz[0][0]
            relative_sigma_xyz = {
                m: (
                    float(r.get("sigma_xyz_nm", float("inf"))) / float(best_sigma_xyz)
                    if np.isfinite(float(r.get("sigma_xyz_nm", float("inf"))))
                    and float(r.get("sigma_xyz_nm", float("inf"))) > 0.0
                    else float("inf")
                )
                for m, r in per_modality.items()
            }
        else:
            best_modality_xyz = None
            relative_sigma_xyz = init_infinite_dict(per_modality)
        out["ordering_xyz"] = ranking_xyz
        out["ranking_xyz"] = ranking_xyz
        out["best_modality_xyz"] = best_modality_xyz
        out["relative_sigma_xyz"] = relative_sigma_xyz

    parent_metadata = combine_parent_statuses(parent_result_metadata_by_modality)
    out["parent_status_metadata"] = parent_metadata
    out["parent_convergence_status_by_modality"] = parent_metadata[
        "parent_convergence_statuses"
    ]
    out["validation_status"] = parent_metadata["validation_status"]
    out["production_grid_diagnostic"] = parent_metadata["production_grid_diagnostic"]
    out["safe_for_ordering"] = parent_metadata["safe_for_ordering"]
    out["safe_for_fusion"] = parent_metadata["safe_for_fusion"]
    out["safe_for_time_allocation"] = parent_metadata["safe_for_time_allocation"]
    out["safe_for_registration"] = parent_metadata["safe_for_registration"]
    out["safe_for_detected_quanta_ranking"] = parent_metadata[
        "safe_for_detected_quanta_ranking"
    ]
    out["status_reason"] = parent_metadata["status_reason"]

    return out

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
    keys = list(derivative_images.keys())
    derivs = [np.asarray(derivative_images[k], dtype=float) for k in keys]
    if not keys:
        raise ValueError("derivative_images must contain at least one axis")
    for key, arr in zip(keys, derivs):
        if arr.shape != mean.shape:
            raise ValueError(f"derivative image {key!r} shape {arr.shape} does not match mean image {mean.shape}")
    requested_mode = str(fisher_mode).strip().lower()
    if requested_mode == FisherMode.MEAN_FISHER_DIAGNOSTIC.value:
        mode = FisherMode.GAUSSIAN_FIXED_VARIANCE.value
    else:
        mode = requested_mode
    warnings = []
    exact_or_diagnostic = "diagnostic" if requested_mode == "mean_fisher_diagnostic" else "exact"
    if requested_mode == "mean_fisher_diagnostic":
        warnings.append("mean_fisher_diagnostic is the legacy fixed-variance diagnostic mode")
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
    poisson_mean_domain = poisson_mean_units_resolved.lower()
    poisson_variance_units = str(noise_variance_units or _variance_units(str(signal_units)))
    poisson_variance_source = ""

    def _poisson_variance_from_declared_mean(mean_arr: np.ndarray) -> np.ndarray:
        nonlocal poisson_variance_source, poisson_variance_units
        if np.any(~np.isfinite(mean_arr)):
            raise ValueError("Poisson Fisher mean_image must contain only finite values.")
        if np.any(mean_arr < 0.0):
            raise ValueError("Poisson Fisher mean_image must be non-negative.")
        detected_quanta_domains = {
            "count",
            "counts",
            "quanta",
            "detected_quanta",
            "photon",
            "photons",
            "photon_count",
            "photon_counts",
            "electron",
            "electrons",
            "electron_count",
            "electron_counts",
        }
        camera_count_domains = {
            "adu",
            "camera_count",
            "camera_counts",
            "detector_count",
            "detector_counts",
        }
        if poisson_mean_domain in camera_count_domains:
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
        if poisson_mean_domain in detected_quanta_domains:
            poisson_variance_source = "detected_quanta_poisson_variance"
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

    if mode == FisherMode.POISSON_EXACT.value:
        var = np.maximum(_poisson_variance_from_declared_mean(mean), variance_floor)
    elif mode == FisherMode.GAUSSIAN_FIXED_VARIANCE.value:
        if variance_image is None:
            raise ValueError("variance_image is required for gaussian_fixed_variance Fisher mode")
        var = np.maximum(np.asarray(variance_image, dtype=float), variance_floor)
    elif mode == FisherMode.POISSON_GAUSSIAN_APPROX.value:
        additive = 0.0 if variance_image is None else np.asarray(variance_image, dtype=float)
        var = np.maximum(_poisson_variance_from_declared_mean(mean) + additive, variance_floor)
        exact_or_diagnostic = "diagnostic"
        warnings.append("Poisson-Gaussian plug-in Fisher ignores variance derivatives")
    elif mode == FisherMode.GAUSSIAN_PARAMETER_DEPENDENT_VARIANCE.value:
        if variance_image is None or variance_derivative_images is None:
            raise ValueError("variance_image and variance_derivative_images are required")
        var = np.maximum(np.asarray(variance_image, dtype=float), variance_floor)
    elif mode == FisherMode.POISSON_GAUSSIAN_NUMERICAL.value:
        raise NotImplementedError("poisson_gaussian_numerical score/Hessian Fisher is not implemented; use explicit diagnostic status")
    else:
        raise ValueError(f"unknown fisher_mode {fisher_mode!r}")
    if np.shape(var) == ():
        var = np.full(mean.shape, float(var), dtype=float)
    elif var.shape != mean.shape:
        raise ValueError(f"variance image shape {var.shape} does not match mean image {mean.shape}")
    F = np.zeros((len(keys), len(keys)), dtype=float)
    inv = 1.0 / var
    for i, di in enumerate(derivs):
        for j, dj in enumerate(derivs[i:], start=i):
            F[i, j] = F[j, i] = float(np.sum(di * dj * inv))
    if mode == FisherMode.GAUSSIAN_PARAMETER_DEPENDENT_VARIANCE.value:
        for i, ki in enumerate(keys):
            dvi = np.asarray(variance_derivative_images[ki], dtype=float)
            for j, kj in enumerate(keys[i:], start=i):
                dvj = np.asarray(variance_derivative_images[kj], dtype=float)
                value = float(0.5 * np.sum(dvi * dvj / (var * var)))
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
        "unit_metadata": {
            "measurement_domain": str(measurement_domain),
            "mean_units": str(signal_units),
            "signal_units": str(signal_units),
            "poisson_mean_units": poisson_mean_units_resolved if mode in poisson_modes else None,
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

def _adaptive_crlb_from_steps(
    steps_by_size: dict[float, Any],
    compute_result_for_step,
    metric_key: str,
    *,
    convergence_tolerance: float,
    min_stable_steps: int,
    source_contract: str,
    modality: str,
) -> dict[str, Any]:
    from .convergence import (
        _matrix_rank_condition,
        _relative_span,
        _select_convergence_status,
        annotate_fisher_result_status,
    )

    per_step: list[dict[str, Any]] = []
    for step in sorted((float(s) for s in steps_by_size), reverse=True):
        result = compute_result_for_step(step, steps_by_size[step])
        fisher = np.asarray(result["fisher_matrix"], dtype=float)
        rank, cond, axes = _matrix_rank_condition(fisher)
        item = dict(result)
        item.update(
            {
                "derivative_step": float(step),
                "fisher_rank": rank,
                "condition_number": cond,
                "singular_axes": axes,
            }
        )
        per_step.append(item)

    status, selected, reason = _select_convergence_status(
        per_step,
        metric_key,
        convergence_tolerance,
        min_stable_steps,
    )
    selected_result = next(
        (
            item
            for item in per_step
            if selected is not None and float(item["derivative_step"]) == float(selected)
        ),
        per_step[-1] if per_step else {},
    )
    final = annotate_fisher_result_status(
        selected_result,
        convergence_status=status,
        source_contract=source_contract,
        modality=modality,
    )
    return {
        "convergence_status": status,
        "selected_step_nm": selected,
        "candidate_steps_nm": [float(item["derivative_step"]) for item in per_step],
        "relative_crlb_span": _relative_span(
            [float(item.get(metric_key, float("nan"))) for item in per_step]
        ),
        "rank_range": [
            min([int(item.get("fisher_rank", 0)) for item in per_step], default=0),
            max([int(item.get("fisher_rank", 0)) for item in per_step], default=0),
        ],
        "reason": reason,
        "per_step_results": per_step,
        "final_result": final,
    }


def adaptive_lateral_crlb_from_rerender_pairs(rerender_pairs_by_step_nm: dict[float, dict[str, np.ndarray]], noise_variance_map: np.ndarray | float, pixel_size_nm: float, *, convergence_tolerance: float = 0.05, min_stable_steps: int = 3, source_contract: str = "Contract-LP", modality: str = "unknown", signal_units: str = "contrast", measurement_domain: str = "contrast", noise_variance_units: str | None = None) -> dict[str, Any]:
    def _compute(step: float, pair: dict[str, np.ndarray]) -> dict[str, Any]:
        return compute_localization_crlb_from_lateral_rerenders(
            pair["x_minus"],
            pair["x_plus"],
            pair["y_minus"],
            pair["y_plus"],
            noise_variance_map,
            pixel_size_nm,
            step,
            signal_units=signal_units,
            measurement_domain=measurement_domain,
            noise_variance_units=noise_variance_units,
        )

    return _adaptive_crlb_from_steps(
        rerender_pairs_by_step_nm,
        _compute,
        metric_key="sigma_xy_nm",
        convergence_tolerance=convergence_tolerance,
        min_stable_steps=min_stable_steps,
        source_contract=source_contract,
        modality=modality,
    )

def compare_modality_information_content_from_crlb_results(
    crlb_by_modality: dict[str, dict[str, Any]],
    *,
    parent_result_metadata_by_modality: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the standard modality-comparison structure from precomputed CRLBs.

    Paper-facing rerendered-derivative workflows first select a converged CRLB
    for each modality.  This helper keeps those selected Fisher matrices as the
    ranking source instead of recomputing stationary detector-grid gradients in
    :func:`compare_modality_information_content`.
    """
    from .convergence import _parent_convergence_statuses, _parent_validation_statuses

    per_modality: dict[str, dict[str, Any]] = {}
    ordering_values: list[tuple[str, float]] = []
    for modality, result in crlb_by_modality.items():
        row = dict(result)
        fisher = np.asarray(row.get("fisher_matrix", np.full((2, 2), np.nan)), dtype=float)
        if fisher.shape == (2, 2) and np.all(np.isfinite(fisher)):
            row.setdefault("fisher_det", float(np.linalg.det(fisher)))
        sigma = float(row.get("sigma_xy_nm", float("inf")))
        if bool(row.get("singular", row.get("fisher_singular", False))):
            sigma = float("inf")
        sigma = _positive_finite_or_inf(sigma)
        per_modality[modality] = row
        ordering_values.append((modality, sigma))

    ordering_xy = sorted(ordering_values, key=_sort_key_finite_then_value)
    finite_ordering = [(m, s) for m, s in ordering_xy if np.isfinite(s) and s > 0.0]
    best_modality_xy = finite_ordering[0][0] if finite_ordering else None
    best_sigma_xy = finite_ordering[0][1] if finite_ordering else float("inf")
    relative_sigma_xy = {
        modality: (
            float(sigma / best_sigma_xy)
            if np.isfinite(sigma) and sigma > 0.0 and np.isfinite(best_sigma_xy) and best_sigma_xy > 0.0
            else float("inf")
        )
        for modality, sigma in ordering_xy
    }
    frames_to_match_best_xy = {
        modality: (
            float(relative_sigma_xy[modality] ** 2)
            if np.isfinite(relative_sigma_xy[modality])
            else float("inf")
        )
        for modality, _ in ordering_xy
    }

    metadata = parent_result_metadata_by_modality
    if metadata is None:
        metadata = {
            modality: {
                "convergence_status": result.get("convergence_status", ConvergenceStatus.UNCHECKED.value),
                "validation_status": result.get("validation_status", ValidationStatus.DIAGNOSTIC_ONLY.value),
                "production_grid_diagnostic": result.get("production_grid_diagnostic", False),
            }
            for modality, result in per_modality.items()
        }
    status_metadata = combine_parent_statuses(metadata)
    return {
        "per_modality": per_modality,
        "ordering_xy": ordering_xy,
        "ranking_xy": ordering_xy,
        "best_modality_xy": best_modality_xy,
        "relative_sigma_xy": relative_sigma_xy,
        "frames_to_match_best_xy": frames_to_match_best_xy,
        "parent_status_metadata": status_metadata,
        "parent_convergence_statuses": _parent_convergence_statuses(metadata),
        "parent_validation_statuses": _parent_validation_statuses(metadata),
        "validation_status": status_metadata["validation_status"],
        "production_grid_diagnostic": status_metadata["production_grid_diagnostic"],
        "safe_for_ordering": status_metadata["safe_for_ordering"],
        "safe_for_fusion": status_metadata["safe_for_fusion"],
        "safe_for_time_allocation": status_metadata["safe_for_time_allocation"],
        "safe_for_registration": status_metadata["safe_for_registration"],
        "safe_for_detected_quanta_ranking": status_metadata["safe_for_detected_quanta_ranking"],
        "derivative_mode": "rerendered_xy_selected_converged",
    }

__all__ = [
    "compute_fisher_information",
    "compute_localization_crlb",
    "compute_localization_crlb_from_lateral_rerenders",
    "crlb_efficiency_ratio",
    "compare_modality_information_content",
    "compare_modality_information_content_from_crlb_results",
    "compute_likelihood_fisher_information",
    "adaptive_lateral_crlb_from_rerender_pairs",
]
