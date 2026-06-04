"""Optional optical backend helpers for broadband and coverslip-aware PSFs."""

from __future__ import annotations

from typing import Any

import numpy as np

from config.runtime import param_value
from optical_params import resolve_probe_wavelength_nm


def _finite_float(value: Any, *, key: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{key} must be finite; got {value!r}.")
    return float(out)


def compute_coverslip_aberration_phase(
    params: dict,
    sin_theta: np.ndarray,
    aperture_mask: np.ndarray,
    *,
    wavelength_nm: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a Gibson-Lanni-style coverslip mismatch OPD phase.

    The implementation is a pupil optical-path-difference term, not a separate
    renderer.  When the configured actual and design coverslip settings match,
    the returned phase is exactly zero after piston removal.
    """
    model = str(param_value(params, 'coverslip_aberration_model')).strip().lower()
    phase = np.zeros_like(np.asarray(sin_theta, dtype=float), dtype=float)
    metadata: dict[str, Any] = {
        "coverslip_aberration_model": model,
        "coverslip_aberration_phase_rms_rad": 0.0,
        "coverslip_aberration_phase_peak_to_peak_rad": 0.0,
    }
    if model in {"", "none", "disabled", "off"}:
        metadata["coverslip_aberration_model"] = "none"
        return phase, metadata
    if model not in {"gibson_lanni", "coverslip_mismatch"}:
        raise ValueError(
            "coverslip_aberration_model must be 'none', 'gibson_lanni', or "
            f"'coverslip_mismatch'; got {model!r}."
        )

    wavelength = _finite_float(
        wavelength_nm if wavelength_nm is not None else resolve_probe_wavelength_nm(params),
        key="wavelength_nm",
    )
    if wavelength <= 0.0:
        raise ValueError("wavelength_nm must be positive.")

    n_medium = _finite_float(param_value(params, "refractive_index_medium"), key="refractive_index_medium")
    n_cs = _finite_float(param_value(params, 'coverslip_refractive_index'), key="coverslip_refractive_index")
    n_design = _finite_float(
        param_value(params, "coverslip_design_refractive_index"),
        key="coverslip_design_refractive_index",
    )
    t_nm = 1000.0 * _finite_float(param_value(params, 'coverslip_thickness_um'), key="coverslip_thickness_um")
    t_design_nm = 1000.0 * _finite_float(
        param_value(params, 'coverslip_design_thickness_um'),
        key="coverslip_design_thickness_um",
    )
    if n_medium <= 0.0 or n_cs <= 0.0 or n_design <= 0.0:
        raise ValueError("coverslip and medium refractive indices must be positive.")
    if t_nm < 0.0 or t_design_nm < 0.0:
        raise ValueError("coverslip thickness values must be non-negative.")

    sin_theta_arr = np.asarray(sin_theta, dtype=float)
    aperture = np.asarray(aperture_mask, dtype=bool)
    u = n_medium * sin_theta_arr
    actual_arg = np.maximum(n_cs * n_cs - u * u, 0.0)
    design_arg = np.maximum(n_design * n_design - u * u, 0.0)
    opd_nm = t_nm * np.sqrt(actual_arg) - t_design_nm * np.sqrt(design_arg)
    phase = (2.0 * np.pi / wavelength) * opd_nm
    phase = np.where(aperture, phase, 0.0)

    if bool(param_value(params, 'coverslip_aberration_subtract_piston')) and np.any(aperture):
        phase = phase.copy()
        phase[aperture] -= float(np.mean(phase[aperture]))

    values = phase[aperture] if np.any(aperture) else np.asarray([], dtype=float)
    if values.size:
        rms = float(np.sqrt(np.mean(values * values)))
        p2p = float(np.max(values) - np.min(values))
    else:
        rms = 0.0
        p2p = 0.0
    metadata.update(
        {
            "coverslip_aberration_model": "gibson_lanni" if model == "coverslip_mismatch" else model,
            "coverslip_thickness_um": float(t_nm / 1000.0),
            "coverslip_design_thickness_um": float(t_design_nm / 1000.0),
            "coverslip_refractive_index": float(n_cs),
            "coverslip_design_refractive_index": float(n_design),
            "coverslip_aberration_subtract_piston": bool(
                param_value(params, 'coverslip_aberration_subtract_piston')
            ),
            "coverslip_aberration_phase_rms_rad": rms,
            "coverslip_aberration_phase_peak_to_peak_rad": p2p,
        }
    )
    return phase, metadata


def _gaussian_weights(wavelengths: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    if fwhm <= 0.0:
        weights = np.zeros_like(wavelengths, dtype=float)
        weights[np.argmin(np.abs(wavelengths - center))] = 1.0
        return weights
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    weights = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    total = float(np.sum(weights))
    return weights / total if total > 0.0 else np.full_like(wavelengths, 1.0 / wavelengths.size)


def broadband_quadrature_channels(params: dict) -> list[dict[str, Any]]:
    """Build normalized spectral channels for broadband quadrature rendering."""
    model = str(param_value(params, 'spectral_integration_model')).strip().lower()
    if model != "broadband_quadrature":
        raise ValueError("broadband_quadrature_channels requires spectral_integration_model='broadband_quadrature'.")

    explicit = param_value(params, 'broadband_wavelengths_nm')
    if explicit is not None:
        wavelengths = np.asarray(explicit, dtype=float).reshape(-1)
    else:
        center = _finite_float(
            param_value(params, "illumination_spectrum_center_nm"),
            key="illumination_spectrum_center_nm",
        )
        fwhm = _finite_float(param_value(params, 'illumination_spectrum_fwhm_nm'), key="illumination_spectrum_fwhm_nm")
        sample_count = int(param_value(params, 'illumination_spectrum_num_samples'))
        if sample_count <= 0:
            raise ValueError("illumination_spectrum_num_samples must be positive.")
        half_width = max(0.5 * fwhm, 1.0)
        wavelengths = np.linspace(center - half_width, center + half_width, sample_count)
    if wavelengths.size == 0 or not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("broadband wavelengths must be a non-empty positive finite sequence.")

    explicit_weights = param_value(params, 'broadband_weights')
    if explicit_weights is not None:
        weights = np.asarray(explicit_weights, dtype=float).reshape(-1)
        if weights.shape != wavelengths.shape:
            raise ValueError("broadband_weights must match broadband_wavelengths_nm length.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("broadband_weights must be finite and non-negative.")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("broadband_weights must have positive sum.")
        weights = weights / total
    else:
        center = float(param_value(params, "illumination_spectrum_center_nm"))
        fwhm = float(param_value(params, "illumination_spectrum_fwhm_nm"))
        weights = _gaussian_weights(wavelengths, center, fwhm)

    detector_model = str(param_value(params, 'detector_spectral_response_model')).strip().lower()
    channels: list[dict[str, Any]] = []
    for idx, (wl, weight) in enumerate(zip(wavelengths, weights)):
        entry: dict[str, Any] = {
            "name": f"broadband_{idx + 1}_{wl:.1f}nm",
            "wavelength_nm": float(wl),
            "probe_wavelength_nm": float(wl),
            "spectral_weight": float(weight),
            "spectral_integration_model": "broadband_quadrature",
        }
        if detector_model == "flat":
            entry["detector_weights_rgb"] = [1.0, 1.0, 1.0]
        elif detector_model == "table":
            raise ValueError("detector_spectral_response_model='table' requires caller-supplied channel weights.")
        elif detector_model != "rgb_heuristic":
            raise ValueError(
                "detector_spectral_response_model must be 'rgb_heuristic', 'flat', or 'table'; "
                f"got {detector_model!r}."
            )
        channels.append(entry)
    return channels


def expand_broadband_quadrature(params: dict) -> dict:
    """Return a params copy whose channels implement the selected spectral model."""
    out = dict(params)
    model = str(out.get("spectral_integration_model", "single_wavelength")).strip().lower()
    if model in {"", "single_wavelength"}:
        return out
    if model == "configured_channels":
        if out.get("channels") is None:
            raise ValueError("spectral_integration_model='configured_channels' requires PARAMS['channels'].")
        return out
    if model == "broadband_quadrature":
        if out.get("channels") is not None and not bool(out.get("allow_broadband_overwrite_channels", False)):
            raise ValueError(
                "spectral_integration_model='broadband_quadrature' generates channels; "
                "clear PARAMS['channels'] or set allow_broadband_overwrite_channels=True."
            )
        out["channels"] = broadband_quadrature_channels(out)
        out["_generated_spectral_channels"] = True
        out["_spectral_channel_count"] = len(out["channels"])
        return out
    raise ValueError(
        "spectral_integration_model must be 'single_wavelength', 'configured_channels', "
        f"or 'broadband_quadrature'; got {model!r}."
    )
