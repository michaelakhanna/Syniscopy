from __future__ import annotations

from typing import Any

import hashlib
import numpy as np
from scipy.fft import fftshift, ifft2, ifftshift

from mie_scattering import mie_S1_S2_from_coefficients, mie_an_bn
from optical_extensions import compute_coverslip_aberration_phase


def _polarization_vector(model: str, rotation_deg: float = 0.0) -> np.ndarray:
    key = str(model).strip().lower()
    if key in {"linear_x", "x"}:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if key in {"linear_y", "y"}:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    raise ValueError(
        "polarization_model must be 'linear_x', 'linear_y', or 'unpolarized' "
        f"for vectorial_debye; got {model!r}."
    )


def _rotate_linear_polarization(
    polarization: np.ndarray,
    rotation_deg: float,
) -> np.ndarray:
    """Rotate a polarization vector around the optical axis by `rotation_deg`.

    The rotation is applied in the x-y plane and preserves the z component.
    """
    angle_rad = np.deg2rad(float(rotation_deg))
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    x, y, z = polarization
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float)


def _coerce_non_negative_float(
    params: dict,
    key: str,
    *,
    default: float,
) -> float:
    value = float(params.get(key, default))
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite; got {value!r}.")
    return float(value)


def _deterministic_seed_from_params(params: dict) -> int:
    """Create a deterministic random seed from vectorial optics inputs."""
    parts: list[str] = [
        f"wavelength_nm={params.get('wavelength_nm')!r}",
        f"refractive_index_medium={params.get('refractive_index_medium')!r}",
        f"numerical_aperture={params.get('numerical_aperture')!r}",
        f"random_seed={params.get('random_seed')!r}",
        f"particle_diameter_nm={params.get('particle_diameter_nm')!r}",
        f"particle_refractive_index={params.get('particle_refractive_index')!r}",
        f"vectorial_pupil_samples={params.get('vectorial_pupil_samples')!r}",
        f"pupil_samples={params.get('pupil_samples')!r}",
        f"vectorial_polarization_rotation_deg={params.get('vectorial_polarization_rotation_deg')!r}",
        f"vectorial_obliquity_apodization={params.get('vectorial_obliquity_apodization')!r}",
        f"apodization_factor={params.get('apodization_factor')!r}",
        f"spherical_aberration_strength={params.get('spherical_aberration_strength')!r}",
        f"random_aberration_strength={params.get('random_aberration_strength')!r}",
        f"coverslip_aberration_model={params.get('coverslip_aberration_model')!r}",
        f"coverslip_thickness_um={params.get('coverslip_thickness_um')!r}",
        f"coverslip_design_thickness_um={params.get('coverslip_design_thickness_um')!r}",
        f"coverslip_refractive_index={params.get('coverslip_refractive_index')!r}",
        f"coverslip_design_refractive_index={params.get('coverslip_design_refractive_index')!r}",
    ]
    seed_src = "|".join(parts)
    return int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:16], 16)


def _component_stack_for_polarization(
    params: dict,
    z_positions_nm: np.ndarray,
    polarization: np.ndarray,
    particle_diameter_nm: float | None = None,
    particle_refractive_index: complex | None = None,
) -> dict[str, np.ndarray]:
    wavelength_nm = float(params["wavelength_nm"])
    n_medium = float(params.get("refractive_index_medium", 1.0))
    NA = float(params["numerical_aperture"])
    if wavelength_nm <= 0.0 or n_medium <= 0.0 or NA <= 0.0:
        raise ValueError("wavelength_nm, refractive_index_medium, and numerical_aperture must be positive.")
    if NA > n_medium:
        raise ValueError(
            "numerical_aperture must be <= refractive_index_medium for vectorial_debye; "
            f"got {NA} > {n_medium}."
        )

    samples_raw = params.get("vectorial_pupil_samples", None)
    samples = int(samples_raw if samples_raw is not None else params.get("pupil_samples", 256))
    if samples <= 0:
        raise ValueError("vectorial_pupil_samples/pupil_samples must be positive.")

    os_factor = float(params.get("psf_oversampling_factor", 1.0))
    detector_pixel_nm = float(params.get("pixel_size_nm", 1.0))
    canvas_pixel_nm = detector_pixel_nm / os_factor
    if canvas_pixel_nm <= 0.0:
        raise ValueError("pixel_size_nm / psf_oversampling_factor must be positive.")

    k0 = 2.0 * np.pi / wavelength_nm
    k_medium = k0 * n_medium
    freq = np.fft.fftfreq(samples, d=canvas_pixel_nm) * 2.0 * np.pi
    kx, ky = np.meshgrid(freq, freq, indexing="xy")
    k_perp2 = kx * kx + ky * ky
    max_k_perp = k0 * NA
    aperture = k_perp2 <= max_k_perp * max_k_perp

    kz = np.zeros_like(kx, dtype=float)
    kz[aperture] = np.sqrt(np.maximum(k_medium * k_medium - k_perp2[aperture], 0.0))

    k_perp = np.sqrt(np.clip(k_perp2, 0.0, np.inf))
    sin_theta = np.zeros_like(kz, dtype=float)
    cos_theta = np.zeros_like(kz, dtype=float)
    sin_phi = np.zeros_like(kz, dtype=float)
    cos_phi = np.zeros_like(kz, dtype=float)

    sin_theta[aperture] = k_perp[aperture] / k_medium
    cos_theta[aperture] = kz[aperture] / k_medium

    nonzero_perp = aperture & (k_perp > 0.0)
    if np.any(nonzero_perp):
        denom = k_perp.copy()
        cos_phi[nonzero_perp] = kx[nonzero_perp] / denom[nonzero_perp]
        sin_phi[nonzero_perp] = ky[nonzero_perp] / denom[nonzero_perp]
    if np.any(aperture & (k_perp == 0.0)):
        cos_phi[aperture & (k_perp == 0.0)] = 1.0

    s_hat = np.zeros((3, samples, samples), dtype=float)
    p_hat = np.zeros((3, samples, samples), dtype=float)

    s_hat[0, aperture] = -sin_phi[aperture]
    s_hat[1, aperture] = cos_phi[aperture]
    s_hat[2, aperture] = 0.0

    p_hat[0, aperture] = cos_theta[aperture] * cos_phi[aperture]
    p_hat[1, aperture] = cos_theta[aperture] * sin_phi[aperture]
    p_hat[2, aperture] = -sin_theta[aperture]

    E_s = (
        polarization[0] * s_hat[0]
        + polarization[1] * s_hat[1]
        + polarization[2] * s_hat[2]
    )
    E_p = (
        polarization[0] * p_hat[0]
        + polarization[1] * p_hat[1]
        + polarization[2] * p_hat[2]
    )

    # Optionally apply Mie amplitudes independently to s/p polarization channels.
    S1 = np.ones_like(E_s, dtype=np.complex128)
    S2 = np.ones_like(E_p, dtype=np.complex128)
    if (
        particle_diameter_nm is not None
        and particle_refractive_index is not None
        and np.isfinite(float(particle_diameter_nm))
        and float(particle_diameter_nm) > 0.0
    ):
        mu = np.zeros_like(cos_theta, dtype=float)
        mu[aperture] = cos_theta[aperture]
        radius_nm = 0.5 * float(particle_diameter_nm)
        wavelength_nm = float(params["wavelength_nm"])
        medium_wavelength_nm = wavelength_nm / n_medium
        x = 2.0 * np.pi * radius_nm / medium_wavelength_nm
        if np.isfinite(x) and x > 0.0:
            m = np.asarray(complex(particle_refractive_index), dtype=complex) / complex(n_medium)
            if np.isfinite(m.real) and np.isfinite(m.imag):
                try:
                    a_n, b_n = mie_an_bn(m, x)
                    mu_vec = mu.astype(float)
                    S1_vec = np.zeros_like(mu_vec, dtype=np.complex128)
                    S2_vec = np.zeros_like(mu_vec, dtype=np.complex128)
                    if np.any(aperture):
                        mie_val = mie_S1_S2_from_coefficients(
                            a_n,
                            b_n,
                            mu_vec[aperture],
                        )
                        if isinstance(mie_val, tuple):
                            S1_masked, S2_masked = mie_val
                            S1_vec[aperture] = np.asarray(S1_masked, dtype=np.complex128)
                            S2_vec[aperture] = np.asarray(S2_masked, dtype=np.complex128)
                    S1 = S1_vec
                    S2 = S2_vec
                except (FloatingPointError, ValueError, ZeroDivisionError):
                    S1 = np.ones_like(E_s, dtype=np.complex128)
                    S2 = np.ones_like(E_p, dtype=np.complex128)

    # Aberration and apodization terms.
    max_sin_theta = NA / n_medium
    rho = np.zeros_like(sin_theta, dtype=float)
    rho[aperture] = sin_theta[aperture] / max_sin_theta
    if max_sin_theta <= 0.0:
        raise ValueError("numerical_aperture and refractive_index_medium imply invalid cone.")

    apodization_factor = _coerce_non_negative_float(
        params,
        "apodization_factor",
        default=0.0,
    )
    pupil_radial_apodization = np.exp(-apodization_factor * rho * rho)

    spherical_aberration_strength = _coerce_non_negative_float(
        params,
        "spherical_aberration_strength",
        default=0.0,
    )
    zernike_spherical = np.sqrt(5.0) * (6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0)
    spherical_phase = spherical_aberration_strength * zernike_spherical * 2.0 * np.pi
    coverslip_phase, _coverslip_metadata = compute_coverslip_aberration_phase(
        params,
        sin_theta,
        aperture,
        wavelength_nm=wavelength_nm,
    )

    random_aberration_strength = _coerce_non_negative_float(
        params,
        "random_aberration_strength",
        default=0.0,
    )
    if random_aberration_strength != 0.0:
        rng = np.random.default_rng(_deterministic_seed_from_params(params))
        random_phase = (rng.random((samples, samples)) - 0.5) * (
            2.0 * np.pi * random_aberration_strength
        )
    else:
        random_phase = 0.0

    if bool(params.get("vectorial_obliquity_apodization", True)):
        obliquity = np.sqrt(np.maximum(cos_theta, 0.0))
    else:
        obliquity = np.ones_like(cos_theta)

    pupil_weight = (
        obliquity
        * pupil_radial_apodization
        * np.exp(1j * (spherical_phase + coverslip_phase + random_phase))
    )

    Ex_hat = E_s * S1 * s_hat[0] + E_p * S2 * p_hat[0]
    Ey_hat = E_s * S1 * s_hat[1] + E_p * S2 * p_hat[1]
    Ez_hat = E_s * S1 * s_hat[2] + E_p * S2 * p_hat[2]
    projected = np.asarray([Ex_hat, Ey_hat, Ez_hat], dtype=np.complex128)
    pupil_components = [
        np.where(aperture, projected[axis] * pupil_weight, 0.0)
        for axis in range(3)
    ]

    z_positions = np.asarray(z_positions_nm, dtype=float).reshape(-1)
    stacks = {
        "Ex": np.empty((z_positions.size, samples, samples), dtype=np.complex128),
        "Ey": np.empty((z_positions.size, samples, samples), dtype=np.complex128),
        "Ez": np.empty((z_positions.size, samples, samples), dtype=np.complex128),
        "_coverslip_metadata": _coverslip_metadata,
    }
    for zi, z_nm in enumerate(z_positions):
        phase = np.exp(1j * kz * float(z_nm))
        for name, pupil in zip(("Ex", "Ey", "Ez"), pupil_components, strict=True):
            stacks[name][zi] = fftshift(ifft2(ifftshift(pupil * phase)))
    return stacks


def _normalize_components(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    intensity = sum(np.abs(components[name]) ** 2 for name in ("Ex", "Ey", "Ez"))
    peak = float(np.max(intensity)) if intensity.size else 0.0
    if peak > 0.0 and np.isfinite(peak):
        scale = 1.0 / np.sqrt(peak)
        return {name: components[name] * scale for name in ("Ex", "Ey", "Ez")}
    return {name: components[name] for name in ("Ex", "Ey", "Ez")}


def compute_vectorial_debye_psf(
    params: dict,
    z_positions_nm,
    particle_diameter_nm: float | None = None,
    particle_refractive_index: complex | None = None,
) -> dict[str, Any]:
    """Compute a vectorial Debye/Richards-Wolf angular-spectrum PSF stack."""
    z_positions = np.asarray(z_positions_nm, dtype=float).reshape(-1)
    if z_positions.size == 0 or not np.all(np.isfinite(z_positions)):
        raise ValueError("z_positions_nm must be a non-empty finite 1D sequence.")

    polarization_model = str(params.get("polarization_model", "linear_x")).strip().lower()
    if polarization_model == "scalar":
        polarization_model = "linear_x"

    rotation_deg = float(params.get("vectorial_polarization_rotation_deg", 0.0))
    if polarization_model == "unpolarized":
        x_components = _component_stack_for_polarization(
            params,
            z_positions,
            _rotate_linear_polarization(_polarization_vector("linear_x"), rotation_deg),
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
        )
        y_components = _component_stack_for_polarization(
            params,
            z_positions,
            _rotate_linear_polarization(_polarization_vector("linear_y"), rotation_deg),
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
        )
        components = {
            name: np.sqrt(0.5 * (np.abs(x_components[name]) ** 2 + np.abs(y_components[name]) ** 2)).astype(
                np.complex128
            )
            for name in ("Ex", "Ey", "Ez")
        }
        coverslip_metadata = dict(x_components.get("_coverslip_metadata", {}))
        unpolarized_average = True
    else:
        components = _component_stack_for_polarization(
            params,
            z_positions,
            _rotate_linear_polarization(
                _polarization_vector(polarization_model),
                rotation_deg,
            ),
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
        )
        coverslip_metadata = dict(components.get("_coverslip_metadata", {}))
        unpolarized_average = False

    components = _normalize_components(components)
    metadata = {
        "backend": "vectorial_debye",
        "polarization_model": polarization_model,
        "vectorial_detection_mode": str(params.get("vectorial_detection_mode", "incoherent_sum")),
        "vectorial_pupil_samples": int(
            params.get("vectorial_pupil_samples") or params.get("pupil_samples", 256)
        ),
        "wavelength_nm": float(params["wavelength_nm"]),
        "numerical_aperture": float(params["numerical_aperture"]),
        "refractive_index_medium": float(params.get("refractive_index_medium", 1.0)),
        "obliquity_apodization": bool(params.get("vectorial_obliquity_apodization", True)),
        "apodization_factor": float(params.get("apodization_factor", 0.0)),
        "spherical_aberration_strength": float(params.get("spherical_aberration_strength", 0.0)),
        "random_aberration_strength": float(params.get("random_aberration_strength", 0.0)),
        **coverslip_metadata,
        "vectorial_polarization_rotation_deg": float(rotation_deg),
        "unpolarized_mode_averages_x_and_y": bool(unpolarized_average),
        "normalization": "peak_vector_intensity_equals_one",
        "z_positions_nm": z_positions.astype(float).tolist(),
    }
    return {**components, "metadata": metadata}
