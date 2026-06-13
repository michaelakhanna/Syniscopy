from __future__ import annotations

from functools import lru_cache
from typing import Any

import hashlib
import numpy as np
from scipy.fft import fftshift, ifft2, ifftshift

from config.runtime import (
    CoverslipAberrationSettings,
    OpticalInstrumentSettings,
    SamplingGeometry,
    VectorialOpticsSettings,
)
from mie_scattering import mie_S1_S2_from_coefficients, mie_an_bn
from optical_extensions import compute_coverslip_aberration_phase
from optical_scattering import (
    OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY,
    OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS,
    OPTICAL_SCATTERING_MIE,
    born_rayleigh_gans_form_factor,
)
from stochastic_runtime import rng_from_seed


@lru_cache(maxsize=128)
def _vectorial_pupil_coordinates(
    samples: int,
    canvas_pixel_nm: float,
    wavelength_nm: float,
    n_medium: float,
    numerical_aperture: float,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k0 = 2.0 * np.pi / float(wavelength_nm)
    k_medium = k0 * float(n_medium)
    freq = np.fft.fftshift(np.fft.fftfreq(int(samples), d=float(canvas_pixel_nm)) * 2.0 * np.pi)
    kx, ky = np.meshgrid(freq, freq, indexing="xy")
    k_perp2 = kx * kx + ky * ky
    max_k_perp = k0 * float(numerical_aperture)
    aperture = k_perp2 <= max_k_perp * max_k_perp
    for arr in (kx, ky, k_perp2, aperture):
        arr.setflags(write=False)
    return k0, k_medium, kx, ky, k_perp2, aperture


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


def _deterministic_seed_from_params(params: dict) -> int:
    """Create a deterministic random seed from vectorial optics inputs."""
    settings = VectorialOpticsSettings.from_params(params)
    # The seed captures system-level optical randomness only. It must not include
    # per-particle properties, which belong to scattering transfer, not to
    # static pupil aberration realization.
    coverslip = CoverslipAberrationSettings.from_params(params)
    parts: list[str] = [
        f"wavelength_nm={settings.instrument.wavelength_nm!r}",
        f"probe_wavelength_nm={settings.instrument.probe_wavelength_nm!r}",
        f"refractive_index_medium={settings.instrument.refractive_index_medium!r}",
        f"numerical_aperture={settings.instrument.numerical_aperture!r}",
        f"optical_aberration_seed={settings.optical_aberration_seed!r}",
        f"vectorial_pupil_samples={settings.instrument.vectorial_pupil_samples!r}",
        f"pupil_samples={settings.instrument.pupil_samples!r}",
        f"vectorial_polarization_rotation_deg={settings.optical.vectorial_polarization_rotation_deg!r}",
        f"vectorial_obliquity_apodization={settings.obliquity_apodization!r}",
        f"apodization_factor={settings.apodization_factor!r}",
        f"spherical_aberration_strength={settings.spherical_aberration_strength!r}",
        f"random_aberration_strength={settings.random_aberration_strength!r}",
        f"coverslip_aberration_model={coverslip.model!r}",
        f"coverslip_thickness_um={coverslip.thickness_um!r}",
        f"coverslip_design_thickness_um={coverslip.design_thickness_um!r}",
        f"coverslip_refractive_index={coverslip.refractive_index!r}",
        f"coverslip_design_refractive_index={coverslip.design_refractive_index!r}",
    ]
    seed_src = "|".join(parts)
    return int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:16], 16)


def _component_stack_for_polarization(
    params: dict,
    z_positions_nm: np.ndarray,
    polarization: np.ndarray,
    particle_diameter_nm: float | None = None,
    particle_refractive_index: complex | None = None,
    apply_mie_amplitudes: bool = True,
    optical_scattering_model: str = OPTICAL_SCATTERING_MIE,
    component_geometry: Any | None = None,
    orientation_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    settings = VectorialOpticsSettings.from_params(params)
    instrument = settings.instrument
    wavelength_nm = instrument.probe_wavelength_nm
    n_medium = instrument.refractive_index_medium
    NA = instrument.numerical_aperture
    if wavelength_nm <= 0.0 or n_medium <= 0.0 or NA <= 0.0:
        raise ValueError("wavelength_nm, refractive_index_medium, and numerical_aperture must be positive.")
    if NA > n_medium:
        raise ValueError(
            "numerical_aperture must be <= refractive_index_medium for vectorial_debye; "
            f"got {NA} > {n_medium}."
        )

    samples = instrument.vectorial_pupil_samples
    if samples <= 0:
        raise ValueError("vectorial_pupil_samples/pupil_samples must be positive.")

    canvas_pixel_nm = settings.sampling.model_canvas_pixel_size_nm
    if canvas_pixel_nm <= 0.0:
        raise ValueError("pixel_size_nm / psf_oversampling_factor must be positive.")

    k0, k_medium, kx, ky, k_perp2, aperture = _vectorial_pupil_coordinates(
        int(samples),
        float(canvas_pixel_nm),
        float(wavelength_nm),
        float(n_medium),
        float(NA),
    )

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

    # Apply the declared scattering transfer in the pupil. Mie keeps separate
    # exact sphere amplitudes for s/p channels. Born/Rayleigh-Gans uses a
    # scalar primitive form factor and therefore multiplies both channels.
    S1 = np.ones_like(E_s, dtype=np.complex128)
    S2 = np.ones_like(E_p, dtype=np.complex128)
    scattering_model = str(optical_scattering_model).strip().lower()
    if (
        bool(apply_mie_amplitudes)
        and scattering_model == OPTICAL_SCATTERING_MIE
        and
        particle_diameter_nm is not None
        and particle_refractive_index is not None
        and np.isfinite(float(particle_diameter_nm))
        and float(particle_diameter_nm) > 0.0
    ):
        mu = np.zeros_like(cos_theta, dtype=float)
        mu[aperture] = cos_theta[aperture]
        radius_nm = 0.5 * float(particle_diameter_nm)
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
    elif scattering_model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS:
        if component_geometry is None:
            raise ValueError("Born/Rayleigh-Gans vectorial scattering requires component_geometry.")
        transfer = born_rayleigh_gans_form_factor(
            component_geometry=component_geometry,
            qx_nm_inv=kx,
            qy_nm_inv=ky,
            qz_nm_inv=kz - k_medium,
            orientation_matrix=orientation_matrix,
        )
        S1 = transfer
        S2 = transfer

    # Aberration and apodization terms.
    max_sin_theta = NA / n_medium
    rho = np.zeros_like(sin_theta, dtype=float)
    rho[aperture] = sin_theta[aperture] / max_sin_theta
    if max_sin_theta <= 0.0:
        raise ValueError("numerical_aperture and refractive_index_medium imply invalid cone.")

    apodization_factor = settings.apodization_factor
    pupil_radial_apodization = np.exp(-apodization_factor * rho * rho)

    spherical_aberration_strength = settings.spherical_aberration_strength
    zernike_spherical = np.sqrt(5.0) * (6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0)
    spherical_phase = spherical_aberration_strength * zernike_spherical * 2.0 * np.pi
    coverslip_phase, _coverslip_metadata = compute_coverslip_aberration_phase(
        params,
        sin_theta,
        aperture,
        wavelength_nm=wavelength_nm,
    )

    random_aberration_strength = settings.random_aberration_strength
    if random_aberration_strength != 0.0:
        rng = rng_from_seed(
            _deterministic_seed_from_params(params),
            stream="vectorial_optics_random_aberration",
        )
        random_phase = (rng.random((samples, samples)) - 0.5) * (
            2.0 * np.pi * random_aberration_strength
        )
    else:
        random_phase = 0.0

    if settings.obliquity_apodization:
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
        if len(pupil_components) != 3:
            raise RuntimeError(
                "Vectorial Debye pupil construction must produce Ex/Ey/Ez components."
            )
        for name, pupil in zip(("Ex", "Ey", "Ez"), pupil_components):
            stacks[name][zi] = fftshift(ifft2(ifftshift(pupil * phase)))
    return stacks


def _normalize_components(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    intensity = sum(np.abs(components[name]) ** 2 for name in ("Ex", "Ey", "Ez"))
    peak = float(np.max(intensity)) if intensity.size else 0.0
    if peak > 0.0 and np.isfinite(peak):
        scale = 1.0 / np.sqrt(peak)
        return {name: components[name] * scale for name in ("Ex", "Ey", "Ez")}
    return {name: components[name] for name in ("Ex", "Ey", "Ez")}


def _component_peak_amplitude(components: dict[str, np.ndarray]) -> float:
    intensity = sum(np.abs(components[name]) ** 2 for name in ("Ex", "Ey", "Ez"))
    peak = float(np.max(intensity)) if intensity.size else 0.0
    return float(np.sqrt(peak)) if peak > 0.0 and np.isfinite(peak) else 1.0


def _vectorial_reference_polarization(settings: VectorialOpticsSettings) -> np.ndarray:
    polarization_model = settings.optical.polarization_model
    if polarization_model in {"scalar", "unpolarized"}:
        polarization_model = "linear_x"
    return _rotate_linear_polarization(
        _polarization_vector(polarization_model),
        settings.optical.vectorial_polarization_rotation_deg,
    )


def compute_vectorial_debye_basis_psf(
    params: dict,
    z_positions_nm,
    *,
    particle_diameter_nm: float | None = None,
    particle_refractive_index: complex | None = None,
    optical_scattering_model: str = OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY,
    component_geometry: Any | None = None,
    orientation_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return unit-dipole vectorial Debye responses for later linear combination.

    The returned basis has shape ``(3, Z, 3, H, W)``.  Axis 0 is the source dipole
    component and axis 2 is the detected field component.  A shared normalization
    is chosen so the configured incident-polarization response has unit peak
    vector intensity; this preserves the existing PSF normalization while keeping
    the Debye operator linear in the dipole vector.
    """

    z_positions = np.asarray(z_positions_nm, dtype=float).reshape(-1)
    if z_positions.size == 0 or not np.all(np.isfinite(z_positions)):
        raise ValueError("z_positions_nm must be a non-empty finite 1D sequence.")
    settings = VectorialOpticsSettings.from_params(params)
    instrument = settings.instrument
    scattering_model = str(optical_scattering_model).strip().lower()
    if scattering_model not in {
        OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY,
        OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS,
    }:
        raise ValueError(
            "compute_vectorial_debye_basis_psf is for linear primitive scattering "
            f"models; got optical_scattering_model={optical_scattering_model!r}."
        )

    basis_vectors = np.eye(3, dtype=np.complex128)
    basis_stacks: list[np.ndarray] = []
    coverslip_metadata: dict[str, Any] = {}
    for vector in basis_vectors:
        components = _component_stack_for_polarization(
            params,
            z_positions,
            vector,
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
            apply_mie_amplitudes=False,
            optical_scattering_model=scattering_model,
            component_geometry=component_geometry,
            orientation_matrix=orientation_matrix,
        )
        if not coverslip_metadata:
            coverslip_metadata = dict(components.get("_coverslip_metadata", {}))
        basis_stacks.append(
            np.stack([components["Ex"], components["Ey"], components["Ez"]], axis=1)
        )
    basis = np.stack(basis_stacks, axis=0).astype(np.complex128)
    reference_polarization = _vectorial_reference_polarization(settings).astype(np.complex128)
    reference_stack = np.tensordot(reference_polarization, basis, axes=(0, 0))
    reference_intensity = np.sum(np.abs(reference_stack) ** 2, axis=1)
    reference_peak = float(np.sqrt(np.max(reference_intensity))) if reference_intensity.size else 0.0
    if not np.isfinite(reference_peak) or reference_peak <= 0.0:
        reference_peak = 1.0
    basis = basis / reference_peak
    metadata = {
        "backend": "vectorial_debye",
        "polarization_model": settings.optical.polarization_model,
        "vectorial_detection_mode": settings.optical.vectorial_detection_mode,
        "vectorial_pupil_samples": instrument.vectorial_pupil_samples,
        "wavelength_nm": instrument.probe_wavelength_nm,
        "numerical_aperture": instrument.numerical_aperture,
        "refractive_index_medium": instrument.refractive_index_medium,
        "obliquity_apodization": settings.obliquity_apodization,
        "apodization_factor": settings.apodization_factor,
        "spherical_aberration_strength": settings.spherical_aberration_strength,
        "random_aberration_strength": settings.random_aberration_strength,
        **coverslip_metadata,
        "vectorial_polarization_rotation_deg": float(settings.optical.vectorial_polarization_rotation_deg),
        "normalization": "basis_normalized_to_incident_polarization_peak_vector_intensity",
        "field_amplitude_scale": 1.0,
        "field_amplitude_scale_semantics": (
            "Analytic polarizability vectorial Debye: unit-dipole basis fields are "
            "linearly combined with the scaled world-frame polarizability dipole "
            "at render-time."
        ),
        "optical_scattering_model": scattering_model,
        "vectorial_basis_response": True,
        "vectorial_basis_axis_order": "source_dipole_xyz_then_field_ExEyEz",
        "z_positions_nm": z_positions.astype(float).tolist(),
    }
    return {"basis": basis, "metadata": metadata}


def _physical_scattered_amplitude_scale(
    params: dict,
    normalized_components: dict[str, np.ndarray],
    z_positions: np.ndarray,
    particle_diameter_nm: float | None,
    particle_refractive_index: complex | None,
    wavelength_nm: float,
    optical_scattering_model: str = OPTICAL_SCATTERING_MIE,
) -> float:
    """Absolute amplitude that ties the rendered scattered field to physics.

    The shape-normalized (Ex,Ey,Ez) carry an arbitrary scale. This returns the
    factor such that, for a unit incident photon per detector pixel, the rendered
    scattered intensity sums to the physical collected Mie scattering
    cross-section:

        sum_pixels (|Ex|^2+|Ey|^2+|Ez|^2) * pixel_area_nm2  ==  sigma_sca_collected

    i.e. ``illumination_count`` / ``background_intensity`` become true incident
    photons per pixel and every modality's absolute signal is physical. The
    cross-section is computed from the validated Mie amplitudes (L02/L08); no
    tuned constant. Falls back to the old peak amplitude when the particle
    optical identity is unavailable (e.g. source-map modalities).
    """
    if str(optical_scattering_model).strip().lower() != OPTICAL_SCATTERING_MIE:
        return 1.0
    if particle_diameter_nm is None or particle_refractive_index is None:
        return _component_peak_amplitude(normalized_components)
    try:
        from mie_scattering import mie_scattering_cross_section_nm2

        instrument = OpticalInstrumentSettings.from_params(params)
        n_medium = instrument.refractive_index_medium
        numerical_aperture = instrument.numerical_aperture
        if not np.isfinite(n_medium) or n_medium <= 0.0:
            return _component_peak_amplitude(normalized_components)
        m_rel = complex(particle_refractive_index) / n_medium
        half_angle = float(np.arcsin(min(numerical_aperture / n_medium, 1.0)))
        sigma_nm2 = mie_scattering_cross_section_nm2(
            m_rel,
            float(particle_diameter_nm),
            float(wavelength_nm),
            n_medium,
            collection_half_angle_rad=half_angle,
        )
        canvas_pitch_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
        pixel_area_nm2 = canvas_pitch_nm * canvas_pitch_nm
        if pixel_area_nm2 <= 0.0:
            return _component_peak_amplitude(normalized_components)
        infocus = int(np.argmin(np.abs(z_positions)))
        shape_power = float(
            sum(
                np.sum(np.abs(normalized_components[name][infocus]) ** 2)
                for name in ("Ex", "Ey", "Ez")
            )
        )
        if not np.isfinite(shape_power) or shape_power <= 0.0:
            return _component_peak_amplitude(normalized_components)
        target_power = sigma_nm2 / pixel_area_nm2
        scale = float(np.sqrt(target_power / shape_power))
        return scale if np.isfinite(scale) and scale > 0.0 else _component_peak_amplitude(normalized_components)
    except Exception:
        return _component_peak_amplitude(normalized_components)


def compute_vectorial_debye_psf(
    params: dict,
    z_positions_nm,
    particle_diameter_nm: float | None = None,
    particle_refractive_index: complex | None = None,
    optical_scattering_model: str = OPTICAL_SCATTERING_MIE,
    component_geometry: Any | None = None,
    orientation_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a vectorial Debye/Richards-Wolf angular-spectrum PSF stack."""
    z_positions = np.asarray(z_positions_nm, dtype=float).reshape(-1)
    if z_positions.size == 0 or not np.all(np.isfinite(z_positions)):
        raise ValueError("z_positions_nm must be a non-empty finite 1D sequence.")
    settings = VectorialOpticsSettings.from_params(params)
    instrument = settings.instrument
    wavelength_nm = instrument.probe_wavelength_nm

    polarization_model = settings.optical.polarization_model
    if polarization_model == "scalar":
        polarization_model = "linear_x"

    rotation_deg = settings.optical.vectorial_polarization_rotation_deg
    scattering_model = str(optical_scattering_model).strip().lower()
    if scattering_model not in {
        OPTICAL_SCATTERING_MIE,
        OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY,
        OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS,
    }:
        raise ValueError(f"Unsupported optical_scattering_model={optical_scattering_model!r}.")
    apply_mie = scattering_model == OPTICAL_SCATTERING_MIE
    if polarization_model == "unpolarized":
        x_components = _component_stack_for_polarization(
            params,
            z_positions,
            _rotate_linear_polarization(_polarization_vector("linear_x"), rotation_deg),
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
            apply_mie_amplitudes=apply_mie,
            optical_scattering_model=scattering_model,
            component_geometry=component_geometry,
            orientation_matrix=orientation_matrix,
        )
        y_components = _component_stack_for_polarization(
            params,
            z_positions,
            _rotate_linear_polarization(_polarization_vector("linear_y"), rotation_deg),
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
            apply_mie_amplitudes=apply_mie,
            optical_scattering_model=scattering_model,
            component_geometry=component_geometry,
            orientation_matrix=orientation_matrix,
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
            apply_mie_amplitudes=apply_mie,
            optical_scattering_model=scattering_model,
            component_geometry=component_geometry,
            orientation_matrix=orientation_matrix,
        )
        coverslip_metadata = dict(components.get("_coverslip_metadata", {}))
        unpolarized_average = False

    components = _normalize_components(components)
    field_amplitude_scale = _physical_scattered_amplitude_scale(
        params,
        components,
        z_positions,
        particle_diameter_nm,
        particle_refractive_index,
        wavelength_nm,
        optical_scattering_model=scattering_model,
    )
    metadata = {
        "backend": "vectorial_debye",
        "polarization_model": polarization_model,
        "vectorial_detection_mode": settings.optical.vectorial_detection_mode,
        "vectorial_pupil_samples": instrument.vectorial_pupil_samples,
        "wavelength_nm": wavelength_nm,
        "numerical_aperture": instrument.numerical_aperture,
        "refractive_index_medium": instrument.refractive_index_medium,
        "obliquity_apodization": settings.obliquity_apodization,
        "apodization_factor": settings.apodization_factor,
        "spherical_aberration_strength": settings.spherical_aberration_strength,
        "random_aberration_strength": settings.random_aberration_strength,
        **coverslip_metadata,
        "vectorial_polarization_rotation_deg": float(rotation_deg),
        "unpolarized_mode_averages_x_and_y": bool(unpolarized_average),
        "normalization": "shape_peak_vector_intensity_equals_one",
        "field_amplitude_scale": float(field_amplitude_scale),
        "field_amplitude_scale_semantics": (
            "Mie: multiply normalized Ex/Ey/Ez fields by this factor so rendered "
            "scattered intensity integrates to the physical collected Mie cross-section. "
            "Analytic polarizability: this is 1.0 and the orientation-dependent "
            "polarizability amplitude is applied during particle stamping. "
            "Born/Rayleigh-Gans: this is 1.0; the primitive form factor is "
            "applied in the pupil and the weak-scattering volume/contrast "
            "amplitude is applied during particle stamping."
        ),
        "optical_scattering_model": scattering_model,
        "z_positions_nm": z_positions.astype(float).tolist(),
    }
    return {**components, "metadata": metadata}


def compute_isotropic_dipole_emission_psf(
    params: dict,
    z_positions_nm,
) -> dict[str, Any]:
    """Compute an incoherent isotropic-dipole fluorescence emission PSF stack."""
    z_positions = np.asarray(z_positions_nm, dtype=float).reshape(-1)
    if z_positions.size == 0 or not np.all(np.isfinite(z_positions)):
        raise ValueError("z_positions_nm must be a non-empty finite 1D sequence.")
    settings = VectorialOpticsSettings.from_params(params)
    instrument = settings.instrument
    wavelength_nm = instrument.probe_wavelength_nm
    dipoles = (
        ("x", np.array([1.0, 0.0, 0.0], dtype=float)),
        ("y", np.array([0.0, 1.0, 0.0], dtype=float)),
        ("z", np.array([0.0, 0.0, 1.0], dtype=float)),
    )
    intensity_accum: np.ndarray | None = None
    coverslip_metadata: dict[str, Any] = {}
    for _name, dipole in dipoles:
        components = _component_stack_for_polarization(params, z_positions, dipole)
        intensity = sum(np.abs(components[name]) ** 2 for name in ("Ex", "Ey", "Ez"))
        if intensity_accum is None:
            intensity_accum = np.asarray(intensity, dtype=float)
            coverslip_metadata = dict(components.get("_coverslip_metadata", {}))
        else:
            intensity_accum = intensity_accum + np.asarray(intensity, dtype=float)
    if intensity_accum is None:
        raise RuntimeError("Isotropic dipole PSF construction produced no components.")
    intensity_stack = np.maximum(intensity_accum / 3.0, 0.0)
    peak = float(np.max(intensity_stack)) if intensity_stack.size else 0.0
    if peak > 0.0 and np.isfinite(peak):
        intensity_stack = intensity_stack / peak
    metadata = {
        "backend": "isotropic_dipole_vectorial_debye_emission",
        "emitter_orientation_model": "isotropic_dipole_incoherent_average_xyz",
        "illumination_polarization_decoupled": True,
        "vectorial_detection_mode": settings.optical.vectorial_detection_mode,
        "vectorial_pupil_samples": instrument.vectorial_pupil_samples,
        "wavelength_nm": wavelength_nm,
        "numerical_aperture": instrument.numerical_aperture,
        "refractive_index_medium": instrument.refractive_index_medium,
        "obliquity_apodization": settings.obliquity_apodization,
        "apodization_factor": settings.apodization_factor,
        "spherical_aberration_strength": settings.spherical_aberration_strength,
        "random_aberration_strength": settings.random_aberration_strength,
        **coverslip_metadata,
        "normalization": "shape_peak_intensity_equals_one",
        "z_positions_nm": z_positions.astype(float).tolist(),
    }
    return {"intensity": intensity_stack, "metadata": metadata}
