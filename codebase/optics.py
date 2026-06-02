import logging

import numpy as np
from scipy.fft import ifft2, fftshift, ifftshift
from tqdm import tqdm

from mie_scattering import (
    mie_S1_S2_from_coefficients,
    mie_S2_from_coefficients,
    mie_an_bn,
    mie_scattering_amplitudes_from_coefficients,
)
from vectorial_optics import compute_vectorial_debye_psf
from optical_extensions import compute_coverslip_aberration_phase
from shared_constants import COHERENT_REFERENCE_MODALITIES


logger = logging.getLogger(__name__)


def _interpolate_z_stack_single(
    z_values: np.ndarray,
    stack: np.ndarray,
    z_min: float,
    z_max: float,
    z_val: float,
) -> np.ndarray:
    """
    Interpolate one z position from a precomputed complex PSF stack.

    Values outside the precomputed range are clamped to the nearest available
    slice so a narrow cache does not make a particle vanish abruptly.
    """
    if z_val < z_min:
        return stack[0]
    if z_val > z_max:
        return stack[-1]

    if z_values.size == 1:
        return stack[0]

    upper_index = int(np.searchsorted(z_values, z_val, side="right"))
    lower_index = upper_index - 1
    if lower_index < 0:
        return stack[0]
    if upper_index >= z_values.size:
        return stack[-1]

    z_lower = float(z_values[lower_index])
    z_upper = float(z_values[upper_index])
    alpha = (z_val - z_lower) / (z_upper - z_lower)
    lower_slice = stack[lower_index]
    upper_slice = stack[upper_index]
    return (1.0 - alpha) * lower_slice + alpha * upper_slice


class ComplexPSFZInterpolator:
    """
    Lightweight 1D interpolator over a precomputed complex PSF Z-stack.

    The underlying data is a 3D array of shape (num_z, height, width) containing
    the complex-valued coherent PSF for each discrete z position in z_values_nm.
    This class performs linear interpolation along z and returns the corresponding
    2D complex field.

    This is specialized for a 1D z-grid with a full 2D field stored at each
    grid point. Queries outside the precomputed z-range are clamped to
    the nearest available slice instead of returning a zero field; the renderer
    should never make a real particle vanish solely because the cache range was
    too narrow.

    Each particle type has its own ComplexPSFZInterpolator instance with a
    type-specific z_grid derived from the realized Brownian trajectories of that
    type plus a safety margin.
    Different types therefore have independent axial coverage tailored to their motion.
    """

    def __init__(self, z_values_nm, ipsf_stack_complex, metadata=None):
        """
        Args:
            z_values_nm (array-like): 1D array of z positions (in nm) at which
                the complex PSF has been precomputed. The grid may be any
                sorted particle-type-specific axial sampling that covers the
                associated trajectory range.
            ipsf_stack_complex (np.ndarray): 3D complex array with shape
                (len(z_values_nm), height, width). The first axis corresponds
                to the z positions.
        """
        z_values = np.asarray(z_values_nm, dtype=float)
        if z_values.ndim != 1 or z_values.size == 0:
            raise ValueError("z_values_nm must be a non-empty 1D array.")

        ipsf_stack = np.asarray(ipsf_stack_complex, dtype=np.complex128)
        if ipsf_stack.shape[0] != z_values.size:
            raise ValueError(
                "First dimension of ipsf_stack_complex must match the length "
                "of z_values_nm."
            )

        order = np.argsort(z_values)
        self.z_values = z_values[order]
        self.ipsf_stack = ipsf_stack[order]
        if self.z_values.size > 1 and np.any(np.diff(self.z_values) <= 0.0):
            raise ValueError("z_values_nm must contain unique z positions.")

        self.z_min = float(self.z_values[0])
        self.z_max = float(self.z_values[-1])
        self.metadata = dict(metadata or {})

    def __call__(self, z_nm):
        """
        Linearly interpolate the iPSF stack along z.

        Args:
            z_nm (float or array-like): Axial position(s) in nanometers.

        Returns:
            np.ndarray:
                - If z_nm is a scalar, returns a 2D complex array of shape
                  (height, width) for that z position.
                - If z_nm is array-like with shape (N,), returns a 3D complex
                  array of shape (N, height, width), where each slice along the
                  first axis corresponds to one input z.
        """
        z = np.asarray(z_nm, dtype=float)

        # Scalar input: return a single 2D iPSF slice.
        if z.ndim == 0:
            return self._interp_single(float(z))

        # Vector input: interpolate each z independently.
        z_flat = z.ravel()
        out = np.empty((z_flat.size,) + self.ipsf_stack.shape[1:], dtype=np.complex128)
        for idx, z_val in enumerate(z_flat):
            out[idx] = self._interp_single(float(z_val))

        # Reshape back to match the input z shape, with PSF dimensions appended.
        new_shape = z.shape + self.ipsf_stack.shape[1:]
        return out.reshape(new_shape)

    def _interp_single(self, z_val):
        """
        Interpolate for a single scalar z position.

        For z values outside the precomputed range, returns the nearest edge
        slice. This is a conservative display/rendering fallback: the preferred
        path is still to build a PSF grid covering the realized trajectory, but
        clamping avoids sudden end-of-video signal collapse when a trajectory or
        sub-frame exposure sample lands just outside the cache.
        """
        # Outside the precomputed z-range: use the nearest computed slice. A
        # zero fill would incorrectly turn an out-of-cache particle into no
        # particle and corrupt mask/video alignment.
        return _interpolate_z_stack_single(
            self.z_values,
            self.ipsf_stack,
            self.z_min,
            self.z_max,
            z_val,
        )


class VectorialPSFZInterpolator:
    """
    1D interpolator over a precomputed vectorial complex PSF stack.

    The stored data has shape ``(num_z, 3, height, width)`` with component
    order ``(Ex, Ey, Ez)``. Interpolation follows the same axial clamping and
    piecewise-linear behavior as :class:`ComplexPSFZInterpolator`.
    """

    def __init__(self, z_values_nm, ipsf_stack_vector, metadata=None):
        z_values = np.asarray(z_values_nm, dtype=float)
        if z_values.ndim != 1 or z_values.size == 0:
            raise ValueError("z_values_nm must be a non-empty 1D array.")

        ipsf_stack = np.asarray(ipsf_stack_vector, dtype=np.complex128)
        if ipsf_stack.ndim != 4 or ipsf_stack.shape[1] != 3:
            raise ValueError(
                "VectorialPSFZInterpolator expects input shape (len(z_values), 3, H, W)."
            )
        if ipsf_stack.shape[0] != z_values.size:
            raise ValueError(
                "First axis of ipsf_stack_vector must match the length of z_values_nm."
            )

        order = np.argsort(z_values)
        self.z_values = z_values[order]
        self.ipsf_stack = ipsf_stack[order]
        if self.z_values.size > 1 and np.any(np.diff(self.z_values) <= 0.0):
            raise ValueError("z_values_nm must contain unique z positions.")

        self.z_min = float(self.z_values[0])
        self.z_max = float(self.z_values[-1])
        self.metadata = dict(metadata or {})

    def __call__(self, z_nm):
        z = np.asarray(z_nm, dtype=float)

        if z.ndim == 0:
            return self._interp_single(float(z))

        z_flat = z.ravel()
        out = np.empty((z_flat.size,) + self.ipsf_stack.shape[1:], dtype=np.complex128)
        for idx, z_val in enumerate(z_flat):
            out[idx] = self._interp_single(float(z_val))

        new_shape = z.shape + self.ipsf_stack.shape[1:]
        return out.reshape(new_shape)

    def _interp_single(self, z_val):
        return _interpolate_z_stack_single(
            self.z_values,
            self.ipsf_stack,
            self.z_min,
            self.z_max,
            z_val,
        )


def compute_complex_psf_stack(params, particle_diameter_nm, particle_refractive_index, z_values_nm):
    """
    Compute a scalar complex coherent Point Spread Function (PSF) stack using a
    pupil-propagation Debye-Born-style integral, calculated via FFT for
    efficiency, and then
    enforce **radial symmetry** of each slice by ring-averaging the complex field
    with **continuous radial interpolation**.

    This default backend is scalar.
    Polarization and explicit vector-field components are tracked when the
    ``optical_field_backend`` selects vectorial Debye.
    The returned field in this backend is a scalar complex scattered-field proxy
    shared by the pluggable imaging models.

    Fundamental architectural decisions:
        - The z-grid is provided explicitly via `z_values_nm` and is specific to
          the associated particle type.  Z-values are supplied per particle type by the caller.

        - Different particle types may therefore have different axial coverage,
          sized from their own realized Brownian trajectories plus a safety margin.
          ComplexPSFZInterpolator stores this z-grid internally. Queries outside
          the range clamp to the nearest slice; the intended path is still to
          build a cache that covers the rendered trajectory.

    Pipeline:
        1. Build the pupil function on a 2D k-space grid using:
             - Circular aperture (NA / n_medium).
             - Mie scattering amplitude S2(mu) across the pupil.
             - Apodization, spherical aberration, random aberration.
        2. Compute the 2D complex Amplitude Spread Function (ASF) via inverse FFT.
        3. For each z-slice in `z_values_nm`:
             - Compute ASF with the appropriate defocus phase.
             - Compute a 1D complex radial profile E_radial[k] via integer
               radius bin averaging.
             - For each pixel, evaluate E(r) at its continuous radius r using
               linear interpolation of E_radial, instead of snapping to the
               nearest radius bin.

    Args:
        params (dict): The main simulation parameter dictionary.
        particle_diameter_nm (float): The diameter of the particle for this PSF.
        particle_refractive_index (complex): The complex refractive index of
            the particle.
        z_values_nm (array-like): 1D array of z positions (in nm) at which to
            compute the PSF stack for this particle type. This is typically a
            type-specific range derived from the realized trajectories.

    Returns:
        ComplexPSFZInterpolator or VectorialPSFZInterpolator: An interpolator
        that can return the complex PSF for a given z-position. For
        vector-aware coherent full-vector paths, this is a 3-component vector interpolator
        with shape ``(3, H, W)``; otherwise a 2D scalar interpolator is
        returned.
    """
    # --- Validate and store z-grid ---
    z_values = np.asarray(z_values_nm, dtype=float)
    if z_values.ndim != 1 or z_values.size == 0:
        raise ValueError("z_values_nm must be a non-empty 1D array.")
    backend = str(params.get("optical_field_backend", "vectorial_debye")).strip().lower()
    if backend == "vectorial_debye":
        vectorial = compute_vectorial_debye_psf(
            params,
            z_values,
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
        )
        dpc_channel_model = str(
            params.get("dpc_channel_model", "vectorial_debye_asymmetric_illumination")
        ).strip().lower()
        active_modality = str(params.get("imaging_model", "")).strip().lower()
        use_full_vectorial_dpc = active_modality in {
            "dpc",
            "differential_phase_contrast",
        } and dpc_channel_model in {
            "vectorial_debye_asymmetric_illumination",
            "two_axis_vectorial_debye_asymmetric_illumination",
            "vectorial",
        }
        detection_mode = str(params.get("vectorial_detection_mode", "incoherent_sum")).strip().lower()
        use_full_vectorial_coherent = (
            detection_mode == "full_vector"
            and active_modality in COHERENT_REFERENCE_MODALITIES
        )
        if use_full_vectorial_coherent:
            polarization_model = str(params.get("polarization_model", "linear_x")).strip().lower()
            if polarization_model == "scalar":
                polarization_model = "linear_x"
            if polarization_model == "unpolarized":
                raise ValueError(
                    "polarization_model='unpolarized' is an incoherent average and "
                    "cannot be used with vectorial_detection_mode='full_vector' for "
                    f"coherent imaging_model={active_modality!r}. Use linear_x, "
                    "linear_y, an analyzer mode, or optical_field_backend='scalar_paraxial'."
                )
        if use_full_vectorial_dpc or use_full_vectorial_coherent:
            vector_stack = np.stack(
                [vectorial["Ex"], vectorial["Ey"], vectorial["Ez"]],
                axis=1,
            )
            metadata = dict(vectorial.get("metadata", {}))
            metadata.update(
                {
                    "scalar_compatibility_reduction": "full_vector_field",
                    "field_representation": "vectorial_coherent_field",
                    "vectorial_detection_requested": detection_mode,
                    "vectorial_field_reason": (
                        "dpc_vectorial_channel"
                        if use_full_vectorial_dpc
                        else "coherent_full_vector_detection"
                    ),
                    "particle_diameter_nm": float(particle_diameter_nm),
                    "particle_refractive_index": {
                        "real": float(complex(particle_refractive_index).real),
                        "imag": float(complex(particle_refractive_index).imag),
                    },
                }
            )
            return VectorialPSFZInterpolator(z_values, vector_stack, metadata=metadata)

        if detection_mode == "analyzer_x":
            scalar_stack = vectorial["Ex"]
            reduction = "analyzer_x_component"
            field_representation = "scalar_coherent_vector_component"
        elif detection_mode == "analyzer_y":
            scalar_stack = vectorial["Ey"]
            reduction = "analyzer_y_component"
            field_representation = "scalar_coherent_vector_component"
        elif detection_mode == "unpolarized":
            if active_modality in COHERENT_REFERENCE_MODALITIES:
                raise ValueError(
                    "vectorial_detection_mode='unpolarized' produces an incoherent "
                    "intensity proxy and cannot be used as a coherent complex field "
                    f"for imaging_model={active_modality!r}. Use analyzer_x, "
                    "analyzer_y, full_vector, or "
                    "the scalar_paraxial backend."
                )
            intensity = (
                np.abs(vectorial["Ex"]) ** 2
                + np.abs(vectorial["Ey"]) ** 2
                + np.abs(vectorial["Ez"]) ** 2
            )
            scalar_stack = np.sqrt(np.maximum(intensity, 0.0)).astype(np.complex128)
            reduction = "sqrt_unpolarized_incoherent_vector_intensity_scalar_proxy"
            field_representation = "incoherent_intensity_proxy"
        elif detection_mode == "incoherent_sum":
            if active_modality in COHERENT_REFERENCE_MODALITIES:
                raise ValueError(
                    "vectorial_detection_mode='incoherent_sum' produces an "
                    "incoherent intensity proxy and cannot be used as a coherent "
                    f"complex field for imaging_model={active_modality!r}. Use "
                    "analyzer_x, analyzer_y, full_vector, or the scalar_paraxial "
                    "backend."
                )
            intensity = (
                np.abs(vectorial["Ex"]) ** 2
                + np.abs(vectorial["Ey"]) ** 2
                + np.abs(vectorial["Ez"]) ** 2
            )
            scalar_stack = np.sqrt(np.maximum(intensity, 0.0)).astype(np.complex128)
            reduction = "sqrt_incoherent_vector_intensity_scalar_proxy"
            field_representation = "incoherent_intensity_proxy"
        elif detection_mode == "full_vector":
            raise ValueError(
                "vectorial_detection_mode='full_vector' requires a vector-aware "
                f"coherent imaging model; got imaging_model={active_modality!r}. "
                "Use analyzer_x/analyzer_y for a coherent scalar projection or "
                "optical_field_backend='scalar_paraxial'."
            )
        else:
            raise ValueError(
                "vectorial_detection_mode must be 'analyzer_x', 'analyzer_y', "
                "'incoherent_sum', 'unpolarized', or 'full_vector'; "
                f"got {detection_mode!r}."
            )
        metadata = dict(vectorial.get("metadata", {}))
        metadata.update(
            {
                "scalar_compatibility_reduction": reduction,
                "field_representation": field_representation,
                "vectorial_detection_requested": detection_mode,
                "particle_diameter_nm": float(particle_diameter_nm),
                "particle_refractive_index": {
                    "real": float(complex(particle_refractive_index).real),
                    "imag": float(complex(particle_refractive_index).imag),
                },
            }
        )
        return ComplexPSFZInterpolator(z_values, scalar_stack, metadata=metadata)
    if backend != "scalar_paraxial":
        raise ValueError(
            "optical_field_backend must be 'scalar_paraxial' or 'vectorial_debye'; "
            f"got {backend!r}."
        )
    z_values_sorted = np.sort(z_values)
    if not np.allclose(z_values_sorted, z_values):
        # Enforce monotonic increasing order to keep interpolation logic simple.
        z_values = z_values_sorted

    # --- Setup k-space coordinates and optical parameters ---
    os_factor = params["psf_oversampling_factor"]
    pupil_samples = params["pupil_samples"]
    psf_size_nm = params["image_size_pixels"] * params["pixel_size_nm"]
    n_medium = params["refractive_index_medium"]
    if n_medium <= 0.0:
        raise ValueError("refractive_index_medium must be positive.")
    NA = float(params["numerical_aperture"])
    if NA <= 0.0:
        raise ValueError("numerical_aperture must be positive.")
    if NA > n_medium:
        raise ValueError(
            "numerical_aperture must not exceed refractive_index_medium. "
            f"Got NA={NA}, n_medium={n_medium}."
        )
    if params["wavelength_nm"] <= 0.0:
        raise ValueError("wavelength_nm must be positive.")
    wavelength_medium_nm = params["wavelength_nm"] / n_medium
    k_medium = 2 * np.pi / wavelength_medium_nm
    if particle_diameter_nm <= 0.0:
        raise ValueError("particle_diameter_nm must be positive.")

    canvas_pitch_nm = float(params["pixel_size_nm"]) / float(os_factor)
    dk = 2 * np.pi * np.fft.fftfreq(int(pupil_samples), d=canvas_pitch_nm)
    kx = np.fft.fftshift(dk)
    ky = np.fft.fftshift(dk)
    Kx, Ky = np.meshgrid(kx, ky)
    K_sq = Kx**2 + Ky**2

    # --- Define the pupil aperture and coordinates ---
    sin_theta = np.sqrt(K_sq) / k_medium
    max_sin_theta = NA / n_medium
    valid_mask = sin_theta <= 1
    aperture_mask = ((sin_theta <= max_sin_theta) & valid_mask).astype(float)

    cos_theta = np.zeros_like(sin_theta)
    cos_theta[valid_mask] = np.sqrt(1 - sin_theta[valid_mask] ** 2)

    # --- Calculate Mie scattering amplitudes across the pupil ---
    m = particle_refractive_index / n_medium
    radius_nm = particle_diameter_nm / 2
    x = 2 * np.pi * radius_nm / wavelength_medium_nm

    mu = np.zeros_like(cos_theta)
    mu[valid_mask] = cos_theta[valid_mask]

    a_n, b_n = mie_an_bn(m, x)
    mie_s2_vec = np.vectorize(
        lambda mu_value: mie_S2_from_coefficients(a_n, b_n, mu_value),
        otypes=[np.complex128],
    )
    S2_vec = mie_s2_vec(mu)

    # --- Define aberration and apodization functions ---
    rho = sin_theta / max_sin_theta
    zernike_spherical = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    spherical_phase = params["spherical_aberration_strength"] * zernike_spherical * 2 * np.pi
    coverslip_phase, coverslip_metadata = compute_coverslip_aberration_phase(
        params,
        sin_theta,
        aperture_mask > 0,
        wavelength_nm=float(params["wavelength_nm"]),
    )
    apodization = np.exp(-params["apodization_factor"] * (rho**2))

    # --- Random aberration phase (static across the entire Z-stack) ---
    # The phase is derived from a dedicated RNG seeded from the particle type
    # key (diameter + complex refractive index) plus PARAMS["random_seed"],
    # so each particle type receives reproducible aberrations independent of
    # the order in which optical types are processed.
    random_aberration_strength = float(params.get("random_aberration_strength", 0.0))
    if random_aberration_strength != 0.0:
        import hashlib

        seed_value = params.get("random_seed", 0)
        seed_int = 0 if seed_value is None else int(seed_value)
        type_key_repr = (
            f"diameter_nm={float(particle_diameter_nm)!r}|"
            f"n_real={float(particle_refractive_index.real)!r}|"
            f"n_imag={float(particle_refractive_index.imag)!r}|"
            f"pupil_samples={int(pupil_samples)}|"
            f"os_factor={int(os_factor)}|"
            f"random_seed={seed_int}"
        )
        type_seed = int(
            hashlib.sha256(type_key_repr.encode("utf-8")).hexdigest()[:16], 16
        )
        local_rng = np.random.default_rng(type_seed)
        random_phase = (
            local_rng.random((pupil_samples, pupil_samples)) - 0.5
        ) * random_aberration_strength * 2 * np.pi
    else:
        random_phase = 0.0

    # --- Precompute radius geometry for radial symmetrization ---
    yy, xx = np.indices((pupil_samples, pupil_samples))
    center = pupil_samples // 2
    r_float = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    r_index = r_float.astype(np.int64)
    max_bin = int(r_index.max())
    r_index_flat = r_index.ravel()
    r_flat = r_float.ravel()

    logger.info("Computing complex PSF stack for %s nm particle...", particle_diameter_nm)
    ipsf_stack_complex = np.zeros((len(z_values), pupil_samples, pupil_samples), dtype=np.complex128)

    # --- Compute the iPSF for each Z-slice ---
    for i, z in enumerate(tqdm(z_values, disable=not logger.isEnabledFor(logging.INFO))):
        defocus_phase = k_medium * z * cos_theta

        # Total phase in the pupil: defocus + configured aberration terms.
        aberration_phase = defocus_phase + spherical_phase + coverslip_phase + random_phase

        pupil_function = (
            -1j * wavelength_medium_nm
        ) * aperture_mask * apodization * S2_vec * np.exp(1j * aberration_phase)

        # The discrete inverse FFT maps the pupil function to the image-plane ASF.
        asf = fftshift(ifft2(ifftshift(pupil_function)))

        # --- Radially symmetrize the ASF with continuous radial interpolation ---
        asf_flat = asf.ravel()

        counts = np.bincount(r_index_flat, minlength=max_bin + 1)
        sum_real = np.bincount(r_index_flat, weights=asf_flat.real, minlength=max_bin + 1)
        sum_imag = np.bincount(r_index_flat, weights=asf_flat.imag, minlength=max_bin + 1)

        E_radial = np.zeros(max_bin + 1, dtype=np.complex128)
        nonzero = counts > 0
        E_radial[nonzero] = (sum_real[nonzero] + 1j * sum_imag[nonzero]) / counts[nonzero]

        r_bins = np.arange(max_bin + 1, dtype=float)

        E_real_interp = np.interp(
            r_flat,
            r_bins,
            E_radial.real,
            left=E_radial.real[0],
            right=E_radial.real[-1],
        )
        E_imag_interp = np.interp(
            r_flat,
            r_bins,
            E_radial.imag,
            left=E_radial.imag[0],
            right=E_radial.imag[-1],
        )

        asf_radial = (E_real_interp + 1j * E_imag_interp).reshape(pupil_samples, pupil_samples)

        ipsf_stack_complex[i, :, :] = asf_radial

    interpolator = ComplexPSFZInterpolator(
        z_values,
        ipsf_stack_complex,
        metadata={
            "backend": "scalar_paraxial",
            "scalar_compatibility_reduction": "native_scalar_paraxial",
            **coverslip_metadata,
        },
    )

    logger.info("Complex PSF stack computation complete.")
    return interpolator
