from functools import lru_cache
import logging

from config.runtime import (
    DpcSettings,
    ModalitySettings,
    OpticalInstrumentSettings,
    OpticalModeSettings,
    SamplingGeometry,
)
from vectorial_detection_contracts import is_dpc_vectorial_field_path

import numpy as np
from scipy.fft import ifft2, fftshift, ifftshift
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else ()

from mie_scattering import (
    mie_S1_S2_from_coefficients,
    mie_S2_from_coefficients,
    mie_an_bn,
    mie_scattering_cross_section_nm2,
    mie_scattering_amplitudes_from_coefficients,
)
from vectorial_optics import compute_vectorial_debye_basis_psf, compute_vectorial_debye_psf
from optical_extensions import compute_coverslip_aberration_phase
from optical_scattering import (
    OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY,
    OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS,
    OPTICAL_SCATTERING_MIE,
    analytic_polarizability_dipole_vector,
    born_rayleigh_gans_form_factor,
    component_dimensions_key,
    scattering_metadata_for_key,
)
from shared_constants import COHERENT_REFERENCE_MODALITIES
from stochastic_runtime import rng_from_seed


logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _scalar_pupil_coordinates(
    pupil_samples: int,
    canvas_pitch_nm: float,
    k_medium: float,
    max_sin_theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dk = 2 * np.pi * np.fft.fftfreq(int(pupil_samples), d=float(canvas_pitch_nm))
    kx = np.fft.fftshift(dk)
    ky = np.fft.fftshift(dk)
    Kx, Ky = np.meshgrid(kx, ky)
    K_sq = Kx**2 + Ky**2
    sin_theta = np.sqrt(K_sq) / float(k_medium)
    valid_mask = sin_theta <= 1
    aperture_mask = ((sin_theta <= float(max_sin_theta)) & valid_mask).astype(float)
    cos_theta = np.zeros_like(sin_theta)
    cos_theta[valid_mask] = np.sqrt(1 - sin_theta[valid_mask] ** 2)
    for arr in (Kx, Ky, sin_theta, aperture_mask, cos_theta):
        arr.setflags(write=False)
    return Kx, Ky, sin_theta, aperture_mask, cos_theta


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

    def field_at(self, z_nm, *, orientation_matrix=None, material_properties=None):
        """Return the field for ``z_nm``; orientation-aware subclasses override."""
        del orientation_matrix, material_properties
        return self(z_nm)

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

    def field_at(self, z_nm, *, orientation_matrix=None, material_properties=None):
        """Return the vector field for ``z_nm``; orientation-aware subclasses override."""
        del orientation_matrix, material_properties
        return self(z_nm)

    def _interp_single(self, z_val):
        return _interpolate_z_stack_single(
            self.z_values,
            self.ipsf_stack,
            self.z_min,
            self.z_max,
            z_val,
        )


class BornRayleighGansPSFZInterpolator:
    """Born/RG PSF interpolator with native orientation-resolved lookup."""

    def __init__(
        self,
        reference_interpolator,
        *,
        params: dict,
        particle_diameter_nm: float,
        particle_refractive_index: complex,
        component_geometry,
    ):
        self._reference_interpolator = reference_interpolator
        self._params = dict(params)
        self._particle_diameter_nm = float(particle_diameter_nm)
        self._particle_refractive_index = complex(particle_refractive_index)
        self._component_geometry = component_geometry
        self.z_min = float(reference_interpolator.z_min)
        self.z_max = float(reference_interpolator.z_max)
        self.metadata = dict(getattr(reference_interpolator, "metadata", {}) or {})
        self.metadata["orientation_resolved_lookup"] = True
        self.metadata["orientation_resolved_lookup_policy"] = (
            "Born/Rayleigh-Gans primitive form factor is rebuilt at render-time "
            "for the stamped component orientation."
        )

    def __call__(self, z_nm):
        return self._reference_interpolator(z_nm)

    def field_at(self, z_nm, *, orientation_matrix=None, material_properties=None):
        del material_properties
        if orientation_matrix is None:
            return self._reference_interpolator(z_nm)
        z = np.asarray(z_nm, dtype=float)
        z_flat = z.reshape(-1)
        if z_flat.size == 0:
            raise ValueError("z_nm must contain at least one position.")
        oriented = compute_complex_psf_stack(
            self._params,
            self._particle_diameter_nm,
            self._particle_refractive_index,
            z_flat,
            optical_scattering_model=OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS,
            component_geometry=self._component_geometry,
            component_orientation_matrix=orientation_matrix,
            orientation_resolved_born_rg=False,
        )
        out = oriented(z_flat)
        return out.reshape(z.shape + out.shape[1:])


class AnalyticPolarizabilityDebyePSFZInterpolator:
    """Orientation-aware vectorial Debye field for analytic-polarizability primitives."""

    def __init__(
        self,
        z_values_nm,
        basis_stack,
        *,
        params: dict,
        component_geometry,
        particle_refractive_index: complex,
        output_reduction: str,
        metadata=None,
    ):
        z_values = np.asarray(z_values_nm, dtype=float)
        basis = np.asarray(basis_stack, dtype=np.complex128)
        if z_values.ndim != 1 or z_values.size == 0:
            raise ValueError("z_values_nm must be a non-empty 1D array.")
        if basis.ndim != 5 or basis.shape[0] != 3 or basis.shape[2] != 3:
            raise ValueError(
                "AnalyticPolarizabilityDebyePSFZInterpolator expects basis shape "
                "(3, len(z_values), 3, H, W)."
            )
        if basis.shape[1] != z_values.size:
            raise ValueError("basis_stack z dimension must match z_values_nm.")
        order = np.argsort(z_values)
        self.z_values = z_values[order]
        self.basis_stack = basis[:, order]
        if self.z_values.size > 1 and np.any(np.diff(self.z_values) <= 0.0):
            raise ValueError("z_values_nm must contain unique z positions.")
        self.z_min = float(self.z_values[0])
        self.z_max = float(self.z_values[-1])
        self._params = dict(params)
        self._component_geometry = component_geometry
        self._particle_refractive_index = complex(particle_refractive_index)
        self._output_reduction = str(output_reduction).strip().lower()
        if self._output_reduction not in {
            "full_vector",
            "analyzer_x",
            "analyzer_y",
            "incoherent_magnitude",
        }:
            raise ValueError(f"Unsupported analytic Debye output reduction {output_reduction!r}.")
        self.metadata = dict(metadata or {})
        self.metadata["orientation_resolved_lookup"] = True
        self.metadata["analytic_polarizability_vectorial_transport"] = "debye_operator_applied_to_dipole"
        self.metadata["analytic_polarizability_output_reduction"] = self._output_reduction

    def __call__(self, z_nm):
        return self.field_at(z_nm)

    def _basis_at_single(self, z_val: float) -> np.ndarray:
        return np.stack(
            [
                _interpolate_z_stack_single(
                    self.z_values,
                    self.basis_stack[source_axis],
                    self.z_min,
                    self.z_max,
                    z_val,
                )
                for source_axis in range(3)
            ],
            axis=0,
        )

    def _reduce_field(self, vector_field: np.ndarray) -> np.ndarray:
        if self._output_reduction == "full_vector":
            return vector_field
        if self._output_reduction == "analyzer_x":
            return vector_field[0]
        if self._output_reduction == "analyzer_y":
            return vector_field[1]
        intensity = np.sum(np.abs(vector_field) ** 2, axis=0)
        return np.sqrt(np.maximum(intensity, 0.0)).astype(np.complex128)

    def field_at(self, z_nm, *, orientation_matrix=None, material_properties=None):
        dipole = analytic_polarizability_dipole_vector(
            self._params,
            component_geometry=self._component_geometry,
            material_properties=material_properties,
            orientation_matrix=orientation_matrix,
            fallback_refractive_index=self._particle_refractive_index,
        )
        z = np.asarray(z_nm, dtype=float)
        z_flat = z.reshape(-1)
        if z_flat.size == 0:
            raise ValueError("z_nm must contain at least one position.")
        reduced_fields = []
        for z_val in z_flat:
            basis = self._basis_at_single(float(z_val))
            vector_field = np.tensordot(dipole, basis, axes=(0, 0))
            reduced_fields.append(self._reduce_field(vector_field))
        out = np.stack(reduced_fields, axis=0)
        if z.ndim == 0:
            return out[0]
        return out.reshape(z.shape + out.shape[1:])


def _scalar_peak_amplitude(scalar_stack: np.ndarray) -> float:
    intensity = np.abs(scalar_stack) ** 2
    peak = float(np.max(intensity)) if intensity.size else 0.0
    return float(np.sqrt(peak)) if peak > 0.0 and np.isfinite(peak) else 1.0


def _normalize_scalar_stack(scalar_stack: np.ndarray) -> tuple[np.ndarray, float]:
    peak_amplitude = _scalar_peak_amplitude(scalar_stack)
    if peak_amplitude > 0.0 and np.isfinite(peak_amplitude):
        return scalar_stack / peak_amplitude, peak_amplitude
    return scalar_stack, 1.0


# Absolute scattered-field normalization for the native scalar backend mirrors
# vectorial_optics._physical_scattered_amplitude_scale: the returned field is
# shape-normalized, while metadata["field_amplitude_scale"] is the single
# absolute multiplier that makes the in-focus rendered power obey
# sum_pixels(|E_sca|^2) * pixel_area_nm2 == sigma_sca_collected. The only
# physical constant used here is mie_scattering_cross_section_nm2(...) over the
# objective collection cone; if the particle optical identity or any numerical
# term is invalid, the fallback reconstructs the old arbitrary peak amplitude.
def _physical_scalar_scattered_amplitude_scale(
    params: dict,
    normalized_scalar_stack: np.ndarray,
    z_values: np.ndarray,
    particle_diameter_nm: float | None,
    particle_refractive_index: complex | None,
    wavelength_nm: float,
    fallback_peak_amplitude: float,
    optical_scattering_model: str = OPTICAL_SCATTERING_MIE,
) -> float:
    if str(optical_scattering_model).strip().lower() != OPTICAL_SCATTERING_MIE:
        return 1.0
    if particle_diameter_nm is None or particle_refractive_index is None:
        return fallback_peak_amplitude
    try:
        instrument = OpticalInstrumentSettings.from_params(params)
        n_medium = instrument.refractive_index_medium
        numerical_aperture = instrument.numerical_aperture
        m_rel = complex(particle_refractive_index) / n_medium
        if not (np.isfinite(m_rel.real) and np.isfinite(m_rel.imag)):
            return fallback_peak_amplitude
        half_angle = float(np.arcsin(min(numerical_aperture / n_medium, 1.0)))
        sigma_nm2 = mie_scattering_cross_section_nm2(
            m_rel,
            float(particle_diameter_nm),
            float(wavelength_nm),
            n_medium,
            collection_half_angle_rad=half_angle,
        )
        if not np.isfinite(sigma_nm2) or sigma_nm2 <= 0.0:
            return fallback_peak_amplitude
        canvas_pitch_nm = SamplingGeometry.from_params(params).model_canvas_pixel_size_nm
        pixel_area_nm2 = canvas_pitch_nm * canvas_pitch_nm
        if not np.isfinite(pixel_area_nm2) or pixel_area_nm2 <= 0.0:
            return fallback_peak_amplitude
        infocus = int(np.argmin(np.abs(z_values)))
        shape_power = float(np.sum(np.abs(normalized_scalar_stack[infocus]) ** 2))
        if not np.isfinite(shape_power) or shape_power <= 0.0:
            return fallback_peak_amplitude
        target_power = sigma_nm2 / pixel_area_nm2
        scale = float(np.sqrt(target_power / shape_power))
        return scale if np.isfinite(scale) and scale > 0.0 else fallback_peak_amplitude
    except Exception:
        return fallback_peak_amplitude


def compute_complex_psf_stack(
    params,
    particle_diameter_nm,
    particle_refractive_index,
    z_values_nm,
    *,
    optical_scattering_model: str = OPTICAL_SCATTERING_MIE,
    component_geometry=None,
    component_orientation_matrix=None,
    orientation_resolved_born_rg: bool = True,
):
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
             - The declared scattering model: exact Mie S2(mu) for spheres,
               Born/Rayleigh-Gans primitive form factor for weak scatterers,
               or a normalized point-scatterer pupil shape for analytic
               polarizability primitives whose amplitude is applied at stamping.
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
    optical_mode = OpticalModeSettings.from_params(params)
    backend = optical_mode.optical_field_backend
    scattering_model = str(optical_scattering_model).strip().lower()
    if scattering_model not in {
        OPTICAL_SCATTERING_MIE,
        OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY,
        OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS,
    }:
        raise ValueError(f"Unsupported optical_scattering_model={optical_scattering_model!r}.")
    geometry_shape = (
        str(getattr(component_geometry, "shape", "sphere")).strip().lower()
        if component_geometry is not None
        else "sphere"
    )
    geometry_dimensions_key = (
        component_dimensions_key(component_geometry)
        if component_geometry is not None
        else ("diameter_nm", float(particle_diameter_nm))
    )
    scattering_metadata = scattering_metadata_for_key(
        (
            "optical_scattering",
            scattering_model,
            geometry_shape,
            geometry_dimensions_key,
            float(particle_diameter_nm),
            float(complex(particle_refractive_index).real),
            float(complex(particle_refractive_index).imag),
        )
    )
    if backend == "vectorial_debye":
        dpc_channel_model = DpcSettings.from_params(params).channel_model
        active_modality = ModalitySettings.from_params(params).modality
        use_vectorial_dpc = is_dpc_vectorial_field_path(active_modality, dpc_channel_model)
        detection_mode = optical_mode.vectorial_detection_mode
        use_full_vectorial_coherent = (
            detection_mode == "full_vector"
            and active_modality in COHERENT_REFERENCE_MODALITIES
        )
        if use_vectorial_dpc or use_full_vectorial_coherent:
            polarization_model = optical_mode.polarization_model
            if polarization_model == "scalar":
                polarization_model = "linear_x"
            if polarization_model == "unpolarized" and use_full_vectorial_coherent:
                raise ValueError(
                    "polarization_model='unpolarized' is an incoherent average and "
                    "cannot be used with vectorial_detection_mode='full_vector' for "
                    f"coherent imaging_model={active_modality!r}. Use linear_x, "
                    "linear_y, an analyzer mode, or optical_field_backend='scalar_paraxial'."
                )
            if polarization_model == "unpolarized" and use_vectorial_dpc:
                raise ValueError(
                    "polarization_model='unpolarized' is an incoherent average and "
                    "cannot be used with full-vector DPC fields. Use linear_x, "
                    "linear_y, an analyzer mode, or optical_field_backend='scalar_paraxial'."
                )
        if (
            scattering_model == OPTICAL_SCATTERING_ANALYTIC_POLARIZABILITY
            and component_geometry is not None
            and (
                use_vectorial_dpc
                or use_full_vectorial_coherent
                or detection_mode in {"analyzer_x", "analyzer_y", "incoherent_sum", "unpolarized"}
            )
        ):
            basis = compute_vectorial_debye_basis_psf(
                params,
                z_values,
                particle_diameter_nm=particle_diameter_nm,
                particle_refractive_index=particle_refractive_index,
                optical_scattering_model=scattering_model,
                component_geometry=component_geometry,
                orientation_matrix=component_orientation_matrix,
            )
            if use_vectorial_dpc or use_full_vectorial_coherent:
                if detection_mode == "full_vector":
                    output_reduction = "full_vector"
                    reduction = "full_vector_field"
                    field_representation = "vectorial_coherent_field"
                elif detection_mode == "analyzer_x":
                    output_reduction = "analyzer_x"
                    reduction = "analyzer_x_component"
                    field_representation = "scalar_coherent_vector_component"
                elif detection_mode == "analyzer_y":
                    output_reduction = "analyzer_y"
                    reduction = "analyzer_y_component"
                    field_representation = "scalar_coherent_vector_component"
                elif detection_mode in {"incoherent_sum", "unpolarized"}:
                    output_reduction = "incoherent_magnitude"
                    reduction = "sqrt_incoherent_vector_intensity_scalar_proxy"
                    field_representation = "incoherent_intensity_proxy"
                else:
                    raise ValueError(f"Unsupported vectorial_detection_mode={detection_mode!r}.")
                vectorial_reason = (
                    "dpc_vectorial_channel_renderer_owned_reduction"
                    if use_vectorial_dpc
                    else "coherent_full_vector_detection"
                )
            elif detection_mode == "analyzer_x":
                output_reduction = "analyzer_x"
                reduction = "analyzer_x_component"
                field_representation = "scalar_coherent_vector_component"
                vectorial_reason = "coherent_analyzer_projection_after_debye_transport"
            elif detection_mode == "analyzer_y":
                output_reduction = "analyzer_y"
                reduction = "analyzer_y_component"
                field_representation = "scalar_coherent_vector_component"
                vectorial_reason = "coherent_analyzer_projection_after_debye_transport"
            elif detection_mode in {"incoherent_sum", "unpolarized"}:
                output_reduction = "incoherent_magnitude"
                reduction = (
                    "sqrt_unpolarized_incoherent_vector_intensity_scalar_proxy"
                    if detection_mode == "unpolarized"
                    else "sqrt_incoherent_vector_intensity_scalar_proxy"
                )
                field_representation = "incoherent_intensity_proxy"
                vectorial_reason = "incoherent_magnitude_after_debye_transport"
            else:
                raise ValueError(
                    "vectorial_detection_mode='full_vector' requires a vector-aware "
                    f"coherent imaging model; got imaging_model={active_modality!r}. "
                    "Use analyzer_x/analyzer_y for a coherent scalar projection or "
                    "optical_field_backend='scalar_paraxial'."
                )
            metadata = dict(basis.get("metadata", {}))
            metadata.update(
                {
                    "scalar_compatibility_reduction": reduction,
                    "field_representation": field_representation,
                    "vectorial_detection_requested": detection_mode,
                    "vectorial_field_reason": vectorial_reason,
                    "particle_diameter_nm": float(particle_diameter_nm),
                    "particle_refractive_index": {
                        "real": float(complex(particle_refractive_index).real),
                        "imag": float(complex(particle_refractive_index).imag),
                    },
                    "analytic_polarizability_vectorial_transport": "debye_operator_applied_to_dipole",
                    "analytic_polarizability_projection_order": (
                        "dipole_first_then_vectorial_debye_transport_then_detection_reduction"
                    ),
                    **scattering_metadata,
                }
            )
            return AnalyticPolarizabilityDebyePSFZInterpolator(
                z_values,
                basis["basis"],
                params=params,
                component_geometry=component_geometry,
                particle_refractive_index=particle_refractive_index,
                output_reduction=output_reduction,
                metadata=metadata,
            )

        vectorial = compute_vectorial_debye_psf(
            params,
            z_values,
            particle_diameter_nm=particle_diameter_nm,
            particle_refractive_index=particle_refractive_index,
            optical_scattering_model=scattering_model,
            component_geometry=component_geometry,
            orientation_matrix=component_orientation_matrix,
        )
        if use_vectorial_dpc or use_full_vectorial_coherent:
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
                        "dpc_vectorial_channel_renderer_owned_reduction"
                        if use_vectorial_dpc
                        else "coherent_full_vector_detection"
                    ),
                    "particle_diameter_nm": float(particle_diameter_nm),
                    "particle_refractive_index": {
                        "real": float(complex(particle_refractive_index).real),
                        "imag": float(complex(particle_refractive_index).imag),
                    },
                    **scattering_metadata,
                }
            )
            vector_interpolator = VectorialPSFZInterpolator(
                z_values,
                vector_stack,
                metadata=metadata,
            )
            if (
                orientation_resolved_born_rg
                and scattering_model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS
            ):
                return BornRayleighGansPSFZInterpolator(
                    vector_interpolator,
                    params=params,
                    particle_diameter_nm=float(particle_diameter_nm),
                    particle_refractive_index=particle_refractive_index,
                    component_geometry=component_geometry,
                )
            return vector_interpolator

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
                **scattering_metadata,
            }
        )
        scalar_interpolator = ComplexPSFZInterpolator(
            z_values,
            scalar_stack,
            metadata=metadata,
        )
        if (
            orientation_resolved_born_rg
            and scattering_model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS
        ):
            return BornRayleighGansPSFZInterpolator(
                scalar_interpolator,
                params=params,
                particle_diameter_nm=float(particle_diameter_nm),
                particle_refractive_index=particle_refractive_index,
                component_geometry=component_geometry,
            )
        return scalar_interpolator
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
    sampling = SamplingGeometry.from_params(params)
    instrument = OpticalInstrumentSettings.from_params(params)
    os_factor = sampling.psf_oversampling_factor
    pupil_samples = instrument.pupil_samples
    n_medium = instrument.refractive_index_medium
    if n_medium <= 0.0:
        raise ValueError("refractive_index_medium must be positive.")
    NA = instrument.numerical_aperture
    if NA <= 0.0:
        raise ValueError("numerical_aperture must be positive.")
    if NA > n_medium:
        raise ValueError(
            "numerical_aperture must not exceed refractive_index_medium. "
            f"Got NA={NA}, n_medium={n_medium}."
        )
    wavelength_nm = instrument.probe_wavelength_nm
    wavelength_medium_nm = wavelength_nm / n_medium
    k_medium = 2 * np.pi / wavelength_medium_nm
    if particle_diameter_nm <= 0.0:
        raise ValueError("particle_diameter_nm must be positive.")

    max_sin_theta = NA / n_medium
    Kx, Ky, sin_theta, aperture_mask, cos_theta = _scalar_pupil_coordinates(
        int(pupil_samples),
        float(sampling.model_canvas_pixel_size_nm),
        float(k_medium),
        float(max_sin_theta),
    )
    valid_mask = sin_theta <= 1

    # --- Calculate scattering transfer across the pupil ---
    if scattering_model == OPTICAL_SCATTERING_MIE:
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
    elif scattering_model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS:
        if component_geometry is None:
            raise ValueError("Born/Rayleigh-Gans scattering requires component_geometry.")
        qz = k_medium * cos_theta - k_medium
        S2_vec = born_rayleigh_gans_form_factor(
            component_geometry=component_geometry,
            qx_nm_inv=Kx,
            qy_nm_inv=Ky,
            qz_nm_inv=qz,
            orientation_matrix=component_orientation_matrix,
        )
    else:
        S2_vec = np.ones_like(cos_theta, dtype=np.complex128)

    # --- Define aberration and apodization functions ---
    # Resolve all three aberration effects through the shared single-source-of-
    # truth resolver so the scalar path cannot drift from the vectorial path
    # (this is the seam where the old `na` NameError hid).
    from config.runtime import AberrationSettings

    aberration = AberrationSettings.from_params(params)
    rho = sin_theta / max_sin_theta
    zernike_spherical = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    spherical_phase = aberration.spherical_aberration_strength * zernike_spherical * 2 * np.pi
    coverslip_phase, coverslip_metadata = compute_coverslip_aberration_phase(
        params,
        sin_theta,
        aperture_mask > 0,
        wavelength_nm=wavelength_nm,
    )
    apodization = np.exp(-aberration.apodization_factor * (rho**2))

    # --- Random aberration phase (static across the entire Z-stack) ---
    # Contract: optical aberration realization is a system-level seed, independent
    # of particle-specific optical properties (diameter/index) and detector/scene
    # randomness. It must only depend on optical-system settings and
    # `AberrationSettings`.
    random_aberration_strength = aberration.random_aberration_strength
    if random_aberration_strength != 0.0:
        import hashlib

        seed_value = aberration.optical_aberration_seed
        seed_repr = "None" if seed_value is None else repr(int(seed_value))
        type_key_repr = (
            f"pupil_samples={int(pupil_samples)}|"
            f"os_factor={int(os_factor)}|"
            f"wavelength_nm={float(wavelength_nm)!r}|"
            f"numerical_aperture={float(NA)!r}|"
            f"refractive_index_medium={float(n_medium)!r}|"
            f"optical_aberration_seed={seed_repr}"
        )
        type_seed = int(
            hashlib.sha256(type_key_repr.encode("utf-8")).hexdigest()[:16], 16
        )
        local_rng = rng_from_seed(type_seed, stream="scalar_optics_random_aberration")
        random_phase = (
            local_rng.random((pupil_samples, pupil_samples)) - 0.5
        ) * random_aberration_strength * 2 * np.pi
    else:
        random_phase = 0.0

    apply_radial_symmetrization = not (
        scattering_model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS
        or float(random_aberration_strength) != 0.0
    )

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

        if apply_radial_symmetrization:
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

            asf = (E_real_interp + 1j * E_imag_interp).reshape(pupil_samples, pupil_samples)

        ipsf_stack_complex[i, :, :] = asf

    scalar_stack, fallback_peak_amplitude = _normalize_scalar_stack(ipsf_stack_complex)
    field_amplitude_scale = _physical_scalar_scattered_amplitude_scale(
        params,
        scalar_stack,
        z_values,
        particle_diameter_nm,
        particle_refractive_index,
        wavelength_nm,
        fallback_peak_amplitude,
        optical_scattering_model=scattering_model,
    )

    interpolator = ComplexPSFZInterpolator(
        z_values,
        scalar_stack,
        metadata={
            "backend": "scalar_paraxial",
            "scalar_compatibility_reduction": "native_scalar_paraxial",
            **coverslip_metadata,
            "normalization": "shape_peak_scalar_intensity_equals_one",
            "field_amplitude_scale": float(field_amplitude_scale),
            "field_amplitude_scale_semantics": (
                "Mie: multiply normalized scalar fields by this factor so rendered "
                "scattered intensity integrates to the physical collected Mie cross-section. "
                "Analytic polarizability: this is 1.0 and the orientation-dependent "
                "polarizability amplitude is applied during particle stamping. "
                "Born/Rayleigh-Gans: this is 1.0; the primitive form factor is "
                "applied in the pupil and the weak-scattering volume/contrast "
                "amplitude is applied during particle stamping."
            ),
            "scalar_radial_symmetrization_applied": bool(apply_radial_symmetrization),
            "scalar_radial_symmetrization_policy": (
                "applied_only_for_radially_symmetric_scalar_pupil_response"
                if apply_radial_symmetrization
                else "disabled_to_preserve_angular_pupil_structure"
            ),
            **scattering_metadata,
        },
    )

    logger.info("Complex PSF stack computation complete.")
    if (
        orientation_resolved_born_rg
        and scattering_model == OPTICAL_SCATTERING_BORN_RAYLEIGH_GANS
    ):
        return BornRayleighGansPSFZInterpolator(
            interpolator,
            params=params,
            particle_diameter_nm=float(particle_diameter_nm),
            particle_refractive_index=particle_refractive_index,
            component_geometry=component_geometry,
        )
    return interpolator
