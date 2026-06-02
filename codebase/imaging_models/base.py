"""
imaging_models/base.py - Shared imaging-model base class for Syniscopy.

This module intentionally contains no concrete imaging models. Keeping the base
class inside the imaging_models package keeps the model contract colocated with
its concrete subclasses.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from substrate import SampleEnvironment

from backend_fidelity import attach_backend_fidelity_metadata


def is_vectorial_field(field: np.ndarray) -> bool:
    """Return True when ``field`` uses component-first Ex/Ey/Ez layout."""
    arr = np.asarray(field)
    return arr.ndim == 3 and arr.shape[0] == 3


def field_intensity(field: np.ndarray) -> np.ndarray:
    """Detector intensity for scalar or component-first vectorial fields."""
    arr = np.asarray(field, dtype=np.complex128)
    if is_vectorial_field(arr):
        return np.sum(np.abs(arr) ** 2, axis=0)
    return np.abs(arr) ** 2


def _coherent_polarization_vector(params: dict) -> np.ndarray:
    """Return the coherent incident polarization unit vector in Ex/Ey/Ez."""
    model = str(params.get("polarization_model", "linear_x")).strip().lower()
    if model == "scalar":
        model = "linear_x"
    theta = np.deg2rad(float(params.get("vectorial_polarization_rotation_deg", 0.0)))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    if model in {"linear_x", "x"}:
        return np.array([c, s, 0.0], dtype=np.complex128)
    if model in {"linear_y", "y"}:
        return np.array([-s, c, 0.0], dtype=np.complex128)
    if model == "unpolarized":
        raise ValueError(
            "polarization_model='unpolarized' is an incoherent average and "
            "cannot define a coherent full-vector reference field."
        )
    raise ValueError(
        "polarization_model must be 'linear_x', 'linear_y', or 'unpolarized'; "
        f"got {model!r}."
    )


def reference_vector_for_scattered(
    background_field: np.ndarray,
    scattered_field: np.ndarray,
    params: dict,
    *,
    scale: complex | float = 1.0,
) -> np.ndarray:
    """Promote a scalar coherent reference to Ex/Ey/Ez when scatter is vectorial."""
    sca = np.asarray(scattered_field, dtype=np.complex128)
    bg = np.asarray(background_field, dtype=np.complex128) * scale
    if not is_vectorial_field(sca):
        return bg
    if is_vectorial_field(bg):
        if bg.shape != sca.shape:
            raise ValueError(
                "Vectorial background reference must match scattered field shape; "
                f"got background {bg.shape!r}, scattered {sca.shape!r}."
            )
        return bg
    if bg.ndim != 2 or bg.shape != sca.shape[-2:]:
        raise ValueError(
            "Scalar background reference for vectorial scattering must have spatial "
            f"shape {sca.shape[-2:]!r}; got {bg.shape!r}."
        )
    pol = _coherent_polarization_vector(params)
    out = np.zeros_like(sca, dtype=np.complex128)
    out[0] = pol[0] * bg
    out[1] = pol[1] * bg
    out[2] = pol[2] * bg
    return out


def coherent_phase_from_reference(E_sum: np.ndarray, E_ref: np.ndarray) -> np.ndarray:
    """Compute arg(E_sum) - arg(E_ref) for scalar or coherent vector fields."""
    total = np.asarray(E_sum, dtype=np.complex128)
    ref = np.asarray(E_ref, dtype=np.complex128)
    if is_vectorial_field(total) or is_vectorial_field(ref):
        if not is_vectorial_field(total) or not is_vectorial_field(ref):
            raise ValueError(
                "Vectorial phase extraction requires both total and reference "
                f"fields to have shape (3, H, W); got {total.shape!r} and {ref.shape!r}."
            )
        if total.shape != ref.shape:
            raise ValueError(
                "Vectorial phase extraction requires matching total/reference shapes; "
                f"got {total.shape!r} and {ref.shape!r}."
            )
        ref_power = np.sum(np.abs(ref) ** 2, axis=0)
        product = np.sum(total * np.conj(ref), axis=0)
        phi = np.zeros(ref_power.shape, dtype=float)
        safe = ref_power > 1e-24
        phi[safe] = np.angle(product[safe])
        return phi
    ref_power = np.abs(ref) ** 2
    safe = ref_power > 1e-24
    phi = np.zeros(total.shape, dtype=float)
    product = total * np.conj(ref)
    phi[safe] = np.angle(product[safe])
    return phi


class ImagingModel:
    """
    Abstract base for all imaging contrast models.

    Subclasses must implement:
        compute_intensity(...)
        compute_per_particle_contrast(...)

    Subclasses may override ``output_type`` to declare what the returned
    array of ``compute_intensity`` represents.  Legal values:

        "intensity"  — dimensionless detector intensity before count scaling.
        "phase"      — phase map in radians (for QPI).
        "fringe"     — carrier-modulated intensity (for off-axis holography).

        The renderer keeps this attribute available for callers that need to
        distinguish intensity-like, phase, and fringe outputs before applying
        the modality's detector-count conversion.
    """

    output_type: str = "intensity"
    uses_sample_environment_pattern: bool = False
    uses_sample_environment: bool = True
    uses_particle_material_sources: bool = False
    requires_complex_optical_psf: bool = True
    requires_optical_scattered_field: bool = True
    requires_pre_crop_optical_filtering: bool = False
    supports_spectral_channels: bool = False

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del particle_instance, frame_index
        if self.uses_particle_material_sources:
            raise RuntimeError(
                f"{type(self).__name__} is a source-map modality and must use "
                "compute_particle_contrast_from_source_map()."
            )
        return self.compute_per_particle_contrast(E_sca_particle, background_field, params)

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray | None:
        del particle_source_map, background_field, params, frame_index
        if self.uses_particle_material_sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares uses_particle_material_sources=True "
                "but does not implement compute_particle_contrast_from_source_map()."
            )
        return None

    def compute_scene_intensity(
        self,
        E_sca_particles: list[np.ndarray],
        particle_instances: list,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_source_maps: list[np.ndarray] | None = None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del E_sca_particles, particle_instances, frame_index
        if self.uses_particle_material_sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares uses_particle_material_sources=True "
                "but does not implement source-map-aware compute_scene_intensity()."
            )
        return self.compute_intensity(E_sca_total, background_field, params)

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del shape, params
        if self.uses_particle_material_sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares uses_particle_material_sources=True "
                "but does not implement initialize_particle_source_canvas()."
            )
        return None

    def accumulate_particle_source(
        self,
        source_canvas,
        *,
        center_x_canvas: float,
        center_y_canvas: float,
        diameter_nm: float,
        pixel_size_nm: float,
        os_factor: int,
        material_properties,
        params: dict,
        particle_z_nm: float | None = None,
        source_multiplier: float = 1.0,
    ) -> None:
        del source_canvas, center_x_canvas, center_y_canvas, diameter_nm, pixel_size_nm
        del os_factor, material_properties, params, particle_z_nm, source_multiplier
        if self.uses_particle_material_sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares uses_particle_material_sources=True "
                "but does not implement accumulate_particle_source()."
            )

    def mask_contrast_image(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Default implementation: identical to per-particle contrast."""
        return self.compute_per_particle_contrast(E_sca_particle, background_field, params)

    def probe_wavelength_nm(self, params: dict) -> float:
        """Detector-domain probe wavelength used by response functions."""
        probe = params.get("probe_wavelength_nm", None)
        if probe is None:
            probe = params.get("wavelength_nm", 532.0)
        return float(probe)

    def illumination_field(self, shape: tuple[int, int], params: dict) -> np.ndarray:
        """Incident-field abstraction; subclasses override geometry-specific cases."""
        amplitude = float(params.get("reference_field_amplitude", 1.0))
        return np.full(shape, amplitude, dtype=np.complex128)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        """Return lightweight response-function metadata for this modality."""
        detector_pixel_size_nm = float(params.get("pixel_size_nm", 1.0))
        oversampling = float(params.get("psf_oversampling_factor", 1.0))
        if not np.isfinite(oversampling) or oversampling <= 0.0:
            raise ValueError(
                f"psf_oversampling_factor must be finite and positive; got {oversampling!r}."
            )
        output_type = str(getattr(self, "output_type", "intensity"))
        if output_type == "phase":
            measurement_domain = "phase"
            signal_units = "radian"
        elif output_type == "fringe":
            measurement_domain = "fringe_count"
            signal_units = "detector_count"
        else:
            measurement_domain = "count"
            signal_units = "detector_count"
        response = {
            "kind": "generic_imaging_model",
            "model_class": self.__class__.__name__,
            "output_type": output_type,
            "optical_field_backend": str(
                params.get("optical_field_backend", "vectorial_debye")
            ),
            "vectorial_detection_mode": str(
                params.get("vectorial_detection_mode", "full_vector")
            ),
            "polarization_model": str(params.get("polarization_model", "linear_x")),
            "measurement_domain": measurement_domain,
            "signal_units": signal_units,
            "probe_wavelength_nm": self.probe_wavelength_nm(params),
            "shape": tuple(shape),
            "detector_pixel_size_nm": detector_pixel_size_nm,
            "model_canvas_pixel_size_nm": detector_pixel_size_nm / oversampling,
            "psf_oversampling_factor": oversampling,
            "uses_particle_material_sources": bool(self.uses_particle_material_sources),
            "requires_complex_optical_psf": bool(self.requires_complex_optical_psf),
            "requires_optical_scattered_field": bool(self.requires_optical_scattered_field),
            "uses_sample_environment_pattern": bool(self.uses_sample_environment_pattern),
            "supports_spectral_channels": bool(self.supports_spectral_channels),
            "fidelity_label": str(
                params.get(
                    "fidelity_label",
                    params.get("profile_fidelity_label", "model_conditional_profile"),
                )
            ),
        }
        return attach_backend_fidelity_metadata(response, params=params)

    def filter_guard_radius_pixels(self, params: dict) -> int | None:
        """Optional pre-crop guard radius for Fourier/probe-domain filtering.

        The renderer uses an optical Airy-tail estimate for scalar optical
        models. Non-optical models such as TEM CTF and SEM probe proxies can
        override this so they do not inherit an optical wavelength/NA-derived
        guard band.
        """
        del params
        return None

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: "SampleEnvironment | None",
    ) -> np.ndarray:
        """Consume substrate, medium, and pattern data for this modality."""
        del E_sca_total, background_field, params, sample_environment
        return intensity

    def compute_noise(
        self,
        frame_counts: np.ndarray,
        params: dict,
        rng: np.random.Generator | None = None,
        *,
        detector_noise_runtime=None,
    ) -> np.ndarray:
        """
        Apply this modality's detector-noise model to a count-domain frame.

        The base implementation delegates to the canonical counts-domain
        camera model. Per-modality differences are supplied through
        ``params["modality_noise"][imaging_model]`` rather than through
        duplicate Poisson/readout code paths.
        """
        from camera_noise import apply_camera_noise_counts

        return apply_camera_noise_counts(
            frame_counts,
            params,
            rng=rng,
            runtime=detector_noise_runtime,
        )

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Convert the model's dimensionless ``intensity`` output into detector
        photon counts.

        Base count scaling for interferometric-scale intensity outputs:
        divide by |E_ref|^2 (the natural scale of the interferometric
        compute_intensity output) and multiply by ``background_final``, which
        is the count-domain reference image constructed from the scalar
        ``background_intensity`` and the substrate pattern. This leaves a
        uniform ~background count level with a small contrast-scale
        perturbation from |E_sca|.

        Models whose output does not live at the same scale as |E_ref|^2
        (dark-field, phase, etc.) must override this method. See the
        per-class docstrings for rationale.
        """
        E_ref_intensity_safe = np.maximum(E_ref_intensity_final, 1e-12)
        counts = background_final * (intensity / E_ref_intensity_safe)
        if np.any(~np.isfinite(counts)):
            raise ValueError(f"{type(self).__name__}.scale_intensity_to_counts produced non-finite counts.")
        if np.any(counts < 0.0):
            raise ValueError(
                f"{type(self).__name__}.scale_intensity_to_counts produced negative counts; "
                "signed-output modalities must override this method."
            )
        return counts
