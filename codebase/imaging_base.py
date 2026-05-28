"""
imaging_base.py - Shared imaging-model base class for Syniscopy.

This module intentionally contains no concrete imaging models. Keeping the base
class here avoids circular imports between imaging_model.py and auxiliary model
modules such as kohler_imaging.py.
"""

from __future__ import annotations

import numpy as np

from substrate import SampleEnvironment


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
    requires_pre_crop_optical_filtering: bool = False

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
        del E_sca_particles, particle_instances, particle_source_maps, frame_index
        return self.compute_intensity(E_sca_total, background_field, params)

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del shape, params
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
    ) -> None:
        del source_canvas, center_x_canvas, center_y_canvas, diameter_nm, pixel_size_nm
        del os_factor, material_properties, params, particle_z_nm

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
        return {
            "kind": "scalar_paraxial",
            "probe_wavelength_nm": self.probe_wavelength_nm(params),
            "shape": tuple(shape),
        }

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        """Consume substrate, medium, and pattern data for this modality."""
        del E_sca_total, background_field, params, sample_environment
        return intensity

    def compute_noise(
        self,
        frame_counts: np.ndarray,
        params: dict,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Apply this modality's detector-noise model to a count-domain frame.

        The base implementation delegates to the canonical counts-domain
        camera model. Per-modality differences are supplied through
        ``params["modality_noise"][imaging_model]`` rather than through
        duplicate Poisson/readout code paths.
        """
        from camera_noise import apply_camera_noise_counts

        return apply_camera_noise_counts(frame_counts, params, rng=rng)

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
        return background_final * (intensity / E_ref_intensity_safe)
