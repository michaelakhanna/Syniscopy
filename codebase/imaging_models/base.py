"""
imaging_models/base.py - Shared imaging-model base class for Syniscopy.

This module intentionally contains no concrete imaging models. Keeping the base
class inside the imaging_models package keeps the model contract colocated with
its concrete subclasses.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from config.runtime import (
    BackendProfileSettings,
    ModalitySettings,
    OpticalInstrumentSettings,
    OpticalModeSettings,
    OpticalScatteringSettings,
    SamplingGeometry,
)

if TYPE_CHECKING:
    from substrate import SampleEnvironment

from backend_fidelity import attach_backend_fidelity_metadata
from detector_frame_conversion import (
    DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS,
    MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY,
    REFERENCE_BASIS_RELATIVE_REFERENCE_INTENSITY,
    VALUE_FORM_ABSOLUTE,
    DetectorFrameConversion,
    convert_model_output_to_detector_frame,
)
from direct_signal_contracts import (
    DirectParticleSignalProduct,
    analysis_contrast_representation,
)
from source_volume_support import (
    SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH,
    SOURCE_Z_BASIS_FOCUS_RELATIVE,
    SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD,
    SOURCE_Z_BASIS_PROJECTED_NO_Z,
    normalize_source_z_basis,
    resolve_entry_surface_depth_nm,
)


def is_vectorial_field(field: np.ndarray) -> bool:
    """Return True when ``field`` uses component-first Ex/Ey/Ez layout."""
    arr = np.asarray(field)
    return arr.ndim == 3 and arr.shape[0] == 3


def field_intensity(field: np.ndarray) -> np.ndarray:
    """Detector intensity for scalar or component-first vectorial fields."""
    arr = np.asarray(field, dtype=np.complex128)
    if is_vectorial_field(arr):
        return np.sum(np.abs(arr) ** 2, axis=0)
    if arr.ndim == 4 and arr.shape[1] == 3:
        return np.sum(np.abs(arr) ** 2, axis=1)
    return np.abs(arr) ** 2


def _coherent_polarization_vector(params: dict) -> np.ndarray:
    """Return the coherent incident polarization unit vector in Ex/Ey/Ez."""
    optical = OpticalModeSettings.from_params(params)
    model = optical.polarization_model
    if model == "scalar":
        model = "linear_x"
    theta = np.deg2rad(optical.vectorial_polarization_rotation_deg)
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


@dataclass(frozen=True)
class SourceCoordinateContext:
    """Coordinate contract for material-source placement and optical response."""

    particle_world_z_nm: float | None
    focus_plane_z_nm: float
    particle_focus_relative_z_nm: float | None
    source_density_z_basis: str = SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD
    optical_response_z_basis: str = SOURCE_Z_BASIS_FOCUS_RELATIVE
    entry_surface_depth_nm: float | None = None
    projected_no_z: bool = False

    @classmethod
    def from_particle_z(
        cls,
        *,
        particle_world_z_nm: float | None,
        focus_plane_z_nm: float,
        source_density_z_basis: str = SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD,
        optical_response_z_basis: str = SOURCE_Z_BASIS_FOCUS_RELATIVE,
        entry_surface_depth_nm: float | None = None,
    ) -> "SourceCoordinateContext":
        world_z = None if particle_world_z_nm is None else float(particle_world_z_nm)
        focus_z = float(focus_plane_z_nm)
        focus_relative = None if world_z is None else world_z - focus_z
        source_basis = normalize_source_z_basis(source_density_z_basis, context="source_density_z_basis")
        optical_basis = normalize_source_z_basis(optical_response_z_basis, context="optical_response_z_basis")
        # Entry-surface depth is a material/source coordinate.  Derive it once
        # in the shared context so SEM/TEM/fluorescence-specific code cannot
        # accidentally reinterpret focus-relative z or a source-grid offset as
        # physical material placement.
        resolved_entry_surface_depth_nm = (
            resolve_entry_surface_depth_nm(
                particle_world_z_nm=world_z,
                entry_surface_depth_nm=entry_surface_depth_nm,
            )
            if source_basis == SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH
            else entry_surface_depth_nm
        )
        return cls(
            particle_world_z_nm=world_z,
            focus_plane_z_nm=focus_z,
            particle_focus_relative_z_nm=focus_relative,
            source_density_z_basis=source_basis,
            optical_response_z_basis=optical_basis,
            entry_surface_depth_nm=resolved_entry_surface_depth_nm,
            projected_no_z=source_basis == SOURCE_Z_BASIS_PROJECTED_NO_Z,
        )

    @property
    def source_density_z_nm(self) -> float | None:
        basis = normalize_source_z_basis(self.source_density_z_basis, context="source_density_z_basis")
        if self.projected_no_z or basis == SOURCE_Z_BASIS_PROJECTED_NO_Z:
            return None
        if basis == SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD:
            return self.particle_world_z_nm
        if basis == SOURCE_Z_BASIS_FOCUS_RELATIVE:
            return self.particle_focus_relative_z_nm
        if basis == SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH:
            if self.entry_surface_depth_nm is None:
                raise ValueError(
                    "SourceCoordinateContext source_density_z_basis='entry_surface_depth' "
                    "requires a resolved entry_surface_depth_nm. This is a physical "
                    "material-depth coordinate, not focus-relative defocus or a display value."
                )
            return self.entry_surface_depth_nm
        raise ValueError(f"unknown source_density_z_basis={self.source_density_z_basis!r}.")

    def optical_response_z_nm_for_physical(self, z_nm: float) -> float | None:
        basis = normalize_source_z_basis(self.optical_response_z_basis, context="optical_response_z_basis")
        if basis == SOURCE_Z_BASIS_PROJECTED_NO_Z:
            return None
        z = float(z_nm)
        if basis == SOURCE_Z_BASIS_FOCUS_RELATIVE:
            return z - self.focus_plane_z_nm
        if basis == SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD:
            return z
        raise ValueError(f"unknown optical_response_z_basis={self.optical_response_z_basis!r}.")


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
    sample_environment_reference_field_only: bool = False
    allow_intensity_sample_environment_fallback: bool = False
    uses_particle_material_sources: bool = False
    requires_complex_optical_psf: bool = True
    requires_optical_scattered_field: bool = True
    requires_pre_crop_optical_filtering: bool = False
    supports_spectral_channels: bool = False
    # Lateral Fisher stationary shifts are a derivative-basis contract, not an
    # optimization detail.  Subclasses with detector/world-fixed carriers or
    # other non-translating scene terms must override these flags so high-level
    # Fisher consumers perturb particle state instead of differentiating the
    # detector grid.
    stationary_lateral_fisher_safe_for_single_uniform_scene: bool = True
    has_detector_fixed_lateral_carrier: bool = False
    requires_rerendered_lateral_fisher: bool = False

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
            raise RuntimeError(
                f"{type(self).__name__} is a source-map modality. Bare direct "
                "contrast arrays are not a stable analysis contract because source "
                "density, secondary-yield, projected-phase, and detector-count "
                "responses use different detector transfers. Use "
                "compute_particle_signal_product_from_source_map() and explicitly "
                "request its Fisher-safe detector/analysis array."
            )
        return None

    def compute_particle_signal_product(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        """Return a typed direct-particle response for Fisher-facing callers.

        This shared method is intentionally separate from the untyped array API.
        The fix-site invariant is that a caller can no longer confuse a
        source-domain diagnostic with a detector/analysis-domain Fisher signal.
        Non-source-map optical modalities keep their historical direct contrast
        semantics here; source-map modalities must override this method with an
        explicit detector-transfer policy.
        """
        del particle_instance, frame_index
        if self.uses_particle_material_sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares uses_particle_material_sources=True "
                "and must implement compute_particle_signal_product()."
            )
        values = self.compute_per_particle_contrast(E_sca_particle, background_field, params)
        representation = analysis_contrast_representation()
        try:
            modality = ModalitySettings.from_params(params).modality
        except KeyError:
            modality = type(self).__name__
        return DirectParticleSignalProduct(
            values=values,
            representation=representation,
            modality=modality,
            producer=f"{type(self).__name__}.compute_particle_signal_product",
            safe_for_fisher=True,
            detector_scale_applied=False,
            background_included=False,
            source_representation=representation,
            conversion_note=(
                "Non-source-map direct optical response preserved as analysis "
                "contrast. Source-map modalities override this method because "
                "their detector transfer is modality-specific."
            ),
        )

    def compute_particle_signal_product_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        del particle_source_map, background_field, params, frame_index
        if self.uses_particle_material_sources:
            raise NotImplementedError(
                f"{type(self).__name__} declares uses_particle_material_sources=True "
                "and must implement compute_particle_signal_product_from_source_map()."
            )
        raise RuntimeError(
            f"{type(self).__name__} does not use source maps; call "
            "compute_particle_signal_product() instead."
        )

    def scattered_field_render_multiplier(
        self,
        params: dict,
        *,
        world_position_nm: np.ndarray,
        diameter_nm: float,
        material_properties=None,
        frame_index: int = 0,
        component_geometry=None,
        orientation_matrix: np.ndarray | None = None,
    ) -> complex:
        del params, world_position_nm, diameter_nm, material_properties, frame_index
        del component_geometry, orientation_matrix
        return 1.0 + 0.0j

    def particle_source_z_basis(self, params: dict) -> str:
        """Coordinate basis expected by ``accumulate_particle_source``.

        Source-map modalities must declare whether ``particle_z_nm`` is a
        physical sample-world coordinate, a focus-relative coordinate, or
        unused. The renderer uses this contract when focus planes shift
        independently of particle trajectories.
        """
        del params
        return SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD

    def source_coordinate_contract(self, params: dict) -> dict:
        source_basis = self.particle_source_z_basis(params)
        return {
            "source_density_z_basis": source_basis,
            "optical_response_z_basis": "focus_relative",
        }

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
        source_coordinate_context: SourceCoordinateContext | None = None,
        source_multiplier: float = 1.0,
        component_geometry=None,
        orientation_matrix: np.ndarray | None = None,
    ) -> None:
        del source_canvas, center_x_canvas, center_y_canvas, diameter_nm, pixel_size_nm
        del os_factor, material_properties, params, particle_z_nm, source_coordinate_context, source_multiplier
        del component_geometry, orientation_matrix
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
        return OpticalInstrumentSettings.from_params(params).probe_wavelength_nm

    def illumination_field(self, shape: tuple[int, int], params: dict) -> np.ndarray:
        """Incident-field abstraction; subclasses override geometry-specific cases."""
        amplitude = OpticalModeSettings.from_params(params).reference_field_amplitude
        return np.full(shape, amplitude, dtype=np.complex128)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        """Return lightweight response-function metadata for this modality."""
        sampling = SamplingGeometry.from_params(params)
        optical = OpticalModeSettings.from_params(params)
        detector_pixel_size_nm = sampling.detector_pixel_size_nm
        oversampling = float(sampling.psf_oversampling_factor)
        if not np.isfinite(oversampling) or oversampling <= 0.0:
            raise ValueError(
                f"psf_oversampling_factor must be finite and positive; got {oversampling!r}."
            )
        output_type = str(getattr(self, "output_type", "intensity"))
        if output_type == "phase":
            measurement_domain = "phase"
            signal_units = "radian"
        elif output_type == "fringe":
            measurement_domain = "count"
            signal_units = "detector_count"
        else:
            measurement_domain = "count"
            signal_units = "detector_count"
        response = {
            "kind": "generic_imaging_model",
            "model_class": self.__class__.__name__,
            "output_type": output_type,
            "observable_subtype": (
                "raw_fringe_interferogram"
                if output_type == "fringe"
                else output_type
            ),
            "optical_field_backend": optical.optical_field_backend,
            "optical_scattering_model": OpticalScatteringSettings.from_params(params).model,
            "vectorial_detection_mode": optical.vectorial_detection_mode,
            "polarization_model": optical.polarization_model,
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
            "stationary_lateral_fisher_safe_for_single_uniform_scene": bool(
                self.stationary_lateral_fisher_safe_for_single_uniform_scene
            ),
            "has_detector_fixed_lateral_carrier": bool(self.has_detector_fixed_lateral_carrier),
            "requires_rerendered_lateral_fisher": bool(self.requires_rerendered_lateral_fisher),
            "fidelity_label": BackendProfileSettings.from_params(params).profile_fidelity_label,
        }
        return attach_backend_fidelity_metadata(response, params=params)

    def filter_guard_radius_pixels(self, params: dict) -> int | None:
        """Optional pre-crop guard radius for Fourier/probe-domain filtering.

        The renderer uses an optical Airy-tail estimate for scalar optical
        models. Non-optical models such as TEM CTF and SEM probe proxies can
        override this so they do not inherit an optical wavelength/NA-derived
        guard band.
        """
        return BackendProfileSettings.from_params(params).filter_guard_radius_pixels()

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
        ``configured parameters["modality_noise"][imaging_model]`` rather than through
        duplicate Poisson/readout code paths.
        """
        from camera_noise import apply_camera_noise_counts

        return apply_camera_noise_counts(
            frame_counts,
            params,
            rng=rng,
            runtime=detector_noise_runtime,
        )

    def convert_model_output_to_detector_frame(
        self,
        model_output: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Convert this model's native output into the renderer detector frame.

        The conversion procedure is centralized in
        ``detector_frame_conversion``; subclasses provide only the conversion
        parameters when their output basis differs from relative-reference
        intensity.
        """
        return convert_model_output_to_detector_frame(
            model_output=model_output,
            background_frame=background_final,
            reference_intensity_frame=E_ref_intensity_final,
            conversion=DetectorFrameConversion(
                model_output_domain=MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY,
                detector_output_domain=DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS,
                value_form=VALUE_FORM_ABSOLUTE,
                reference_basis=REFERENCE_BASIS_RELATIVE_REFERENCE_INTENSITY,
            ),
            params=params,
            context=f"{type(self).__name__}.convert_model_output_to_detector_frame",
        )
