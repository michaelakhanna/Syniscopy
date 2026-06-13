"""fluorescence widefield imaging model."""

from __future__ import annotations
from configured_parameters import configured_assign

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    SourceCoordinateContext,
    _mean_normalized_map,
    np,
)
from backend_fidelity import attach_backend_fidelity_metadata
from direct_signal_contracts import (
    DirectParticleSignalProduct,
    detector_count_delta_representation,
    direct_signal_identity_from_model,
    fluorescence_emission_source_representation,
)
from config.runtime import (
    FluorescenceSettings,
    FocusPlaneState,
    MotionDynamicsSettings,
    OpticalInstrumentSettings,
    SampleEnvironmentSettings,
    SamplingGeometry,
)
from detector_frame_conversion import (
    DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS,
    MODEL_OUTPUT_DOMAIN_EMISSION_DENSITY,
    REFERENCE_BASIS_NONE,
    VALUE_FORM_ABSOLUTE,
    DetectorFrameConversion,
    convert_model_output_to_detector_frame,
)
from .fluorescence_backends import VectorialPhotophysicsFluorescenceBackend
from source_convolution_contracts import (
    SOURCE_CONVOLUTION_CONTRACT_ID,
    SourceConvolutionBoundaryMode,
    SourceConvolutionContext,
)
from .fluorescence_source_layers import (
    FLUORESCENCE_SOURCE_BASIS_EMITTER_DENSITY,
    FLUORESCENCE_SOURCE_ROLE_SAMPLE_ENVIRONMENT_AUTOFLUORESCENCE,
    FLUORESCENCE_SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD,
    FluorescenceSourceLayerPlacement,
)
from simulation_runtime_state import get_source_volume_support
from .source_rasterization import (
    normalize_sliced_source_to_projected_chord,
    primitive_footprint_patch,
)


def _convolve1d_reflect_same(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    row = np.asarray(values, dtype=float)
    ker = np.asarray(kernel, dtype=float)
    if row.size <= 1 or ker.size <= 1:
        return row * float(np.sum(ker))
    radius = ker.size // 2
    padded = np.pad(row, radius, mode="reflect")
    return np.convolve(padded, ker, mode="valid")


_FLUORESCENCE_DIRECT_SIGNAL_MODEL_CLASSES = {
    "fluorescence_widefield": ("FluorescenceWidefieldImagingModel",),
    "tirf_fluorescence": ("TIRFFluorescenceImagingModel",),
}


class FluorescenceWidefieldImagingModel(ImagingModel):
    """
    Widefield epi-fluorescence imaging model.

    Fluorescence is rendered from material-property source maps, not from
    coherent scattering intensity. During rendering each particle/sub-particle
    contributes a projected emitter-density profile weighted by its chord
    length through the sphere, MaterialProperties.fluorophore_density, and
    excitation/emission spectral overlap. The scene source map is then blurred
    by the emission PSF and scaled to detector counts by the physical
    fluorescence photon budget.
    """

    output_type = "intensity"
    counts_are_exposure_integrated = True
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_complex_optical_psf = False
    requires_optical_scattered_field = False
    requires_pre_crop_optical_filtering = True
    supports_spectral_channels = True

    def __init__(
        self,
        params: dict,
        *,
        fluorescence_settings: FluorescenceSettings | None = None,
        vectorial_numerical_aperture: float | None = None,
    ) -> None:
        settings = fluorescence_settings or FluorescenceSettings.from_params(params)
        self._fluorescence_settings = settings
        self._fluorescence_backend = settings.backend
        self._excitation_wavelength_nm = settings.excitation_wavelength_nm
        self._emission_wavelength_nm = settings.emission_wavelength_nm
        self._Qf = settings.quantum_yield
        self._excitation = settings.excitation_scale
        self._absorbed_excitation_photons_per_fluorophore = (
            settings.absorbed_excitation_photons_per_fluorophore_per_frame
        )
        self._collection_efficiency = settings.collection_efficiency
        self._detector_qe = settings.detector_qe
        sampling = SamplingGeometry.from_params(params)
        canvas_pitch_nm = sampling.model_canvas_pixel_size_nm
        if not np.isfinite(canvas_pitch_nm) or canvas_pitch_nm <= 0.0:
            raise ValueError(
                "parameters['pixel_size_nm'] / parameters['psf_oversampling_factor'] "
                f"must resolve to a positive fluorescence canvas pitch; got {canvas_pitch_nm} nm."
            )
        self._canvas_pitch_nm = float(canvas_pitch_nm)
        self._detector_pixel_area_nm2 = float(sampling.detector_pixel_size_nm) ** 2
        if (
            not np.isfinite(self._detector_pixel_area_nm2)
            or self._detector_pixel_area_nm2 <= 0.0
        ):
            raise ValueError(
                "parameters['pixel_size_nm'] must resolve to a positive detector "
                "pixel area for fluorescence; got "
                f"{self._detector_pixel_area_nm2} nm^2."
        )
        self._source_map_area_element_nm2 = self._canvas_pitch_nm ** 2
        self._detector_pixel_size_nm = float(sampling.detector_pixel_size_nm)
        self._source_representation = settings.source_representation
        configured_volume_slices = settings.volume_slices
        self._configured_volume_slices = configured_volume_slices
        source_volume_support = get_source_volume_support(params, "fluorescence")
        self._volume_slices = (
            configured_volume_slices
            if source_volume_support is None
            else int(source_volume_support.slice_count)
        )
        if self._volume_slices <= 0:
            raise ValueError("parameters['fluorescence_volume_slices'] must resolve to a positive source-stack depth.")
        slice_thickness_raw = (
            float(source_volume_support.slice_thickness_nm)
            if source_volume_support is not None
            else settings.volume_slice_thickness_nm
        )
        if slice_thickness_raw is None:
            source_span_nm = max(
                MotionDynamicsSettings.from_params(params).initial_z_span_nm,
                canvas_pitch_nm,
            )
            self._volume_slice_thickness_nm = source_span_nm / float(configured_volume_slices)
        else:
            self._volume_slice_thickness_nm = float(slice_thickness_raw)
            if not np.isfinite(self._volume_slice_thickness_nm) or self._volume_slice_thickness_nm <= 0.0:
                raise ValueError(
                    "parameters['fluorescence_volume_slice_thickness_nm'] must be "
                    "finite and positive when set."
                )
        source_z_center_raw = None if source_volume_support is None else source_volume_support.z_center_nm
        self._source_z_center_nm = 0.0 if source_z_center_raw is None else float(source_z_center_raw)
        if not np.isfinite(self._source_z_center_nm):
            raise ValueError(
                "Resolved internal fluorescence source-volume z center must be finite; "
                f"got {source_z_center_raw!r}."
            )
        # Run-scope source-volume support fix: the source stack indexes physical
        # emitter-density z, while focus-relative defocus is applied later by
        # _optical_response_z_positions_nm().  Do not recenter these planes on
        # optical focus or TIRF interface height.
        self._source_z_planes_nm = (
            np.arange(self._volume_slices, dtype=float)
            - 0.5 * float(self._volume_slices - 1)
        ) * self._volume_slice_thickness_nm + self._source_z_center_nm
        self._resolved_source_volume_configured_slices = (
            None if source_volume_support is None else int(source_volume_support.configured_slice_count)
        )
        self._resolved_source_volume_required_slices = (
            None if source_volume_support is None else int(source_volume_support.required_slice_count)
        )
        self._source_z_min_nm = None if source_volume_support is None else float(source_volume_support.z_min_nm)
        self._source_z_max_nm = None if source_volume_support is None else float(source_volume_support.z_max_nm)
        self._source_z_envelope_min_nm = (
            None if source_volume_support is None else float(source_volume_support.envelope_min_nm)
        )
        self._source_z_envelope_max_nm = (
            None if source_volume_support is None else float(source_volume_support.envelope_max_nm)
        )
        self._source_z_support_policy = None if source_volume_support is None else str(source_volume_support.policy)
        self._source_z_preserved_configured_center = (
            None if source_volume_support is None else bool(source_volume_support.preserved_configured_center)
        )
        if settings.emission_psf_sigma_nm is not None:
            sigma_nm = float(settings.emission_psf_sigma_nm)
            self._emission_sigma_source = "nm"
        else:
            sigma_nm = float(settings.emission_psf_sigma_px) * self._detector_pixel_size_nm
            self._emission_sigma_source = "detector_pixels"
        self._emission_sigma_nm = float(sigma_nm)
        self._emission_sigma_px = self._emission_sigma_nm / canvas_pitch_nm
        self._uniform_background = settings.background_count
        self._spectral_bandwidth_nm = settings.spectral_bandwidth_nm
        self._bleaching_rate_per_frame = settings.bleaching_rate_per_frame
        self._vectorial_photophysics_backend = None
        if self._fluorescence_backend == "vectorial_photophysics":
            self._vectorial_photophysics_backend = VectorialPhotophysicsFluorescenceBackend(
                params,
                fluorescence_settings=settings,
                canvas_pitch_nm=canvas_pitch_nm,
                detector_pixel_area_nm2=self._detector_pixel_area_nm2,
                base_emission_sigma_px=self._emission_sigma_px,
                quantum_yield=self._Qf,
                excitation_scale=self._excitation,
                collection_efficiency=self._collection_efficiency,
                detector_qe=self._detector_qe,
                absorbed_excitation_photons_per_fluorophore=(
                    self._absorbed_excitation_photons_per_fluorophore
                ),
                uniform_background=self._uniform_background,
                vectorial_numerical_aperture=vectorial_numerical_aperture,
            )

    def _uses_volume_source(self) -> bool:
        return self._source_representation == "volume"

    def source_coordinate_contract(self, params: dict) -> dict:
        del params
        return {
            "source_density_z_basis": (
                "physical_sample_world" if self._uses_volume_source() else "projected_no_z"
            ),
            "source_z_planes_basis": (
                "physical_sample_world" if self._uses_volume_source() else "projected_no_z"
            ),
            "optical_response_z_basis": (
                "focus_relative"
                if self._uses_volume_source()
                and self._vectorial_photophysics_backend is not None
                else "projected_no_z"
            ),
        }

    def _source_slice_bounds_nm(self) -> tuple[np.ndarray, np.ndarray]:
        half = 0.5 * self._volume_slice_thickness_nm
        return self._source_z_planes_nm - half, self._source_z_planes_nm + half

    def _source_slice_overlaps_nm(
        self,
        center_z_nm: float,
        z_lower_rel_nm: np.ndarray,
        z_upper_rel_nm: np.ndarray,
    ) -> np.ndarray:
        z0 = float(center_z_nm) + np.asarray(z_lower_rel_nm, dtype=float)
        z1 = float(center_z_nm) + np.asarray(z_upper_rel_nm, dtype=float)
        lower, upper = self._source_slice_bounds_nm()
        overlap = np.maximum(
            np.minimum(z1[None, :, :], upper[:, None, None])
            - np.maximum(z0[None, :, :], lower[:, None, None]),
            0.0,
        )
        target_total = np.maximum(z1 - z0, 0.0)
        covered = np.sum(overlap, axis=0)
        missing = np.maximum(target_total - covered, 0.0)
        tolerance = 1.0e-9 * max(float(np.max(target_total)) if target_total.size else 0.0, 1.0)
        if np.any(missing > tolerance):
            raise ValueError(
                "Fluorescence volume source extends outside the configured z-slice "
                "support. Increase fluorescence_volume_slices, increase "
                "fluorescence_volume_slice_thickness_nm, or use projected source "
                "representation; uncovered source mass must not be reassigned to "
                "a different physical z plane."
            )
        return overlap

    def _emission_blur(self, arr: np.ndarray) -> np.ndarray:
        if self._emission_sigma_px == 0.0:
            return arr
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(arr, sigma=self._emission_sigma_px)
        except ImportError:
            sigma = self._emission_sigma_px
            radius = max(int(4 * sigma), 1)
            x = np.arange(-radius, radius + 1, dtype=float)
            k1d = np.exp(-0.5 * (x / sigma) ** 2)
            k1d /= k1d.sum()
            out = arr.astype(float, copy=True)
            for axis in (0, 1):
                out = np.apply_along_axis(
                    _convolve1d_reflect_same,
                    axis, out,
                    kernel=k1d,
                )
            return out

    def _bleach_factor(self, frame_index: int = 0) -> float:
        if self._bleaching_rate_per_frame <= 0.0:
            return 1.0
        t = float(frame_index)
        return float(np.exp(-self._bleaching_rate_per_frame * t))

    def _physical_count_scale(self) -> float:
        return float(
            self._absorbed_excitation_photons_per_fluorophore
            * self._excitation
            * self._Qf
            * self._collection_efficiency
            * self._detector_qe
        )

    def _areal_density_count_scale(self) -> float:
        return float(self._physical_count_scale() * self._detector_pixel_area_nm2)

    def _spectral_factor(self, peak_nm: float | None, wavelength_nm: float) -> float:
        if peak_nm is None:
            return 1.0
        peak = float(peak_nm)
        if peak <= 0.0:
            raise ValueError("Material excitation/emission peak wavelengths must be positive when set.")
        delta = (float(wavelength_nm) - peak) / self._spectral_bandwidth_nm
        return float(np.exp(-0.5 * delta * delta))

    def _material_source_scale(self, material, params: dict) -> float:
        if material is None:
            return 0.0
        density = float(getattr(material, "fluorophore_density", 0.0))
        if density <= 0.0:
            return 0.0
        excitation_nm = self._excitation_wavelength_nm
        emission_nm = self.probe_wavelength_nm(params)
        return (
            density
            * self._spectral_factor(getattr(material, "excitation_peak_nm", None), excitation_nm)
            * self._spectral_factor(getattr(material, "emission_peak_nm", None), emission_nm)
        )

    def _material_source_scale_for_particle(
        self,
        material,
        params: dict,
        *,
        particle_z_nm: float | None = None,
    ) -> float:
        del particle_z_nm
        return self._material_source_scale(material, params)

    def probe_wavelength_nm(self, params: dict) -> float:
        del params
        return self._emission_wavelength_nm

    def _default_direct_signal_modality(self) -> str:
        return "fluorescence_widefield"

    def _direct_signal_source_metadata(self, params: dict) -> dict[str, object]:
        contract = self.source_coordinate_contract(params)
        return {
            "source_input_kind": (
                "z_sliced_fluorophore_emitter_density"
                if self._uses_volume_source()
                else "projected_2d_fluorophore_emitter_density"
            ),
            "source_z_basis": contract.get("source_density_z_basis"),
            "source_projection_policy": (
                "emitter_density_preserved_by_axial_source_slices"
                if self._uses_volume_source()
                else "cell_integrated_emitter_density_chord_before_emission_psf"
            ),
            "source_map_ndim": 3 if self._uses_volume_source() else 2,
            "source_axis_order": "zyx" if self._uses_volume_source() else "yx",
        }

    def _direct_signal_identity(self, params: dict, *, method_name: str):
        metadata = self._direct_signal_source_metadata(params)
        extra = {
            key: value
            for key, value in metadata.items()
            if key not in {"source_input_kind", "source_z_basis", "source_projection_policy"}
        }
        return direct_signal_identity_from_model(
            self,
            params,
            method_name=method_name,
            default_modality=self._default_direct_signal_modality(),
            expected_model_classes_by_modality=_FLUORESCENCE_DIRECT_SIGNAL_MODEL_CLASSES,
            source_input_kind=metadata.get("source_input_kind"),
            source_z_basis=metadata.get("source_z_basis"),
            source_projection_policy=metadata.get("source_projection_policy"),
            extra_provenance=extra,
        )

    def _source_convolution_context_for_signal(
        self,
        source: np.ndarray,
        params: dict,
        *,
        purpose: str,
    ) -> SourceConvolutionContext | None:
        if self._vectorial_photophysics_backend is None:
            return None
        arr = np.asarray(source)
        if arr.ndim not in {2, 3}:
            return SourceConvolutionContext(
                boundary_mode=SourceConvolutionBoundaryMode.LINEAR_ZERO_PADDED_SAME.value,
                source_extent_role="finite_fov_source_map",
                producer=f"{type(self).__name__}.{purpose}",
                notes="non-2D/3D source shape falls back to finite-FOV semantics before backend validation",
            )
        h, w = int(arr.shape[-2]), int(arr.shape[-1])
        os_size = SamplingGeometry.from_params(params).model_canvas_shape[0]
        if purpose == "rendered_scene" and h == w and h > os_size and (h - os_size) % 2 == 0:
            guard = int((h - os_size) // 2)
            return SourceConvolutionContext(
                boundary_mode=SourceConvolutionBoundaryMode.PRE_CROP_GUARDED_FFT.value,
                source_extent_role="pre_crop_guarded_render_canvas",
                producer=f"{type(self).__name__}.{purpose}",
                guard_radius_pixels=guard,
                crop_slices=((guard, guard + os_size), (guard, guard + os_size)),
                notes=(
                    "rendering/frame_loop supplied an oversized pre-crop canvas; "
                    "circular FFT is allowed only because the wrapped region is outside the saved crop"
                ),
            )
        return SourceConvolutionContext(
            boundary_mode=SourceConvolutionBoundaryMode.LINEAR_ZERO_PADDED_SAME.value,
            source_extent_role=(
                "direct_finite_fov_source_map"
                if purpose.startswith("direct_")
                else "finite_fov_render_canvas"
            ),
            producer=f"{type(self).__name__}.{purpose}",
            notes="finite source extent uses zero-padded linear same convolution; no periodic wrapping is implied",
        )

    def illumination_field(self, shape: tuple[int, int], params: dict) -> np.ndarray:
        del params
        return np.ones(shape, dtype=float)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        physical_count_scale = self._physical_count_scale()
        areal_density_count_scale = self._areal_density_count_scale()
        response.update({
            "kind": "fluorescence_emission_psf",
            "excitation_wavelength_nm": self._excitation_wavelength_nm,
            "emission_wavelength_nm": self.probe_wavelength_nm(params),
            "emission_sigma_canvas_px": self._emission_sigma_px,
            "emission_sigma_detector_px": self._emission_sigma_nm / self._detector_pixel_size_nm,
            "emission_sigma_nm": self._emission_sigma_nm,
            "emission_sigma_source": self._emission_sigma_source,
            "emission_psf_boundary_mode": (
                "source_convolution_context_dependent"
                if self._vectorial_photophysics_backend is not None
                else "reflect_sum_preserving"
            ),
            "fluorescence_quantum_yield": self._Qf,
            "fluorescence_excitation_scale": self._excitation,
            "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame": (
                self._absorbed_excitation_photons_per_fluorophore
            ),
            "fluorescence_collection_efficiency": self._collection_efficiency,
            "detector_qe": FluorescenceSettings.from_params(params).detector_qe,
            "fluorescence_detector_qe": self._detector_qe,
            "fluorescence_absolute_scale": "physical_absorbed_excitation_photon_budget",
            "fluorescence_background_counts_per_pixel": self._uniform_background,
            "fluorescence_background_units": "detected_counts_per_pixel",
            "fluorescence_photon_budget_source": (
                "fluorescence_absorbed_excitation_photons_per_fluorophore_per_frame"
            ),
            "fluorescence_photon_budget_semantics": "absorbed_excitation_photons_before_quantum_yield",
            "source_map_units": "fluorophore_areal_density_per_nm2",
            "source_map_area_element_nm2": self._source_map_area_element_nm2,
            "fluorescence_detector_pixel_area_nm2": self._detector_pixel_area_nm2,
            "fluorescence_physical_count_scale": physical_count_scale,
            "fluorescence_areal_density_count_scale": areal_density_count_scale,
            "count_scale": areal_density_count_scale,
            "count_scaling_mode": "areal_density_times_detector_pixel_area_times_photon_budget",
            "spectral_bandwidth_nm": self._spectral_bandwidth_nm,
            "filter_guard_radius_pixels": self.filter_guard_radius_pixels(
                params,
                model_canvas_shape=shape,
            ),
            "source_input_kind": (
                "z_sliced_fluorophore_emitter_density"
                if self._uses_volume_source()
                else "projected_2d_fluorophore_emitter_density"
            ),
            "source_map_ndim": 3 if self._uses_volume_source() else 2,
            "source_axis_order": "zyx" if self._uses_volume_source() else "yx",
            **self.source_coordinate_contract(params),
            "source_projection_policy": (
                "emitter_density_preserved_by_axial_source_slices"
                if self._uses_volume_source()
                else "cell_integrated_emitter_density_chord_before_emission_psf"
            ),
            "source_convolution_contract_id": SOURCE_CONVOLUTION_CONTRACT_ID,
            "source_convolution_boundary_policy": (
                "explicit_context_required_for_vectorial_backend"
                if self._vectorial_photophysics_backend is not None
                else "parametric_backend_native_boundary"
            ),
            "backend_consumes_volume_source": bool(
                self._uses_volume_source()
                and self._vectorial_photophysics_backend is not None
            ),
            "volume_transport_model": (
                "z_sliced_isotropic_dipole_emission_psf"
                if self._uses_volume_source()
                and self._vectorial_photophysics_backend is not None
                else (
                    "source_stack_projected_before_parametric_emission_psf"
                    if self._uses_volume_source()
                    else "emission_psf_from_projected_emitter_density"
                )
            ),
            "fluorescence_source_representation": self._source_representation,
            "source_z_planes_nm": (
                self._source_z_planes_nm.astype(float).tolist()
                if self._uses_volume_source()
                else None
            ),
            "emission_psf_z_positions_nm": (
                (self._source_z_planes_nm - FocusPlaneState.from_params(params).z_nm).astype(float).tolist()
                if self._uses_volume_source()
                and self._vectorial_photophysics_backend is not None
                else None
            ),
            "source_slice_thickness_nm": (
                float(self._volume_slice_thickness_nm)
                if self._uses_volume_source()
                else None
            ),
            "source_z_center_nm": (
                float(self._source_z_center_nm)
                if self._uses_volume_source()
                else None
            ),
            "source_z_min_nm": (
                None
                if not self._uses_volume_source() or self._source_z_min_nm is None
                else float(self._source_z_min_nm)
            ),
            "source_z_max_nm": (
                None
                if not self._uses_volume_source() or self._source_z_max_nm is None
                else float(self._source_z_max_nm)
            ),
            "source_z_envelope_min_nm": (
                None
                if not self._uses_volume_source() or self._source_z_envelope_min_nm is None
                else float(self._source_z_envelope_min_nm)
            ),
            "source_z_envelope_max_nm": (
                None
                if not self._uses_volume_source() or self._source_z_envelope_max_nm is None
                else float(self._source_z_envelope_max_nm)
            ),
            "source_volume_configured_slices": (
                None
                if not self._uses_volume_source()
                else (
                    int(self._resolved_source_volume_configured_slices)
                    if self._resolved_source_volume_configured_slices is not None
                    else int(self._configured_volume_slices)
                )
            ),
            "source_volume_required_slices_for_rendered_z": (
                None
                if not self._uses_volume_source() or self._resolved_source_volume_required_slices is None
                else int(self._resolved_source_volume_required_slices)
            ),
            "source_z_support_policy": (
                self._source_z_support_policy
                if self._uses_volume_source()
                else None
            ),
            "source_z_preserved_configured_center": (
                self._source_z_preserved_configured_center
                if self._uses_volume_source()
                else None
            ),
            "source_stack_out_of_range_policy": (
                "error_on_uncovered_chord_length_outside_source_stack"
                if self._uses_volume_source()
                else None
            ),
            "sample_environment_autofluorescence_layer_metadata": (
                self._sample_environment_autofluorescence_layer(params).metadata()
            ),
            "sample_environment_autofluorescence_world_z_nm": (
                self._sample_environment_autofluorescence_layer_world_z_nm(params)
            ),
            "sample_environment_autofluorescence_z_basis": "physical_sample_world",
            "sample_environment_autofluorescence_volume_policy": (
                "explicit_source_layer_world_z_to_source_slice"
                if self._uses_volume_source()
                else "projected_with_particle_source"
            ),
        })
        if self._vectorial_photophysics_backend is not None:
            self._vectorial_photophysics_backend._vectorial_psf(tuple(shape), z_nm=0.0)
            response.update(
                self._vectorial_photophysics_backend.metadata(
                    params,
                    source_ndim=3 if self._uses_volume_source() else 2,
                )
            )
        else:
            response["fluorescence_backend"] = self._fluorescence_backend
            response = attach_backend_fidelity_metadata(
                response,
                params=params,
                backend_name=self._fluorescence_backend,
                equations_or_model_family="parametric fluorescence PSF proxy",
                implemented_approximation_level="proxy",
                native_operating_assumptions="material fluorophore-source map convolved with emission PSF",
                comparison_contract_id=response.get("comparison_contract_id", "Contract-NR"),
                artifact_provenance_id=response.get("artifact_provenance_id"),
            )
        return response

    def compute_noise(
        self,
        frame_counts: np.ndarray,
        params: dict,
        rng: np.random.Generator | None = None,
        *,
        detector_noise_runtime=None,
    ) -> np.ndarray:
        """Apply fluorescence-specific detector noise with explicit QE control."""
        from camera_noise import (
            canonicalize_detector_frame_noise_params,
            modality_noise_overrides_from_params,
            noise_model_overrides_from_params,
        )

        local_params = dict(params)
        configured_assign(local_params, 'detector_qe', self._detector_qe)
        noise_model = noise_model_overrides_from_params(local_params)
        noise_model["detector_qe"] = self._detector_qe
        configured_assign(local_params, 'noise_model', noise_model)
        modality_noise = modality_noise_overrides_from_params(local_params)
        for key, value in list(modality_noise.items()):
            if isinstance(value, dict):
                override = dict(value)
                override["detector_qe"] = self._detector_qe
                modality_noise[key] = override
        configured_assign(local_params, 'modality_noise', modality_noise)
        # Fluorescence outputs detected counts after the photon budget has used
        # detector QE.  Delegate the incident-quanta rejection to the shared
        # detector-frame contract so stochastic rendering, deterministic means,
        # reports, calibration, and Fisher likelihoods all enforce one rule.
        local_params = canonicalize_detector_frame_noise_params(
            local_params,
            context="FluorescenceWidefieldImagingModel.compute_noise",
        )
        return super().compute_noise(
            frame_counts,
            local_params,
            rng=rng,
            detector_noise_runtime=detector_noise_runtime,
        )

    def filter_guard_radius_pixels(
        self,
        params: dict,
        *,
        model_canvas_shape: tuple[int, int] | None = None,
    ) -> int | None:
        guard = int(np.ceil(max(4.0 * self._emission_sigma_px, 2.0)))
        if self._vectorial_photophysics_backend is None:
            return guard
        if model_canvas_shape is None:
            os_size = max(1, SamplingGeometry.from_params(params).model_canvas_shape[0])
        else:
            os_size = max(
                1,
                int(model_canvas_shape[0]),
                int(model_canvas_shape[1]),
            )
        instrument = OpticalInstrumentSettings.from_params(params)
        max_auto_guard = max(os_size, 4 * instrument.vectorial_pupil_samples, 64, guard)
        z_positions = (
            self._source_z_planes_nm - FocusPlaneState.from_params(params).z_nm
            if self._uses_volume_source()
            else None
        )
        for _ in range(10):
            shape_size = os_size + 2 * guard
            support = self._vectorial_photophysics_backend.psf_support_radius_pixels(
                (shape_size, shape_size),
                z_positions_nm=z_positions,
            )
            next_guard = int(np.ceil(min(max_auto_guard, max(guard, support + 2))))
            if next_guard <= guard:
                break
            guard = next_guard
        return guard

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del params
        if self._uses_volume_source():
            return np.zeros((self._volume_slices, int(shape[0]), int(shape[1])), dtype=float)
        return np.zeros(shape, dtype=float)

    def accumulate_particle_source(
        self,
        source_canvas,
        *,
        center_x_canvas: int,
        center_y_canvas: int,
        diameter_nm: float,
        pixel_size_nm: float,
        os_factor: int,
        material_properties,
        params: dict,
        particle_z_nm: float | None = None,
        source_coordinate_context: SourceCoordinateContext | None = None,
        source_multiplier: float = 1.0,
        component_geometry=None,
        orientation_matrix=None,
    ) -> None:
        if source_canvas is None:
            return
        if component_geometry is None:
            raise ValueError("Fluorescence source accumulation requires component_geometry.")
        if source_coordinate_context is not None:
            particle_z_nm = source_coordinate_context.source_density_z_nm
        scale = self._material_source_scale_for_particle(
            material_properties,
            params,
            particle_z_nm=particle_z_nm,
        )
        if scale <= 0.0:
            return
        source_arr = np.asarray(source_canvas)
        h, w = source_arr.shape[-2:]
        footprint = primitive_footprint_patch(
            component_geometry=component_geometry,
            center_x_canvas=float(center_x_canvas),
            center_y_canvas=float(center_y_canvas),
            pixel_size_nm=float(pixel_size_nm),
            os_factor=int(os_factor),
            canvas_shape=(h, w),
            orientation_matrix=orientation_matrix,
        )
        if footprint is None:
            return
        projected_chord_nm = footprint.projected_chord_nm()
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        if source_arr.ndim == 3:
            center_z_nm = 0.0 if particle_z_nm is None else float(particle_z_nm)
            overlaps_nm = footprint.average_over_samples(
                lambda z_lower_rel_nm, z_upper_rel_nm: self._source_slice_overlaps_nm(
                    center_z_nm,
                    z_lower_rel_nm,
                    z_upper_rel_nm,
                )
            )
            overlaps_nm = normalize_sliced_source_to_projected_chord(
                overlaps_nm,
                projected_chord_nm,
            )
            source_canvas[:, footprint.y0:footprint.y1, footprint.x0:footprint.x1] += (
                multiplier * scale * overlaps_nm
            )
        elif source_arr.ndim == 2:
            source_canvas[footprint.y0:footprint.y1, footprint.x0:footprint.x1] += (
                multiplier * scale * projected_chord_nm
            )
        else:
            raise ValueError(
                "Fluorescence source canvas must be 2D (y, x) or 3D (z, y, x); "
                f"got shape {source_arr.shape!r}."
            )

    @staticmethod
    def _source_from_particle_source_maps(
        E_sca_total: np.ndarray,
        particle_source_maps: list[np.ndarray] | None,
    ) -> np.ndarray:
        if particle_source_maps is None or len(particle_source_maps) == 0:
            return np.zeros_like(E_sca_total, dtype=float)
        source_raw = np.sum(
            [
                np.asarray(source_map, dtype=np.complex128)
                for source_map in particle_source_maps
            ],
            axis=0,
        )
        if np.iscomplexobj(source_raw):
            imag_max = float(np.max(np.abs(source_raw.imag))) if source_raw.size else 0.0
            real_scale = max(
                float(np.max(np.abs(source_raw.real))) if source_raw.size else 0.0,
                1.0,
            )
            if imag_max > 1.0e-12 * real_scale:
                raise ValueError(
                    "Fluorescence source maps must remain real-valued emitter-density "
                    f"sources; max imaginary component is {imag_max}."
                )
            source = np.asarray(source_raw.real, dtype=float)
        else:
            source = np.asarray(source_raw, dtype=float)
        return np.maximum(source, 0.0)

    def _sample_environment_autofluorescence_layer_world_z_nm(self, params: dict) -> float:
        del params
        return 0.0

    def _sample_environment_autofluorescence_layer(
        self,
        params: dict,
    ) -> FluorescenceSourceLayerPlacement:
        return FluorescenceSourceLayerPlacement(
            role=FLUORESCENCE_SOURCE_ROLE_SAMPLE_ENVIRONMENT_AUTOFLUORESCENCE,
            source_basis=FLUORESCENCE_SOURCE_BASIS_EMITTER_DENSITY,
            z_basis=FLUORESCENCE_SOURCE_Z_BASIS_PHYSICAL_SAMPLE_WORLD,
            world_z_nm=self._sample_environment_autofluorescence_layer_world_z_nm(params),
        )

    def _add_sample_environment_autofluorescence_to_volume(
        self,
        source: np.ndarray,
        autofl_source: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        out = np.asarray(source, dtype=float).copy()
        layer = self._sample_environment_autofluorescence_layer(params)
        # Surface/source-layer placement is a physical source-stack contract,
        # not a display or PSF shortcut.  TIRF overrides the layer world-z so an
        # interface offset changes the vectorial defocus instead of being lost
        # by a hard-coded world-z=0 insertion.
        layer_idx = layer.volume_slice_index(
            self._source_z_planes_nm,
            source_slice_thickness_nm=self._volume_slice_thickness_nm,
        )
        out[layer_idx] += np.asarray(autofl_source, dtype=float)
        return out

    def _source_with_sample_environment(
        self,
        source: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        source = np.asarray(source, dtype=float)
        if sample_environment is None:
            return source
        excitation_nm = self._excitation_wavelength_nm
        reflection = sample_environment.substrate.reflection_amplitude(excitation_nm)
        modulation = _mean_normalized_map(np.abs(1.0 + reflection) ** 2)
        sample_environment_settings = SampleEnvironmentSettings.from_params(params)
        mod_gain = sample_environment_settings.fluorescence_excitation_modulation_gain
        autofl_gain = sample_environment_settings.fluorescence_autofluorescence_gain
        excitation_factor = np.maximum(1.0 + mod_gain * (modulation - 1.0), 0.0)
        autofl_source = autofl_gain * np.maximum(
            sample_environment.substrate.autofluorescence_density(),
            0.0,
        )
        if source.ndim == 3:
            out = source * excitation_factor[None, :, :]
            out = self._add_sample_environment_autofluorescence_to_volume(
                out,
                autofl_source,
                params,
            )
            return np.maximum(out, 0.0)
        return np.maximum(source * excitation_factor + autofl_source, 0.0)

    def _optical_response_z_positions_nm(
        self,
        source: np.ndarray,
        params: dict,
    ) -> np.ndarray | None:
        if not self._uses_volume_source() or np.asarray(source).ndim != 3:
            return None
        return self._source_z_planes_nm - FocusPlaneState.from_params(params).z_nm

    def _detector_signal_from_source(
        self,
        source: np.ndarray,
        params: dict,
        *,
        frame_index: int,
        include_background: bool = True,
        source_convolution_context: SourceConvolutionContext | None = None,
    ) -> np.ndarray:
        del include_background
        if self._vectorial_photophysics_backend is not None:
            return self._vectorial_photophysics_backend.source_to_emission_density(
                source,
                frame_index=frame_index,
                z_positions_nm=self._optical_response_z_positions_nm(source, params),
                convolution_context=(
                    source_convolution_context
                    if source_convolution_context is not None
                    else self._source_convolution_context_for_signal(
                        source,
                        params,
                        purpose="finite_fov_internal_default",
                    )
                ),
            )
        if np.asarray(source).ndim == 3:
            source = np.sum(np.asarray(source, dtype=float), axis=0)
        emission = self._emission_blur(source)
        bleach = self._bleach_factor(frame_index=frame_index)
        intensity = bleach * emission
        return np.maximum(intensity, 0.0)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_field
        # Direct calls without render-supplied material source maps have no
        # particle fluorescence source. Background is added at the count layer.
        return np.zeros(E_sca_total.shape, dtype=float)

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
        del E_sca_particles, particle_instances, background_field
        source = self._source_from_particle_source_maps(E_sca_total, particle_source_maps)
        return self._detector_signal_from_source(
            source,
            params,
            frame_index=frame_index,
            source_convolution_context=self._source_convolution_context_for_signal(
                source,
                params,
                purpose="rendered_scene",
            ),
        )

    def compute_scene_intensity_with_sample_environment(
        self,
        E_sca_particles: list[np.ndarray],
        particle_instances: list,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_source_maps: list[np.ndarray] | None = None,
        *,
        frame_index: int = 0,
        sample_environment: SampleEnvironment | None = None,
    ) -> np.ndarray:
        del E_sca_particles, particle_instances, background_field
        source = self._source_from_particle_source_maps(E_sca_total, particle_source_maps)
        source = self._source_with_sample_environment(source, params, sample_environment)
        return self._detector_signal_from_source(
            source,
            params,
            frame_index=frame_index,
            source_convolution_context=self._source_convolution_context_for_signal(
                source,
                params,
                purpose="rendered_scene",
            ),
        )

    def _direct_signal_product_from_source(
        self,
        source: np.ndarray,
        params: dict,
        *,
        frame_index: int,
        method_name: str,
    ) -> DirectParticleSignalProduct:
        # The fix-site invariant is detector-transfer ownership: fluorescence
        # source maps are emitter/emission density, while Fisher derivatives in
        # count-domain likelihoods must use the detector-count contribution.
        # Uniform background remains outside the derivative image and belongs in
        # the noise/reference model, so this product stores a counts-delta frame.
        source_convolution_context = self._source_convolution_context_for_signal(
            source,
            params,
            purpose=f"direct_{method_name}",
        )
        emission_density = self._detector_signal_from_source(
            source,
            params,
            frame_index=frame_index,
            include_background=False,
            source_convolution_context=source_convolution_context,
        )
        scale = self._areal_density_count_scale()
        counts_delta = scale * np.asarray(emission_density, dtype=float)
        identity = self._direct_signal_identity(params, method_name=method_name)
        provenance = identity.provenance_payload(frame_index=int(frame_index))
        if source_convolution_context is not None:
            provenance.update(source_convolution_context.to_metadata())
        return DirectParticleSignalProduct(
            values=counts_delta,
            representation=detector_count_delta_representation(),
            modality=identity.modality,
            producer=identity.producer,
            safe_for_fisher=True,
            detector_scale_applied=True,
            background_included=False,
            source_representation=fluorescence_emission_source_representation(),
            detector_scale_factor=float(scale),
            conversion_note=(
                "Converted fluorescence emission-density response to detector "
                "count contribution. Uniform fluorescence background is not part "
                "of the particle-position derivative and must enter through the "
                "typed noise/reference model."
            ),
            provenance=provenance,
        )

    def compute_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del E_sca_particle, background_field, params, particle_instance, frame_index
        raise RuntimeError(
            "Fluorescence direct particle contrast no longer returns a bare array. "
            "Use compute_particle_signal_product(); its metadata records the "
            "emitter-density to detector-count transfer before Fisher use."
        )

    def _zero_direct_signal_product(
        self,
        shape: tuple[int, int],
        params: dict,
        *,
        frame_index: int,
        method_name: str,
        conversion_note: str,
    ) -> DirectParticleSignalProduct:
        # Direct zero products still cross the same metadata seam as nonzero
        # products.  Resolving identity here prevents TIRF's no-particle and
        # zero-fluorophore branches from inheriting a widefield modality label.
        identity = self._direct_signal_identity(params, method_name=method_name)
        return DirectParticleSignalProduct(
            values=np.zeros(shape, dtype=float),
            representation=detector_count_delta_representation(),
            modality=identity.modality,
            producer=identity.producer,
            safe_for_fisher=True,
            detector_scale_applied=True,
            background_included=False,
            source_representation=fluorescence_emission_source_representation(),
            detector_scale_factor=float(self._areal_density_count_scale()),
            conversion_note=conversion_note,
            provenance=identity.provenance_payload(frame_index=int(frame_index)),
        )

    def compute_particle_signal_product(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        del background_field
        shape = E_sca_particle.shape[-2:]
        if particle_instance is None:
            return self._zero_direct_signal_product(
                shape,
                params,
                frame_index=frame_index,
                method_name="compute_particle_signal_product",
                conversion_note="No particle instance was supplied; particle derivative contribution is zero.",
            )
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct fluorescence particle signal for composite particles "
                "requires a rendered source map; use compute_particle_signal_product_from_source_map()."
            )
        source = self.initialize_particle_source_canvas(shape, params)
        material = getattr(particle_instance, "material_properties", None)
        scale = self._material_source_scale(material, params)
        if scale <= 0.0:
            return self._zero_direct_signal_product(
                shape,
                params,
                frame_index=frame_index,
                method_name="compute_particle_signal_product",
                conversion_note="Material fluorescence source scale is zero; particle derivative contribution is zero.",
            )
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(frame_index)
        frame_idx = int(np.clip(frame_idx, 0, traj.shape[0] - 1))
        sampling = SamplingGeometry.from_params(params)
        px = float(traj[frame_idx, 0]) / sampling.detector_pixel_size_nm * float(sampling.psf_oversampling_factor)
        py = float(traj[frame_idx, 1]) / sampling.detector_pixel_size_nm * float(sampling.psf_oversampling_factor)
        pz = float(traj[frame_idx, 2]) if traj.shape[1] >= 3 else 0.0
        self.accumulate_particle_source(
            source,
            center_x_canvas=px,
            center_y_canvas=py,
            diameter_nm=float(particle_instance.particle_type.diameter_nm),
            pixel_size_nm=sampling.detector_pixel_size_nm,
            os_factor=sampling.psf_oversampling_factor,
            material_properties=material,
            params=params,
            particle_z_nm=pz,
            component_geometry=particle_instance.component_geometry,
            orientation_matrix=None,
        )
        return self._direct_signal_product_from_source(
            source,
            params,
            frame_index=frame_idx,
            method_name="compute_particle_signal_product",
        )

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del particle_source_map, background_field, params, frame_index
        raise RuntimeError(
            "Fluorescence source-map contrast no longer returns a bare array. "
            "Use compute_particle_signal_product_from_source_map(); it converts "
            "emission density to detector-count contribution exactly once."
        )

    def compute_particle_signal_product_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        del background_field
        source = np.asarray(particle_source_map, dtype=float)
        return self._direct_signal_product_from_source(
            source,
            params,
            frame_index=frame_index,
            method_name="compute_particle_signal_product_from_source_map",
        )

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_field, params
        return np.zeros_like(E_sca_particle, dtype=float)

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        del E_sca_total, background_field
        if sample_environment is None:
            return intensity
        excitation_nm = self._excitation_wavelength_nm
        reflection = sample_environment.substrate.reflection_amplitude(excitation_nm)
        modulation = _mean_normalized_map(np.abs(1.0 + reflection) ** 2)
        sample_environment_settings = SampleEnvironmentSettings.from_params(params)
        mod_gain = sample_environment_settings.fluorescence_excitation_modulation_gain
        autofl_gain = sample_environment_settings.fluorescence_autofluorescence_gain
        autofl = autofl_gain * self._emission_blur(
            sample_environment.substrate.autofluorescence_density()
        )
        return np.maximum(intensity * (1.0 + mod_gain * (modulation - 1.0)) + autofl, 0.0)

    def convert_model_output_to_detector_frame(
        self,
        model_output: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_final, E_ref_intensity_final
        del params
        return convert_model_output_to_detector_frame(
            model_output=model_output,
            background_frame=None,
            reference_intensity_frame=None,
            conversion=DetectorFrameConversion(
                model_output_domain=MODEL_OUTPUT_DOMAIN_EMISSION_DENSITY,
                detector_output_domain=DETECTOR_OUTPUT_DOMAIN_CAMERA_COUNTS,
                value_form=VALUE_FORM_ABSOLUTE,
                reference_basis=REFERENCE_BASIS_NONE,
                scale=self._areal_density_count_scale(),
                offset=self._uniform_background,
            ),
            params=None,
            context="FluorescenceWidefieldImagingModel.convert_model_output_to_detector_frame",
        )

__all__ = ['FluorescenceWidefieldImagingModel']
