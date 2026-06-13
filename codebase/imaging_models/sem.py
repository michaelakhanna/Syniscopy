"""sem imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    SourceCoordinateContext,
    np,
)
from backend_fidelity import attach_backend_fidelity_metadata
from detector_frame_conversion import (
    DETECTOR_OUTPUT_DOMAIN_ELECTRON_COUNT,
    MODEL_OUTPUT_DOMAIN_ELECTRON_YIELD,
    REFERENCE_BASIS_NONE,
    VALUE_FORM_ABSOLUTE,
    DetectorFrameConversion,
    convert_model_output_to_detector_frame,
)
from source_volume_support import (
    SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH,
    SOURCE_Z_BASIS_PROJECTED_NO_Z,
    SOURCE_Z_FRAME_CONTRACT_VERSION,
    require_source_density_z_basis,
)
from config.runtime import (
    FocusPlaneState,
    SampleEnvironmentSettings,
    SemSettings,
)
from electron_optics import electron_wavelength_m
from direct_signal_contracts import (
    DirectParticleSignalProduct,
    electron_count_delta_representation,
    sem_secondary_electron_source_representation,
)
from .sem_source import (
    SEMMaterialSourceCanvas,
    source_like_numeric_array,
    source_like_sum,
)
from .sem_backends import (
    GaussianProbeSEMProxyBackend,
    InteractionVolumeSEMProxyBackend,
    MonteCarloSEMTransportBackend,
    PhysicalMonteCarloSEMTransportBackend,
    ReferenceKernelSEMBackend,
    SyniscopyTransportSEMBackend,
)
from .sem_depth_grid import (
    sem_depth_grid_from_params,
    sem_source_volume_support_from_params,
)
from simulation_runtime_state import (
    config_without_runtime_state,
    get_source_volume_support,
    set_source_volume_support,
)
from .source_rasterization import primitive_footprint_patch

class ScanningElectronMicroscopyImagingModel(ImagingModel):
    """
    Scanning electron microscopy (SEM) imaging model — secondary-electron
    topography contrast.

    SEM differs from TEM in two important ways that together define its
    characteristic image appearance:

    * The signal is not a transmitted electron wave but a per-pixel
      *secondary-electron yield* proxy.  The implemented source is weighted by
      the particle material's nominal secondary-electron yield coefficient and
      by the gradient of that projected source, which gives the expected
      edge-brightening behavior without claiming a full surface-transport
      calculation.

    * The beam is raster-scanned with a finite-size probe and the
    resulting signal is the convolution of the material's per-point
    emission with the probe intensity profile.  The probe size, not
    the illumination wavelength, is what ultimately limits SEM
    resolution.

    Syniscopy field representation
    ------------------------------
    Syniscopy's hybrid-assembly pipeline provides a material source weighted
    by the particle material's nominal secondary-electron yield.  The default
    Monte Carlo backend consumes a sliced source volume with axis order
    ``(z, y, x)`` and applies a stochastic interaction-volume transport kernel.
    Legacy projected-source proxy, transport-lite, and reference-kernel routes
    remain selectable and report their projection policy in metadata.

    Parameters (parameters keys, optional, nominal defaults)
    ----------------------------------------------------
    The defaults set a stable moderate-contrast synthetic SEM regime; use
    calibrated values for instrument-specific studies.

    - ``sem_probe_sigma_nm``        Gaussian probe spot size
    - ``sem_edge_contrast_gain``    (default 10.0) weight on the gradient-
                                    magnitude term (secondary-emission edge
                                    enhancement).
    - ``sem_bulk_contrast_gain``    (default 1.0) weight on the material
                                    source term (bulk/Z-like contribution).
    - ``sem_baseline_yield``        (default 0.05) yield from the substrate
                                    with no particle present.
    - ``sem_electrons_per_pixel``   (default 1000.0) dose scale used by the
                                    model-output detector-frame conversion.
    """

    output_type = "intensity"
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_complex_optical_psf = False
    requires_optical_scattered_field = False
    requires_pre_crop_optical_filtering = True
    supports_spectral_channels = False
    counts_are_exposure_integrated = True

    @staticmethod
    def _configured_incident_electrons_per_pixel(params: dict) -> tuple[float, str]:
        return SemSettings.from_params(params).configured_incident_electrons_per_pixel()

    def __init__(self, params: dict) -> None:
        settings = SemSettings.from_params(params)
        self._sem_settings = settings
        self._sem_model = settings.model
        self._sem_backend = settings.backend
        self._sem_transport_backend = None
        self._sem_reference_kernel_backend = None
        self._sem_reference_backend = None
        self._sem_proxy_backend = None
        physical_backends = {
            "monte_carlo_physical",
            "monte_carlo_transport",
            "syniscopy_transport_lite",
            "reference_kernel_table",
        }
        self._sem_source_resolution = settings.source_resolution
        self._sem_source_representation = self._sem_source_resolution.requested
        self._sem_effective_source_representation = self._sem_source_resolution.effective
        self._sem_source_projection_policy = self._sem_source_resolution.source_projection_policy
        self._sem_source_z_origin = settings.source_z_origin
        self._sem_source_z_offset_nm = settings.source_z_offset_nm
        self._sem_volume_slices = settings.volume_slices
        if self._sem_effective_source_representation == "volume":
            self._sem_depth_grid = sem_depth_grid_from_params(params, backend_name=self._sem_backend)
            # The depth grid may be widened at run scope by the shared SEM
            # source-support resolver.  Store the effective values here so
            # response metadata, source canvases, and backend kernels report and
            # consume one physical z interval instead of the public defaults.
            self._sem_volume_slices = self._sem_depth_grid.slice_count
            self._sem_volume_slice_thickness_nm = self._sem_depth_grid.slice_thickness_nm
            self._sem_source_z_offset_nm = self._sem_depth_grid.offset_nm
        else:
            self._sem_depth_grid = None
            self._sem_volume_slice_thickness_nm = settings.volume_slice_thickness_for_backend(self._sem_backend)
        canvas_pitch_nm = settings.sampling.model_canvas_pixel_size_nm
        if not np.isfinite(canvas_pitch_nm) or canvas_pitch_nm <= 0.0:
            raise ValueError(
                "parameters['pixel_size_nm'] / parameters['psf_oversampling_factor'] "
                f"must resolve to a positive SEM canvas pitch; got {canvas_pitch_nm} nm."
            )
        self._probe_sigma_source = "pixels"
        if settings.probe_sigma_nm is not None:
            self._probe_sigma_px = settings.probe_sigma_nm / canvas_pitch_nm
            self._probe_sigma_source = "nm"
        else:
            self._probe_sigma_px = settings.probe_sigma_px
        self._probe_sigma_nm = self._probe_sigma_px * canvas_pitch_nm
        self._canvas_pitch_nm = canvas_pitch_nm

        if self._sem_backend == "monte_carlo_transport":
            self._sem_transport_backend = MonteCarloSEMTransportBackend(
                params,
                canvas_pitch_nm=self._canvas_pitch_nm,
                probe_sigma_px=self._probe_sigma_px,
            )
        elif self._sem_backend == "monte_carlo_physical":
            self._sem_transport_backend = PhysicalMonteCarloSEMTransportBackend(
                params,
                canvas_pitch_nm=self._canvas_pitch_nm,
                probe_sigma_px=self._probe_sigma_px,
            )
        elif self._sem_backend == "syniscopy_transport_lite":
            self._sem_transport_backend = SyniscopyTransportSEMBackend(
                params,
                canvas_pitch_nm=self._canvas_pitch_nm,
                probe_sigma_px=self._probe_sigma_px,
            )
        elif self._sem_backend == "reference_kernel_table":
            self._sem_reference_kernel_backend = ReferenceKernelSEMBackend(
                params,
                canvas_pitch_nm=self._canvas_pitch_nm,
                probe_sigma_px=self._probe_sigma_px,
            )
            self._sem_reference_backend = self._sem_reference_kernel_backend
        if (
            self._sem_backend == "monte_carlo_physical"
            and self._sem_transport_backend is not None
            and hasattr(self._sem_transport_backend, "precompute_material_kernels_from_params")
        ):
            self._sem_transport_backend.precompute_material_kernels_from_params(
                params,
                require_volume=self._sem_effective_source_representation == "volume",
            )
        self._interaction_volume_nm = settings.interaction_volume_nm
        self._detector_direction_xy = np.asarray(settings.detector_direction_xy, dtype=float)
        self._edge_gain = settings.edge_contrast_gain
        self._bulk_gain = settings.bulk_contrast_gain
        self._topography_gain = settings.topography_contrast_gain
        self._baseline = settings.baseline_yield
        self._acceleration_kV = settings.acceleration_kV
        if self._sem_transport_backend is None and self._sem_reference_kernel_backend is None:
            if self._sem_model == "interaction_volume_proxy":
                self._sem_proxy_backend = InteractionVolumeSEMProxyBackend(
                    probe_sigma_px=self._probe_sigma_px,
                    canvas_pitch_nm=self._canvas_pitch_nm,
                    interaction_volume_nm=self._interaction_volume_nm,
                    detector_direction_xy=self._detector_direction_xy,
                    edge_gain=self._edge_gain,
                    bulk_gain=self._bulk_gain,
                    topography_gain=self._topography_gain,
                    baseline=self._baseline,
                )
            else:
                self._sem_proxy_backend = GaussianProbeSEMProxyBackend(
                    probe_sigma_px=self._probe_sigma_px,
                    canvas_pitch_nm=self._canvas_pitch_nm,
                    edge_gain=self._edge_gain,
                    bulk_gain=self._bulk_gain,
                    baseline=self._baseline,
                )
    def _backend_response_contract(self) -> dict:
        if self._sem_backend == "gaussian_probe_proxy":
            edge_convention = "gradient_magnitude_per_nm"
            topography_convention = "not_supported"
            detector_direction_role = "not_used"
            topography_supported = False
            detector_direction_used = False
        elif self._sem_backend == "interaction_volume_proxy":
            edge_convention = "positive_directed_detector_gradient_per_nm"
            topography_convention = "gradient_magnitude_per_nm"
            detector_direction_role = "edge_term_positive_projection"
            topography_supported = True
            detector_direction_used = self._edge_gain > 0.0
        else:
            edge_convention = "gradient_magnitude_per_nm"
            topography_convention = "positive_directed_detector_gradient_per_nm"
            detector_direction_role = "topography_term_positive_projection"
            topography_supported = True
            detector_direction_used = self._topography_gain > 0.0

        return {
            "sem_backend_edge_convention": edge_convention,
            "sem_backend_topography_convention": topography_convention,
            "sem_backend_detector_direction_role": detector_direction_role,
            "sem_backend_active_terms": {
                "bulk": self._bulk_gain > 0.0,
                "edge": self._edge_gain > 0.0,
                "topography": topography_supported and self._topography_gain > 0.0,
                "detector_direction": detector_direction_used,
            },
            "sem_backend_consumes_topography_gain": topography_supported,
            "sem_backend_consumes_detector_direction": detector_direction_role != "not_used",
            "sem_sample_environment_z_policy": (
                "surface_source_first_slice"
                if self._sem_effective_source_representation == "volume"
                else "projected_surface_source"
            ),
            "sem_sample_environment_uses_particle_world_z": False,
        }
        
    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        sem_support = get_source_volume_support(params, "sem")
        response.update({
            "kind": (
                "sem_interaction_volume_proxy"
                if self._sem_model == "interaction_volume_proxy"
                else (
                    "sem_physical_electron_transport"
                    if self._sem_model == "physical_electron_transport"
                    else "sem_gaussian_probe"
                )
            ),
            "sem_model": self._sem_model,
            "sem_backend": self._sem_backend,
            "measurement_domain": "electron_count",
            "signal_units": "electron_count",
            "contrast_frame_units": "electron_count_difference",
            "model_output_domain": "secondary_electron_yield",
            "model_signal_units": "dimensionless_yield",
            "pre_count_contrast_units": "secondary_electron_yield_difference",
            "final_measurement_domain": "electron_count",
            "final_signal_units": "electron_count",
            "count_scaling_mode": "sem_electrons_per_pixel",
            "forward_observable": (
                "Monte Carlo interaction-volume transport from sliced SEM source volume"
                if self._sem_effective_source_representation == "volume"
                else (
                    "Gaussian probe plus interaction-volume, edge, and topography secondary-electron proxy"
                    if self._sem_model == "interaction_volume_proxy"
                    else "Gaussian-probe blurred secondary-electron yield proxy"
                )
            ),
            "acceleration_kV": self._acceleration_kV,
            "probe_sigma_canvas_pixels": self._probe_sigma_px,
            "probe_sigma_nm": self._probe_sigma_nm,
            "interaction_volume_nm": self._interaction_volume_nm,
            "detector_direction_xy": [float(v) for v in self._detector_direction_xy],
            "probe_sigma_source": self._probe_sigma_source,
            "edge_contrast_gain": self._edge_gain,
            "bulk_contrast_gain": self._bulk_gain,
            "topography_contrast_gain": self._topography_gain,
            "baseline_yield": self._baseline,
            "electrons_per_pixel": SemSettings.from_params(params).electrons_per_pixel,
            "filter_guard_radius_pixels": self.filter_guard_radius_pixels(params),
            "source_input_kind": (
                "sliced_sem_source_volume"
                if self._sem_effective_source_representation == "volume"
                else "projected_2d_source_map"
            ),
            "source_map_ndim": 3 if self._sem_effective_source_representation == "volume" else 2,
            "source_axis_order": "zyx" if self._sem_effective_source_representation == "volume" else "yx",
            "source_z_frame_contract_version": SOURCE_Z_FRAME_CONTRACT_VERSION,
            "source_z_basis": self.particle_source_z_basis(params),
            "material_source_z_frame": self.particle_source_z_basis(params),
            "sem_material_source_z_contract": (
                "entry_surface_depth_required_for_volume_transport"
                if self._sem_effective_source_representation == "volume"
                else "projected_no_z_source"
            ),
            "source_z_origin": (
                self._sem_source_z_origin
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_depth_grid_contract_version": (
                self._sem_depth_grid.contract_version
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_depth_grid_offset_policy": (
                self._sem_depth_grid.offset_policy
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_z_offset_nm": self._sem_source_z_offset_nm,
            "source_slice_thickness_nm": (
                self._sem_depth_grid.slice_thickness_nm
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_z_edges_nm": (
                self._sem_depth_grid.edges_nm
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_z_planes_nm": (
                self._sem_depth_grid.centers_nm
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            # Volume SEM converts particle world z to entry-surface material
            # depth through SourceCoordinateContext; projected SEM intentionally
            # discards particle z at the source-map seam.
            "source_z_uses_particle_world_z": self._sem_effective_source_representation == "volume",
            "source_z_particle_world_to_entry_surface_policy": (
                "world_z_nm_is_entry_surface_depth_nm"
                if self._sem_effective_source_representation == "volume"
                else "projected_source_discards_particle_z"
            ),
            # The shared SEM source resolver owns requested-vs-effective basis
            # semantics.  An explicit volume request can no longer be silently
            # projected by a backend that consumes only 2D source maps.
            "source_representation_request_satisfied": self._sem_source_resolution.request_satisfied,
            "source_projection_policy": self._sem_source_projection_policy,
            "sem_source_backend_capability": self._sem_source_resolution.backend_source_capability,
            "sem_effective_source_representation": self._sem_effective_source_representation,
            "sem_source_representation_resolution_mode": (
                "explicit" if self._sem_source_resolution.requested_is_explicit else "auto"
            ),
            "sem_source_units": (
                "geometry_reference_normalized_slice_overlap_fraction"
                if self._sem_effective_source_representation == "volume"
                else "geometry_reference_normalized_projected_chord_fraction"
            ),
            "sem_source_normalization": (
                "per_slice_overlap_nm_divided_by_component_reference_length_nm"
                if self._sem_effective_source_representation == "volume"
                else "projected_chord_nm_divided_by_component_reference_length_nm"
            ),
            "sem_source_absolute_occupancy": False,
            "sem_source_physical_extent_claim": (
                "proxy_normalized_material_source_not_absolute_sphere_volume"
            ),
            "sem_sample_environment_source_dimensionality": (
                "sliced_volume_surface_layer"
                if self._sem_effective_source_representation == "volume"
                else "projected_2d"
            ),
            "sem_sample_environment_projection_policy": (
                "surface_yield_topography_first_source_slice"
                if self._sem_effective_source_representation == "volume"
                else "projected_2d_yield_topography"
            ),
            "backend_consumes_volume_source": self._sem_source_resolution.backend_consumes_volume_source,
            "volume_transport_model": (
                "3d_monte_carlo_interaction_kernel"
                if self._sem_effective_source_representation == "volume"
                else "none_projected_2d"
            ),
            "sem_requested_source_representation": self._sem_source_representation,
            "sem_volume_slices": self._sem_volume_slices,
            "sem_volume_slice_thickness_nm": self._sem_volume_slice_thickness_nm,
            "sem_volume_depth_nm": self._sem_volume_slices * self._sem_volume_slice_thickness_nm,
            "sem_source_volume_configured_slices": (
                int(sem_support.configured_slice_count)
                if self._sem_effective_source_representation == "volume" and sem_support is not None
                else None
            ),
            "sem_source_volume_required_slices_for_rendered_z": (
                int(sem_support.required_slice_count)
                if self._sem_effective_source_representation == "volume" and sem_support is not None
                else None
            ),
            "sem_source_z_envelope_min_nm": (
                float(sem_support.envelope_min_nm)
                if self._sem_effective_source_representation == "volume" and sem_support is not None
                else None
            ),
            "sem_source_z_envelope_max_nm": (
                float(sem_support.envelope_max_nm)
                if self._sem_effective_source_representation == "volume" and sem_support is not None
                else None
            ),
            "sem_source_z_support_policy": (
                str(sem_support.policy)
                if self._sem_effective_source_representation == "volume" and sem_support is not None
                else None
            ),
            "sem_source_z_preserved_configured_center": (
                bool(sem_support.preserved_configured_center)
                if self._sem_effective_source_representation == "volume" and sem_support is not None
                else None
            ),
            "fidelity_label": (
                "syniscopy_monte_carlo_sem_volume_transport"
                if self._sem_effective_source_representation == "volume"
                else (
                    "interaction_volume_secondary_electron_yield_proxy"
                    if self._sem_model == "interaction_volume_proxy"
                    else "sem_gaussian_probe_secondary_yield_proxy"
                )
            ),
        })
        response.update(self._backend_response_contract())
        if self._sem_transport_backend is not None:
            response.update(self._sem_transport_backend.metadata(params))
        elif self._sem_reference_kernel_backend is not None:
            response.update(self._sem_reference_kernel_backend.metadata(params))
        elif self._sem_reference_backend is not None:
            response.update(self._sem_reference_backend.metadata())
        else:
            response["sem_backend"] = self._sem_backend
            response = attach_backend_fidelity_metadata(
                response,
                params=params,
                backend_name=self._sem_backend,
                equations_or_model_family=(
                    "interaction-volume secondary electron proxy"
                    if self._sem_model == "interaction_volume_proxy"
                    else "gaussian probe secondary electron yield proxy"
                ),
                implemented_approximation_level="proxy",
                native_operating_assumptions="SEM secondary electron proxy on projected source with optional topography and probe blur",
                comparison_contract_id=response.get("comparison_contract_id", "Contract-NR"),
                artifact_provenance_id=response.get("artifact_provenance_id"),
            )
        # Backend metadata can report its raw requested value; the top-level
        # SEM model is the public owner of the effective numeric source basis.
        # Reapply the resolver metadata after backend metadata so downstream
        # reports cannot mistake auto/projected backends for z-y-x volume output.
        response.update(self._sem_source_resolution.metadata())
        settings = SemSettings.from_params(params)
        if self._sem_transport_backend is not None:
            response["electrons_per_pixel"] = self._sem_transport_backend.electrons_per_pixel()
            electron_source = "transport_backend"
        elif self._sem_reference_kernel_backend is not None:
            response["electrons_per_pixel"] = self._sem_reference_kernel_backend.electrons_per_pixel()
            electron_source = "reference_kernel_backend"
        else:
            response["electrons_per_pixel"], electron_source = self._configured_incident_electrons_per_pixel(params)
        if settings.beam_current_nA > 0.0 and settings.dwell_time_us > 0.0:
            response["count_scaling_mode"] = "sem_beam_current_nA_times_sem_dwell_time_us"
            response["incident_primary_electrons_source"] = electron_source
        else:
            response["count_scaling_mode"] = "sem_electrons_per_pixel"
            response["incident_primary_electrons_source"] = electron_source
        return response

    def probe_wavelength_nm(self, params: dict) -> float:
        return float(electron_wavelength_m(SemSettings.from_params(params).acceleration_kV) * 1.0e9)

    def filter_guard_radius_pixels(self, params: dict) -> int | None:
        backend_guard = None
        if self._sem_transport_backend is not None and hasattr(self._sem_transport_backend, "guard_radius_pixels"):
            backend_guard = float(self._sem_transport_backend.guard_radius_pixels())
        return SemSettings.from_params(params).filter_guard_radius_pixels(
            probe_sigma_px=self._probe_sigma_px,
            backend_guard_radius=backend_guard,
        )

    # -- Contract methods -------------------------------------------------

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Return the SEM secondary-electron yield image (dimensionless)."""
        rho = np.abs(E_sca_total) ** 2
        return self._yield_from_source(rho)

    def _proxy_numeric_source(self, source) -> np.ndarray:
        if isinstance(source, SEMMaterialSourceCanvas):
            out = np.zeros(source.shape, dtype=float)
            for key, value in source.channels.items():
                out += key.se_yield_coefficient * np.asarray(value, dtype=float)
            if out.ndim == 3 and self._sem_effective_source_representation != "volume":
                return np.sum(out, axis=0)
            return out
        return self._source_for_selected_backend(source)

    def _yield_from_source(self, source) -> np.ndarray:
        source = self._source_for_selected_backend(source)
        if self._sem_transport_backend is not None:
            transport_source = (
                source if self._sem_backend == "monte_carlo_physical" else self._proxy_numeric_source(source)
            )
            return self._sem_transport_backend.yield_from_source(transport_source, baseline=self._baseline)
        if self._sem_reference_kernel_backend is not None:
            return self._sem_reference_kernel_backend.yield_from_source(source, baseline=self._baseline)
        if self._sem_reference_backend is not None:
            return self._sem_reference_backend.yield_from_source(source, baseline=self._baseline)
        if self._sem_proxy_backend is None:
            raise RuntimeError("SEM proxy backend was not initialized.")
        return self._sem_proxy_backend.yield_from_source(
            self._proxy_numeric_source(source),
            baseline=self._baseline,
        )

    def _contrast_from_source(self, source) -> np.ndarray:
        source = self._source_for_selected_backend(source)
        if self._sem_transport_backend is not None:
            transport_source = (
                source if self._sem_backend == "monte_carlo_physical" else self._proxy_numeric_source(source)
            )
            return self._sem_transport_backend.contrast_from_source(transport_source)
        if self._sem_reference_kernel_backend is not None:
            return self._sem_reference_kernel_backend.contrast_from_source(source)
        if self._sem_reference_backend is not None:
            return self._sem_reference_backend.contrast_from_source(source)
        if self._sem_proxy_backend is None:
            raise RuntimeError("SEM proxy backend was not initialized.")
        return self._sem_proxy_backend.contrast_from_source(self._proxy_numeric_source(source))

    def _source_for_selected_backend(self, source):
        if isinstance(source, SEMMaterialSourceCanvas):
            if source.ndim == 3 and self._sem_effective_source_representation != "volume":
                projected = SEMMaterialSourceCanvas(shape=source.shape[-2:])
                for key, value in source.channels.items():
                    projected.channels[key] = np.sum(np.asarray(value, dtype=float), axis=0)
                return projected
            return source
        src = np.asarray(source, dtype=float)
        if src.ndim == 3 and self._sem_effective_source_representation != "volume":
            return np.sum(src, axis=0)
        return src

    def particle_source_z_basis(self, params: dict) -> str:
        del params
        if self._sem_effective_source_representation != "volume":
            return SOURCE_Z_BASIS_PROJECTED_NO_Z
        # SEM z-y-x source volumes are material-density/yield volumes. Focus-relative
        # z belongs to the imaging response/probe-defocus contract, so allowing it
        # here would translate static material density through transport-depth kernels.
        require_source_density_z_basis(
            SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH,
            allowed_bases={SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH},
            source_input_kind="sliced_sem_source_volume",
            modality_name="sem_secondary_electron",
            backend_name=self._sem_backend,
        )
        return SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del params
        if self._sem_effective_source_representation == "volume":
            return SEMMaterialSourceCanvas(shape=(self._sem_volume_slices, *shape))
        return SEMMaterialSourceCanvas(shape=shape)

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
        orientation_matrix=None,
    ) -> None:
        del params
        if component_geometry is None:
            raise ValueError("SEM source accumulation requires component_geometry.")
        if source_coordinate_context is not None:
            particle_z_nm = source_coordinate_context.source_density_z_nm
        if source_canvas is None:
            return
        if isinstance(source_canvas, SEMMaterialSourceCanvas):
            target_canvas = source_canvas.channel_for(material_properties)
        else:
            target_canvas = source_canvas
        if np.asarray(target_canvas).ndim == 3:
            _, h, w = target_canvas.shape
        else:
            h, w = target_canvas.shape
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
        reference_length_nm = float(component_geometry.source_normalization_length_nm)
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        if np.asarray(target_canvas).ndim == 3:
            context_basis = (
                source_coordinate_context.source_density_z_basis
                if source_coordinate_context is not None
                else self.particle_source_z_basis({})
            )
            require_source_density_z_basis(
                context_basis,
                allowed_bases={SOURCE_Z_BASIS_ENTRY_SURFACE_DEPTH},
                source_input_kind="sliced_sem_source_volume",
                modality_name="sem_secondary_electron",
                backend_name=self._sem_backend,
            )

            if particle_z_nm is None:
                raise ValueError(
                    "SEM volume source accumulation requires resolved entry-surface "
                    "particle depth; projected SEM sources are the only SEM source "
                    "basis allowed to discard particle z."
                )
            entry_surface_depth_center_nm = float(particle_z_nm)
            if not np.isfinite(entry_surface_depth_center_nm):
                raise ValueError(
                    "SEM volume entry-surface particle depth must be finite; "
                    f"got {particle_z_nm!r}."
                )

            def overlap_with_slice(z_lower_rel_nm, z_upper_rel_nm, slice_z0, slice_z1):
                # The source stack uses physical entry-surface depth.  The sphere
                # cross-section generalized here is an oriented primitive interval
                # in world/source z, not a diameter-derived symmetric chord.
                return np.maximum(
                    np.minimum(entry_surface_depth_center_nm + z_upper_rel_nm, slice_z1)
                    - np.maximum(entry_surface_depth_center_nm + z_lower_rel_nm, slice_z0),
                    0.0,
                )
            for slice_idx in range(target_canvas.shape[0]):
                # SEMDepthGrid owns sem_source_z_offset_nm.  Particle/source bounds
                # stay in physical entry-surface depth, while slice_bounds_nm()
                # returns the same shifted interval reported in metadata and used
                # by transport kernels.  Do not add the offset to both quantities.
                slice_z0, slice_z1 = self._sem_depth_grid.slice_bounds_nm(slice_idx)
                overlap_nm = footprint.average_over_samples(
                    lambda z_lower_rel_nm, z_upper_rel_nm: overlap_with_slice(
                        z_lower_rel_nm,
                        z_upper_rel_nm,
                        slice_z0,
                        slice_z1,
                    )
                )
                if np.any(overlap_nm > 0.0):
                    target_canvas[slice_idx, footprint.y0:footprint.y1, footprint.x0:footprint.x1] += (
                        multiplier
                        * (overlap_nm / max(reference_length_nm, 1e-12))
                    )
            return
        target_canvas[footprint.y0:footprint.y1, footprint.x0:footprint.x1] += (
            multiplier * (projected_chord_nm / max(reference_length_nm, 1e-12))
        )

    def _merged_source_from_particles(self, particle_source_maps, E_sca_total):
        if particle_source_maps is None or len(particle_source_maps) == 0:
            if self._sem_backend == "monte_carlo_physical":
                image_shape = tuple(np.asarray(E_sca_total).shape[-2:])
                if self._sem_effective_source_representation == "volume":
                    return SEMMaterialSourceCanvas(shape=(self._sem_volume_slices, *image_shape))
                return SEMMaterialSourceCanvas(shape=image_shape)
            return np.abs(E_sca_total) ** 2
        return source_like_sum(particle_source_maps)

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
        del E_sca_particles, particle_instances, background_field, params, frame_index
        source = self._merged_source_from_particles(particle_source_maps, E_sca_total)
        return self._yield_from_source(source)

    def _sample_environment_source(
        self,
        sample_environment: SampleEnvironment,
        params: dict,
        image_shape: tuple[int, int],
    ) -> SEMMaterialSourceCanvas:
        topo = np.asarray(sample_environment.substrate.topography_gradient(), dtype=float)
        frac = np.asarray(sample_environment.substrate.material_fraction_map, dtype=float)
        height = np.asarray(sample_environment.substrate.height_map_nm, dtype=float)
        if topo.shape != tuple(image_shape) or frac.shape != tuple(image_shape):
            raise ValueError(
                "SEM sample-environment maps must match the SEM source image shape; "
                f"got topography {topo.shape}, material fraction {frac.shape}, expected {tuple(image_shape)}."
            )
        edge_gain = SampleEnvironmentSettings.from_params(params).sem_edge_gain
        frac = np.where(height > 0.0, frac, 0.0)
        layer_source = frac + edge_gain * topo
        substrate_source = 1.0 - frac
        if self._sem_effective_source_representation != "volume":
            source = SEMMaterialSourceCanvas(shape=tuple(image_shape))
            source.channel_for(sample_environment.substrate.material_layer)[:, :] += layer_source
            source.channel_for(sample_environment.substrate.material_substrate)[:, :] += substrate_source
            return source
        source = SEMMaterialSourceCanvas(shape=(self._sem_volume_slices, *tuple(image_shape)))
        # Sample-environment topography is a surface source at physical z=0 nm;
        # route it through SEMDepthGrid so a nonzero source slice-grid offset
        # cannot silently relabel a surface layer as a deeper material slice.
        surface_slice_idx = self._sem_depth_grid.surface_slice_index()
        source.channel_for(sample_environment.substrate.material_layer)[surface_slice_idx] += layer_source
        source.channel_for(sample_environment.substrate.material_substrate)[surface_slice_idx] += substrate_source
        return source

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
        del E_sca_particles, particle_instances, background_field, frame_index
        source = self._merged_source_from_particles(particle_source_maps, E_sca_total)
        if sample_environment is not None:
            source_arr = source_like_numeric_array(source)
            image_shape = tuple(source_arr.shape[-2:])
            environment_source = self._sample_environment_source(sample_environment, params, image_shape)
            if isinstance(source, SEMMaterialSourceCanvas):
                source = source_like_sum([source, environment_source])
            elif np.any(source_arr):
                raise ValueError(
                    "SEM sample-environment rendering with non-material particle sources is unsupported."
                )
            else:
                source = environment_source
        return self._yield_from_source(source)

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
        image_shape = tuple(np.asarray(intensity).shape[-2:])
        substrate = self._sample_environment_source(sample_environment, params, image_shape)
        substrate_contrast = self._contrast_from_source(substrate)
        return np.maximum(intensity + substrate_contrast, 0.0)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Per-particle SEM contrast: baseline-subtracted SE image."""
        rho = np.abs(E_sca_particle) ** 2
        return self._contrast_from_source(rho)

    def _sem_direct_detector_scale(self, params: dict) -> tuple[float, str]:
        if self._sem_transport_backend is not None:
            return float(self._sem_transport_backend.electrons_per_pixel()), "transport_backend_electrons_per_pixel"
        if self._sem_reference_kernel_backend is not None:
            return float(self._sem_reference_kernel_backend.electrons_per_pixel()), "reference_kernel_electrons_per_pixel"
        electrons_per_pixel, source = self._configured_incident_electrons_per_pixel(params)
        return float(electrons_per_pixel), source

    def _direct_signal_product_from_source(
        self,
        source,
        params: dict,
        *,
        producer: str,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        # The fix-site invariant is detector-transfer ownership: SEM source
        # responses are secondary-electron yield/contrast, while count-domain
        # Fisher derivatives must use the primary-electron dose multiplied by
        # that yield.  Keeping this transfer in a typed product prevents the
        # same source map from being interpreted as both yield and electrons.
        yield_delta = self._contrast_from_source(source)
        scale, scale_source = self._sem_direct_detector_scale(params)
        electron_delta = scale * np.asarray(yield_delta, dtype=float)
        return DirectParticleSignalProduct(
            values=electron_delta,
            representation=electron_count_delta_representation(),
            modality="sem_secondary_electron",
            producer=producer,
            safe_for_fisher=True,
            detector_scale_applied=True,
            background_included=False,
            source_representation=sem_secondary_electron_source_representation(),
            detector_scale_factor=float(scale),
            conversion_note=(
                "Converted SEM secondary-electron yield response to electron-count "
                "contribution using the active beam-dose owner."
            ),
            provenance={"frame_index": int(frame_index), "detector_scale_source": scale_source},
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
            "SEM direct particle contrast no longer returns a bare array. Use "
            "compute_particle_signal_product(); its metadata records the "
            "secondary-yield to electron-count transfer before Fisher use."
        )

    def _params_with_direct_sem_source_support(
        self,
        params: dict,
        *,
        particle_z_nm: float,
        component_geometry,
    ) -> tuple[dict, bool]:
        if self._sem_effective_source_representation != "volume":
            return params, False
        if get_source_volume_support(params, "sem") is not None:
            return params, False
        radius_nm = float(component_geometry.axial_half_extent_nm(None))
        center_nm = float(particle_z_nm)
        if not np.isfinite(radius_nm) or radius_nm < 0.0 or not np.isfinite(center_nm):
            raise ValueError(
                "Direct SEM source support requires finite non-negative radius and "
                f"finite entry-surface center depth; got radius={radius_nm!r}, z={center_nm!r}."
            )
        support = sem_source_volume_support_from_params(
            params,
            backend_name=self._sem_backend,
            envelope_min_nm=center_nm - radius_nm,
            envelope_max_nm=center_nm + radius_nm,
            policy="auto_from_direct_sem_particle_envelope",
        )
        resolved = config_without_runtime_state(params)
        # Direct single-particle SEM signal products do not pass through the
        # frame-loop run-scope resolver.  Recreate the same internal support
        # contract here before allocating the source canvas or backend kernels;
        # otherwise diagnostics/Fisher products can clip material even though
        # video rendering is correct.
        set_source_volume_support(resolved, "sem", support)
        return resolved, True


    def compute_particle_signal_product(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        if particle_instance is None:
            rho = np.abs(E_sca_particle) ** 2
            return self._direct_signal_product_from_source(
                rho,
                params,
                producer="ScanningElectronMicroscopyImagingModel.compute_particle_signal_product",
                frame_index=frame_index,
            )
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct SEM particle signal for composite particles requires a "
                "rendered source map; use compute_particle_signal_product_from_source_map()."
            )
        material = getattr(particle_instance, "material_properties", None)
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(np.clip(int(frame_index), 0, traj.shape[0] - 1))
        sampling = SemSettings.from_params(params).sampling
        px = float(traj[frame_idx, 0]) / sampling.detector_pixel_size_nm * float(sampling.psf_oversampling_factor)
        py = float(traj[frame_idx, 1]) / sampling.detector_pixel_size_nm * float(sampling.psf_oversampling_factor)
        pz = float(traj[frame_idx, 2]) if traj.shape[1] >= 3 else 0.0
        params, rebound_required = self._params_with_direct_sem_source_support(
            params,
            particle_z_nm=pz,
            component_geometry=particle_instance.component_geometry,
        )
        if rebound_required:
            resolved_model = type(self)(params)
            return resolved_model.compute_particle_signal_product(
                E_sca_particle,
                background_field,
                params,
                particle_instance=particle_instance,
                frame_index=frame_idx,
            )
        source = self.initialize_particle_source_canvas(E_sca_particle.shape[-2:], params)
        # Direct single-particle SEM source rendering must use the same
        # entry-surface material-depth resolver as the frame-loop path; otherwise
        # response diagnostics and rendered scenes would disagree on z placement.
        source_coordinate_context = SourceCoordinateContext.from_particle_z(
            particle_world_z_nm=pz,
            focus_plane_z_nm=FocusPlaneState.from_params(params).z_nm,
            source_density_z_basis=self.particle_source_z_basis(params),
            optical_response_z_basis="projected_no_z",
        )
        self.accumulate_particle_source(
            source,
            center_x_canvas=px,
            center_y_canvas=py,
            diameter_nm=float(particle_instance.particle_type.diameter_nm),
            pixel_size_nm=sampling.detector_pixel_size_nm,
            os_factor=sampling.psf_oversampling_factor,
            material_properties=material,
            params=params,
            particle_z_nm=source_coordinate_context.source_density_z_nm,
            source_coordinate_context=source_coordinate_context,
            component_geometry=particle_instance.component_geometry,
            orientation_matrix=None,
        )
        return self._direct_signal_product_from_source(
            source,
            params,
            producer="ScanningElectronMicroscopyImagingModel.compute_particle_signal_product",
            frame_index=frame_idx,
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
            "SEM source-map contrast no longer returns a bare array. Use "
            "compute_particle_signal_product_from_source_map(); it converts "
            "secondary-electron yield to electron-count contribution exactly once."
        )

    def compute_particle_signal_product_from_source_map(
        self,
        particle_source_map,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        del background_field
        return self._direct_signal_product_from_source(
            particle_source_map,
            params,
            producer="ScanningElectronMicroscopyImagingModel.compute_particle_signal_product_from_source_map",
            frame_index=frame_index,
        )

    def convert_model_output_to_detector_frame(
        self,
        model_output: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Convert dimensionless SE yield to detector electron counts.

        SEM detectors report integrated electron counts per pixel; we
        multiply the dimensionless yield by the per-pixel dose.  No
        reference-beam division is involved (no E_ref in SEM).
        """
        if self._sem_transport_backend is not None:
            electrons_per_pixel = self._sem_transport_backend.electrons_per_pixel()
        elif self._sem_reference_kernel_backend is not None:
            electrons_per_pixel = self._sem_reference_kernel_backend.electrons_per_pixel()
        else:
            electrons_per_pixel, _ = self._configured_incident_electrons_per_pixel(params)
        return convert_model_output_to_detector_frame(
            model_output=model_output,
            background_frame=background_final,
            reference_intensity_frame=E_ref_intensity_final,
            conversion=DetectorFrameConversion(
                model_output_domain=MODEL_OUTPUT_DOMAIN_ELECTRON_YIELD,
                detector_output_domain=DETECTOR_OUTPUT_DOMAIN_ELECTRON_COUNT,
                value_form=VALUE_FORM_ABSOLUTE,
                reference_basis=REFERENCE_BASIS_NONE,
                scale=electrons_per_pixel,
                measurement_domain="electron_count",
                signal_units="electron_count",
            ),
            params=params,
            context="ScanningElectronMicroscopyImagingModel.convert_model_output_to_detector_frame",
        )

__all__ = ['ScanningElectronMicroscopyImagingModel']
