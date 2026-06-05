"""sem imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    np,
)
from backend_fidelity import attach_backend_fidelity_metadata
from config.runtime import SemSettings, param_value
from .electron_constants import electron_wavelength_m
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

    Parameters (PARAMS keys, optional, nominal defaults)
    ----------------------------------------------------
    The defaults set a stable moderate-contrast synthetic SEM regime; use
    calibrated values for instrument-specific studies.

    - ``sem_probe_sigma_pixels``    (default 1.0) Gaussian probe spot size
    - ``sem_edge_contrast_gain``    (default 10.0) weight on the gradient-
                                    magnitude term (secondary-emission edge
                                    enhancement).
    - ``sem_bulk_contrast_gain``    (default 1.0) weight on the material
                                    source term (bulk/Z-like contribution).
    - ``sem_baseline_yield``        (default 0.05) yield from the substrate
                                    with no particle present.
    - ``sem_electrons_per_pixel``   (default 1000.0) dose scale used by
                                    scale_intensity_to_counts.
    """

    output_type = "intensity"
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_complex_optical_psf = False
    requires_optical_scattered_field = False
    requires_pre_crop_optical_filtering = True
    supports_spectral_channels = False

    def __init__(self, params: dict) -> None:
        settings = SemSettings.from_params(params)
        _sem_model_raw = str(
            param_value(params, "sem_model")
        ).strip().lower()
        self._sem_model = _sem_model_raw
        if self._sem_model not in {
            "gaussian_probe_secondary_yield",
            "interaction_volume_proxy",
            "physical_electron_transport",
        }:
            raise ValueError(
                "PARAMS['sem_model'] must be 'gaussian_probe_secondary_yield' "
                "'interaction_volume_proxy', or 'physical_electron_transport'; "
                f"got {self._sem_model!r}."
            )
        self._sem_backend = str(param_value(params, "sem_backend")).strip().lower()
        if self._sem_backend not in {
            "gaussian_probe_proxy",
            "interaction_volume_proxy",
            "monte_carlo_transport",
            "monte_carlo_physical",
            "syniscopy_transport_lite",
            "reference_kernel_table",
        }:
            raise ValueError(
                "PARAMS['sem_backend'] must be 'gaussian_probe_proxy', "
                f"'interaction_volume_proxy', 'monte_carlo_transport', 'monte_carlo_physical', "
                "'syniscopy_transport_lite', or 'reference_kernel_table'; got "
                f"{self._sem_backend!r}."
            )
        self._sem_transport_backend = None
        self._sem_reference_kernel_backend = None
        self._sem_reference_backend = None
        self._sem_proxy_backend = None
        if self._sem_model == "interaction_volume_proxy" and self._sem_backend != "interaction_volume_proxy":
            raise ValueError(
                "PARAMS['sem_model']='interaction_volume_proxy' requires "
                "PARAMS['sem_backend']='interaction_volume_proxy'."
            )
        if self._sem_backend == "interaction_volume_proxy" and self._sem_model != "interaction_volume_proxy":
            raise ValueError(
                "PARAMS['sem_backend']='interaction_volume_proxy' requires "
                "PARAMS['sem_model']='interaction_volume_proxy'."
            )
        if self._sem_model == "physical_electron_transport" and self._sem_backend != "monte_carlo_physical":
            raise ValueError(
                "PARAMS['sem_model']='physical_electron_transport' requires "
                "PARAMS['sem_backend']='monte_carlo_physical'."
            )
        if self._sem_backend == "monte_carlo_physical" and self._sem_model != "physical_electron_transport":
            raise ValueError(
                "PARAMS['sem_backend']='monte_carlo_physical' requires "
                "PARAMS['sem_model']='physical_electron_transport'."
            )
        self._sem_source_representation = str(param_value(params, "sem_source_representation")).strip().lower()
        if self._sem_source_representation not in {"projected", "volume"}:
            raise ValueError(
                "PARAMS['sem_source_representation'] must be 'projected' or "
                f"'volume'; got {self._sem_source_representation!r}."
            )
        self._sem_effective_source_representation = (
            "volume"
            if self._sem_backend in {"monte_carlo_transport", "monte_carlo_physical"}
            and self._sem_source_representation == "volume"
            else "projected"
        )
        self._sem_source_projection_policy = (
            "backend_native_volume_transport"
            if self._sem_effective_source_representation == "volume"
            else (
                "user_selected_projected_source"
                if self._sem_source_representation == "projected"
                else "projected_for_backend_without_volume_transport"
            )
        )
        self._sem_source_z_origin = settings.source_z_origin
        if self._sem_source_z_origin not in {"entry_surface_depth", "focus_plane_relative"}:
            raise ValueError(
                "PARAMS['sem_source_z_origin'] must be 'entry_surface_depth' or "
                "'focus_plane_relative'."
            )
        self._sem_source_z_offset_nm = settings.source_z_offset_nm
        self._sem_volume_slices = settings.volume_slices
        canvas_pitch_nm = settings.sampling.model_canvas_pixel_size_nm
        if not np.isfinite(canvas_pitch_nm) or canvas_pitch_nm <= 0.0:
            raise ValueError(
                "PARAMS['pixel_size_nm'] / PARAMS['psf_oversampling_factor'] "
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
        self._interaction_volume_nm = float(param_value(params, "sem_interaction_volume_nm"))
        if not np.isfinite(self._interaction_volume_nm) or self._interaction_volume_nm < 0.0:
            raise ValueError(
                "PARAMS['sem_interaction_volume_nm'] must be finite and non-negative; "
                f"got {self._interaction_volume_nm}."
            )
        direction = np.asarray(param_value(params, "sem_detector_direction_xy"), dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("PARAMS['sem_detector_direction_xy'] must be a finite length-2 vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("PARAMS['sem_detector_direction_xy'] must have nonzero norm.")
        self._detector_direction_xy = direction / norm
        self._edge_gain = float(param_value(params, "sem_edge_contrast_gain"))
        self._bulk_gain = float(param_value(params, "sem_bulk_contrast_gain"))
        self._topography_gain = float(param_value(params, "sem_topography_contrast_gain"))
        self._baseline = float(param_value(params, "sem_baseline_yield"))
        if self._baseline < 0.0:
            raise ValueError(
                f"PARAMS['sem_baseline_yield'] must be non-negative; "
                f"got {self._baseline}."
            )
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
                    edge_gain=self._edge_gain,
                    bulk_gain=self._bulk_gain,
                    baseline=self._baseline,
                )
        slice_raw = param_value(params, "sem_volume_slice_thickness_nm")
        if slice_raw is None:
            interaction_depth = float(param_value(params, "sem_interaction_volume_nm"))
            self._sem_volume_slice_thickness_nm = max(
                interaction_depth / float(self._sem_volume_slices),
                1e-9,
            )
        else:
            self._sem_volume_slice_thickness_nm = float(slice_raw)
            if (
                not np.isfinite(self._sem_volume_slice_thickness_nm)
                or self._sem_volume_slice_thickness_nm <= 0.0
            ):
                raise ValueError(
                    "PARAMS['sem_volume_slice_thickness_nm'] must be positive when supplied."
                )

    def _backend_response_contract(self) -> dict:
        if self._sem_backend == "gaussian_probe_proxy":
            edge_convention = "gradient_magnitude"
            topography_convention = "not_supported"
            detector_direction_role = "not_used"
            topography_supported = False
            detector_direction_used = False
        elif self._sem_backend == "interaction_volume_proxy":
            edge_convention = "positive_directed_detector_gradient"
            topography_convention = "gradient_magnitude"
            detector_direction_role = "edge_term_positive_projection"
            topography_supported = True
            detector_direction_used = self._edge_gain > 0.0
        else:
            edge_convention = "gradient_magnitude"
            topography_convention = "absolute_directed_detector_gradient"
            detector_direction_role = "topography_term_absolute_projection"
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
        response.update({
            "kind": "sem_interaction_volume_proxy" if self._sem_model == "interaction_volume_proxy" else "sem_gaussian_probe",
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
            "acceleration_kV": float(param_value(params, "sem_acceleration_kV")),
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
            "source_z_origin": (
                self._sem_source_z_origin
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_z_offset_nm": self._sem_source_z_offset_nm,
            "source_slice_thickness_nm": (
                self._sem_volume_slice_thickness_nm
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_z_planes_nm": (
                [
                    (idx + 0.5) * self._sem_volume_slice_thickness_nm
                    + self._sem_source_z_offset_nm
                    for idx in range(self._sem_volume_slices)
                ]
                if self._sem_effective_source_representation == "volume"
                else None
            ),
            "source_z_uses_particle_world_z": (
                self._sem_source_z_origin == "focus_plane_relative"
                and self._sem_effective_source_representation == "volume"
            ),
            "source_representation_request_satisfied": (
                self._sem_source_representation == self._sem_effective_source_representation
            ),
            "source_projection_policy": self._sem_source_projection_policy,
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
            "backend_consumes_volume_source": self._sem_effective_source_representation == "volume",
            "volume_transport_model": (
                "3d_monte_carlo_interaction_kernel"
                if self._sem_effective_source_representation == "volume"
                else "none_projected_2d"
            ),
            "sem_source_representation": self._sem_effective_source_representation,
            "sem_requested_source_representation": self._sem_source_representation,
            "sem_volume_slices": self._sem_volume_slices,
            "sem_volume_slice_thickness_nm": self._sem_volume_slice_thickness_nm,
            "sem_volume_depth_nm": self._sem_volume_slices * self._sem_volume_slice_thickness_nm,
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
        beam_current_nA = float(param_value(params, "sem_beam_current_nA"))
        dwell_time_us = float(param_value(params, "sem_dwell_time_us"))
        if self._sem_transport_backend is not None:
            response["electrons_per_pixel"] = self._sem_transport_backend.electrons_per_pixel()
        elif self._sem_reference_kernel_backend is not None:
            response["electrons_per_pixel"] = self._sem_reference_kernel_backend.electrons_per_pixel()
        else:
            response["electrons_per_pixel"] = SemSettings.from_params(params).electrons_per_pixel
        if beam_current_nA > 0.0 and dwell_time_us > 0.0:
            response["count_scaling_mode"] = "sem_beam_current_nA_times_sem_dwell_time_us"
            response["incident_primary_electrons_source"] = "beam_current_and_dwell_time"
        else:
            response["count_scaling_mode"] = "sem_electrons_per_pixel"
            response["incident_primary_electrons_source"] = "sem_electrons_per_pixel"
        return response

    def probe_wavelength_nm(self, params: dict) -> float:
        acceleration_kV = float(param_value(params, 'sem_acceleration_kV'))
        return float(electron_wavelength_m(acceleration_kV) * 1.0e9)

    def filter_guard_radius_pixels(self, params: dict) -> int | None:
        raw = param_value(params, 'sem_filter_guard_pixels')
        if raw is None:
            raw = max(4.0 * self._probe_sigma_px, 2.0)
            if self._sem_transport_backend is not None and hasattr(self._sem_transport_backend, "guard_radius_pixels"):
                raw = max(raw, float(self._sem_transport_backend.guard_radius_pixels()))
        guard = float(raw)
        if not np.isfinite(guard) or guard < 0.0:
            raise ValueError(
                "PARAMS['sem_filter_guard_pixels'] must be non-negative and finite; "
                f"got {raw!r}."
            )
        return int(np.ceil(guard))

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
            return self._sem_reference_kernel_backend.yield_from_source(
                self._proxy_numeric_source(source),
                baseline=self._baseline,
            )
        if self._sem_reference_backend is not None:
            return self._sem_reference_backend.yield_from_source(
                self._proxy_numeric_source(source),
                baseline=self._baseline,
            )
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
            return self._sem_reference_kernel_backend.contrast_from_source(self._proxy_numeric_source(source))
        if self._sem_reference_backend is not None:
            return self._sem_reference_backend.contrast_from_source(self._proxy_numeric_source(source))
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
        source_multiplier: float = 1.0,
    ) -> None:
        del params
        if source_canvas is None:
            return
        radius_px = max(0.5, 0.5 * float(diameter_nm) / float(pixel_size_nm) * float(os_factor))
        if isinstance(source_canvas, SEMMaterialSourceCanvas):
            target_canvas = source_canvas.channel_for(material_properties)
        else:
            target_canvas = source_canvas
        if np.asarray(target_canvas).ndim == 3:
            _, h, w = target_canvas.shape
        else:
            h, w = target_canvas.shape
        x0 = max(0, int(np.floor(center_x_canvas - radius_px - 2)))
        x1 = min(w, int(np.ceil(center_x_canvas + radius_px + 3)))
        y0 = max(0, int(np.floor(center_y_canvas - radius_px - 2)))
        y1 = min(h, int(np.ceil(center_y_canvas + radius_px + 3)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
        dx = xx + x0 - float(center_x_canvas)
        dy = yy + y0 - float(center_y_canvas)
        r = np.sqrt(dx * dx + dy * dy)
        inside = r <= radius_px
        thickness_px = np.zeros_like(r, dtype=float)
        thickness_px[inside] = 2.0 * np.sqrt(np.maximum(radius_px ** 2 - r[inside] ** 2, 0.0))
        edge_width = max(0.75, 0.5 * float(os_factor))
        taper = np.clip((radius_px + edge_width - r) / max(edge_width, 1e-9), 0.0, 1.0)
        # Normalize by diameter so ``se_yield_coefficient`` remains the main
        # material-scale control rather than growing quadratically with size.
        diameter_px = max(2.0 * radius_px, 1.0)
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        if np.asarray(target_canvas).ndim == 3:
            radius_nm = 0.5 * float(diameter_nm)
            lateral_nm = r * (float(pixel_size_nm) / float(os_factor))
            chord_half_nm = np.zeros_like(lateral_nm, dtype=float)
            inside_nm = lateral_nm <= radius_nm
            chord_half_nm[inside_nm] = np.sqrt(
                np.maximum(radius_nm * radius_nm - lateral_nm[inside_nm] ** 2, 0.0)
            )
            if self._sem_source_z_origin == "focus_plane_relative":
                z_center_nm = (
                    float(particle_z_nm) if particle_z_nm is not None else 0.0
                ) + self._sem_source_z_offset_nm
                z_top_nm = z_center_nm - chord_half_nm
                z_bottom_nm = z_center_nm + chord_half_nm
            else:
                z_top_nm = radius_nm - chord_half_nm + self._sem_source_z_offset_nm
                z_bottom_nm = radius_nm + chord_half_nm + self._sem_source_z_offset_nm
            for slice_idx in range(target_canvas.shape[0]):
                slice_z0 = slice_idx * self._sem_volume_slice_thickness_nm
                slice_z1 = slice_z0 + self._sem_volume_slice_thickness_nm
                overlap_nm = np.maximum(
                    np.minimum(z_bottom_nm, slice_z1) - np.maximum(z_top_nm, slice_z0),
                    0.0,
                )
                if np.any(overlap_nm > 0.0):
                    target_canvas[slice_idx, y0:y1, x0:x1] += (
                        multiplier
                        * (overlap_nm / max(float(diameter_nm), 1e-12))
                        * taper
                    )
            return
        target_canvas[y0:y1, x0:x1] += multiplier * (thickness_px / diameter_px) * taper

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
        edge_gain = float(param_value(params, "sem_sample_environment_edge_gain"))
        frac = np.where(height > 0.0, frac, 0.0)
        layer_source = frac + edge_gain * topo
        substrate_source = 1.0 - frac
        if self._sem_effective_source_representation != "volume":
            source = SEMMaterialSourceCanvas(shape=tuple(image_shape))
            source.channel_for(sample_environment.substrate.material_layer)[:, :] += layer_source
            source.channel_for(sample_environment.substrate.material_substrate)[:, :] += substrate_source
            return source
        source = SEMMaterialSourceCanvas(shape=(self._sem_volume_slices, *tuple(image_shape)))
        source.channel_for(sample_environment.substrate.material_layer)[0] += layer_source
        source.channel_for(sample_environment.substrate.material_substrate)[0] += substrate_source
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

    def compute_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        if particle_instance is None:
            return self.compute_per_particle_contrast(E_sca_particle, background_field, params)
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct SEM particle contrast for composite particles requires a "
                "rendered source map; use compute_particle_contrast_from_source_map()."
            )
        del background_field
        source = self.initialize_particle_source_canvas(E_sca_particle.shape[-2:], params)
        material = getattr(particle_instance, "material_properties", None)
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(np.clip(int(frame_index), 0, traj.shape[0] - 1))
        sampling = SemSettings.from_params(params).sampling
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
        )
        return self._contrast_from_source(source)

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del background_field, params, frame_index
        return self._contrast_from_source(np.asarray(particle_source_map, dtype=float))

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
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
            return self._sem_transport_backend.electrons_per_pixel() * intensity
        if self._sem_reference_kernel_backend is not None:
            return self._sem_reference_kernel_backend.electrons_per_pixel() * intensity
        return SemSettings.from_params(params).electrons_per_pixel * intensity

__all__ = ['ScanningElectronMicroscopyImagingModel']
