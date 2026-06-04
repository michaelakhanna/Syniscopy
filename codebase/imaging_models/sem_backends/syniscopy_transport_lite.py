"""SyniscopyTransportSEMBackend SEM backend."""

from __future__ import annotations

from config.runtime import SemSettings, param_value

from ._metadata import (
    Any,
    SEMTransportBackendError,
    SEMTransportMetadata,
    _electrons_from_beam_current,
    _finite_nonnegative,
    _gaussian_blur,
    _gradient_components,
    _gradient_magnitude,
    attach_backend_fidelity_metadata,
    np,
)

class SyniscopyTransportSEMBackend:
    """Syniscopy-native raster-transport SEM backend (lite).

    This backend models SEM secondary-electron yield as a raster-scanned probe
    interacting with a projected secondary-emission source, then converting to
    expected detected counts with explicit beam-energy, dwell-time, and current
    bookkeeping.
    """

    backend_mode = "syniscopy_transport_lite"

    def __init__(
        self,
        params: dict,
        *,
        canvas_pitch_nm: float,
        probe_sigma_px: float,
    ) -> None:
        self.canvas_pitch_nm = _finite_nonnegative("canvas_pitch_nm", canvas_pitch_nm, minimum=1e-12)
        self.probe_sigma_px = _finite_nonnegative("sem probe sigma", probe_sigma_px, minimum=0.0)
        self.backend_mode = self.__class__.backend_mode
        self._acceleration_kV = _finite_nonnegative("sem_acceleration_kV", param_value(params, 'sem_acceleration_kV'), minimum=1e-9)
        self._baseline = _finite_nonnegative("sem_baseline_yield", param_value(params, 'sem_baseline_yield'), minimum=0.0)
        self._edge_gain = _finite_nonnegative("sem_edge_contrast_gain", param_value(params, 'sem_edge_contrast_gain'), minimum=0.0)
        self._bulk_gain = _finite_nonnegative("sem_bulk_contrast_gain", param_value(params, 'sem_bulk_contrast_gain'), minimum=0.0)
        self._topography_gain = _finite_nonnegative(
            "sem_topography_contrast_gain",
            param_value(params, 'sem_topography_contrast_gain'),
            minimum=0.0,
        )
        self._detector_acceptance = _finite_nonnegative(
            "sem_detector_acceptance",
            param_value(params, 'sem_detector_acceptance'),
            minimum=0.0,
        )
        self._takeoff_angle_deg = _finite_nonnegative(
            "sem_detector_takeoff_angle_deg",
            param_value(params, 'sem_detector_takeoff_angle_deg'),
            minimum=0.0,
        )
        self._escape_depth_nm = _finite_nonnegative(
            "sem_escape_depth_nm",
            param_value(params, 'sem_escape_depth_nm'),
            minimum=0.0,
        )
        self._backscatter_fraction = _finite_nonnegative(
            "sem_backscatter_fraction",
            param_value(params, 'sem_backscatter_fraction'),
            minimum=0.0,
        )
        self._material_scale = _finite_nonnegative(
            "sem_transport_material_scale",
            param_value(params, 'sem_transport_material_scale'),
            minimum=0.0,
        )
        self._source_exponent = _finite_nonnegative(
            "sem_transport_source_exponent",
            param_value(params, 'sem_transport_source_exponent'),
            minimum=0.05,
        )
        self._topography_source_exponent = _finite_nonnegative(
            "sem_transport_topography_exponent",
            param_value(params, 'sem_transport_topography_exponent'),
            minimum=0.05,
        )
        self._beam_current_nA = _finite_nonnegative("sem_beam_current_nA", param_value(params, 'sem_beam_current_nA'), minimum=0.0)
        self._dwell_time_us = _finite_nonnegative(
            "sem_dwell_time_us",
            param_value(params, 'sem_dwell_time_us'),
            minimum=0.0,
        )
        self._electrons_per_pixel_reference = _finite_nonnegative(
            "sem_electrons_per_pixel",
            SemSettings.from_params(params).electrons_per_pixel,
            minimum=0.0,
        )
        self._beam_energy_gain = float(np.sqrt(self._acceleration_kV / 5.0))
        direction_raw = param_value(params, 'sem_detector_direction_xy')
        direction = np.asarray(direction_raw, dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise SEMTransportBackendError("PARAMS['sem_detector_direction_xy'] must be a finite length-2 vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise SEMTransportBackendError("PARAMS['sem_detector_direction_xy'] must have nonzero norm.")
        self._detector_direction_xy = direction / norm

    def _electrons_from_beam_current(self) -> float | None:
        return _electrons_from_beam_current(self._beam_current_nA, self._dwell_time_us)

    def electrons_per_pixel(self) -> float:
        return self._electrons_from_beam_current() or self._electrons_per_pixel_reference

    def _effective_probe_sigma_px(self) -> float:
        if self._escape_depth_nm <= 0.0:
            return float(self.probe_sigma_px)
        escape_sigma_px = (self._escape_depth_nm / self.canvas_pitch_nm) * self._beam_energy_gain * 0.1
        return float(np.sqrt(self.probe_sigma_px ** 2 + escape_sigma_px ** 2))

    def _detector_geometry_gain(self) -> float:
        takeoff_term = max(np.cos(np.deg2rad(self._takeoff_angle_deg)), 0.0)
        return float(self._detector_acceptance * takeoff_term)

    def _material_response(self, source: np.ndarray) -> np.ndarray:
        material_scale = self._material_scale
        if material_scale == 1.0:
            return source
        if material_scale <= 0.0:
            return np.zeros_like(source)
        return source * material_scale

    def _topography_term(self, source: np.ndarray) -> np.ndarray:
        gx, gy = _gradient_components(source)
        if self._topography_gain <= 0.0:
            return np.zeros_like(source)
        directed = gy * self._detector_direction_xy[1] + gx * self._detector_direction_xy[0]
        topo = np.abs(directed)
        if self._topography_source_exponent != 1.0:
            topo = np.power(topo, self._topography_source_exponent)
        topo_blur = _gaussian_blur(topo, self._probe_sigma_px)
        return self._topography_gain * topo_blur

    def yield_from_source(self, source: np.ndarray, *, baseline: float = 0.0) -> np.ndarray:
        src = np.asarray(source, dtype=float)
        if not np.all(np.isfinite(src)):
            raise FloatingPointError("SEM transport source contains non-finite values.")
        source_positive = np.maximum(src, 0.0)
        if self._source_exponent != 1.0:
            source_model = np.power(source_positive, self._source_exponent)
        else:
            source_model = source_positive
        source_model = self._material_response(source_model)
        blur_sigma = self._effective_probe_sigma_px()
        source_blur = _gaussian_blur(source_model, blur_sigma)
        edge = _gradient_magnitude(source_model)
        edge_blur = _gaussian_blur(edge, blur_sigma)
        backscatter = _gaussian_blur(source_model, max(0.5 * blur_sigma, 0.0))

        transport = (
            self._bulk_gain * source_blur
            + self._edge_gain * edge_blur
            + self._topography_term(source_model)
        )
        if self._backscatter_fraction > 0.0:
            transport = transport + self._backscatter_fraction * backscatter
        transport = np.maximum(transport, 0.0)
        transport = self._detector_geometry_gain() * transport
        output = np.maximum(float(baseline) + transport, 0.0)
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("SEM transport backend produced non-finite yield map.")
        return output

    def contrast_from_source(self, source: np.ndarray) -> np.ndarray:
        src = np.asarray(source, dtype=float)
        return self.yield_from_source(src, baseline=0.0) - self.yield_from_source(
            np.zeros_like(src),
            baseline=0.0,
        )

    def metadata(self, params: dict | None = None) -> dict[str, Any]:
        raw = params or {}
        meta = SEMTransportMetadata(
            backend_mode=self.backend_mode,
            backend_fidelity_level="physics_based",
            backend_name=self.backend_mode,
            equations_or_model_family="syniscopy_transport_sem_scan",
            implemented_approximation_level="physics_based_transport_lite",
            native_operating_assumptions=(
                "raster-scanned Gaussian probe with beam-energy-dependent blur,"
                " source-depth attenuation proxy, and geometry-scaled emission acceptance"
            ),
        ).to_dict()
        meta.update(
            {
                "kind": "sem_syniscopy_transport_lite",
                "sem_backend": self.backend_mode,
                "backend_mode": self.backend_mode,
                "backend_fidelity_level": "physics_based",
                "forward_observable": "detected secondary-electron counts from raster transport",
                "acceleration_kV": self._acceleration_kV,
                "beam_current_nA": self._beam_current_nA,
                "dwell_time_us": self._dwell_time_us,
                "electrons_per_pixel": self.electrons_per_pixel(),
                "probe_sigma_canvas_pixels": self.probe_sigma_px,
                "probe_sigma_nm": self.probe_sigma_px * self.canvas_pitch_nm,
                "escape_depth_nm": self._escape_depth_nm,
                "detector_takeoff_angle_deg": self._takeoff_angle_deg,
                "detector_acceptance": self._detector_acceptance,
                "detector_direction_xy": [float(v) for v in self._detector_direction_xy],
                "fidelity_label": "syniscopy_transport_sem_lite_physics_based",
                "reference_backend_metadata": None,
                "validation_status": "diagnostic_only",
                "comparison_contract_id": str(raw.get("comparison_contract_id", "Contract-NR")),
                "artifact_provenance_id": raw.get("artifact_provenance_id", None),
            }
        )
        return attach_backend_fidelity_metadata(
            meta,
            params=raw,
            backend_name=self.backend_mode,
            equations_or_model_family="syniscopy_transport_sem_scan",
            implemented_approximation_level="physics_based_transport_lite",
            native_operating_assumptions=meta["native_operating_assumptions"],
            comparison_contract_id=str(raw.get("comparison_contract_id", "Contract-NR")),
            artifact_provenance_id=raw.get("artifact_provenance_id", None),
        )


__all__ = ["SyniscopyTransportSEMBackend"]
