"""MonteCarloSEMTransportBackend SEM backend."""

from __future__ import annotations

from ._metadata import (
    Any,
    SEMTransportBackendError,
    SEMTransportMetadata,
    _electrons_from_beam_current,
    _fft_convolve_centered,
    _finite_nonnegative,
    _gradient_components,
    _gradient_magnitude,
    attach_backend_fidelity_metadata,
    np,
)

class MonteCarloSEMTransportBackend:
    """Syniscopy-owned stochastic SEM interaction-volume transport backend.

    This backend precomputes a deterministic Monte Carlo interaction kernel
    from electron random walks through the near-surface volume. Runtime rendering
    stays cheap by convolving source maps with the sampled kernel rather than
    tracing electron histories independently for every detector pixel. When the
    renderer supplies a depth stack ``(z, y, x)``, the backend uses a separate
    interaction kernel per source-depth slice and sums the detected yield after
    depth-dependent escape and energy-loss weighting.
    """

    backend_mode = "monte_carlo_transport"

    def __init__(
        self,
        params: dict,
        *,
        canvas_pitch_nm: float,
        probe_sigma_px: float,
    ) -> None:
        self.canvas_pitch_nm = _finite_nonnegative("canvas_pitch_nm", canvas_pitch_nm, minimum=1e-12)
        self.probe_sigma_px = _finite_nonnegative("sem_probe_sigma_px", probe_sigma_px, minimum=0.0)
        self.backend_mode = self.__class__.backend_mode
        self._acceleration_kV = _finite_nonnegative("sem_acceleration_kV", params.get("sem_acceleration_kV", 5.0), minimum=1e-9)
        self._baseline = _finite_nonnegative("sem_baseline_yield", params.get("sem_baseline_yield", 0.05), minimum=0.0)
        self._edge_gain = _finite_nonnegative("sem_edge_contrast_gain", params.get("sem_edge_contrast_gain", 10.0), minimum=0.0)
        self._bulk_gain = _finite_nonnegative("sem_bulk_contrast_gain", params.get("sem_bulk_contrast_gain", 1.0), minimum=0.0)
        self._topography_gain = _finite_nonnegative("sem_topography_contrast_gain", params.get("sem_topography_contrast_gain", 0.0), minimum=0.0)
        self._detector_acceptance = _finite_nonnegative("sem_detector_acceptance", params.get("sem_detector_acceptance", 1.0), minimum=0.0)
        self._takeoff_angle_deg = _finite_nonnegative("sem_detector_takeoff_angle_deg", params.get("sem_detector_takeoff_angle_deg", 45.0), minimum=0.0)
        self._escape_depth_nm = _finite_nonnegative("sem_escape_depth_nm", params.get("sem_escape_depth_nm", 20.0), minimum=0.0)
        self._backscatter_fraction = _finite_nonnegative("sem_backscatter_fraction", params.get("sem_backscatter_fraction", 0.05), minimum=0.0)
        self._material_scale = _finite_nonnegative("sem_transport_material_scale", params.get("sem_transport_material_scale", 1.0), minimum=0.0)
        self._source_exponent = _finite_nonnegative("sem_transport_source_exponent", params.get("sem_transport_source_exponent", 1.0), minimum=0.05)
        self._topography_source_exponent = _finite_nonnegative("sem_transport_topography_exponent", params.get("sem_transport_topography_exponent", 1.0), minimum=0.05)
        self._beam_current_nA = _finite_nonnegative("sem_beam_current_nA", params.get("sem_beam_current_nA", 0.0), minimum=0.0)
        self._dwell_time_us = _finite_nonnegative("sem_dwell_time_us", params.get("sem_dwell_time_us", 0.0), minimum=0.0)
        self._electrons_per_pixel_reference = _finite_nonnegative("sem_electrons_per_pixel", params.get("sem_electrons_per_pixel", 1000.0), minimum=0.0)
        self._source_representation = str(params.get("sem_source_representation", "volume")).strip().lower()
        if self._source_representation in {"projected_2d", "projected-source", "projected_source"}:
            self._source_representation = "projected"
        if self._source_representation in {"sliced_volume", "voxel_volume", "volume_zyx"}:
            self._source_representation = "volume"
        if self._source_representation not in {"projected", "volume"}:
            raise SEMTransportBackendError(
                "PARAMS['sem_source_representation'] must be 'projected' or 'volume'."
            )

        trajectories = params.get("sem_monte_carlo_trajectories", 4096)
        steps = params.get("sem_monte_carlo_steps", 64)
        self._trajectory_count = int(trajectories)
        self._step_count = int(steps)
        if self._trajectory_count <= 0:
            raise SEMTransportBackendError("PARAMS['sem_monte_carlo_trajectories'] must be positive.")
        if self._step_count <= 0:
            raise SEMTransportBackendError("PARAMS['sem_monte_carlo_steps'] must be positive.")
        range_raw = params.get("sem_monte_carlo_range_nm", None)
        base_range_nm = float(params.get("sem_interaction_volume_nm", 30.0)) if range_raw is None else float(range_raw)
        energy_scale = np.sqrt(max(self._acceleration_kV, 1e-9) / 5.0)
        self._range_nm = max(base_range_nm * energy_scale, self.canvas_pitch_nm)
        step_raw = params.get("sem_monte_carlo_step_nm", None)
        self._step_nm = (
            max(self._range_nm / float(self._step_count), 0.05)
            if step_raw is None
            else float(step_raw)
        )
        if not np.isfinite(self._step_nm) or self._step_nm <= 0.0:
            raise SEMTransportBackendError("PARAMS['sem_monte_carlo_step_nm'] must be positive when supplied.")
        scatter_deg = float(params.get("sem_monte_carlo_scatter_std_deg", 8.0))
        if not np.isfinite(scatter_deg) or scatter_deg < 0.0:
            raise SEMTransportBackendError("PARAMS['sem_monte_carlo_scatter_std_deg'] must be finite and non-negative.")
        self._scatter_std_rad = np.deg2rad(scatter_deg) / max(energy_scale, 1e-6)
        seed_raw = params.get("sem_monte_carlo_seed", params.get("random_seed", 0))
        self._seed = 0 if seed_raw is None else int(seed_raw)
        kernel_raw = params.get("sem_monte_carlo_kernel_size_px", None)
        if kernel_raw is None:
            radius_px = int(np.ceil((3.0 * self.probe_sigma_px) + (1.5 * self._range_nm / self.canvas_pitch_nm) + 4.0))
            self._kernel_size_px = max(9, 2 * radius_px + 1)
        else:
            self._kernel_size_px = int(kernel_raw)
            if self._kernel_size_px <= 0:
                raise SEMTransportBackendError("PARAMS['sem_monte_carlo_kernel_size_px'] must be positive.")
            if self._kernel_size_px % 2 == 0:
                self._kernel_size_px += 1
        volume_slices = int(params.get("sem_volume_slices", 8))
        if volume_slices <= 0:
            raise SEMTransportBackendError("PARAMS['sem_volume_slices'] must be positive.")
        self._volume_slices = volume_slices
        slice_raw = params.get("sem_volume_slice_thickness_nm", None)
        if slice_raw is None:
            self._volume_slice_thickness_nm = max(self._range_nm / float(self._volume_slices), 1e-9)
        else:
            self._volume_slice_thickness_nm = float(slice_raw)
            if not np.isfinite(self._volume_slice_thickness_nm) or self._volume_slice_thickness_nm <= 0.0:
                raise SEMTransportBackendError(
                    "PARAMS['sem_volume_slice_thickness_nm'] must be positive when supplied."
                )

        direction_raw = params.get("sem_detector_direction_xy", [1.0, 0.0])
        direction = np.asarray(direction_raw, dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise SEMTransportBackendError("PARAMS['sem_detector_direction_xy'] must be a finite length-2 vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise SEMTransportBackendError("PARAMS['sem_detector_direction_xy'] must have nonzero norm.")
        self._detector_direction_xy = direction / norm
        self._kernel_cache: np.ndarray | None = None
        self._kernel_stack_cache: dict[tuple[int, float], np.ndarray] = {}
        self._kernel_weight_sum = 0.0
        self._kernel_stack_weight_sum = 0.0

    def _electrons_from_beam_current(self) -> float | None:
        return _electrons_from_beam_current(self._beam_current_nA, self._dwell_time_us)

    def electrons_per_pixel(self) -> float:
        return self._electrons_from_beam_current() or self._electrons_per_pixel_reference

    def _detector_geometry_gain(self) -> float:
        takeoff_term = max(np.cos(np.deg2rad(self._takeoff_angle_deg)), 0.0)
        return float(self._detector_acceptance * takeoff_term)

    def _deposit(self, kernel: np.ndarray, x_nm: float, y_nm: float, weight: float) -> None:
        if weight <= 0.0 or not np.isfinite(weight):
            return
        center = self._kernel_size_px // 2
        x = center + x_nm / self.canvas_pitch_nm
        y = center + y_nm / self.canvas_pitch_nm
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        fx = x - x0
        fy = y - y0
        for yy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            if yy < 0 or yy >= self._kernel_size_px:
                continue
            for xx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if 0 <= xx < self._kernel_size_px:
                    kernel[yy, xx] += weight * wx * wy

    def _interaction_kernel(self) -> np.ndarray:
        if self._kernel_cache is not None:
            return self._kernel_cache
        rng = np.random.default_rng(self._seed)
        kernel = np.zeros((self._kernel_size_px, self._kernel_size_px), dtype=float)
        probe_sigma_nm = self.probe_sigma_px * self.canvas_pitch_nm
        escape_depth = max(self._escape_depth_nm, 1e-9)
        for _ in range(self._trajectory_count):
            x_nm = float(rng.normal(0.0, probe_sigma_nm)) if probe_sigma_nm > 0.0 else 0.0
            y_nm = float(rng.normal(0.0, probe_sigma_nm)) if probe_sigma_nm > 0.0 else 0.0
            ux = 0.0
            uy = 0.0
            z_nm = 0.0
            for _step in range(self._step_count):
                z_nm += self._step_nm
                ux += float(rng.normal(0.0, self._scatter_std_rad))
                uy += float(rng.normal(0.0, self._scatter_std_rad))
                x_nm += ux * self._step_nm
                y_nm += uy * self._step_nm
                if z_nm > self._range_nm:
                    break
                escape_weight = np.exp(-z_nm / escape_depth)
                energy_weight = max(1.0 - z_nm / max(self._range_nm, 1e-12), 0.0)
                self._deposit(kernel, x_nm, y_nm, escape_weight * energy_weight)
                if self._backscatter_fraction > 0.0 and rng.random() < self._backscatter_fraction / max(self._step_count, 1):
                    self._deposit(
                        kernel,
                        -0.5 * x_nm,
                        -0.5 * y_nm,
                        self._backscatter_fraction * escape_weight,
                    )
                    break
        total = float(kernel.sum())
        self._kernel_weight_sum = total
        if total <= 0.0 or not np.isfinite(total):
            raise SEMTransportBackendError("SEM Monte Carlo interaction kernel had non-positive energy.")
        kernel /= total
        self._kernel_cache = kernel
        return kernel

    def _interaction_kernel_stack(
        self,
        num_slices: int,
        slice_thickness_nm: float | None = None,
    ) -> np.ndarray:
        num_slices = int(num_slices)
        if num_slices <= 0:
            raise SEMTransportBackendError("SEM volume source stack must contain at least one slice.")
        slice_thickness = (
            self._volume_slice_thickness_nm
            if slice_thickness_nm is None
            else float(slice_thickness_nm)
        )
        if not np.isfinite(slice_thickness) or slice_thickness <= 0.0:
            raise SEMTransportBackendError("SEM volume slice thickness must be positive.")
        key = (num_slices, float(slice_thickness))
        cached = self._kernel_stack_cache.get(key)
        if cached is not None:
            return cached

        rng = np.random.default_rng(self._seed)
        kernels = np.zeros(
            (num_slices, self._kernel_size_px, self._kernel_size_px),
            dtype=float,
        )
        probe_sigma_nm = self.probe_sigma_px * self.canvas_pitch_nm
        escape_depth = max(self._escape_depth_nm, 1e-9)
        max_depth_nm = num_slices * slice_thickness
        for _ in range(self._trajectory_count):
            x_nm = float(rng.normal(0.0, probe_sigma_nm)) if probe_sigma_nm > 0.0 else 0.0
            y_nm = float(rng.normal(0.0, probe_sigma_nm)) if probe_sigma_nm > 0.0 else 0.0
            ux = 0.0
            uy = 0.0
            z_nm = 0.0
            for _step in range(self._step_count):
                z_nm += self._step_nm
                ux += float(rng.normal(0.0, self._scatter_std_rad))
                uy += float(rng.normal(0.0, self._scatter_std_rad))
                x_nm += ux * self._step_nm
                y_nm += uy * self._step_nm
                if z_nm > self._range_nm:
                    break
                if z_nm > max_depth_nm:
                    continue
                slice_idx = int(np.floor(z_nm / slice_thickness))
                if slice_idx < 0 or slice_idx >= num_slices:
                    continue
                escape_weight = np.exp(-z_nm / escape_depth)
                energy_weight = max(1.0 - z_nm / max(self._range_nm, 1e-12), 0.0)
                self._deposit(kernels[slice_idx], x_nm, y_nm, escape_weight * energy_weight)
                if self._backscatter_fraction > 0.0 and rng.random() < self._backscatter_fraction / max(self._step_count, 1):
                    self._deposit(
                        kernels[slice_idx],
                        -0.5 * x_nm,
                        -0.5 * y_nm,
                        self._backscatter_fraction * escape_weight,
                    )
                    break
        total = float(kernels.sum())
        self._kernel_stack_weight_sum = total
        if total <= 0.0 or not np.isfinite(total):
            raise SEMTransportBackendError("SEM Monte Carlo volume kernel stack had non-positive energy.")
        kernels /= total
        self._kernel_stack_cache[key] = kernels
        return kernels

    def _material_response(self, source: np.ndarray) -> np.ndarray:
        source_positive = np.maximum(np.asarray(source, dtype=float), 0.0)
        if self._source_exponent != 1.0:
            source_positive = np.power(source_positive, self._source_exponent)
        return self._material_scale * source_positive

    def _kernel_blur(self, arr: np.ndarray) -> np.ndarray:
        return np.maximum(_fft_convolve_centered(arr, self._interaction_kernel()), 0.0)

    def _kernel_blur_volume(self, source_stack: np.ndarray) -> np.ndarray:
        stack = np.asarray(source_stack, dtype=float)
        if stack.ndim != 3:
            raise SEMTransportBackendError(
                f"SEM volume source must have shape (z, y, x); got ndim={stack.ndim}."
            )
        kernels = self._interaction_kernel_stack(stack.shape[0])
        out = np.zeros(stack.shape[-2:], dtype=float)
        for idx in range(stack.shape[0]):
            out += _fft_convolve_centered(stack[idx], kernels[idx])
        return np.maximum(out, 0.0)

    def _edge_volume(self, source_stack: np.ndarray) -> np.ndarray:
        stack = np.asarray(source_stack, dtype=float)
        kernels = self._interaction_kernel_stack(stack.shape[0])
        out = np.zeros(stack.shape[-2:], dtype=float)
        for idx in range(stack.shape[0]):
            out += _fft_convolve_centered(_gradient_magnitude(stack[idx]), kernels[idx])
        return np.maximum(out, 0.0)

    def _topography_term(self, source: np.ndarray) -> np.ndarray:
        if self._topography_gain <= 0.0:
            return np.zeros_like(source)
        gx, gy = _gradient_components(source)
        directed = gy * self._detector_direction_xy[1] + gx * self._detector_direction_xy[0]
        topo = np.abs(directed)
        if self._topography_source_exponent != 1.0:
            topo = np.power(topo, self._topography_source_exponent)
        return self._topography_gain * self._kernel_blur(topo)

    def yield_from_source(self, source: np.ndarray, *, baseline: float = 0.0) -> np.ndarray:
        src = np.asarray(source, dtype=float)
        if not np.all(np.isfinite(src)):
            raise FloatingPointError("SEM Monte Carlo source contains non-finite values.")
        source_model = self._material_response(src)
        if source_model.ndim == 3:
            bulk = self._kernel_blur_volume(source_model)
            edge = self._edge_volume(source_model)
            topography_source = np.max(source_model, axis=0)
        elif source_model.ndim == 2:
            bulk = self._kernel_blur(source_model)
            edge = self._kernel_blur(_gradient_magnitude(source_model))
            topography_source = source_model
        else:
            raise SEMTransportBackendError(
                f"SEM Monte Carlo source must be 2D or 3D; got shape {source_model.shape!r}."
            )
        transport = (
            self._bulk_gain * bulk
            + self._edge_gain * edge
            + self._topography_term(topography_source)
        )
        output = np.maximum(float(baseline) + self._detector_geometry_gain() * transport, 0.0)
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("SEM Monte Carlo backend produced non-finite yield map.")
        return output

    def contrast_from_source(self, source: np.ndarray) -> np.ndarray:
        src = np.asarray(source, dtype=float)
        return self.yield_from_source(src, baseline=0.0) - self.yield_from_source(
            np.zeros_like(src),
            baseline=0.0,
        )

    def guard_radius_pixels(self) -> int:
        return int(np.ceil(0.5 * self._kernel_size_px + 2.0))

    def metadata(self, params: dict | None = None) -> dict[str, Any]:
        raw = params or {}
        kernel = self._interaction_kernel()
        volume_kernels = (
            self._interaction_kernel_stack(self._volume_slices)
            if self._source_representation == "volume"
            else None
        )
        meta = SEMTransportMetadata(
            backend_mode=self.backend_mode,
            backend_fidelity_level="physics_based",
            backend_name=self.backend_mode,
            equations_or_model_family="stochastic_sem_monte_carlo_interaction_volume",
            implemented_approximation_level="physics_based_monte_carlo_interaction_volume",
            native_operating_assumptions=(
                "precomputed stochastic electron random-walk interaction kernel with "
                "escape-depth weighting, backscatter events, probe spread, and detector geometry"
            ),
        ).to_dict()
        meta.update(
            {
                "kind": "sem_monte_carlo_transport",
                "sem_backend": self.backend_mode,
                "backend_mode": self.backend_mode,
                "backend_fidelity_level": "physics_based",
                "forward_observable": "detected secondary-electron counts from Monte Carlo interaction-volume transport",
                "acceleration_kV": self._acceleration_kV,
                "beam_current_nA": self._beam_current_nA,
                "dwell_time_us": self._dwell_time_us,
                "electrons_per_pixel": self.electrons_per_pixel(),
                "probe_sigma_canvas_pixels": self.probe_sigma_px,
                "probe_sigma_nm": self.probe_sigma_px * self.canvas_pitch_nm,
                "monte_carlo_trajectories": self._trajectory_count,
                "monte_carlo_steps": self._step_count,
                "monte_carlo_step_nm": self._step_nm,
                "monte_carlo_range_nm": self._range_nm,
                "monte_carlo_kernel_size_px": self._kernel_size_px,
                "monte_carlo_kernel_weight_sum": self._kernel_weight_sum,
                "monte_carlo_volume_kernel_weight_sum": self._kernel_stack_weight_sum,
                "monte_carlo_seed": self._seed,
                "monte_carlo_scatter_std_rad": self._scatter_std_rad,
                "monte_carlo_kernel_peak": float(np.max(kernel)),
                "monte_carlo_volume_kernel_peak": (
                    float(np.max(volume_kernels)) if volume_kernels is not None else None
                ),
                "source_representation": self._source_representation,
                "sem_source_representation": self._source_representation,
                "sem_volume_slices": self._volume_slices,
                "sem_volume_slice_thickness_nm": self._volume_slice_thickness_nm,
                "sem_volume_depth_nm": self._volume_slices * self._volume_slice_thickness_nm,
                "escape_depth_nm": self._escape_depth_nm,
                "backscatter_fraction": self._backscatter_fraction,
                "detector_takeoff_angle_deg": self._takeoff_angle_deg,
                "detector_acceptance": self._detector_acceptance,
                "detector_direction_xy": [float(v) for v in self._detector_direction_xy],
                "fidelity_label": "syniscopy_monte_carlo_sem_physics_based",
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
            equations_or_model_family="stochastic_sem_monte_carlo_interaction_volume",
            implemented_approximation_level="physics_based_monte_carlo_interaction_volume",
            native_operating_assumptions=meta["native_operating_assumptions"],
            comparison_contract_id=str(raw.get("comparison_contract_id", "Contract-NR")),
            artifact_provenance_id=raw.get("artifact_provenance_id", None),
        )


__all__ = ["MonteCarloSEMTransportBackend"]
