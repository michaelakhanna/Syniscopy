"""fluorescence widefield imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    _mean_normalized_map,
    np,
)
from backend_fidelity import attach_backend_fidelity_metadata
from .fluorescence_backends import VectorialPhotophysicsFluorescenceBackend

class FluorescenceWidefieldImagingModel(ImagingModel):
    """
    Widefield epi-fluorescence imaging model.

    Fluorescence is rendered from material-property source maps, not from
    coherent scattering intensity. During rendering each particle/sub-particle
    contributes a projected emitter-density profile weighted by its chord
    length through the sphere, MaterialProperties.fluorophore_density, and
    excitation/emission spectral overlap. The scene source map is then blurred
    by the emission PSF and scaled to detector counts by scale_intensity_to_counts.
    """

    output_type = "intensity"
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_complex_optical_psf = False
    requires_optical_scattered_field = False
    requires_pre_crop_optical_filtering = True
    supports_spectral_channels = True

    def __init__(self, params: dict) -> None:
        self._fluorescence_backend = str(params.get("fluorescence_backend", "vectorial_photophysics")).strip().lower()
        if self._fluorescence_backend not in {"parametric_psf", "vectorial_photophysics"}:
            raise ValueError(
                "PARAMS['fluorescence_backend'] must be 'parametric_psf' or "
                f"'vectorial_photophysics'; got {self._fluorescence_backend!r}."
            )
        self._Qf = float(params.get("fluorescence_quantum_yield", 0.5))
        if not (0.0 <= self._Qf <= 1.0):
            raise ValueError(
                f"PARAMS['fluorescence_quantum_yield'] must be in [0, 1]; "
                f"got {self._Qf}."
            )
        self._excitation = float(params.get("fluorescence_excitation_scale", 1.0))
        if self._excitation < 0.0:
            raise ValueError(
                f"PARAMS['fluorescence_excitation_scale'] must be non-negative; "
                f"got {self._excitation}."
            )
        self._photons_per_fluorophore = params.get(
            "fluorescence_photons_per_fluorophore_per_frame",
            None,
        )
        if self._photons_per_fluorophore is not None:
            self._photons_per_fluorophore = float(self._photons_per_fluorophore)
            if not np.isfinite(self._photons_per_fluorophore) or self._photons_per_fluorophore < 0.0:
                raise ValueError(
                    "PARAMS['fluorescence_photons_per_fluorophore_per_frame'] "
                    f"must be finite and non-negative; got {self._photons_per_fluorophore}."
                )
        self._collection_efficiency = float(params.get("fluorescence_collection_efficiency", 1.0))
        self._detector_qe = float(
            params.get("fluorescence_detector_qe", params.get("detector_qe", 1.0))
        )
        if not (0.0 <= self._collection_efficiency <= 1.0):
            raise ValueError("PARAMS['fluorescence_collection_efficiency'] must be in [0, 1].")
        if not (0.0 <= self._detector_qe <= 1.0):
            raise ValueError(
                "PARAMS['fluorescence_detector_qe'] must be in [0, 1] (or "
                "PARAMS['detector_qe'] when fallback is used)."
            )
        canvas_pitch_nm = float(params.get("pixel_size_nm", 1.0)) / max(
            float(params.get("psf_oversampling_factor", 1.0)),
            1.0,
        )
        if not np.isfinite(canvas_pitch_nm) or canvas_pitch_nm <= 0.0:
            raise ValueError(
                "PARAMS['pixel_size_nm'] / PARAMS['psf_oversampling_factor'] "
                f"must resolve to a positive fluorescence canvas pitch; got {canvas_pitch_nm} nm."
            )
        self._emission_sigma_source = "pixels"
        sigma_nm_raw = params.get("fluorescence_emission_psf_sigma_nm", None)
        if sigma_nm_raw is not None:
            sigma_nm = float(sigma_nm_raw)
            if sigma_nm < 0.0:
                raise ValueError(
                    "PARAMS['fluorescence_emission_psf_sigma_nm'] must be "
                    f"non-negative; got {sigma_nm}."
                )
            self._emission_sigma_px = sigma_nm / canvas_pitch_nm
            self._emission_sigma_source = "nm"
        else:
            self._emission_sigma_px = float(
                params.get("fluorescence_emission_psf_sigma_px", 1.0)
            )
        if self._emission_sigma_px < 0.0:
            raise ValueError(
                f"PARAMS['fluorescence_emission_psf_sigma_px'] must be "
                f"non-negative; got {self._emission_sigma_px}."
            )
        self._emission_sigma_nm = self._emission_sigma_px * canvas_pitch_nm
        self._uniform_background = float(params.get("fluorescence_background", 0.0))
        if self._uniform_background < 0.0:
            raise ValueError(
                f"PARAMS['fluorescence_background'] must be non-negative; "
                f"got {self._uniform_background}."
            )
        self._spectral_bandwidth_nm = float(params.get("fluorescence_spectral_bandwidth_nm", 40.0))
        if self._spectral_bandwidth_nm <= 0.0:
            raise ValueError("PARAMS['fluorescence_spectral_bandwidth_nm'] must be positive.")
        self._tau_frames = params.get("fluorescence_photobleach_tau_frames", None)
        if self._tau_frames is not None:
            self._tau_frames = float(self._tau_frames)
            if self._tau_frames <= 0.0:
                raise ValueError(
                    f"PARAMS['fluorescence_photobleach_tau_frames'] must be "
                    f"positive when set; got {self._tau_frames}."
                )
        self._vectorial_photophysics_backend = None
        if self._fluorescence_backend == "vectorial_photophysics":
            self._vectorial_photophysics_backend = VectorialPhotophysicsFluorescenceBackend(
                params,
                canvas_pitch_nm=canvas_pitch_nm,
                base_emission_sigma_px=self._emission_sigma_px,
                quantum_yield=self._Qf,
                excitation_scale=self._excitation,
                collection_efficiency=self._collection_efficiency,
                detector_qe=self._detector_qe,
                photons_per_fluorophore=self._photons_per_fluorophore,
                uniform_background=self._uniform_background,
            )

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
                    lambda v: np.convolve(v, k1d, mode="same"),
                    axis, out,
                )
            return out

    def _bleach_factor(self, frame_index: int = 0) -> float:
        if self._tau_frames is None:
            return 1.0
        t = float(frame_index)
        return float(np.exp(-t / self._tau_frames))

    def _count_scale(self, params: dict) -> tuple[float, str]:
        if self._photons_per_fluorophore is not None:
            return (
                float(
                    self._photons_per_fluorophore
                    * self._collection_efficiency
                    * self._detector_qe
                ),
                "physical_fluorophore_photon_budget",
            )
        return (
            float(
                params.get(
                    "fluorescence_photon_count_scale",
                    float(params.get("background_intensity", 500.0)),
                )
                * self._collection_efficiency
                * self._detector_qe
            ),
            "legacy_emitted_photon_scale_before_collection_and_qe",
        )

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
        excitation_nm = float(params.get("fluorescence_excitation_wavelength_nm", 488.0))
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
        return float(params.get("fluorescence_emission_wavelength_nm", 520.0))

    def illumination_field(self, shape: tuple[int, int], params: dict) -> np.ndarray:
        del params
        return np.ones(shape, dtype=float)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        response.update({
            "kind": "fluorescence_emission_psf",
            "excitation_wavelength_nm": float(params.get("fluorescence_excitation_wavelength_nm", 488.0)),
            "emission_wavelength_nm": self.probe_wavelength_nm(params),
            "emission_sigma_canvas_px": self._emission_sigma_px,
            "emission_sigma_nm": self._emission_sigma_nm,
            "emission_sigma_source": self._emission_sigma_source,
            "emission_psf_boundary_mode": (
                "circular_fft_convolution"
                if self._vectorial_photophysics_backend is not None
                else "reflect_sum_preserving"
            ),
            "fluorescence_quantum_yield": self._Qf,
            "fluorescence_excitation_scale": self._excitation,
            "fluorescence_photons_per_fluorophore_per_frame": self._photons_per_fluorophore,
            "fluorescence_collection_efficiency": self._collection_efficiency,
            "detector_qe": float(params.get("detector_qe", 1.0)),
            "fluorescence_detector_qe": self._detector_qe,
            "fluorescence_background_counts_per_pixel": self._uniform_background,
            "fluorescence_background_units": "detected_counts_per_pixel",
            "fluorescence_photon_count_scale_contract": "emitted_photon_scale_before_collection_and_qe",
            "fluorescence_count_scale": self._count_scale(params)[0],
            "fluorescence_count_scaling_mode": self._count_scale(params)[1],
            "spectral_bandwidth_nm": self._spectral_bandwidth_nm,
            "filter_guard_radius_pixels": self.filter_guard_radius_pixels(params),
            "source_input_kind": "projected_2d_fluorophore_emitter_density",
            "source_map_ndim": 2,
            "source_axis_order": "yx",
            "source_projection_policy": "emitter_density_chord_integrated_before_emission_psf",
            "backend_consumes_volume_source": False,
            "volume_transport_model": "emission_psf_from_projected_emitter_density",
        })
        if self._vectorial_photophysics_backend is not None:
            self._vectorial_photophysics_backend._vectorial_psf(tuple(shape))
            response.update(self._vectorial_photophysics_backend.metadata(params))
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
        local_params = dict(params)
        if bool(local_params.get("detector_input_is_incident_quanta", False)):
            raise ValueError(
                "Fluorescence model outputs are detected counts. "
                "detector_input_is_incident_quanta=True would apply detector QE a "
                "second time; supply incident-quanta frames before the fluorescence "
                "backend or leave this option False."
            )
        local_params["detector_qe"] = self._detector_qe
        local_params["detector_input_is_incident_quanta"] = False
        return super().compute_noise(
            frame_counts,
            local_params,
            rng=rng,
            detector_noise_runtime=detector_noise_runtime,
        )

    def filter_guard_radius_pixels(self, params: dict) -> int | None:
        del params
        return int(np.ceil(max(4.0 * self._emission_sigma_px, 2.0)))

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del params
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
        source_multiplier: float = 1.0,
    ) -> None:
        if source_canvas is None:
            return
        scale = self._material_source_scale_for_particle(
            material_properties,
            params,
            particle_z_nm=particle_z_nm,
        )
        if scale <= 0.0:
            return
        radius_px = max(0.5, 0.5 * float(diameter_nm) / float(pixel_size_nm) * float(os_factor))
        h, w = source_canvas.shape
        x0 = max(0, int(np.floor(center_x_canvas - radius_px - 1)))
        x1 = min(w, int(np.ceil(center_x_canvas + radius_px + 2)))
        y0 = max(0, int(np.floor(center_y_canvas - radius_px - 1)))
        y1 = min(h, int(np.ceil(center_y_canvas + radius_px + 2)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
        dx = xx + x0 - float(center_x_canvas)
        dy = yy + y0 - float(center_y_canvas)
        r = np.sqrt(dx * dx + dy * dy)
        inside = r <= radius_px
        thickness_px = np.zeros_like(r, dtype=float)
        thickness_px[inside] = 2.0 * np.sqrt(np.maximum(radius_px ** 2 - r[inside] ** 2, 0.0))
        thickness_nm = thickness_px * float(pixel_size_nm) / float(os_factor)
        edge_width = max(0.75, 0.5 * float(os_factor))
        disk = np.clip((radius_px + edge_width - r) / max(edge_width, 1e-9), 0.0, 1.0)
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        source_canvas[y0:y1, x0:x1] += multiplier * scale * thickness_nm * disk

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

    def _source_with_sample_environment(
        self,
        source: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        source = np.asarray(source, dtype=float)
        if sample_environment is None:
            return source
        excitation_nm = float(params.get("fluorescence_excitation_wavelength_nm", 488.0))
        reflection = sample_environment.substrate.reflection_amplitude(excitation_nm)
        modulation = _mean_normalized_map(np.abs(1.0 + reflection) ** 2)
        mod_gain = float(params.get("fluorescence_sample_environment_excitation_modulation_gain", 0.25))
        autofl_gain = float(params.get("fluorescence_sample_environment_autofluorescence_gain", 1.0))
        excitation_factor = np.maximum(1.0 + mod_gain * (modulation - 1.0), 0.0)
        autofl_source = autofl_gain * np.maximum(
            sample_environment.substrate.autofluorescence_density(),
            0.0,
        )
        return np.maximum(source * excitation_factor + autofl_source, 0.0)

    def _detector_signal_from_source(
        self,
        source: np.ndarray,
        params: dict,
        *,
        frame_index: int,
        include_background: bool = True,
    ) -> np.ndarray:
        if self._vectorial_photophysics_backend is not None:
            return self._vectorial_photophysics_backend.source_to_detector_counts(
                source,
                frame_index=frame_index,
                include_background=include_background,
            )
        emission = self._emission_blur(source)
        bleach = self._bleach_factor(frame_index=frame_index)
        intensity = self._Qf * self._excitation * bleach * emission
        return np.maximum(intensity, 0.0)

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_field
        # Direct calls without render-supplied material source maps have no
        # particle fluorescence source. Background is added at the count layer
        # for the parametric backend and inside source_to_detector_counts for
        # the vectorial backend.
        if self._vectorial_photophysics_backend is not None:
            return np.full(E_sca_total.shape, self._uniform_background, dtype=float)
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
        del background_field
        if particle_instance is None:
            return np.zeros_like(E_sca_particle, dtype=float)
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct fluorescence particle contrast for composite particles "
                "requires a rendered source map; use compute_particle_contrast_from_source_map()."
            )
        shape = E_sca_particle.shape
        source = np.zeros(shape, dtype=float)
        material = getattr(particle_instance, "material_properties", None)
        scale = self._material_source_scale(material, params)
        if scale <= 0.0:
            return source
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(frame_index)
        frame_idx = int(np.clip(frame_idx, 0, traj.shape[0] - 1))
        px = float(traj[frame_idx, 0]) / float(params["pixel_size_nm"]) * float(params.get("psf_oversampling_factor", 1))
        py = float(traj[frame_idx, 1]) / float(params["pixel_size_nm"]) * float(params.get("psf_oversampling_factor", 1))
        pz = float(traj[frame_idx, 2]) if traj.shape[1] >= 3 else 0.0
        self.accumulate_particle_source(
            source,
            center_x_canvas=px,
            center_y_canvas=py,
            diameter_nm=float(particle_instance.particle_type.diameter_nm),
            pixel_size_nm=float(params["pixel_size_nm"]),
            os_factor=int(params.get("psf_oversampling_factor", 1)),
            material_properties=material,
            params=params,
            particle_z_nm=pz,
        )
        return self._detector_signal_from_source(
            source,
            params,
            frame_index=frame_idx,
            include_background=False,
        )

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del background_field, params
        source = np.asarray(particle_source_map, dtype=float)
        return self._detector_signal_from_source(
            source,
            {},
            frame_index=frame_index,
            include_background=False,
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
        excitation_nm = float(params.get("fluorescence_excitation_wavelength_nm", 488.0))
        reflection = sample_environment.substrate.reflection_amplitude(excitation_nm)
        modulation = _mean_normalized_map(np.abs(1.0 + reflection) ** 2)
        mod_gain = float(params.get("fluorescence_sample_environment_excitation_modulation_gain", 0.25))
        autofl_gain = float(params.get("fluorescence_sample_environment_autofluorescence_gain", 1.0))
        autofl = autofl_gain * self._emission_blur(
            sample_environment.substrate.autofluorescence_density()
        )
        return np.maximum(intensity * (1.0 + mod_gain * (modulation - 1.0)) + autofl, 0.0)

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        del background_final, E_ref_intensity_final
        if self._vectorial_photophysics_backend is not None:
            return np.asarray(intensity, dtype=float)
        scale, _ = self._count_scale(params)
        return scale * intensity + self._uniform_background

__all__ = ['FluorescenceWidefieldImagingModel']
