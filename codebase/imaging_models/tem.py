"""tem imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    np,
)
from backend_fidelity import attach_backend_fidelity_metadata
from config.runtime import (
    OpticalPsfSupportSettings,
    SampleEnvironmentSettings,
    TemSettings,
)
from simulation_runtime_state import get_source_volume_support
from detector_frame_conversion import (
    DETECTOR_OUTPUT_DOMAIN_ELECTRON_COUNT,
    MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY,
    REFERENCE_BASIS_NONE,
    VALUE_FORM_ABSOLUTE,
    DetectorFrameConversion,
    convert_model_output_to_detector_frame,
)
from direct_signal_contracts import (
    DirectParticleSignalProduct,
    electron_count_delta_representation,
    tem_projected_phase_source_representation,
)
from .tem_backends import (
    CTFProxyTEMBackend,
    MultisliceLiteTEMBackend,
    PhysicalMultisliceTEMBackend,
    SyniscopyMultisliceTEMBackend,
)
from .source_rasterization import primitive_footprint_patch

class TransmissionElectronMicroscopyImagingModel(ImagingModel):
    """
    Transmission electron microscopy (TEM) phase-contrast imaging model.

    This model converts a projected material-potential source map into a TEM
    phase-contrast image by applying the standard electron contrast transfer
    function (CTF) in Fourier space.
    The underlying physics is the weak-phase-object approximation used
    throughout electron microscopy: a thin specimen imparts a small phase
    shift proportional to its projected electrostatic potential, and the
    detector records a filtered image of that phase shift.

    Under this approximation the complex exit wave is
        psi_exit(r) ~= 1 + i sigma V_proj(r),
    and after the objective lens and detector sampling the recorded
    intensity becomes
        I(r) ~= 1 - 2 sigma V_proj * PSF_TEM(r),
    with the TEM point-spread function given in Fourier space by
        CTF(k)    = 2 sin(chi(k)) * E(k),
        chi(k)    = pi C_s lambda^3 k^4 / 2  -  pi lambda Delta_f k^2,
        E(k)      = exp(-pi^2 alpha^2 (C_s lambda^2 k^3 - Delta_f k)^2),
    where lambda is the relativistic de Broglie wavelength of the electron,
    C_s is the spherical aberration coefficient, Delta_f is the defocus,
    and alpha is the illumination-angle half-width controlling partial
    coherence.  These formulas follow Kirkland (2010), Chap. 5.

    Syniscopy source representation
    -------------------------------
    During rendering, particle material properties are accumulated into a
    projected phase-shift source map ``sigma * V_proj(r)``. The imaging model
    applies the CTF in Fourier space and returns the linearized weak-phase
    intensity ``1 + CTF * source``, clamped at zero for count-domain noise
    sampling. This is the configurable weak-phase proxy path; when
    ``tem_backend='multislice_physical'``, Syniscopy uses its physical
    Cowley-Moodie/Kirkland-style multislice TEM backend.

    Parameters (parameters keys, all optional with nominal defaults)
    ------------------------------------------------------------
    The defaults define a stable moderate-contrast synthetic TEM regime; use
    calibrated values for instrument-specific studies.

    - ``tem_acceleration_kV``         (default 300.0) accelerating voltage
    - ``tem_Cs_mm``                   (default 0.5)   spherical aberration
    - ``tem_defocus_nm``              (default: Scherzer) defocus Delta_f
    - ``tem_partial_coherence_alpha_mrad`` (default 0.1) illumination half-angle
    - ``tem_phase_shift_per_volt_nm`` (default: relativistic electron
      interaction parameter for ``tem_acceleration_kV``) projected phase scale
      multiplying material mean inner potential and projected thickness.
    - ``tem_pixel_size_pm``           optional compatibility assertion for
      the CTF Fourier-grid pitch. When supplied, it must match the actual
      rendered model-canvas pitch ``pixel_size_nm / psf_oversampling_factor``.
    - ``tem_dose_per_pixel``          (default 100)    mean electron
      count per pixel for the unscattered beam.  Used by
      the model-output detector-frame conversion to convert the dimensionless
      weak-phase image into detector electron counts.

    Output
    ------
    Returns an ``intensity`` output in the same dimensionless (reference=1)
    scale as the other intensity-output imaging models, so the standard
    noise and quantization layers apply directly.

    Validation
    ----------
    Accelerating voltage must be > 0.  Cs, alpha, dose must be
    non-negative.  Defocus may be positive (underfocus) or negative
    (overfocus) per the Scherzer convention.
    """

    output_type = "intensity"
    counts_are_exposure_integrated = True
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_complex_optical_psf = False
    requires_optical_scattered_field = False
    requires_pre_crop_optical_filtering = True
    supports_spectral_channels = False
    _TEM_POTENTIAL_SOURCE_MATERIAL = "material_projected_inner_potential"
    _TEM_POTENTIAL_SOURCE_SAMPLE_ENVIRONMENT = "sample_environment_projected_potential"
    _TEM_POTENTIAL_SOURCE_COMPOSITE = "material_plus_sample_environment"

    @staticmethod
    def _required_multislice_slices_from_particles(params: dict, slice_thickness_nm: float | None) -> int:
        if slice_thickness_nm is None:
            return 1
        dz = float(slice_thickness_nm)
        if not np.isfinite(dz) or dz <= 0.0:
            return 1
        from particle_specs import get_particle_specs

        max_depth_nm = 0.0
        for spec in get_particle_specs(params):
            z_low = np.inf
            z_high = -np.inf
            for component in spec.components:
                radius_nm = float(component.bounding_radius_nm)
                offset_z_nm = float(component.offset_nm[2])
                z_low = min(z_low, offset_z_nm - radius_nm)
                z_high = max(z_high, offset_z_nm + radius_nm)
            if np.isfinite(z_low) and np.isfinite(z_high):
                max_depth_nm = max(max_depth_nm, z_high - z_low)
        if max_depth_nm <= 0.0:
            return 1
        return max(1, int(np.ceil(max_depth_nm / dz)))

    def __init__(self, params: dict) -> None:
        settings = TemSettings.from_params(params)
        self._tem_settings = settings
        self._tem_model = settings.model
        self._tem_backend = settings.backend
        self._tem_potential_source = settings.potential_source
        self._V_kV = settings.acceleration_kV
        self._Cs_mm = settings.spherical_aberration_mm
        self._alpha_mrad = settings.partial_coherence_alpha_mrad
        self._lambda_m = settings.electron_wavelength_m
        self._defocus_m = settings.defocus_m
        self._phase_shift_per_volt_nm = settings.phase_shift_per_volt_nm
        self._projected_potential_scale = settings.projected_potential_scale

        self._sampling = settings.sampling
        canvas_pitch_nm = self._sampling.model_canvas_pixel_size_nm
        if not np.isfinite(canvas_pitch_nm) or canvas_pitch_nm <= 0.0:
            raise ValueError(
                "parameters['pixel_size_nm'] / parameters['psf_oversampling_factor'] must "
                f"resolve to a positive pitch; got {canvas_pitch_nm} nm."
            )

        # Fourier-grid pitch is the physical pitch of the rendered model canvas.
        self._pixel_size_m = 1.0e-9 * canvas_pitch_nm
        settings.assert_canvas_pixel_pitch()
        if not np.isfinite(self._pixel_size_m) or self._pixel_size_m <= 0.0:
            raise ValueError(
                "parameters['pixel_size_nm'] must resolve "
                f"to a positive pixel pitch; got {self._pixel_size_m} m."
            )

        self._dose_per_pixel = settings.dose_per_pixel
        self._configured_multislice_slices = settings.multislice_slices
        self._objective_aperture_mrad = settings.objective_aperture_mrad
        self._slice_thickness_nm = settings.slice_thickness_nm
        self._required_multislice_slices = self._required_multislice_slices_from_particles(
            params,
            self._slice_thickness_nm,
        )
        source_volume_support = get_source_volume_support(params, "tem")
        resolved_source_slices = None if source_volume_support is None else int(source_volume_support.slice_count)
        resolved_required_slices = (
            None if source_volume_support is None else int(source_volume_support.required_slice_count)
        )
        self._resolved_source_volume_slices = (
            None if resolved_source_slices is None else int(resolved_source_slices)
        )
        self._required_multislice_slices_for_rendered_z = (
            None if resolved_required_slices is None else int(resolved_required_slices)
        )
        self._multislice_slices = max(
            int(self._configured_multislice_slices),
            int(self._required_multislice_slices),
            1
            if self._resolved_source_volume_slices is None
            else int(self._resolved_source_volume_slices),
        )
        source_z_center = None if source_volume_support is None else float(source_volume_support.z_center_nm)
        self._source_z_center_nm = 0.0 if source_z_center is None else float(source_z_center)
        if not np.isfinite(self._source_z_center_nm):
            raise ValueError(
                "Resolved internal TEM source-volume z center must be finite; "
                f"got {source_z_center!r}."
            )
        self._source_z_envelope_min_nm = (
            None if source_volume_support is None else float(source_volume_support.envelope_min_nm)
        )
        self._source_z_envelope_max_nm = (
            None if source_volume_support is None else float(source_volume_support.envelope_max_nm)
        )
        # Cache the CTF array per frame shape. The CTF depends only on
        # the shape, pixel pitch, lambda, Cs, defocus, alpha, so once
        # computed it is reused across all frames of a run.
        self._ctf_cache: dict = {}
        self._ctf_backend = CTFProxyTEMBackend(
            pixel_size_m=self._pixel_size_m,
            electron_wavelength_m=self._lambda_m,
            Cs_mm=self._Cs_mm,
            defocus_m=self._defocus_m,
            partial_coherence_alpha_mrad=self._alpha_mrad,
            objective_aperture_mrad=self._objective_aperture_mrad,
        )
        self._multislice_lite_backend = MultisliceLiteTEMBackend(
            ctf_backend=self._ctf_backend,
            electron_wavelength_m=self._lambda_m,
            pixel_size_m=self._pixel_size_m,
            multislice_slices=self._multislice_slices,
            slice_thickness_nm=self._slice_thickness_nm,
        )
        self._tem_high_fidelity_backend = None
        if self._tem_backend == "multislice_physical":
            self._tem_high_fidelity_backend = PhysicalMultisliceTEMBackend(
                ctf_backend=self._ctf_backend,
                tem_settings=settings,
                electron_wavelength_m=self._lambda_m,
                pixel_size_m=self._pixel_size_m,
                dose_per_pixel=self._dose_per_pixel,
                default_slice_count=self._multislice_slices,
            )
        elif self._tem_backend == "syniscopy_multislice":
            self._tem_high_fidelity_backend = SyniscopyMultisliceTEMBackend(
                pixel_size_m=self._pixel_size_m,
                tem_settings=settings,
                electron_wavelength_m=self._lambda_m,
                Cs_mm=self._Cs_mm,
                defocus_m=self._defocus_m,
                partial_coherence_alpha_mrad=self._alpha_mrad,
                dose_per_pixel=self._dose_per_pixel,
                default_slice_count=self._multislice_slices,
            )

    def _uses_sliced_source_stack(self) -> bool:
        return self._tem_high_fidelity_backend is not None and self._slice_thickness_nm is not None

    def particle_source_z_basis(self, params: dict) -> str:
        del params
        return "physical_sample_world" if self._uses_sliced_source_stack() else "projected_no_z"

    def source_coordinate_contract(self, params: dict) -> dict:
        del params
        source_z_basis = self.particle_source_z_basis({})
        # TEM has two distinct source-coordinate regimes. Weak-phase CTF and
        # multislice-lite consume a projected phase/potential map, so absolute
        # particle z must not cross the source-density seam. Physical multislice
        # consumes a slice-resolved source stack and must retain world-z placement.
        return {
            "source_density_z_basis": source_z_basis,
            "source_z_planes_basis": source_z_basis,
            "optical_response_z_basis": "physical_sample_world" if self._uses_sliced_source_stack() else "projected_no_z",
            "tem_source_coordinate_regime": (
                "slice_resolved_physical_world_z"
                if self._uses_sliced_source_stack()
                else "projected_phase_no_particle_z"
            ),
        }

    @staticmethod
    def _as_sample_environment_potential(
        sample_environment: SampleEnvironment | None,
        phase_shift_per_volt_nm: float,
        params: dict,
    ) -> np.ndarray:
        if sample_environment is None:
            raise ValueError(
                "TEM potential synthesis requested sample-environment terms "
                "but sample_environment is not configured."
            )
        scale = SampleEnvironmentSettings.from_params(params).tem_potential_scale
        return scale * phase_shift_per_volt_nm * sample_environment.substrate.projected_potential_V_nm()

    def _compose_tem_projected_potential_source(
        self,
        particle_source: np.ndarray,
        sample_environment: SampleEnvironment | None,
        params: dict,
    ) -> np.ndarray:
        particle_raw = np.asarray(particle_source)
        if np.iscomplexobj(particle_raw):
            imag_max = float(np.max(np.abs(particle_raw.imag))) if particle_raw.size else 0.0
            real_scale = max(
                float(np.max(np.abs(particle_raw.real))) if particle_raw.size else 0.0,
                1.0,
            )
            if imag_max > 1.0e-12 * real_scale:
                raise ValueError(
                    "TEM projected potential source must be real-valued; "
                    f"max imaginary component is {imag_max}."
                )
            particle = np.asarray(particle_raw.real, dtype=float)
        else:
            particle = np.asarray(particle_raw, dtype=float)
        if particle.ndim not in {2, 3}:
            raise ValueError(
                "TEM projected potential source must be a 2D array for proxy paths "
                "or a 3D stack for multislice paths."
            )
        include_particles = self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_MATERIAL or \
            self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_COMPOSITE
        include_environment = self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_SAMPLE_ENVIRONMENT or \
            self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_COMPOSITE
        if not (include_particles or include_environment):
            raise ValueError(
                "Unsupported TEM potential source mode; allowed values are "
                f"{self._TEM_POTENTIAL_SOURCE_MATERIAL}, "
                f"{self._TEM_POTENTIAL_SOURCE_SAMPLE_ENVIRONMENT}, "
                f"{self._TEM_POTENTIAL_SOURCE_COMPOSITE}."
            )
        if include_environment:
            substrate_phase = self._as_sample_environment_potential(
                sample_environment,
                self._phase_shift_per_volt_nm,
                params,
            )
            if particle.ndim == 3:
                substrate = np.broadcast_to(
                    np.asarray(substrate_phase, dtype=float) / float(particle.shape[0]),
                    particle.shape,
                )
            else:
                substrate = np.asarray(substrate_phase, dtype=float)
        else:
            substrate = 0.0
        if include_particles and include_environment:
            return particle + substrate
        if include_particles:
            return particle
        return np.asarray(substrate, dtype=float)

    # -- CTF construction -------------------------------------------------







    def _intensity_from_projected_phase(self, source: np.ndarray) -> np.ndarray:
        if self._tem_high_fidelity_backend is not None:
            return self._tem_high_fidelity_backend.intensity_from_projected_phase(source)
        if self._tem_model == "multislice_lite":
            return self._multislice_lite_backend.intensity_from_projected_phase(source)
        return self._ctf_backend.intensity_from_projected_phase(source)

    def _contrast_from_projected_phase(self, source: np.ndarray) -> np.ndarray:
        if self._tem_high_fidelity_backend is not None:
            return self._tem_high_fidelity_backend.contrast_from_projected_phase(source)
        if self._tem_model == "multislice_lite":
            return self._multislice_lite_backend.contrast_from_projected_phase(source)
        return self._ctf_backend.contrast_from_projected_phase(source)

    @staticmethod
    def _centered_l1_support_radius_pixels(
        response: np.ndarray,
        *,
        tail_fraction: float,
    ) -> int:
        arr = np.maximum(np.asarray(response, dtype=float), 0.0)
        total = float(np.sum(arr))
        if total <= 0.0 or not np.isfinite(total):
            return 0
        h, w = arr.shape[-2:]
        yy, xx = np.indices((h, w), dtype=float)
        cy = h // 2
        cx = w // 2
        radius = np.sqrt((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2).ravel()
        values = arr.ravel()
        order = np.argsort(radius)
        cumulative = np.cumsum(values[order])
        target = total * (1.0 - float(np.clip(tail_fraction, 0.0, 1.0)))
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), order.size - 1)
        return int(np.ceil(float(radius[order[idx]])))

    def _automatic_filter_guard_radius_pixels(self, params: dict) -> int:
        os_size = max(1, self._sampling.model_canvas_shape[0])
        probe_size = max(os_size, 128)
        ctf = self._ctf_backend.ctf((probe_size, probe_size))
        impulse = np.abs(np.fft.fftshift(np.fft.ifft2(ctf)))
        threshold = OpticalPsfSupportSettings.from_params(params).intensity_fraction_threshold
        support = self._centered_l1_support_radius_pixels(
            impulse,
            tail_fraction=threshold,
        )
        return int(np.ceil(min(max(support + 2, 64), max(os_size, 64))))

    def filter_guard_radius_pixels(self, params: dict) -> int | None:
        return self._tem_settings.filter_guard_radius_pixels(
            automatic_guard_radius=self._automatic_filter_guard_radius_pixels(params)
        )


    def _projected_phase_source(
        self,
        *,
        shape: tuple[int, int],
        center_x_canvas: float,
        center_y_canvas: float,
        diameter_nm: float,
        pixel_size_nm: float,
        os_factor: int,
        material_properties,
        params: dict,
        particle_z_nm: float | None = None,
        component_geometry=None,
        orientation_matrix=None,
    ) -> np.ndarray:
        source = self.initialize_particle_source_canvas(shape, params)
        self.accumulate_particle_source(
            source,
            center_x_canvas=center_x_canvas,
            center_y_canvas=center_y_canvas,
            diameter_nm=diameter_nm,
            pixel_size_nm=pixel_size_nm,
            os_factor=os_factor,
            material_properties=material_properties,
            params=params,
            particle_z_nm=particle_z_nm,
            component_geometry=component_geometry,
            orientation_matrix=orientation_matrix,
        )
        return source

    def probe_wavelength_nm(self, params: dict) -> float:
        del params
        return float(self._lambda_m * 1.0e9)

    def _source_z_planes_nm(self) -> list[float] | None:
        if self._slice_thickness_nm is None:
            return None
        dz_nm = float(self._slice_thickness_nm)
        return [
            float(self._source_z_center_nm + (idx - 0.5 * (self._multislice_slices - 1)) * dz_nm)
            for idx in range(self._multislice_slices)
        ]

    def _source_z_bounds_nm(self) -> tuple[float, float] | None:
        if self._slice_thickness_nm is None:
            return None
        half_extent_nm = 0.5 * float(self._multislice_slices) * float(self._slice_thickness_nm)
        return (
            float(self._source_z_center_nm - half_extent_nm),
            float(self._source_z_center_nm + half_extent_nm),
        )

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        source_z_planes_nm = self._source_z_planes_nm()
        source_z_bounds_nm = self._source_z_bounds_nm()
        kind_by_model = {
            "weak_phase_ctf": "tem_ctf",
            "multislice_lite": "tem_multislice_lite",
            "syniscopy_multislice": "tem_multislice",
            "multislice_physical": "tem_multislice_physical",
        }
        response.update({
            "kind": kind_by_model[self._tem_model],
            "tem_model": self._tem_model,
            "tem_backend": self._tem_backend,
            "tem_potential_source": self._tem_potential_source,
            "measurement_domain": "electron_count",
            "signal_units": "electron_count",
            "contrast_frame_units": "electron_count_difference",
            "model_output_domain": "relative_direct_beam_intensity",
            "model_signal_units": "relative_intensity",
            "pre_count_contrast_units": "relative_intensity_difference",
            "final_measurement_domain": "electron_count",
            "final_signal_units": "electron_count",
            "count_scaling_mode": "incident_electron_dose_per_pixel_times_relative_direct_beam_intensity",
            "forward_observable": (
                "|multislice-lite exit wave with objective CTF readout|^2"
                if self._tem_model == "multislice_lite"
                else "|physical multislice exit wave with objective transfer|^2"
                if self._tem_model == "multislice_physical"
                else "1 + CTF(projected electrostatic phase)"
            ),
            "acceleration_kV": self._V_kV,
            "electron_wavelength_pm": float(self._lambda_m * 1.0e12),
            "interaction_parameter_rad_per_V_nm": self._phase_shift_per_volt_nm,
            "Cs_mm": self._Cs_mm,
            "defocus_m": self._defocus_m,
            "defocus_nm": float(self._defocus_m * 1.0e9),
            "partial_coherence_alpha_mrad": self._alpha_mrad,
            "ctf_pixel_size_nm": float(self._pixel_size_m * 1.0e9),
            "dose_per_pixel": self._dose_per_pixel,
            "multislice_slices": self._multislice_slices,
            "configured_multislice_slices": self._configured_multislice_slices,
            "required_multislice_slices_for_particle_depth": self._required_multislice_slices,
            "required_multislice_slices_for_rendered_z": self._required_multislice_slices_for_rendered_z,
            "resolved_tem_source_volume": bool(self._resolved_source_volume_slices is not None),
            "resolved_tem_source_volume_slices": self._resolved_source_volume_slices,
            "slice_thickness_nm": self._slice_thickness_nm,
            "multislice_source_extent_nm": (
                float(self._multislice_slices * self._slice_thickness_nm)
                if self._slice_thickness_nm is not None
                else None
            ),
            "source_z_center_nm": float(self._source_z_center_nm),
            "source_z_min_nm": None if source_z_bounds_nm is None else float(source_z_bounds_nm[0]),
            "source_z_max_nm": None if source_z_bounds_nm is None else float(source_z_bounds_nm[1]),
            "source_z_envelope_min_nm": (
                None
                if self._source_z_envelope_min_nm is None
                else float(self._source_z_envelope_min_nm)
            ),
            "source_z_envelope_max_nm": (
                None
                if self._source_z_envelope_max_nm is None
                else float(self._source_z_envelope_max_nm)
            ),
            "filter_guard_radius_pixels": self.filter_guard_radius_pixels(params),
            "fidelity_label": (
                "multislice_lite_projected_phase_proxy"
                if self._tem_model == "multislice_lite"
                else "electron_ctf_proxy"
            ),
        })
        if self._tem_high_fidelity_backend is not None:
            response.update(self._tem_high_fidelity_backend.metadata(params))
            response["reference_backend_metadata"] = response.get("reference_backend_metadata") or {
                "reference_status": response.get("reference_status"),
                "reference_validation_hash": response.get("reference_validation_hash"),
            }
        else:
            response["tem_backend"] = self._tem_backend
            response = attach_backend_fidelity_metadata(
                response,
                params=params,
                backend_name=self._tem_backend,
                equations_or_model_family=(
                    "multislice-lite projected phase proxy"
                    if self._tem_model == "multislice_lite"
                    else "weak-phase CTF projected potential proxy"
                ),
                implemented_approximation_level=(
                    "physics_based"
                    if self._tem_model == "multislice_lite"
                    else "proxy"
                ),
                native_operating_assumptions=(
                    "3D projected-potential proxy with reduced-slice Fresnel propagation"
                    if self._tem_model == "multislice_lite"
                    else "weak-phase CTF linearized projected-potential approximation"
                ),
                comparison_contract_id=response.get("comparison_contract_id", "Contract-NR"),
                artifact_provenance_id=response.get("artifact_provenance_id"),
            )
        tem_uses_slice_stack = self._tem_high_fidelity_backend is not None and self._slice_thickness_nm is not None
        if tem_uses_slice_stack and self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_MATERIAL:
            response.update(
                source_input_kind="slice_resolved_tem_material_potential_stack",
                source_map_ndim=3,
                source_axis_order="zyx",
                source_projection_policy="backend_native_slice_stack",
                backend_consumes_volume_source=True,
                volume_transport_model="split_step_multislice",
                tem_environment_source_dimensionality="none",
                tem_projected_source_fallback=False,
                source_slice_thickness_nm=self._slice_thickness_nm,
                source_z_planes_nm=source_z_planes_nm,
                source_z_origin="resolved_world_z_multislice_stack",
                source_z_uses_particle_world_z=True,
                source_z_out_of_range_policy="raise_outside_resolved_multislice_slab",
            )
        elif tem_uses_slice_stack and self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_COMPOSITE:
            response.update(
                source_input_kind="slice_resolved_tem_material_stack_plus_projected_environment",
                source_map_ndim=3,
                source_axis_order="zyx",
                source_projection_policy="projected_environment_broadcast_across_material_slices",
                backend_consumes_volume_source=True,
                volume_transport_model="split_step_multislice_with_projected_environment_broadcast",
                tem_environment_source_dimensionality="projected_2d_broadcast_to_slices",
                tem_projected_source_fallback=True,
                source_slice_thickness_nm=self._slice_thickness_nm,
                source_z_planes_nm=source_z_planes_nm,
                source_z_origin="resolved_world_z_multislice_stack",
                source_z_uses_particle_world_z=True,
                source_z_out_of_range_policy="raise_outside_resolved_multislice_slab",
            )
        elif self._tem_high_fidelity_backend is not None:
            if self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_MATERIAL:
                fallback_source_kind = "projected_2d_tem_material_potential"
                environment_dimensionality = "none"
                projection_policy = "projected_material_source_evenly_split_across_multislice_slices"
            elif self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_COMPOSITE:
                fallback_source_kind = "projected_2d_tem_material_plus_environment_potential"
                environment_dimensionality = "projected_2d"
                projection_policy = "projected_material_plus_environment_source_evenly_split_across_multislice_slices"
            else:
                fallback_source_kind = "projected_2d_tem_environment_potential"
                environment_dimensionality = "projected_2d"
                projection_policy = "projected_environment_source_evenly_split_across_multislice_slices"
            response.update(
                source_input_kind=fallback_source_kind,
                source_map_ndim=2,
                source_axis_order="yx",
                source_projection_policy=projection_policy,
                backend_consumes_volume_source=False,
                volume_transport_model="split_step_multislice_projected_source_fallback",
                tem_environment_source_dimensionality=environment_dimensionality,
                tem_projected_source_fallback=True,
                source_z_origin="projected_2d_no_slice_depth",
                source_z_uses_particle_world_z=False,
            )
        else:
            response.update(
                source_input_kind="projected_2d_tem_potential",
                source_map_ndim=2,
                source_axis_order="yx",
                source_projection_policy="projected_source_ctf_or_lite_backend",
                backend_consumes_volume_source=False,
                volume_transport_model="projected_phase_transfer",
                tem_environment_source_dimensionality=(
                    "projected_2d"
                    if self._tem_potential_source != self._TEM_POTENTIAL_SOURCE_MATERIAL
                    else "none"
                ),
                tem_projected_source_fallback=True,
                source_z_origin="projected_2d_no_slice_depth",
                source_z_uses_particle_world_z=False,
            )
        if self._tem_high_fidelity_backend is None:
            response.update(self._ctf_backend.diagnostics(tuple(shape)))
        else:
            response["weak_phase_ctf_diagnostics"] = self._ctf_backend.diagnostics(tuple(shape))
        return response

    # -- Contract methods -------------------------------------------------

    def compute_intensity(
        self,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        TEM phase-contrast intensity.  With |E_direct|=1 and weak-phase
        approximation, the recorded intensity equals
            I(r) = 1 - 2 sigma V_proj * PSF_TEM(r)
        with E_sca representing sigma*V_proj becomes
            I(r) = 1 + CTF(k) * E_sca  (applied in Fourier space, real part).

        We return 1 + CTF-filtered E_sca (clamped at 0 from below to keep
        the downstream shot-noise layer physically valid).
        """
        return self._intensity_from_projected_phase(E_sca_total)

    def initialize_particle_source_canvas(self, shape: tuple[int, int], params: dict):
        del params
        if self._tem_high_fidelity_backend is not None and self._slice_thickness_nm is not None:
            return np.zeros((self._multislice_slices, *shape), dtype=float)
        return np.zeros(shape, dtype=float)

    def _accumulate_particle_source_multislice_stack(
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
        component_geometry=None,
        orientation_matrix=None,
    ) -> None:
        if component_geometry is None:
            raise ValueError("TEM multislice source accumulation requires component_geometry.")
        if not self._slice_thickness_nm:
            raise ValueError(
                "parameters['tem_slice_thickness_nm'] must be set for high-fidelity "
                "multislice source accumulation."
            )
        if source_canvas.ndim != 3:
            raise ValueError("3D source canvas required for syniscopy_multislice accumulation.")
        n_slices, h, w = source_canvas.shape
        if n_slices != self._multislice_slices:
            raise ValueError(
                "TEM syniscopy_multislice expects source stack depth to match "
                f"tem_multislice_slices={self._multislice_slices}; got {n_slices}."
            )
        if source_canvas is None:
            return
        mip = float(getattr(material_properties, "mean_inner_potential_V", 0.0))
        if mip <= 0.0 or self._phase_shift_per_volt_nm <= 0.0:
            return
        scale = self._projected_potential_scale
        dz_nm = float(self._slice_thickness_nm)
        if not np.isfinite(dz_nm) or dz_nm <= 0.0:
            raise ValueError(
                "parameters['tem_slice_thickness_nm'] must be positive and finite; "
                f"got {dz_nm!r}."
            )

        if particle_z_nm is None:
            particle_z_nm = 0.0
        else:
            particle_z_nm = float(particle_z_nm)
            if not np.isfinite(particle_z_nm):
                raise ValueError(f"particle_z_nm must be finite for TEM multislice source placement; got {particle_z_nm!r}.")

        radius_nm = float(component_geometry.axial_half_extent_nm(orientation_matrix))
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
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        edge_scale = self._phase_shift_per_volt_nm * mip * scale * multiplier

        slab_centers_nm = self._source_z_center_nm + (
            np.arange(float(n_slices), dtype=float) - 0.5 * (n_slices - 1)
        ) * dz_nm
        slab_half = 0.5 * dz_nm
        slab_low = slab_centers_nm - slab_half
        slab_high = slab_centers_nm + slab_half
        z_center = np.float64(particle_z_nm)
        # Do not silently rewrite trajectory z. A particle outside the resolved
        # multislice slab means the requested scene cannot be represented by this
        # source volume; callers should expand the source volume or reduce axial
        # motion instead of accepting clipped physics.
        slab_min_center = float(slab_low[0]) + radius_nm
        slab_max_center = float(slab_high[-1]) - radius_nm
        if np.isfinite(slab_min_center) and np.isfinite(slab_max_center) and slab_min_center <= slab_max_center:
            z_float = float(z_center)
            tol_nm = max(
                1.0e-9,
                1.0e-12
                * max(abs(slab_min_center), abs(slab_max_center), abs(z_float), 1.0),
            )
            if z_float < slab_min_center - tol_nm or z_float > slab_max_center + tol_nm:
                raise ValueError(
                    "TEM particle_z_nm lies outside the resolved multislice slab: "
                    f"z={z_float} nm, allowed center range=[{slab_min_center}, {slab_max_center}] nm. "
                    "Increase parameters['tem_multislice_slices'], reduce axial motion/span, "
                    "or adjust parameters['tem_slice_thickness_nm']."
                )
        elif np.isfinite(radius_nm) and radius_nm > 0.0:
            raise ValueError(
                "TEM particle primitive is thicker than the resolved multislice slab: "
                f"axial half extent={float(radius_nm)} nm, slab thickness={float(n_slices * dz_nm)} nm. "
                "Increase parameters['tem_multislice_slices'] or parameters['tem_slice_thickness_nm']."
        )
        for idx in range(n_slices):
            slab_lower = float(slab_low[idx])
            slab_upper = float(slab_high[idx])
            thickness_nm = footprint.average_over_samples(
                lambda z_lower_rel_nm, z_upper_rel_nm: np.maximum(
                    np.minimum(slab_upper, z_center + z_upper_rel_nm)
                    - np.maximum(slab_lower, z_center + z_lower_rel_nm),
                    0.0,
                )
            )
            if np.any(thickness_nm > 0.0):
                source_canvas[idx, footprint.y0:footprint.y1, footprint.x0:footprint.x1] += (
                    edge_scale * thickness_nm
                )


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
        source_coordinate_context=None,
        source_multiplier: float = 1.0,
        component_geometry=None,
        orientation_matrix=None,
    ) -> None:
        if component_geometry is None:
            raise ValueError("TEM source accumulation requires component_geometry.")
        if source_coordinate_context is not None:
            particle_z_nm = source_coordinate_context.source_density_z_nm
        if source_canvas is None:
            return
        if self._tem_high_fidelity_backend is not None and self._slice_thickness_nm is not None:
            self._accumulate_particle_source_multislice_stack(
                source_canvas,
                center_x_canvas=center_x_canvas,
                center_y_canvas=center_y_canvas,
                diameter_nm=diameter_nm,
                pixel_size_nm=pixel_size_nm,
                os_factor=os_factor,
                material_properties=material_properties,
                params=params,
                particle_z_nm=particle_z_nm,
                source_multiplier=source_multiplier,
                component_geometry=component_geometry,
                orientation_matrix=orientation_matrix,
            )
            return
        if particle_z_nm is not None:
            z_float = float(particle_z_nm)
            if not np.isfinite(z_float):
                raise ValueError(f"particle_z_nm must be finite for TEM projected-source accumulation; got {particle_z_nm!r}.")
            if abs(z_float) > 1e-12:
                raise ValueError(
                    "TEM projected 2D source accumulation cannot represent nonzero particle_z_nm "
                    f"({z_float} nm). Use a slice-resolved multislice TEM source or set particle z to 0."
                )
        mip = float(getattr(material_properties, "mean_inner_potential_V", 0.0))
        if mip <= 0.0 or self._phase_shift_per_volt_nm <= 0.0:
            return
        scale = self._projected_potential_scale
        h, w = source_canvas.shape
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
        thickness_nm = footprint.projected_chord_nm()
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        phase = multiplier * scale * self._phase_shift_per_volt_nm * mip * thickness_nm
        source_canvas[footprint.y0:footprint.y1, footprint.x0:footprint.x1] += phase

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
        del E_sca_particles, particle_instances, background_field, frame_index
        source = self._source_from_particle_source_maps(E_sca_total, particle_source_maps)
        return self._intensity_from_projected_phase(source)

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
        source = self._source_from_particle_source_maps(E_sca_total, particle_source_maps)
        if (
            sample_environment is not None
            and self._tem_potential_source != self._TEM_POTENTIAL_SOURCE_MATERIAL
        ):
            source = self._compose_tem_projected_potential_source(
                source,
                sample_environment,
                params,
            )
        return np.maximum(self._intensity_from_projected_phase(source), 0.0)

    @staticmethod
    def _source_from_particle_source_maps(
        E_sca_total: np.ndarray,
        particle_source_maps: list[np.ndarray] | None,
    ) -> np.ndarray:
        if particle_source_maps is None or len(particle_source_maps) == 0:
            return np.zeros_like(E_sca_total, dtype=float)
        source_raw = np.sum(
            [np.asarray(source_map, dtype=np.complex128) for source_map in particle_source_maps],
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
                    "TEM particle source maps must remain real-valued projected "
                    f"phase/potential sources; max imaginary component is {imag_max}."
                )
            return np.asarray(source_raw.real, dtype=float)
        return np.asarray(source_raw, dtype=float)

    def apply_sample_environment(
        self,
        intensity: np.ndarray,
        E_sca_total: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        sample_environment: SampleEnvironment | None,
    ) -> np.ndarray:
        del background_field
        if sample_environment is None or self._tem_potential_source == self._TEM_POTENTIAL_SOURCE_MATERIAL:
            return np.asarray(intensity, dtype=float)
        source = self._compose_tem_projected_potential_source(E_sca_total, sample_environment, params)
        return np.maximum(self._intensity_from_projected_phase(source), 0.0)

    def compute_per_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Legacy direct-call contrast: CTF-filtered projected phase source."""
        del background_field, params
        return self._contrast_from_projected_phase(E_sca_particle)

    def _tem_direct_detector_scale(self, params: dict) -> float:
        dose = TemSettings.from_params(params).dose_per_pixel
        if not np.isfinite(dose) or dose < 0.0:
            raise ValueError(
                "parameters['tem_dose_per_pixel'] must be finite and non-negative; "
                f"got {dose}."
            )
        return float(dose)

    def _direct_signal_product_from_projected_source(
        self,
        source: np.ndarray,
        params: dict,
        *,
        producer: str,
        frame_index: int = 0,
    ) -> DirectParticleSignalProduct:
        # The fix-site invariant is detector-transfer ownership: TEM projected
        # phase/relative contrast is not an electron-count derivative until the
        # incident dose is applied.  This product performs that dose transfer
        # exactly once and keeps projected-phase provenance attached.
        projected = np.asarray(source, dtype=float)
        if projected.ndim == 3 and self._tem_high_fidelity_backend is None:
            projected = np.sum(projected, axis=0)
        contrast = self._contrast_from_projected_phase(projected)
        dose = self._tem_direct_detector_scale(params)
        electron_delta = dose * np.asarray(contrast, dtype=float)
        return DirectParticleSignalProduct(
            values=electron_delta,
            representation=electron_count_delta_representation(),
            modality="tem_phase_contrast",
            producer=producer,
            safe_for_fisher=True,
            detector_scale_applied=True,
            background_included=False,
            source_representation=tem_projected_phase_source_representation(),
            detector_scale_factor=float(dose),
            conversion_note=(
                "Converted TEM projected-phase/relative contrast response to "
                "electron-count contribution using tem_dose_per_pixel."
            ),
            provenance={"frame_index": int(frame_index)},
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
            "TEM direct particle contrast no longer returns a bare array. Use "
            "compute_particle_signal_product(); its metadata records the "
            "projected-phase to electron-count transfer before Fisher use."
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
        if particle_instance is None:
            return self._direct_signal_product_from_projected_source(
                E_sca_particle,
                params,
                producer="TransmissionElectronMicroscopyImagingModel.compute_particle_signal_product",
                frame_index=frame_index,
            )
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct TEM particle signal for composite particles requires a "
                "rendered source map; use compute_particle_signal_product_from_source_map()."
            )
        shape = E_sca_particle.shape
        traj = np.asarray(particle_instance.trajectory_nm, dtype=float)
        frame_idx = int(np.clip(int(frame_index), 0, traj.shape[0] - 1))
        sampling = TemSettings.from_params(params).sampling
        os_factor = sampling.psf_oversampling_factor
        px = float(traj[frame_idx, 0]) / sampling.detector_pixel_size_nm * float(os_factor)
        py = float(traj[frame_idx, 1]) / sampling.detector_pixel_size_nm * float(os_factor)
        pz = float(traj[frame_idx, 2]) if traj.shape[1] >= 3 else 0.0
        source = self._projected_phase_source(
            shape=shape,
            center_x_canvas=px,
            center_y_canvas=py,
            diameter_nm=float(particle_instance.particle_type.diameter_nm),
            pixel_size_nm=sampling.detector_pixel_size_nm,
            os_factor=os_factor,
            material_properties=getattr(particle_instance, "material_properties", None),
            params=params,
            particle_z_nm=pz,
            component_geometry=getattr(particle_instance, "component_geometry", None),
            orientation_matrix=None,
        )
        return self._direct_signal_product_from_projected_source(
            source,
            params,
            producer="TransmissionElectronMicroscopyImagingModel.compute_particle_signal_product",
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
            "TEM source-map contrast no longer returns a bare array. Use "
            "compute_particle_signal_product_from_source_map(); it converts "
            "projected-phase response to electron-count contribution exactly once."
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
        return self._direct_signal_product_from_projected_source(
            np.asarray(particle_source_map, dtype=float),
            params,
            producer="TransmissionElectronMicroscopyImagingModel.compute_particle_signal_product_from_source_map",
            frame_index=frame_index,
        )

    def convert_model_output_to_detector_frame(
        self,
        model_output: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        TEM intensity-to-counts conversion.

        The TEM model's compute_intensity returns a dimensionless relative
        direct-beam intensity. ``tem_dose_per_pixel`` is the incident primary
        electron dose per pixel, so detected counts are the incident dose
        multiplied by the relative transmitted/objective-filtered intensity.
        Do not renormalize by the frame mean here: attenuation, aperture loss,
        and energy/objective filtering are physical changes to detected counts.
        """
        dose = TemSettings.from_params(params).dose_per_pixel
        if not np.isfinite(dose) or dose < 0.0:
            raise ValueError(
                "parameters['tem_dose_per_pixel'] must be finite and non-negative; "
                f"got {dose}."
            )
        return convert_model_output_to_detector_frame(
            model_output=model_output,
            background_frame=background_final,
            reference_intensity_frame=E_ref_intensity_final,
            conversion=DetectorFrameConversion(
                model_output_domain=MODEL_OUTPUT_DOMAIN_RELATIVE_INTENSITY,
                detector_output_domain=DETECTOR_OUTPUT_DOMAIN_ELECTRON_COUNT,
                value_form=VALUE_FORM_ABSOLUTE,
                reference_basis=REFERENCE_BASIS_NONE,
                scale=dose,
                measurement_domain="electron_count",
                signal_units="electron_count",
            ),
            params=params,
            context="TransmissionElectronMicroscopyImagingModel.convert_model_output_to_detector_frame",
        )

__all__ = ['TransmissionElectronMicroscopyImagingModel']
