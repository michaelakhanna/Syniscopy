"""tem imaging model."""

from __future__ import annotations

from ._shared import (
    ImagingModel,
    SampleEnvironment,
    np,
)
from backend_fidelity import attach_backend_fidelity_metadata
from config.runtime import TemSettings, param_value
from .electron_constants import (
    electron_interaction_parameter_rad_per_V_nm,
    electron_wavelength_m,
    scherzer_defocus_m,
)
from .tem_backends import (
    CTFProxyTEMBackend,
    MultisliceLiteTEMBackend,
    SyniscopyMultisliceTEMBackend,
)

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
        CTF(k)    = -2 sin(chi(k)) * E(k),
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
    ``tem_backend='syniscopy_multislice'``, Syniscopy uses the native split-step
    multislice TEM backend.

    Parameters (PARAMS keys, all optional with nominal defaults)
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
      scale_intensity_to_counts to convert the dimensionless weak-phase
      image into detector counts.

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
    uses_sample_environment_pattern = True
    uses_particle_material_sources = True
    requires_complex_optical_psf = False
    requires_optical_scattered_field = False
    requires_pre_crop_optical_filtering = True
    supports_spectral_channels = False
    _TEM_POTENTIAL_SOURCE_MATERIAL = "material_projected_inner_potential"
    _TEM_POTENTIAL_SOURCE_SAMPLE_ENVIRONMENT = "sample_environment_projected_potential"
    _TEM_POTENTIAL_SOURCE_COMPOSITE = "material_plus_sample_environment"

    def __init__(self, params: dict) -> None:
        settings = TemSettings.from_params(params)
        self._tem_model = str(param_value(params, "tem_model")).strip().lower()
        if self._tem_model not in {"weak_phase_ctf", "multislice_lite", "syniscopy_multislice"}:
            raise ValueError(
                "PARAMS['tem_model'] must be 'weak_phase_ctf', "
                "'multislice_lite', or 'syniscopy_multislice'; "
                f"got {self._tem_model!r}."
            )
        self._tem_backend = str(param_value(params, "tem_backend")).strip().lower()
        if self._tem_backend not in {"ctf_proxy", "multislice_lite", "syniscopy_multislice"}:
            raise ValueError(
                "PARAMS['tem_backend'] must be 'ctf_proxy', 'multislice_lite', "
                f"'syniscopy_multislice'; got {self._tem_backend!r}."
            )
        if self._tem_backend == "multislice_lite" and self._tem_model != "multislice_lite":
            raise ValueError(
                "PARAMS['tem_backend']='multislice_lite' requires "
                "PARAMS['tem_model']='multislice_lite'."
            )
        if self._tem_model == "syniscopy_multislice" and self._tem_backend != "syniscopy_multislice":
            raise ValueError(
                "PARAMS['tem_model']='syniscopy_multislice' requires "
                "PARAMS['tem_backend']='syniscopy_multislice'."
            )
        if self._tem_model == "weak_phase_ctf" and self._tem_backend != "ctf_proxy":
            raise ValueError(
                "PARAMS['tem_model']='weak_phase_ctf' requires "
                "PARAMS['tem_backend']='ctf_proxy'."
            )
        if self._tem_backend == "ctf_proxy" and self._tem_model != "weak_phase_ctf":
            raise ValueError(
                "PARAMS['tem_backend']='ctf_proxy' requires "
                "PARAMS['tem_model']='weak_phase_ctf'."
            )
        self._tem_potential_source = self._resolve_tem_potential_source(
            param_value(params, "tem_potential_source")
        )
        self._V_kV = float(param_value(params, "tem_acceleration_kV"))
        if self._V_kV <= 0.0:
            raise ValueError(
                f"PARAMS['tem_acceleration_kV'] must be positive; got {self._V_kV}."
            )
        self._Cs_mm = float(param_value(params, "tem_Cs_mm"))
        if self._Cs_mm < 0.0:
            raise ValueError(
                f"PARAMS['tem_Cs_mm'] must be non-negative; got {self._Cs_mm}."
            )
        self._alpha_mrad = float(param_value(params, "tem_partial_coherence_alpha_mrad"))
        if self._alpha_mrad < 0.0:
            raise ValueError(
                f"PARAMS['tem_partial_coherence_alpha_mrad'] must be non-negative; "
                f"got {self._alpha_mrad}."
            )

        # Resolve wavelength and defocus (defocus defaults to Scherzer).
        self._lambda_m = electron_wavelength_m(self._V_kV)
        if "tem_defocus_nm" in params and params["tem_defocus_nm"] is not None:
            self._defocus_m = 1.0e-9 * float(params["tem_defocus_nm"])
        else:
            self._defocus_m = scherzer_defocus_m(self._V_kV, self._Cs_mm)

        phase_shift_raw = param_value(params, "tem_phase_shift_per_volt_nm")
        if phase_shift_raw is None:
            self._phase_shift_per_volt_nm = electron_interaction_parameter_rad_per_V_nm(self._V_kV)
        else:
            self._phase_shift_per_volt_nm = float(phase_shift_raw)
        if (
            not np.isfinite(self._phase_shift_per_volt_nm)
            or self._phase_shift_per_volt_nm < 0.0
        ):
            raise ValueError(
                "PARAMS['tem_phase_shift_per_volt_nm'] must be finite and "
                f"non-negative; got {self._phase_shift_per_volt_nm}."
            )

        canvas_pitch_nm = settings.sampling.model_canvas_pixel_size_nm
        if not np.isfinite(canvas_pitch_nm) or canvas_pitch_nm <= 0.0:
            raise ValueError(
                "PARAMS['pixel_size_nm'] / PARAMS['psf_oversampling_factor'] must "
                f"resolve to a positive pitch; got {canvas_pitch_nm} nm."
            )

        # Fourier-grid pitch is the physical pitch of the rendered model canvas.
        # tem_pixel_size_pm is retained only as a compatibility assertion so it
        # cannot silently make the CTF grid disagree with detector/Fisher units.
        self._pixel_size_m = 1.0e-9 * canvas_pitch_nm
        if "tem_pixel_size_pm" in params and params["tem_pixel_size_pm"] is not None:
            requested_m = 1.0e-12 * float(params["tem_pixel_size_pm"])
            if (
                not np.isfinite(requested_m)
                or requested_m <= 0.0
                or not np.isclose(requested_m, self._pixel_size_m, rtol=1e-6, atol=1e-15)
            ):
                raise ValueError(
                    "PARAMS['tem_pixel_size_pm'] must match the rendered model-canvas "
                    "pitch pixel_size_nm / psf_oversampling_factor. "
                    f"Got tem_pixel_size_pm={params['tem_pixel_size_pm']} pm and "
                    f"canvas pitch={canvas_pitch_nm * 1000.0:.6g} pm."
                )
        if not np.isfinite(self._pixel_size_m) or self._pixel_size_m <= 0.0:
            raise ValueError(
                "PARAMS['tem_pixel_size_pm'] or PARAMS['pixel_size_nm'] must resolve "
                f"to a positive pixel pitch; got {self._pixel_size_m} m."
            )

        self._dose_per_pixel = settings.dose_per_pixel
        self._multislice_slices = settings.multislice_slices
        self._objective_aperture_mrad = param_value(params, "tem_objective_aperture_mrad")
        if self._objective_aperture_mrad is not None:
            self._objective_aperture_mrad = float(self._objective_aperture_mrad)
            if not np.isfinite(self._objective_aperture_mrad) or self._objective_aperture_mrad <= 0.0:
                raise ValueError(
                    "PARAMS['tem_objective_aperture_mrad'] must be positive when set."
                )
        slice_thickness_raw = param_value(params, "tem_slice_thickness_nm")
        self._slice_thickness_nm = None if slice_thickness_raw is None else float(slice_thickness_raw)
        if self._slice_thickness_nm is not None and (
            not np.isfinite(self._slice_thickness_nm) or self._slice_thickness_nm <= 0.0
        ):
            raise ValueError(
                "PARAMS['tem_slice_thickness_nm'] must be positive and finite when set; "
                f"got {slice_thickness_raw!r}."
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
        )
        self._multislice_lite_backend = MultisliceLiteTEMBackend(
            ctf_backend=self._ctf_backend,
            electron_wavelength_m=self._lambda_m,
            pixel_size_m=self._pixel_size_m,
            multislice_slices=self._multislice_slices,
            slice_thickness_nm=self._slice_thickness_nm,
        )
        self._tem_high_fidelity_backend = None
        if self._tem_backend == "syniscopy_multislice":
            self._tem_high_fidelity_backend = SyniscopyMultisliceTEMBackend(
                params,
                pixel_size_m=self._pixel_size_m,
                electron_wavelength_m=self._lambda_m,
                Cs_mm=self._Cs_mm,
                defocus_m=self._defocus_m,
                partial_coherence_alpha_mrad=self._alpha_mrad,
                dose_per_pixel=self._dose_per_pixel,
                default_slice_count=self._multislice_slices,
                default_slice_thickness_nm=self._slice_thickness_nm,
                default_objective_aperture_mrad=self._objective_aperture_mrad,
            )

    @classmethod
    def _resolve_tem_potential_source(cls, raw: object) -> str:
        source = str(raw).strip().lower()
        if source in {
            cls._TEM_POTENTIAL_SOURCE_MATERIAL,
            "material",
            "particle_only",
            "material_projected_potential",
        }:
            return cls._TEM_POTENTIAL_SOURCE_MATERIAL
        if source in {
            cls._TEM_POTENTIAL_SOURCE_SAMPLE_ENVIRONMENT,
            "sample_environment",
            "sample_environment_only",
            "environment_only",
        }:
            return cls._TEM_POTENTIAL_SOURCE_SAMPLE_ENVIRONMENT
        if source in {
            cls._TEM_POTENTIAL_SOURCE_COMPOSITE,
            "material_plus_environment",
            "composite",
            "material_and_sample_environment",
            "particle_plus_sample_environment",
        }:
            return cls._TEM_POTENTIAL_SOURCE_COMPOSITE
        return source

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
        scale = float(param_value(params, "tem_sample_environment_potential_scale"))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "PARAMS['tem_sample_environment_potential_scale'] must be finite "
                f"and non-negative; got {scale!r}."
            )
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

    def filter_guard_radius_pixels(self, params: dict) -> int | None:
        raw = param_value(params, 'tem_filter_guard_pixels')
        if raw is None:
            return 64
        guard = float(raw)
        if not np.isfinite(guard) or guard < 0.0:
            raise ValueError(
                "PARAMS['tem_filter_guard_pixels'] must be non-negative and finite; "
                f"got {raw!r}."
            )
        return int(np.ceil(guard))


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
        )
        return source

    def probe_wavelength_nm(self, params: dict) -> float:
        del params
        return float(self._lambda_m * 1.0e9)

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        kind_by_model = {
            "weak_phase_ctf": "tem_ctf",
            "multislice_lite": "tem_multislice_lite",
            "syniscopy_multislice": "tem_multislice",
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
            "count_scaling_mode": "electron_dose_per_pixel",
            "forward_observable": (
                "|multislice-lite exit wave with objective CTF readout|^2"
                if self._tem_model == "multislice_lite"
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
            "slice_thickness_nm": self._slice_thickness_nm,
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
                    "physics_based_unvalidated"
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
                source_z_planes_nm=[
                    (idx - 0.5 * (self._multislice_slices - 1)) * float(self._slice_thickness_nm)
                    for idx in range(self._multislice_slices)
                ] if self._slice_thickness_nm is not None else None,
                source_z_origin="particle_centered_object_depth",
                source_z_uses_particle_world_z=False,
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
                source_z_planes_nm=[
                    (idx - 0.5 * (self._multislice_slices - 1)) * float(self._slice_thickness_nm)
                    for idx in range(self._multislice_slices)
                ] if self._slice_thickness_nm is not None else None,
                source_z_origin="particle_centered_object_depth",
                source_z_uses_particle_world_z=False,
            )
        elif self._tem_high_fidelity_backend is not None:
            response.update(
                source_input_kind="projected_2d_tem_environment_potential",
                source_map_ndim=2,
                source_axis_order="yx",
                source_projection_policy="projected_source_evenly_split_across_multislice_slices",
                backend_consumes_volume_source=False,
                volume_transport_model="split_step_multislice_projected_source_fallback",
                tem_environment_source_dimensionality="projected_2d",
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
    ) -> None:
        if not self._slice_thickness_nm:
            raise ValueError(
                "PARAMS['tem_slice_thickness_nm'] must be set for high-fidelity "
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
        scale = float(param_value(params, 'tem_projected_potential_scale'))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "PARAMS['tem_projected_potential_scale'] must be finite and "
                f"non-negative; got {scale}."
            )
        dz_nm = float(self._slice_thickness_nm)
        if not np.isfinite(dz_nm) or dz_nm <= 0.0:
            raise ValueError(
                "PARAMS['tem_slice_thickness_nm'] must be positive and finite; "
                f"got {dz_nm!r}."
            )

        if particle_z_nm is None:
            particle_z_nm = 0.0
        else:
            particle_z_nm = float(particle_z_nm)
            if not np.isfinite(particle_z_nm):
                particle_z_nm = 0.0

        radius_px = max(0.5, 0.5 * float(diameter_nm) / float(pixel_size_nm) * float(os_factor))
        radius_nm = 0.5 * float(diameter_nm)
        y0 = max(0, int(np.floor(center_y_canvas - radius_px - 2)))
        y1 = min(h, int(np.ceil(center_y_canvas + radius_px + 3)))
        x0 = max(0, int(np.floor(center_x_canvas - radius_px - 2)))
        x1 = min(w, int(np.ceil(center_x_canvas + radius_px + 3)))
        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
        dx = xx + x0 - float(center_x_canvas)
        dy = yy + y0 - float(center_y_canvas)
        r_px = np.sqrt(dx * dx + dy * dy)
        r_nm = r_px * float(pixel_size_nm) / float(os_factor)
        chord_half_nm = np.sqrt(np.maximum(radius_nm * radius_nm - r_nm * r_nm, 0.0))
        if not np.any(chord_half_nm > 0.0):
            return
        taper = np.clip((radius_px + max(0.75, 0.5 * float(os_factor)) - r_px) / max(0.5 * float(os_factor), 1e-9), 0.0, 1.0)
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        edge_scale = self._phase_shift_per_volt_nm * mip * scale * multiplier

        slab_centers_nm = (np.arange(float(n_slices), dtype=float) - 0.5 * (n_slices - 1)) * dz_nm
        slab_half = 0.5 * dz_nm
        slab_low = slab_centers_nm - slab_half
        slab_high = slab_centers_nm + slab_half
        # The slice stack is an object-internal material-thickness
        # representation. World/lab z controls optical defocus and trajectories
        # elsewhere; it must not translate the particle out of its own TEM
        # source stack.
        z_center = np.float64(0.0)
        for idx in range(n_slices):
            slab_lower = float(slab_low[idx])
            slab_upper = float(slab_high[idx])
            z_low = z_center - chord_half_nm
            z_high = z_center + chord_half_nm
            overlap = np.minimum(slab_upper, z_high) - np.maximum(slab_lower, z_low)
            thickness_nm = np.maximum(overlap, 0.0)
            if np.any(thickness_nm > 0.0):
                source_canvas[idx, y0:y1, x0:x1] += edge_scale * thickness_nm * taper


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
            )
            return
        mip = float(getattr(material_properties, "mean_inner_potential_V", 0.0))
        if mip <= 0.0 or self._phase_shift_per_volt_nm <= 0.0:
            return
        scale = float(param_value(params, 'tem_projected_potential_scale'))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "PARAMS['tem_projected_potential_scale'] must be finite and "
                f"non-negative; got {scale}."
            )
        radius_px = max(0.5, 0.5 * float(diameter_nm) / float(pixel_size_nm) * float(os_factor))
        h, w = source_canvas.shape
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
        thickness_nm = thickness_px * float(pixel_size_nm) / float(os_factor)
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        phase = multiplier * scale * self._phase_shift_per_volt_nm * mip * thickness_nm
        edge_width = max(0.75, 0.5 * float(os_factor))
        taper = np.clip((radius_px + edge_width - r) / max(edge_width, 1e-9), 0.0, 1.0)
        source_canvas[y0:y1, x0:x1] += phase * taper

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

    def compute_particle_contrast(
        self,
        E_sca_particle: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        particle_instance=None,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        background = background_field
        if particle_instance is None:
            return self.compute_per_particle_contrast(E_sca_particle, background, params)
        if bool(getattr(getattr(particle_instance, "particle_type", None), "is_composite", False)):
            raise ValueError(
                "Direct TEM particle contrast for composite particles requires a "
                "rendered source map; use compute_particle_contrast_from_source_map()."
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
        )
        return self._contrast_from_projected_phase(source)

    def compute_particle_contrast_from_source_map(
        self,
        particle_source_map: np.ndarray,
        background_field: np.ndarray,
        params: dict,
        *,
        frame_index: int = 0,
    ) -> np.ndarray:
        del background_field, params, frame_index
        source = np.asarray(particle_source_map, dtype=float)
        if source.ndim == 3 and self._tem_high_fidelity_backend is None:
            source = np.sum(source, axis=0)
        return self._contrast_from_projected_phase(source)

    def scale_intensity_to_counts(
        self,
        intensity: np.ndarray,
        background_final: np.ndarray,
        E_ref_intensity_final: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        TEM intensity-to-counts conversion.

        The TEM model's compute_intensity returns a dimensionless image
        normalized such that the unscattered direct beam has intensity
        one. We convert to detector counts by multiplying by the resolved
        TEM electron dose per pixel.
        """
        dose = TemSettings.from_params(params).dose_per_pixel
        if not np.isfinite(dose) or dose < 0.0:
            raise ValueError(
                "PARAMS['tem_dose_per_pixel'] must be finite and non-negative; "
                f"got {dose}."
            )
        return dose * intensity

__all__ = ['TransmissionElectronMicroscopyImagingModel']
