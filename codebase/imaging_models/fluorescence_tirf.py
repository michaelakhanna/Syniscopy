"""fluorescence tirf imaging model."""

from __future__ import annotations

from config.runtime import OpticalInstrumentSettings, TirfSettings

from ._shared import SourceCoordinateContext, np
from .fluorescence_widefield import FluorescenceWidefieldImagingModel, _convolve1d_reflect_same
from .source_rasterization import (
    normalize_sliced_source_to_projected_chord,
    primitive_footprint_patch,
)

class TIRFFluorescenceImagingModel(FluorescenceWidefieldImagingModel):
    """TIRF fluorescence with evanescent excitation applied to material source maps."""

    @staticmethod
    def penetration_depth_nm(params: dict) -> float:
        return TirfSettings.from_params(params).penetration_depth_nm

    def __init__(self, params: dict) -> None:
        self._tirf_settings = TirfSettings.from_params(params)
        settings = self._tirf_settings
        self._tirf_vectorial_effective_na_applied = settings.vectorial_effective_na_applied
        super().__init__(
            params,
            fluorescence_settings=settings.fluorescence,
            vectorial_numerical_aperture=(
                settings.effective_numerical_aperture
                if settings.vectorial_effective_na_applied
                else None
            ),
        )
        effective_na = settings.effective_numerical_aperture
        if effective_na is None:
            self._tirf_emission_sigma_multiplier = 1.0
        else:
            effective_na = float(effective_na)
            if not np.isfinite(effective_na) or effective_na <= 0.0:
                raise ValueError("parameters['tirf_effective_numerical_aperture'] must be positive when set.")
            detection_na = OpticalInstrumentSettings.from_params(params).numerical_aperture
            if not np.isfinite(detection_na) or detection_na <= 0.0:
                raise ValueError(
                    "parameters['numerical_aperture'] must be positive when TIRF "
                    "effective NA is set."
                )
            self._tirf_emission_sigma_multiplier = max(detection_na / effective_na, 1e-6)

    def _penetration_depth_nm(self) -> float:
        return self._tirf_settings.penetration_depth_nm

    def source_coordinate_contract(self, params: dict) -> dict:
        del params
        return {
            "source_density_z_basis": "physical_sample_world",
            "source_z_planes_basis": (
                "physical_sample_world" if self._uses_volume_source() else "projected_no_z"
            ),
            "optical_response_z_basis": (
                "focus_relative"
                if self._uses_volume_source()
                and self._vectorial_photophysics_backend is not None
                else "projected_no_z"
            ),
            "tirf_excitation_z_basis": "physical_interface_height",
        }

    def _default_direct_signal_modality(self) -> str:
        return "tirf_fluorescence"

    def _direct_signal_source_metadata(self, params: dict) -> dict[str, object]:
        contract = self.source_coordinate_contract(params)
        return {
            "source_input_kind": (
                "z_sliced_tirf_evanescently_weighted_emitter_density"
                if self._uses_volume_source()
                else "projected_2d_tirf_evanescently_weighted_emitter_density"
            ),
            "source_z_basis": contract.get("source_density_z_basis"),
            "source_projection_policy": (
                "z_sliced_sample_side_evanescent_line_integral_before_emission_psf"
                if self._uses_volume_source()
                else "sample_side_evanescent_excitation_line_integrated_over_particle_chord_before_emission_psf"
            ),
            "source_map_ndim": 3 if self._uses_volume_source() else 2,
            "source_axis_order": "zyx" if self._uses_volume_source() else "yx",
            "tirf_penetration_depth_nm": self._penetration_depth_nm(),
            "tirf_height_offset_nm": self._tirf_settings.height_offset_nm,
            "tirf_excitation_z_basis": contract.get("tirf_excitation_z_basis"),
        }

    def _sample_environment_autofluorescence_layer_world_z_nm(self, params: dict) -> float:
        del params
        # Interface-bound substrate autofluorescence is a physical source layer,
        # while TIRF excitation height is h = z_world + tirf_height_offset_nm.
        # Therefore the interface layer itself lives at z_world=-offset; using
        # z_world=0 would give vectorial volume TIRF the wrong emission defocus.
        return -self._tirf_settings.height_offset_nm

    def _effective_particle_height_nm(
        self,
        particle_z_nm: float | None,
    ) -> float:
        if particle_z_nm is None:
            raise ValueError(
                "TIRF excitation requires particle trajectory z. Particle "
                "height above the interface is not a separate public scalar; "
                "set particles[*].motion.initial_position_nm[2] or provide "
                "trajectory z through the renderer source-coordinate context. "
                "tirf_height_offset_nm may be used only as a static interface "
                "offset."
            )
        # TIRF single-height-authority fix: evanescent excitation is evaluated
        # from one physical z coordinate. The particle trajectory owns z;
        # tirf_height_offset_nm shifts the interface/frame only. This prevents
        # the over-specified state where a public particle-height scalar could
        # disagree with Brownian z, saved trajectories, source maps, or
        # focus-relative optical response.
        return float(particle_z_nm) + self._tirf_settings.height_offset_nm

    def _emission_blur(self, arr: np.ndarray) -> np.ndarray:
        sigma = self._emission_sigma_px * self._tirf_emission_sigma_multiplier
        if sigma == 0.0:
            return arr
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(arr, sigma=sigma)
        except ImportError:
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

    def _material_source_scale_for_particle(
        self,
        material,
        params: dict,
        *,
        particle_z_nm: float | None = None,
    ) -> float:
        base = super()._material_source_scale_for_particle(
            material,
            params,
            particle_z_nm=particle_z_nm,
        )
        penetration_nm = self._penetration_depth_nm()
        particle_height_nm = self._effective_particle_height_nm(particle_z_nm)
        excitation_factor = np.exp(-max(particle_height_nm, 0.0) / penetration_nm)
        return float(base * excitation_factor)

    @staticmethod
    def _evanescent_chord_integral_nm(
        z_lower_height_nm: np.ndarray,
        z_upper_height_nm: np.ndarray,
        penetration_nm: float,
    ) -> np.ndarray:
        z0 = np.asarray(z_lower_height_nm, dtype=float)
        z1 = np.asarray(z_upper_height_nm, dtype=float)
        depth = max(float(penetration_nm), 1e-12)
        integral = np.zeros_like(z0, dtype=float)
        below_interface = z1 <= 0.0
        above_interface = z0 >= 0.0
        crossing = ~(below_interface | above_interface)
        integral[below_interface] = 0.0
        integral[above_interface] = depth * (
            np.exp(-z0[above_interface] / depth)
            - np.exp(-z1[above_interface] / depth)
        )
        integral[crossing] = (
            depth * (1.0 - np.exp(-np.maximum(z1[crossing], 0.0) / depth))
        )
        return np.maximum(integral, 0.0)

    def _evanescent_slice_integrals_nm(
        self,
        center_z_nm: float,
        z_lower_rel_nm: np.ndarray,
        z_upper_rel_nm: np.ndarray,
    ) -> np.ndarray:
        lower, upper = self._source_slice_bounds_nm()
        z0 = float(center_z_nm) + np.asarray(z_lower_rel_nm, dtype=float)
        z1 = float(center_z_nm) + np.asarray(z_upper_rel_nm, dtype=float)
        lo_source = np.maximum(z0[None, :, :], lower[:, None, None])
        hi_source = np.minimum(z1[None, :, :], upper[:, None, None])
        valid = hi_source > lo_source

        height_offset_nm = self._tirf_settings.height_offset_nm
        lo_height = lo_source + height_offset_nm
        hi_height = hi_source + height_offset_nm
        depth = max(self._penetration_depth_nm(), 1e-12)

        lo_clip = np.maximum(lo_height, 0.0)
        hi_clip = np.maximum(hi_height, 0.0)
        integrals = depth * (np.exp(-lo_clip / depth) - np.exp(-hi_clip / depth))
        integrals = np.where(valid, np.maximum(integrals, 0.0), 0.0)

        projected = self._evanescent_chord_integral_nm(
            z0 + height_offset_nm,
            z1 + height_offset_nm,
            depth,
        )
        missing = np.maximum(projected - np.sum(integrals, axis=0), 0.0)
        tolerance = 1.0e-9 * max(float(np.max(projected)) if projected.size else 0.0, 1.0)
        if np.any(missing > tolerance):
            raise ValueError(
                "TIRF volume source extends outside the configured z-slice support. "
                "Increase tirf_volume_slices, increase tirf_volume_slice_thickness_nm, "
                "or use projected source representation; evanescent source mass must "
                "not be reassigned to a different physical/interface height."
            )
        return integrals

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
            raise ValueError("TIRF source accumulation requires component_geometry.")
        if source_coordinate_context is not None:
            particle_z_nm = source_coordinate_context.source_density_z_nm
        scale = FluorescenceWidefieldImagingModel._material_source_scale(
            self,
            material_properties,
            params,
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
        center_height_nm = self._effective_particle_height_nm(particle_z_nm)
        # Weighted projected source maps must share the same projected-primitive
        # normalization contract as unweighted fluorescence. In the large
        # penetration-depth limit TIRF becomes uniform excitation, so the source
        # mass must reduce to the exact primitive volume instead of raw subpixel
        # quadrature; otherwise counts and Fisher/CRLB depend on pixel phase.
        excitation_projection = footprint.weighted_projected_source_integral(
            lambda z_lower_rel_nm, z_upper_rel_nm: self._evanescent_chord_integral_nm(
                center_height_nm + z_lower_rel_nm,
                center_height_nm + z_upper_rel_nm,
                self._penetration_depth_nm(),
            ),
            source_basis="tirf_evanescent_projected_primitive",
        )
        excitation_integral_nm = excitation_projection.cell_integral_nm
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        if source_arr.ndim == 3:
            center_z_nm = center_height_nm - self._tirf_settings.height_offset_nm
            raw_excitation_integrals_nm = footprint.average_over_samples(
                lambda z_lower_rel_nm, z_upper_rel_nm: self._evanescent_slice_integrals_nm(
                    center_z_nm,
                    z_lower_rel_nm,
                    z_upper_rel_nm,
                )
            )
            excitation_integrals_nm = normalize_sliced_source_to_projected_chord(
                raw_excitation_integrals_nm,
                excitation_integral_nm,
            )
            source_canvas[:, footprint.y0:footprint.y1, footprint.x0:footprint.x1] += (
                multiplier
                * scale
                * excitation_integrals_nm
            )
        elif source_arr.ndim == 2:
            source_canvas[footprint.y0:footprint.y1, footprint.x0:footprint.x1] += (
                multiplier * scale * excitation_integral_nm
            )
        else:
            raise ValueError(
                "TIRF fluorescence source canvas must be 2D (y, x) or 3D "
                f"(z, y, x); got shape {source_arr.shape!r}."
            )

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        if self._vectorial_photophysics_backend is not None:
            response.update(
                self._vectorial_photophysics_backend.metadata(
                    params,
                    tirf_depth_nm=self._penetration_depth_nm(),
                    source_ndim=3 if self._uses_volume_source() else 2,
                )
            )
        effective_sigma_px = self._emission_sigma_px * self._tirf_emission_sigma_multiplier
        effective_sigma_nm = effective_sigma_px * self._canvas_pitch_nm
        response.update(
            kind="tirf_evanescent_fluorescence",
            tirf_fluorescence_backend=self._fluorescence_backend,
            tirf_source_representation=self._source_representation,
            penetration_depth_nm=self._penetration_depth_nm(),
            tirf_incident_angle_deg=self._tirf_settings.incident_angle_deg,
            tirf_effective_numerical_aperture=self._tirf_settings.effective_numerical_aperture,
            tirf_effective_na_applied_to_vectorial_psf=bool(
                self._tirf_vectorial_effective_na_applied
            ),
            tirf_critical_angle_deg=self._tirf_settings.critical_angle_deg,
            tirf_prism_refractive_index=self._tirf_settings.prism_refractive_index,
            tirf_sample_refractive_index=self._tirf_settings.sample_refractive_index,
            tirf_particle_height_source="particle_trajectory_z_nm",
            tirf_height_offset_nm=self._tirf_settings.height_offset_nm,
            tirf_interface_world_z_nm=self._sample_environment_autofluorescence_layer_world_z_nm(params),
            tirf_sample_environment_autofluorescence_layer_role="interface_bound_physical_source_layer",
            base_emission_sigma_canvas_px=self._emission_sigma_px,
            base_emission_sigma_detector_px=self._emission_sigma_nm / self._detector_pixel_size_nm,
            base_emission_sigma_nm=self._emission_sigma_nm,
            emission_sigma_canvas_px=effective_sigma_px,
            emission_sigma_detector_px=effective_sigma_nm / self._detector_pixel_size_nm,
            emission_sigma_nm=effective_sigma_nm,
            emission_sigma_multiplier=self._tirf_emission_sigma_multiplier,
            source_input_kind=(
                "z_sliced_tirf_evanescently_weighted_emitter_density"
                if self._uses_volume_source()
                else "projected_2d_tirf_evanescently_weighted_emitter_density"
            ),
            source_map_ndim=3 if self._uses_volume_source() else 2,
            source_axis_order="zyx" if self._uses_volume_source() else "yx",
            source_projection_policy=(
                "z_sliced_sample_side_evanescent_line_integral_before_emission_psf"
                if self._uses_volume_source()
                else "sample_side_evanescent_excitation_line_integrated_over_particle_chord_before_emission_psf"
            ),
            source_stack_out_of_range_policy=(
                "error_on_uncovered_source_integral_outside_source_stack"
                if self._uses_volume_source()
                else None
            ),
            tirf_excitation_depth_model="cell_integrated_primitive_interval_sample_side_evanescent_line_integral",
        )
        return response

__all__ = ['TIRFFluorescenceImagingModel']
