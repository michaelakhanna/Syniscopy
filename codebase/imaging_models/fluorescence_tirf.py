"""fluorescence tirf imaging model."""

from __future__ import annotations

from ._shared import np
from .fluorescence_widefield import FluorescenceWidefieldImagingModel

class TIRFFluorescenceImagingModel(FluorescenceWidefieldImagingModel):
    """TIRF fluorescence with evanescent excitation applied to material source maps."""

    @staticmethod
    def penetration_depth_nm(params: dict) -> float:
        if params.get("tirf_use_angle_derived_penetration_depth", False):
            wavelength_nm = float(params.get("fluorescence_excitation_wavelength_nm", 488.0))
            n_prism = float(params.get("tirf_prism_refractive_index", 1.518))
            n_sample = float(params.get("tirf_sample_refractive_index", params.get("refractive_index_medium", 1.333)))
            angle_rad = np.deg2rad(float(params.get("tirf_incident_angle_deg", 66.0)))
            sin_term = n_prism * np.sin(angle_rad)
            under_root = sin_term * sin_term - n_sample * n_sample
            if under_root <= 0.0:
                raise ValueError(
                    "TIRF incident angle must exceed the critical angle when "
                    "'tirf_use_angle_derived_penetration_depth' is enabled."
                )
            return float(wavelength_nm / (4.0 * np.pi * np.sqrt(under_root)))
        penetration_nm = float(params.get("tirf_penetration_depth_nm", 120.0))
        if penetration_nm <= 0.0:
            raise ValueError("PARAMS['tirf_penetration_depth_nm'] must be positive.")
        return penetration_nm

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        effective_na = params.get("tirf_effective_numerical_aperture", None)
        if effective_na is None:
            self._tirf_emission_sigma_multiplier = 1.0
        else:
            effective_na = float(effective_na)
            if effective_na <= 0.0:
                raise ValueError("PARAMS['tirf_effective_numerical_aperture'] must be positive when set.")
            detection_na = float(params.get("numerical_aperture", effective_na))
            if detection_na <= 0.0:
                raise ValueError(
                    "PARAMS['numerical_aperture'] must be positive when TIRF "
                    "effective NA is set."
                )
            self._tirf_emission_sigma_multiplier = max(detection_na / effective_na, 1e-6)

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
                    lambda v: np.convolve(v, k1d, mode="same"),
                    axis, out,
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
        penetration_nm = self.penetration_depth_nm(params)
        if particle_z_nm is None:
            particle_height_nm = float(params.get("tirf_particle_height_nm", 0.0))
        else:
            particle_height_nm = float(particle_z_nm) + float(params.get("tirf_height_offset_nm", 0.0))
        excitation_factor = np.exp(-max(particle_height_nm, 0.0) / penetration_nm)
        return float(base * excitation_factor)

    @staticmethod
    def _evanescent_chord_integral_nm(
        center_height_nm: float,
        half_chord_nm: np.ndarray,
        penetration_nm: float,
    ) -> np.ndarray:
        half = np.maximum(np.asarray(half_chord_nm, dtype=float), 0.0)
        z0 = float(center_height_nm) - half
        z1 = float(center_height_nm) + half
        depth = max(float(penetration_nm), 1e-12)
        integral = np.zeros_like(half, dtype=float)
        below_interface = z1 <= 0.0
        above_interface = z0 >= 0.0
        crossing = ~(below_interface | above_interface)
        integral[below_interface] = np.maximum(z1[below_interface] - z0[below_interface], 0.0)
        integral[above_interface] = depth * (
            np.exp(-z0[above_interface] / depth)
            - np.exp(-z1[above_interface] / depth)
        )
        integral[crossing] = (
            np.maximum(-z0[crossing], 0.0)
            + depth * (1.0 - np.exp(-np.maximum(z1[crossing], 0.0) / depth))
        )
        return np.maximum(integral, 0.0)

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
        scale = FluorescenceWidefieldImagingModel._material_source_scale(
            self,
            material_properties,
            params,
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
        r_px = np.sqrt(dx * dx + dy * dy)
        inside = r_px <= radius_px
        lateral_nm = r_px * float(pixel_size_nm) / float(os_factor)
        radius_nm = 0.5 * float(diameter_nm)
        half_chord_nm = np.zeros_like(lateral_nm, dtype=float)
        half_chord_nm[inside] = np.sqrt(
            np.maximum(radius_nm * radius_nm - lateral_nm[inside] ** 2, 0.0)
        )
        if particle_z_nm is None:
            center_height_nm = float(params.get("tirf_particle_height_nm", 0.0))
        else:
            center_height_nm = float(particle_z_nm) + float(params.get("tirf_height_offset_nm", 0.0))
        excitation_integral_nm = self._evanescent_chord_integral_nm(
            center_height_nm,
            half_chord_nm,
            self.penetration_depth_nm(params),
        )
        edge_width = max(0.75, 0.5 * float(os_factor))
        disk = np.clip((radius_px + edge_width - r_px) / max(edge_width, 1e-9), 0.0, 1.0)
        multiplier = float(source_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(f"source_multiplier must be finite and non-negative; got {source_multiplier!r}.")
        source_canvas[y0:y1, x0:x1] += multiplier * scale * excitation_integral_nm * disk

    def compute_response_function(self, shape: tuple[int, int], params: dict) -> dict:
        response = super().compute_response_function(shape, params)
        if self._vectorial_photophysics_backend is not None:
            response.update(
                self._vectorial_photophysics_backend.metadata(
                    params,
                    tirf_depth_nm=self.penetration_depth_nm(params),
                )
            )
        response.update(
            kind="tirf_evanescent_fluorescence",
            penetration_depth_nm=self.penetration_depth_nm(params),
            tirf_incident_angle_deg=float(params.get("tirf_incident_angle_deg", 66.0)),
            tirf_critical_angle_deg=float(
                np.rad2deg(
                    np.arcsin(
                        min(
                            float(params.get("tirf_sample_refractive_index", params.get("refractive_index_medium", 1.333)))
                            / max(float(params.get("tirf_prism_refractive_index", 1.518)), 1e-12),
                            1.0,
                        )
                    )
                )
            ),
            tirf_prism_refractive_index=float(params.get("tirf_prism_refractive_index", 1.518)),
            tirf_sample_refractive_index=float(params.get("tirf_sample_refractive_index", params.get("refractive_index_medium", 1.333))),
            tirf_particle_height_nm=float(params.get("tirf_particle_height_nm", 0.0)),
            tirf_height_offset_nm=float(params.get("tirf_height_offset_nm", 0.0)),
            emission_sigma_multiplier=self._tirf_emission_sigma_multiplier,
            source_input_kind="projected_2d_tirf_evanescently_weighted_emitter_density",
            source_projection_policy="evanescent_excitation_line_integrated_over_particle_chord_before_emission_psf",
            tirf_excitation_depth_model="per_pixel_sphere_chord_evanescent_line_integral",
        )
        return response

__all__ = ['TIRFFluorescenceImagingModel']
