"""Substrate pattern, substrate, and sample-environment containers."""
from __future__ import annotations

from config import SampleEnvironmentSettings, SamplingGeometry
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable
from param_schema.sample_environment import PATTERN_PRESET_SPECS
from simulation_runtime_state import runtime_state_or_default

from .materials import (
    MaterialProperties,
    SIO2,
    WATER,
    _material_with_param_overrides,
    material_from_name,
)


@dataclass
class Pattern:
    """Modality-agnostic pattern overlay with height and material maps."""

    height_map_nm: np.ndarray
    material_fraction_map: np.ndarray
    pixel_size_nm: float
    kind: str = "none"

    @staticmethod
    def uniform(shape: tuple[int, int], pixel_size_nm: float, *, height_nm: float = 0.0) -> "Pattern":
        material_fraction = 1.0 if float(height_nm) > 0.0 else 0.0
        return Pattern(
            height_map_nm=np.full(shape, float(height_nm), dtype=float),
            material_fraction_map=np.full(shape, material_fraction, dtype=float),
            pixel_size_nm=float(pixel_size_nm),
            kind="uniform",
        )

    @staticmethod
    def hexagonal_hole_array(
        shape: tuple[int, int],
        pixel_size_nm: float,
        *,
        pitch_nm: float,
        hole_diameter_nm: float,
        layer_thickness_nm: float,
    ) -> "Pattern":
        h, w = shape
        yy, xx = np.indices(shape, dtype=float)
        x_nm = (xx - (w - 1) / 2.0) * pixel_size_nm
        y_nm = (yy - (h - 1) / 2.0) * pixel_size_nm
        pitch_nm = max(float(pitch_nm), 1.0)
        row_pitch = np.sqrt(3.0) * pitch_nm / 2.0
        nearest = np.zeros(shape, dtype=float) + np.inf
        n_rows = int(np.ceil((h * pixel_size_nm) / row_pitch)) + 3
        n_cols = int(np.ceil((w * pixel_size_nm) / pitch_nm)) + 3
        for r in range(-n_rows, n_rows + 1):
            cy = r * row_pitch
            offset = 0.5 * pitch_nm if (r & 1) else 0.0
            for c in range(-n_cols, n_cols + 1):
                cx = c * pitch_nm + offset
                d2 = (x_nm - cx) ** 2 + (y_nm - cy) ** 2
                nearest = np.minimum(nearest, d2)
        holes = nearest <= (0.5 * float(hole_diameter_nm)) ** 2
        height = np.full(shape, float(layer_thickness_nm), dtype=float)
        height[holes] = 0.0
        frac = np.ones(shape, dtype=float)
        frac[holes] = 0.0
        return Pattern(height, frac, float(pixel_size_nm), kind="hexagonal_hole_array")

    @staticmethod
    def square_grid(
        shape: tuple[int, int],
        pixel_size_nm: float,
        *,
        pitch_nm: float,
        bar_width_nm: float,
        layer_thickness_nm: float,
    ) -> "Pattern":
        h, w = shape
        yy, xx = np.indices(shape, dtype=float)
        x_nm = np.mod((xx - (w - 1) / 2.0) * pixel_size_nm + pitch_nm / 2.0, pitch_nm) - pitch_nm / 2.0
        y_nm = np.mod((yy - (h - 1) / 2.0) * pixel_size_nm + pitch_nm / 2.0, pitch_nm) - pitch_nm / 2.0
        bars = (np.abs(x_nm) <= bar_width_nm / 2.0) | (np.abs(y_nm) <= bar_width_nm / 2.0)
        height = np.zeros(shape, dtype=float)
        height[bars] = float(layer_thickness_nm)
        return Pattern(height, bars.astype(float), float(pixel_size_nm), kind="square_grid")

    @staticmethod
    def fiducial_dot_array(
        shape: tuple[int, int],
        pixel_size_nm: float,
        *,
        pitch_nm: float,
        dot_diameter_nm: float,
        layer_thickness_nm: float,
    ) -> "Pattern":
        pattern = Pattern.square_grid(
            shape,
            pixel_size_nm,
            pitch_nm=pitch_nm,
            bar_width_nm=dot_diameter_nm,
            layer_thickness_nm=0.0,
        )
        h, w = shape
        yy, xx = np.indices(shape, dtype=float)
        x_nm = np.mod((xx - (w - 1) / 2.0) * pixel_size_nm + pitch_nm / 2.0, pitch_nm) - pitch_nm / 2.0
        y_nm = np.mod((yy - (h - 1) / 2.0) * pixel_size_nm + pitch_nm / 2.0, pitch_nm) - pitch_nm / 2.0
        dots = x_nm**2 + y_nm**2 <= (dot_diameter_nm / 2.0) ** 2
        height = np.zeros(shape, dtype=float)
        height[dots] = float(layer_thickness_nm)
        pattern.height_map_nm = height
        pattern.material_fraction_map = dots.astype(float)
        pattern.kind = "fiducial_dot_array"
        return pattern


@dataclass
class Substrate:
    """Structured substrate shared by all imaging models."""

    height_map_nm: np.ndarray
    material_top: MaterialProperties
    material_layer: MaterialProperties
    material_substrate: MaterialProperties
    pixel_size_nm: float
    material_fraction_map: np.ndarray | None = None
    kind: str = "thin_film"

    def __post_init__(self) -> None:
        self.height_map_nm = np.asarray(self.height_map_nm, dtype=float)
        if self.material_fraction_map is None:
            self.material_fraction_map = np.where(self.height_map_nm > 0.0, 1.0, 0.0)
        else:
            self.material_fraction_map = np.asarray(self.material_fraction_map, dtype=float)
            if self.material_fraction_map.shape != self.height_map_nm.shape:
                raise ValueError("material_fraction_map shape must match height_map_nm.")

    def transmission_phase(self, wavelength_nm: float) -> np.ndarray:
        """Thin-film transmission phase from the patterned layer thickness."""
        n_layer = self.material_layer.n_complex(wavelength_nm)
        n_top = self.material_top.n_complex(wavelength_nm)
        opl_nm = (n_layer - n_top) * self.height_map_nm
        return np.exp(1j * 2.0 * np.pi * opl_nm / float(wavelength_nm))

    def reflection_amplitude(self, wavelength_nm: float) -> np.ndarray:
        """Two-interface normal-incidence thin-film reflection amplitude."""
        n0 = self.material_top.n_complex(wavelength_nm)
        n1 = self.material_layer.n_complex(wavelength_nm)
        n2 = self.material_substrate.n_complex(wavelength_nm)
        r01 = (n0 - n1) / (n0 + n1)
        r12 = (n1 - n2) / (n1 + n2)
        beta = 2.0 * np.pi * n1 * self.height_map_nm / float(wavelength_nm)
        round_trip = np.exp(2j * beta)
        denom = 1.0 + r01 * r12 * round_trip
        return (r01 + r12 * round_trip) / np.where(np.abs(denom) > 1e-12, denom, 1e-12)

    def projected_potential_V_nm(self) -> np.ndarray:
        """Projected mean inner potential contribution for TEM-style models."""
        return self.material_layer.mean_inner_potential_V * self.height_map_nm

    def topography_gradient(self) -> np.ndarray:
        """Magnitude of the height-map gradient for SEM-style edge contrast."""
        gy, gx = np.gradient(self.height_map_nm, float(self.pixel_size_nm))
        return np.sqrt(gx**2 + gy**2)

    def secondary_electron_yield_map(self) -> np.ndarray:
        """Material-dependent secondary-electron yield baseline map."""
        frac = np.asarray(self.material_fraction_map, dtype=float)
        frac = np.where(np.asarray(self.height_map_nm, dtype=float) > 0.0, frac, 0.0)
        return (
            frac * self.material_layer.se_yield_coefficient
            + (1.0 - frac) * self.material_substrate.se_yield_coefficient
        )

    def autofluorescence_density(self) -> np.ndarray:
        """Relative substrate autofluorescence source density."""
        return self.material_layer.autofluorescence_per_nm * np.maximum(self.height_map_nm, 0.0)


@dataclass
class SampleEnvironment:
    """Everything in the scene that is not the particle."""

    substrate: Substrate
    medium: MaterialProperties = WATER
    pattern: Pattern | None = None
    description: str = "sample environment"

    @property
    def mounting_interface(self) -> Substrate:
        return self.substrate

def sample_environment_from_params(
    params: dict,
    shape: tuple[int, int],
    *,
    pixel_size_nm: float | None = None,
) -> SampleEnvironment:
    """Build a lightweight environment from parameters."""

    px = float(
        pixel_size_nm
        if pixel_size_nm is not None
        else SamplingGeometry.from_params(params).detector_pixel_size_nm
    )
    settings = SampleEnvironmentSettings.from_params(params)

    def _dimension_nm(key: str) -> float:
        value = float(settings.dimension(key))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"sample_environment_pattern_dimensions[{key!r}] must be finite and non-negative.")
        return value

    def _pattern_layer_defaults(kind: str) -> tuple[str, float]:
        if kind not in PATTERN_PRESET_SPECS:
            raise ValueError(f"Unsupported sample_environment_pattern {kind!r}.")
        spec = PATTERN_PRESET_SPECS[kind]
        material_override = settings.pattern_material
        material_name = material_override if material_override is not None else spec["material"]
        thickness_nm = _dimension_nm(str(spec["thickness_dimension_key"]))
        return str(material_name), float(thickness_nm)

    layer_thickness_nm = settings.mounting_interface_thickness_nm
    layer_material_name = settings.mounting_interface_material
    medium = _material_with_param_overrides(
        material_from_name(settings.medium_material, WATER),
        params,
        "medium",
    )
    base = _material_with_param_overrides(
        material_from_name(settings.bulk_substrate_material, SIO2),
        params,
        "support",
    )

    if not settings.pattern_active:
        pattern = Pattern.uniform(shape, px, height_nm=0.0)
    else:
        from substrate.patterns import (
            canonical_sample_environment_pattern_and_preset,
            generate_sample_environment_pattern_maps,
        )

        kind, preset = canonical_sample_environment_pattern_and_preset(
            settings.pattern,
            settings.pattern_preset,
        )
        if kind == "none" or preset == "empty_background":
            pattern = Pattern.uniform(shape, px, height_nm=0.0)
        else:
            layer_material_name, layer_thickness_nm = _pattern_layer_defaults(kind)
            layout_extent_nm = runtime_state_or_default(params).substrate_pattern_layout_extent_nm
            height_map, material_fraction_map, pattern_kind = generate_sample_environment_pattern_maps(
                params,
                shape,
                px,
                layer_thickness_nm,
                layout_extent_nm=layout_extent_nm,
            )
            pattern = Pattern(
                height_map_nm=height_map,
                material_fraction_map=material_fraction_map,
                pixel_size_nm=px,
                kind=pattern_kind,
            )

    layer = _material_with_param_overrides(
        material_from_name(layer_material_name, SIO2),
        params,
        "mounting_interface",
    )

    substrate = Substrate(
        height_map_nm=pattern.height_map_nm,
        material_top=medium,
        material_layer=layer,
        material_substrate=base,
        pixel_size_nm=px,
        material_fraction_map=pattern.material_fraction_map,
        kind=pattern.kind,
    )
    return SampleEnvironment(
        substrate=substrate,
        medium=medium,
        pattern=pattern,
        description=f"{pattern.kind} in {medium.name}",
    )

def build_substrate_nuisance_basis(
    sample_environment: SampleEnvironment,
    *,
    basis: str = "height_gradient_material",
) -> dict[str, np.ndarray]:
    """
    Build signed substrate/background basis maps for nuisance-Fisher diagnostics.

    The returned maps are detector-grid covariates; imaging-model-specific
    response transforms can be applied by callers before passing them to the
    Fisher Schur-complement helper.
    """
    if basis != "height_gradient_material":
        raise ValueError(
            "Supported basis values: 'height_gradient_material'; "
            f"got {basis!r}."
        )
    substrate = sample_environment.substrate
    height = np.asarray(substrate.height_map_nm, dtype=float)
    gy, gx = np.gradient(height, float(substrate.pixel_size_nm))
    return {
        "height": height,
        "height_grad_x": gx.astype(float),
        "height_grad_y": gy.astype(float),
        "material_fraction": np.asarray(substrate.material_fraction_map, dtype=float),
        "topography_gradient": substrate.topography_gradient().astype(float),
        "secondary_electron_yield": substrate.secondary_electron_yield_map().astype(float),
        "autofluorescence_density": substrate.autofluorescence_density().astype(float),
    }

__all__ = [
    "Pattern",
    "SampleEnvironment",
    "Substrate",
    "build_substrate_nuisance_basis",
    "sample_environment_from_params",
]
