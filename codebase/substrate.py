"""Sample-environment abstractions for modality-agnostic rendering.

A sample environment contains the substrate, surrounding medium, and
optional pattern overlay. This module gives every imaging model the same
geometry/material object and lets each modality decide how to consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np


ComplexIndexModel = complex | Callable[[float], complex]


@dataclass(frozen=True)
class MaterialProperties:
    """
    Material properties used by optical, fluorescence, and electron models.

    Optical indices are dimensionless. Mean inner potential is in volts,
    density is g/cm^3, fluorescence/autofluorescence scales are relative source
    terms, and spectral peaks are wavelengths in nanometres.
    """

    name: str
    n_complex_visible: ComplexIndexModel = 1.0 + 0.0j
    mean_inner_potential_V: float = 0.0
    density_g_cm3: float = 0.0
    se_yield_coefficient: float = 0.0
    autofluorescence_per_nm: float = 0.0
    fluorophore_density: float = 0.0
    emission_peak_nm: float | None = None
    excitation_peak_nm: float | None = None
    polarizability_tensor: tuple[tuple[float, float, float], ...] | None = None

    def n_complex(self, wavelength_nm: float) -> complex:
        """Return the complex refractive index at ``wavelength_nm``."""
        model = self.n_complex_visible
        if callable(model):
            return complex(model(float(wavelength_nm)))
        return complex(model)


AIR = MaterialProperties("air", 1.00 + 0.0j)
VACUUM = MaterialProperties("vacuum", 1.00 + 0.0j)
WATER = MaterialProperties("water", 1.33 + 0.0j, mean_inner_potential_V=4.0, density_g_cm3=1.00)
SIO2 = MaterialProperties(
    "SiO2",
    1.46 + 0.0j,
    mean_inner_potential_V=10.1,
    density_g_cm3=2.20,
    se_yield_coefficient=0.10,
    autofluorescence_per_nm=0.02,
)
SI = MaterialProperties(
    "Si",
    3.88 + 0.02j,
    mean_inner_potential_V=11.7,
    density_g_cm3=2.33,
    se_yield_coefficient=0.13,
)
CARBON = MaterialProperties(
    "carbon",
    2.42 + 0.0j,
    mean_inner_potential_V=8.7,
    density_g_cm3=2.0,
    se_yield_coefficient=0.08,
)
GOLD = MaterialProperties(
    "gold",
    0.47 + 2.41j,
    mean_inner_potential_V=25.0,
    density_g_cm3=19.3,
    se_yield_coefficient=0.18,
)
SILVER = MaterialProperties(
    "silver",
    0.14 + 3.98j,
    mean_inner_potential_V=22.0,
    density_g_cm3=10.5,
    se_yield_coefficient=0.16,
)
GLASS = MaterialProperties("glass", 1.52 + 0.0j, mean_inner_potential_V=9.5, density_g_cm3=2.5)
PET = MaterialProperties(
    "PET",
    1.57 + 0.0j,
    mean_inner_potential_V=8.0,
    density_g_cm3=1.38,
    se_yield_coefficient=0.05,
)
POLYETHYLENE = MaterialProperties(
    "polyethylene",
    1.51 + 0.0j,
    mean_inner_potential_V=7.5,
    density_g_cm3=0.94,
    se_yield_coefficient=0.05,
)
POLYPROPYLENE = MaterialProperties(
    "polypropylene",
    1.49 + 0.0j,
    mean_inner_potential_V=7.5,
    density_g_cm3=0.90,
    se_yield_coefficient=0.05,
)
POLYSTYRENE = MaterialProperties(
    "polystyrene",
    1.59 + 0.0j,
    mean_inner_potential_V=8.0,
    density_g_cm3=1.05,
    se_yield_coefficient=0.05,
)
FLUORESCENT_POLYSTYRENE = replace(
    POLYSTYRENE,
    name="fluorescent_polystyrene",
    fluorophore_density=1.0,
    excitation_peak_nm=488.0,
    emission_peak_nm=520.0,
)
PROTEIN = MaterialProperties(
    "protein",
    1.45 + 0.0j,
    mean_inner_potential_V=6.0,
    density_g_cm3=1.35,
    se_yield_coefficient=0.04,
)
LIPID = MaterialProperties(
    "lipid",
    1.47 + 0.0j,
    mean_inner_potential_V=4.5,
    density_g_cm3=0.92,
    se_yield_coefficient=0.04,
)


_MATERIALS = {
    "air": AIR,
    "vacuum": VACUUM,
    "water": WATER,
    "buffer": WATER,
    "sio2": SIO2,
    "silica": SIO2,
    "silicon_dioxide": SIO2,
    "si": SI,
    "silicon": SI,
    "carbon": CARBON,
    "holey_carbon": CARBON,
    "gold": GOLD,
    "au": GOLD,
    "silver": SILVER,
    "ag": SILVER,
    "glass": GLASS,
    "pet": PET,
    "polyethylene_terephthalate": PET,
    "polyethylene": POLYETHYLENE,
    "polypropylene": POLYPROPYLENE,
    "polystyrene": POLYSTYRENE,
    "ps": POLYSTYRENE,
    "fluorescent_polystyrene": FLUORESCENT_POLYSTYRENE,
    "fluorescent_polystyrene_bead": FLUORESCENT_POLYSTYRENE,
    "protein": PROTEIN,
    "lipid": LIPID,
}


def material_from_name(name: str | MaterialProperties | None, default: MaterialProperties) -> MaterialProperties:
    if isinstance(name, MaterialProperties):
        return name
    if name is None:
        return default
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _MATERIALS:
        known = ", ".join(sorted(_MATERIALS))
        raise ValueError(
            f"Unknown sample-environment material {name!r}. "
            f"Known materials/aliases are: {known}."
        )
    return _MATERIALS[key]


def fresnel_reflection_amplitude(
    material_top: str | MaterialProperties | None,
    material_bottom: str | MaterialProperties | None,
    wavelength_nm: float,
    *,
    default_top: MaterialProperties = WATER,
    default_bottom: MaterialProperties = GLASS,
) -> complex:
    """Normal-incidence Fresnel reflection amplitude from top to bottom medium."""
    top = material_from_name(material_top, default_top)
    bottom = material_from_name(material_bottom, default_bottom)
    n_top = top.n_complex(float(wavelength_nm))
    n_bottom = bottom.n_complex(float(wavelength_nm))
    denom = n_top + n_bottom
    if abs(denom) <= 1e-12:
        raise ValueError(
            "Fresnel reflection denominator is near zero for "
            f"n_top={n_top!r}, n_bottom={n_bottom!r}."
        )
    return (n_top - n_bottom) / denom


def _material_with_param_overrides(material: MaterialProperties, params: dict, prefix: str) -> MaterialProperties:
    updates = {}
    for field_name in (
        "autofluorescence_per_nm",
        "fluorophore_density",
        "emission_peak_nm",
        "excitation_peak_nm",
        "se_yield_coefficient",
        "mean_inner_potential_V",
        "density_g_cm3",
    ):
        key = f"{prefix}_{field_name}"
        if key in params:
            value = params[key]
            updates[field_name] = None if value is None else float(value)
    if not updates:
        return material
    return replace(material, **updates)


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
        denom = 1.0 + r01 * r12 * np.exp(-2j * beta)
        return (r01 + r12 * np.exp(-2j * beta)) / np.where(np.abs(denom) > 1e-12, denom, 1e-12)

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
    """Build a lightweight environment from PARAMS."""

    px = float(pixel_size_nm if pixel_size_nm is not None else params["pixel_size_nm"])
    environment_enabled = bool(params.get("sample_environment_enabled", True))
    pattern_enabled = bool(params.get("sample_environment_pattern_enabled", False))
    enabled = environment_enabled and pattern_enabled
    layer_thickness_nm = float(params["mounting_interface_thickness_nm"])
    medium = _material_with_param_overrides(
        material_from_name(params.get("medium_material", "water"), WATER),
        params,
        "medium",
    )
    layer = _material_with_param_overrides(
        material_from_name(params.get("mounting_interface_material", "glass"), SIO2),
        params,
        "mounting_interface",
    )
    base = _material_with_param_overrides(
        material_from_name(params.get("bulk_substrate_material", "glass"), SIO2),
        params,
        "support",
    )

    if not enabled:
        pattern = Pattern.uniform(shape, px, height_nm=0.0)
    else:
        kind = str(params.get("sample_environment_pattern", "none")).strip().lower()
        preset = str(
            params.get("sample_environment_pattern_preset", "empty_background")
        ).strip().lower()
        if kind == "none" or preset == "empty_background":
            pattern = Pattern.uniform(shape, px, height_nm=0.0)
        elif kind in {"gold_holes", "nanopillars"}:
            from substrate_pattern import generate_sample_environment_pattern_maps

            layout_extent_nm = params.get("_substrate_pattern_layout_extent_nm", None)
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
        else:
            raise ValueError(
                f"Unsupported sample_environment_pattern '{kind}'. Supported models "
                "are 'none', 'gold_holes', and 'nanopillars'."
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
    "MaterialProperties",
    "Pattern",
    "Substrate",
    "SampleEnvironment",
    "AIR",
    "VACUUM",
    "WATER",
    "SIO2",
    "SI",
    "CARBON",
    "GOLD",
    "GLASS",
    "material_from_name",
    "fresnel_reflection_amplitude",
    "sample_environment_from_params",
    "build_substrate_nuisance_basis",
]
