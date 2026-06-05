"""Shared imports and helper functions for imaging model implementations."""
from __future__ import annotations

import numpy as np

from config.runtime import param_value
from optical_params import resolve_probe_wavelength_nm
from substrate import MaterialProperties, SampleEnvironment, fresnel_reflection_amplitude
from .base import (
    ImagingModel,
    coherent_phase_from_reference,
    field_intensity,
    is_vectorial_field,
    reference_vector_for_scattered,
)
from modality_registry import (
    CANONICAL_COHERENT_MODALITIES,
    LABEL_FREE_OPTICAL_MODALITIES,
    RELATIVE_REFERENCE_CONTRAST_MODALITIES,
    SUPPORTED_MODALITIES,
    canonical_modality_name as _canonical_modality_name,
)



def _ricm_particle_reflection_material(params: dict) -> str | MaterialProperties:
    explicit = param_value(params, 'ricm_particle_material')
    if isinstance(explicit, MaterialProperties):
        return explicit
    explicit_text = "" if explicit is None else str(explicit).strip()
    if explicit_text.lower() not in ("", "none", "particle_material", "primary_particle"):
        return explicit_text

    from particle_specs import get_particle_specs
    from particle_material_resolution import resolve_component_material_properties

    specs = get_particle_specs(params)
    primary = specs[0].primary_component
    if (
        primary.material not in (None, "")
        or primary.refractive_index is not None
        or primary.material_properties is not None
    ):
        return resolve_component_material_properties(params, primary)
    raise ValueError(
        "Particle material properties could not be resolved from PARAMS['particles']. "
        "Set the primary component material/material_properties/refractive_index, "
        "or use the modality-specific material parameter."
    )

def _mean_normalized_map(arr: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """Return ``arr`` divided by its positive finite mean."""
    out = np.asarray(arr, dtype=float)
    finite = np.isfinite(out)
    mean = float(out[finite].mean()) if np.any(finite) else 0.0
    if abs(mean) <= floor:
        return np.ones_like(out, dtype=float)
    return out / mean

def _complex_from_param(value, *, default: complex = 1.0 + 0.0j) -> complex:
    """Coerce a config value into a complex scalar."""
    if value is None:
        return complex(default)
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float, np.number)):
        return complex(float(value), 0.0)
    if isinstance(value, str):
        text = value.strip().replace("i", "j")
        return complex(text)
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise TypeError(f"Cannot interpret {value!r} as a complex scalar.")

def _optical_pupil_frequency_grid(
    shape: tuple[int, int],
    pixel_size_nm: float,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return Fourier coordinates normalized by the objective pupil cutoff."""
    if len(shape) != 2:
        raise ValueError(f"Optical pupil grid requires a 2D shape; got {shape!r}.")
    H, W = int(shape[0]), int(shape[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"Optical pupil grid requires positive image dimensions; got {shape!r}.")
    pixel_nm = float(pixel_size_nm)
    if not np.isfinite(pixel_nm) or pixel_nm <= 0.0:
        raise ValueError(f"pixel_size_nm must be finite and positive; got {pixel_size_nm!r}.")
    wavelength_nm = resolve_probe_wavelength_nm(params)
    numerical_aperture = float(param_value(params, "numerical_aperture"))
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError(f"Optical pupil wavelength must be finite and positive; got {wavelength_nm!r}.")
    if not np.isfinite(numerical_aperture) or numerical_aperture <= 0.0:
        raise ValueError(
            f"PARAMS['numerical_aperture'] must be finite and positive; got {numerical_aperture!r}."
        )
    fy = np.fft.fftfreq(H, d=pixel_nm)
    fx = np.fft.fftfreq(W, d=pixel_nm)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    cutoff_cycles_per_nm = max(numerical_aperture / wavelength_nm, 1e-30)
    rho = np.sqrt(FX * FX + FY * FY) / cutoff_cycles_per_nm
    return FX / cutoff_cycles_per_nm, FY / cutoff_cycles_per_nm, rho, cutoff_cycles_per_nm

__all__ = [
    "CANONICAL_COHERENT_MODALITIES",
    "ImagingModel",
    "LABEL_FREE_OPTICAL_MODALITIES",
    "MaterialProperties",
    "RELATIVE_REFERENCE_CONTRAST_MODALITIES",
    "SUPPORTED_MODALITIES",
    "SampleEnvironment",
    "_canonical_modality_name",
    "_complex_from_param",
    "_mean_normalized_map",
    "_optical_pupil_frequency_grid",
    "_ricm_particle_reflection_material",
    "coherent_phase_from_reference",
    "field_intensity",
    "fresnel_reflection_amplitude",
    "is_vectorial_field",
    "np",
    "reference_vector_for_scattered",
]
