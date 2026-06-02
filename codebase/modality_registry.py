"""Import-light modality registry and capability metadata."""

from __future__ import annotations

from registry_utils import AliasRegistry

SUPPORTED_MODALITIES = (
    "bright_field",
    "fluorescence_widefield",
    "tirf_fluorescence",
    "dark_field",
    "zernike_phase_contrast",
    "differential_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
    "tem_phase_contrast",
    "sem_secondary_electron",
    "partially_coherent_bright_field",
    "coherent_bright_field",
    "coherent_dark_field",
)

MODALITY_ALIASES = {
    "bf": "bright_field",
    "brightfield": "bright_field",
    "darkfield": "dark_field",
    "qpi": "quantitative_phase",
    "dpc": "differential_phase_contrast",
    "tem": "tem_phase_contrast",
    "sem": "sem_secondary_electron",
    "partially_coherent_brightfield": "partially_coherent_bright_field",
    "coherent_brightfield": "coherent_bright_field",
    "coherent_darkfield": "coherent_dark_field",
}

LABEL_FREE_OPTICAL_MODALITIES = (
    "bright_field",
    "partially_coherent_bright_field",
    "coherent_bright_field",
    "dark_field",
    "coherent_dark_field",
    "zernike_phase_contrast",
    "differential_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
)

CANONICAL_COHERENT_MODALITIES = (
    "coherent_bright_field",
    "coherent_dark_field",
)

RELATIVE_REFERENCE_CONTRAST_MODALITIES = (
    "interferometric",
    "bright_field",
    "partially_coherent_bright_field",
    "coherent_bright_field",
    "ricm",
)

VECTORIAL_FULL_FIELD_MODALITIES = (
    "bright_field",
    "partially_coherent_bright_field",
    "coherent_bright_field",
    "dark_field",
    "coherent_dark_field",
    "zernike_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
)

ELECTRON_MODALITIES = (
    "tem_phase_contrast",
    "sem_secondary_electron",
)

LAB_OPTICAL_MODALITIES = (
    "bright_field",
    "coherent_bright_field",
    "fluorescence_widefield",
    "tirf_fluorescence",
    "dark_field",
    "coherent_dark_field",
    "zernike_phase_contrast",
    "differential_phase_contrast",
    "quantitative_phase",
    "off_axis_holography",
    "ricm",
    "interferometric",
)

LAB_DEFAULT_MODALITIES = LAB_OPTICAL_MODALITIES + ELECTRON_MODALITIES

MODALITY_DISPLAY_NAMES = {
    "bright_field": "partially coherent Kohler bright-field",
    "fluorescence_widefield": "widefield fluorescence",
    "tirf_fluorescence": "TIRF fluorescence",
    "dark_field": "annular Kohler dark-field",
    "partially_coherent_bright_field": "partially coherent Kohler bright-field",
    "coherent_bright_field": "coherent bright-field (COBRI)",
    "coherent_dark_field": "coherent dark-field (zero-order blocked)",
    "zernike_phase_contrast": "Zernike phase contrast",
    "differential_phase_contrast": "differential phase contrast (DPC)",
    "quantitative_phase": "quantitative phase imaging (QPI)",
    "off_axis_holography": "off-axis digital holography (DHM)",
    "ricm": "reflection interference contrast (RICM)",
    "interferometric": "interferometric scattering (iSCAT)",
    "tem_phase_contrast": "TEM phase contrast",
    "sem_secondary_electron": "SEM secondary-electron",
}

MODALITY_REGISTRY = AliasRegistry(
    supported=SUPPORTED_MODALITIES,
    aliases=MODALITY_ALIASES,
    display_names=MODALITY_DISPLAY_NAMES,
)


def canonical_modality_name(model_name: object) -> str:
    """Return the canonical public spelling for an imaging-model name."""
    return MODALITY_REGISTRY.canonical_name(model_name)


def modality_display_name(model_name: object) -> str:
    """Return a human-readable modality name for reports and generated tables."""
    return MODALITY_REGISTRY.display_name(model_name)


def modality_uses_relative_reference_contrast(model_name: object) -> bool:
    """Whether reference-frame subtraction should use ``(signal - reference) / reference``."""
    return canonical_modality_name(model_name) in RELATIVE_REFERENCE_CONTRAST_MODALITIES


def modality_alias_closure(modalities: tuple[str, ...] | frozenset[str]) -> frozenset[str]:
    """Return canonical modality names plus aliases that resolve into them."""
    return MODALITY_REGISTRY.alias_closure(modalities)
