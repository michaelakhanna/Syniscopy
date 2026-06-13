"""Import-light single-source modality registry and capability metadata."""

from __future__ import annotations

from dataclasses import dataclass

from registry_utils import NameRegistry
from modality_parameter_surface import (
    ModalityParameterSurface,
    REPORT_COMMON_OPTICAL_PARAM_KEYS,
    REPORT_DETECTOR_PARAM_KEYS,
    REPORT_ELECTRON_PARAM_KEYS,
    REPORT_FLUORESCENCE_PARAM_KEYS,
    REPORT_MODALITY_SPECIFIC_ELECTRON_PARAM_KEYS,
    REPORT_MODALITY_SPECIFIC_OPTICAL_PARAM_KEYS,
    REPORT_OPTICAL_PARAM_KEYS,
    REPORT_SHARED_PARAM_KEYS,
    REPORT_TIRF_PARAM_KEYS,
    modality_parameter_surface,
)


@dataclass(frozen=True)
class ModalitySpec:
    """Single definition of a public imaging modality."""

    id: str
    display_name: str
    implementation_class: str
    label_free_optical: bool = False
    coherent_reference: bool = False
    relative_reference_contrast: bool = False
    vectorial_full_field: bool = False
    electron: bool = False
    fluorescence: bool = False
    lab_optical: bool = False
    comparison_identity: str | None = None


MODALITY_SPECS: dict[str, ModalitySpec] = {
    "bright_field": ModalitySpec(
        "bright_field",
        "partially coherent Kohler bright-field",
        "imaging_models.kohler:PartiallyCoherentBrightfieldImagingModel",
        label_free_optical=True,
        relative_reference_contrast=True,
        vectorial_full_field=True,
        lab_optical=True,
        # Report ranking compares physical modality profiles, not public
        # spellings. ``bright_field`` is the historical public spelling for
        # the same partially coherent Koehler bright-field profile exposed by
        # ``partially_coherent_bright_field``; they must not both enter one
        # best-candidate table as distinct scientific candidates.
        comparison_identity="partially_coherent_kohler_bright_field",
    ),
    "fluorescence_widefield": ModalitySpec(
        "fluorescence_widefield",
        "widefield fluorescence",
        "imaging_models.fluorescence_widefield:FluorescenceWidefieldImagingModel",
        fluorescence=True,
        lab_optical=True,
    ),
    "tirf_fluorescence": ModalitySpec(
        "tirf_fluorescence",
        "TIRF fluorescence",
        "imaging_models.fluorescence_tirf:TIRFFluorescenceImagingModel",
        fluorescence=True,
        lab_optical=True,
    ),
    "dark_field": ModalitySpec(
        "dark_field",
        "annular Kohler dark-field",
        "imaging_models.kohler:AnnularDarkFieldImagingModel",
        label_free_optical=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "zernike_phase_contrast": ModalitySpec(
        "zernike_phase_contrast",
        "Zernike phase contrast",
        "imaging_models.zernike_phase:ZernikePhaseContrastImagingModel",
        label_free_optical=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "differential_phase_contrast": ModalitySpec(
        "differential_phase_contrast",
        "differential phase contrast (DPC)",
        "imaging_models.dpc:DifferentialPhaseContrastImagingModel",
        label_free_optical=True,
        lab_optical=True,
    ),
    "quantitative_phase": ModalitySpec(
        "quantitative_phase",
        "quantitative phase imaging (QPI)",
        "imaging_models.qpi:QuantitativePhaseImagingModel",
        label_free_optical=True,
        coherent_reference=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "off_axis_holography": ModalitySpec(
        "off_axis_holography",
        "off-axis digital holography (DHM)",
        "imaging_models.off_axis_holography:OffAxisHolographyImagingModel",
        label_free_optical=True,
        coherent_reference=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "ricm": ModalitySpec(
        "ricm",
        "reflection interference contrast (RICM)",
        "imaging_models.ricm:ReflectionInterferenceContrastImagingModel",
        label_free_optical=True,
        coherent_reference=True,
        relative_reference_contrast=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "interferometric": ModalitySpec(
        "interferometric",
        "interferometric scattering (iSCAT)",
        "imaging_models.interferometric:InterferometricImagingModel",
        label_free_optical=True,
        coherent_reference=True,
        relative_reference_contrast=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "tem_phase_contrast": ModalitySpec(
        "tem_phase_contrast",
        "TEM phase contrast",
        "imaging_models.tem:TransmissionElectronMicroscopyImagingModel",
        electron=True,
    ),
    "sem_secondary_electron": ModalitySpec(
        "sem_secondary_electron",
        "SEM secondary-electron",
        "imaging_models.sem:ScanningElectronMicroscopyImagingModel",
        electron=True,
    ),
    "partially_coherent_bright_field": ModalitySpec(
        "partially_coherent_bright_field",
        "partially coherent Kohler bright-field",
        "imaging_models.kohler:PartiallyCoherentBrightfieldImagingModel",
        label_free_optical=True,
        relative_reference_contrast=True,
        vectorial_full_field=True,
        lab_optical=True,
        # This explicit spelling is retained for public API clarity, but it is
        # the same report-comparison identity as ``bright_field``. The lab
        # report resolver coalesces identities so aliases cannot receive
        # different modality ranks for identical Fisher/CRLB evidence.
        comparison_identity="partially_coherent_kohler_bright_field",
    ),
    "coherent_bright_field": ModalitySpec(
        "coherent_bright_field",
        "coherent bright-field (COBRI)",
        "imaging_models.coherent_brightfield:CoherentBrightfieldImagingModel",
        label_free_optical=True,
        coherent_reference=True,
        relative_reference_contrast=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "coherent_dark_field": ModalitySpec(
        "coherent_dark_field",
        "coherent dark-field (zero-order blocked)",
        "imaging_models.coherent_darkfield:CoherentDarkFieldImagingModel",
        label_free_optical=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
}

SUPPORTED_MODALITIES = tuple(MODALITY_SPECS)
LABEL_FREE_OPTICAL_MODALITIES = tuple(
    spec.id for spec in MODALITY_SPECS.values() if spec.label_free_optical
)
CANONICAL_COHERENT_MODALITIES = tuple(
    spec.id for spec in MODALITY_SPECS.values() if spec.coherent_reference
)
RELATIVE_REFERENCE_CONTRAST_MODALITIES = tuple(
    spec.id for spec in MODALITY_SPECS.values() if spec.relative_reference_contrast
)
VECTORIAL_FULL_FIELD_MODALITIES = tuple(
    spec.id for spec in MODALITY_SPECS.values() if spec.vectorial_full_field
)
ELECTRON_MODALITIES = tuple(spec.id for spec in MODALITY_SPECS.values() if spec.electron)
FLUORESCENCE_MODALITIES = tuple(
    spec.id for spec in MODALITY_SPECS.values() if spec.fluorescence
)
LAB_OPTICAL_MODALITIES = tuple(spec.id for spec in MODALITY_SPECS.values() if spec.lab_optical)
LAB_DEFAULT_MODALITIES = LAB_OPTICAL_MODALITIES + ELECTRON_MODALITIES
MODALITY_DISPLAY_NAMES = {
    spec.id: spec.display_name for spec in MODALITY_SPECS.values()
}
MODALITY_COMPARISON_IDENTITIES = {
    spec.id: spec.comparison_identity or spec.id for spec in MODALITY_SPECS.values()
}


def modality_report_parameter_surface(model_name: object) -> ModalityParameterSurface:
    """Return the public report-facing parameter surface for one modality."""

    spec = modality_spec(model_name)
    return modality_parameter_surface(
        modality=spec.id,
        flags={
            "label_free_optical": spec.label_free_optical,
            "coherent_reference": spec.coherent_reference,
            "lab_optical": spec.lab_optical,
            "fluorescence": spec.fluorescence,
            "electron": spec.electron,
        },
    )


def relevant_param_keys(model_name: object) -> frozenset[str]:
    """Return canonical public report-facing params relevant to one modality.

    The returned keys are consumed by microscope JSON template generation and
    microscope-overlay diagnostics. They must therefore be canonical public
    parameters keys, not renderer-private metadata or historical aliases. The
    modality parameter-surface object owns this contract so templates and
    warnings cannot drift apart and cause valid configurations to be hidden or
    falsely reported as irrelevant.
    """

    return modality_report_parameter_surface(model_name).public_keys

MODALITY_REGISTRY = NameRegistry(
    supported=SUPPORTED_MODALITIES,
    display_names=MODALITY_DISPLAY_NAMES,
)


def canonical_modality_name(model_name: object) -> str:
    """Return the canonical public spelling for an imaging-model name."""
    return MODALITY_REGISTRY.canonical_name(model_name)


def normalize_modality_key(model_name: object) -> str:
    """Normalize modality-like text without checking support."""
    return canonical_modality_name(model_name)


def require_modality_name(model_name: object, *, item_label: str = "imaging modality") -> str:
    """Return a supported canonical modality name or raise a clear error."""
    key = normalize_modality_key(model_name)
    if key in MODALITY_SPECS:
        return key
    raise ValueError(
        f"Unknown {item_label} {model_name!r}. Supported values are: {list(SUPPORTED_MODALITIES)}."
    )


def modality_spec(model_name: object) -> ModalitySpec:
    """Return the canonical modality spec."""
    return MODALITY_SPECS[require_modality_name(model_name)]


def modality_display_name(model_name: object) -> str:
    """Return a human-readable modality name for reports and generated tables."""
    return modality_spec(model_name).display_name


def modality_comparison_identity(model_name: object) -> str:
    """Return the physical comparison identity for report-facing rankings."""
    key = require_modality_name(model_name)
    return MODALITY_COMPARISON_IDENTITIES[key]


def modality_uses_relative_reference_contrast(model_name: object) -> bool:
    """Whether reference-frame subtraction should use ``(signal - reference) / reference``."""
    return modality_spec(model_name).relative_reference_contrast


def is_electron_modality(model_name: object) -> bool:
    """Whether a modality measures electron-count-domain signals."""
    return modality_spec(model_name).electron


def is_fluorescence_modality(model_name: object) -> bool:
    """Whether a modality uses fluorescence-specific detector/source settings."""
    return modality_spec(model_name).fluorescence


def is_vectorial_full_field_modality(model_name: object) -> bool:
    """Whether a modality can use the vectorial full-field optical backend."""
    return modality_spec(model_name).vectorial_full_field


def modality_name_set(modalities: tuple[str, ...] | frozenset[str]) -> frozenset[str]:
    """Return the canonical modality-name set."""
    return frozenset(canonical_modality_name(name) for name in modalities)


__all__ = [
    "CANONICAL_COHERENT_MODALITIES",
    "ELECTRON_MODALITIES",
    "FLUORESCENCE_MODALITIES",
    "LABEL_FREE_OPTICAL_MODALITIES",
    "LAB_DEFAULT_MODALITIES",
    "LAB_OPTICAL_MODALITIES",
    "MODALITY_DISPLAY_NAMES",
    "MODALITY_COMPARISON_IDENTITIES",
    "MODALITY_REGISTRY",
    "REPORT_COMMON_OPTICAL_PARAM_KEYS",
    "REPORT_DETECTOR_PARAM_KEYS",
    "REPORT_ELECTRON_PARAM_KEYS",
    "REPORT_FLUORESCENCE_PARAM_KEYS",
    "REPORT_MODALITY_SPECIFIC_ELECTRON_PARAM_KEYS",
    "REPORT_MODALITY_SPECIFIC_OPTICAL_PARAM_KEYS",
    "REPORT_OPTICAL_PARAM_KEYS",
    "REPORT_TIRF_PARAM_KEYS",
    "REPORT_SHARED_PARAM_KEYS",
    "MODALITY_SPECS",
    "ModalityParameterSurface",
    "ModalitySpec",
    "RELATIVE_REFERENCE_CONTRAST_MODALITIES",
    "SUPPORTED_MODALITIES",
    "VECTORIAL_FULL_FIELD_MODALITIES",
    "canonical_modality_name",
    "is_electron_modality",
    "is_fluorescence_modality",
    "is_vectorial_full_field_modality",
    "modality_name_set",
    "modality_display_name",
    "modality_comparison_identity",
    "modality_spec",
    "modality_uses_relative_reference_contrast",
    "modality_report_parameter_surface",
    "relevant_param_keys",
    "normalize_modality_key",
    "require_modality_name",
]
