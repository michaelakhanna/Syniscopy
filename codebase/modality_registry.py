"""Import-light single-source modality registry and capability metadata."""

from __future__ import annotations

from dataclasses import dataclass

from registry_utils import NameRegistry


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


MODALITY_SPECS: dict[str, ModalitySpec] = {
    "bright_field": ModalitySpec(
        "bright_field",
        "partially coherent Kohler bright-field",
        "imaging_models.kohler:PartiallyCoherentBrightfieldImagingModel",
        label_free_optical=True,
        relative_reference_contrast=True,
        vectorial_full_field=True,
        lab_optical=True,
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
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "off_axis_holography": ModalitySpec(
        "off_axis_holography",
        "off-axis digital holography (DHM)",
        "imaging_models.off_axis_holography:OffAxisHolographyImagingModel",
        label_free_optical=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "ricm": ModalitySpec(
        "ricm",
        "reflection interference contrast (RICM)",
        "imaging_models.ricm:ReflectionInterferenceContrastImagingModel",
        label_free_optical=True,
        relative_reference_contrast=True,
        vectorial_full_field=True,
        lab_optical=True,
    ),
    "interferometric": ModalitySpec(
        "interferometric",
        "interferometric scattering (iSCAT)",
        "imaging_models.interferometric:InterferometricImagingModel",
        label_free_optical=True,
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
        coherent_reference=True,
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

MODALITY_REGISTRY = NameRegistry(
    supported=SUPPORTED_MODALITIES,
    display_names=MODALITY_DISPLAY_NAMES,
)


def canonical_modality_name(model_name: object) -> str:
    """Return the canonical public spelling for an imaging-model name."""
    return MODALITY_REGISTRY.canonical_name(model_name)


def modality_spec(model_name: object) -> ModalitySpec:
    """Return the canonical modality spec."""
    key = canonical_modality_name(model_name)
    try:
        return MODALITY_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown imaging modality {model_name!r}. Supported values are: {list(SUPPORTED_MODALITIES)}."
        ) from exc


def modality_display_name(model_name: object) -> str:
    """Return a human-readable modality name for reports and generated tables."""
    return modality_spec(model_name).display_name


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
    "MODALITY_REGISTRY",
    "MODALITY_SPECS",
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
    "modality_spec",
    "modality_uses_relative_reference_contrast",
]
