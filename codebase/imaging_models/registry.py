"""Imaging model registry and modality capability helpers."""

from __future__ import annotations

from .base import ImagingModel
from modality_registry import (
    CANONICAL_COHERENT_MODALITIES,
    LABEL_FREE_OPTICAL_MODALITIES,
    MODALITY_ALIASES,
    RELATIVE_REFERENCE_CONTRAST_MODALITIES,
    SUPPORTED_MODALITIES,
    canonical_modality_name as _canonical_modality_name,
    modality_uses_relative_reference_contrast,
)
from registry_utils import ObjectRegistry
from .kohler import (
    AnnularDarkFieldImagingModel,
    PartiallyCoherentBrightfieldImagingModel,
)

from .coherent_brightfield import CoherentBrightfieldImagingModel
from .coherent_darkfield import CoherentDarkFieldImagingModel
from .dpc import DifferentialPhaseContrastImagingModel
from .fluorescence_tirf import TIRFFluorescenceImagingModel
from .fluorescence_widefield import FluorescenceWidefieldImagingModel
from .interferometric import InterferometricImagingModel
from .off_axis_holography import OffAxisHolographyImagingModel
from .qpi import QuantitativePhaseImagingModel
from .ricm import ReflectionInterferenceContrastImagingModel
from .sem import ScanningElectronMicroscopyImagingModel
from .tem import TransmissionElectronMicroscopyImagingModel
from .zernike_phase import ZernikePhaseContrastImagingModel


_MODEL_REGISTRY: dict[str, type[ImagingModel]] = {
    "bright_field": PartiallyCoherentBrightfieldImagingModel,
    "fluorescence_widefield": FluorescenceWidefieldImagingModel,
    "tirf_fluorescence": TIRFFluorescenceImagingModel,
    "dark_field": AnnularDarkFieldImagingModel,
    "zernike_phase_contrast": ZernikePhaseContrastImagingModel,
    "differential_phase_contrast": DifferentialPhaseContrastImagingModel,
    "quantitative_phase": QuantitativePhaseImagingModel,
    "off_axis_holography": OffAxisHolographyImagingModel,
    "ricm": ReflectionInterferenceContrastImagingModel,
    "interferometric": InterferometricImagingModel,
    "tem_phase_contrast": TransmissionElectronMicroscopyImagingModel,
    "sem_secondary_electron": ScanningElectronMicroscopyImagingModel,
    "partially_coherent_bright_field": PartiallyCoherentBrightfieldImagingModel,
    "coherent_bright_field": CoherentBrightfieldImagingModel,
    "coherent_dark_field": CoherentDarkFieldImagingModel,
}

_IMAGING_MODEL_REGISTRY = ObjectRegistry[type[ImagingModel]](
    entries=_MODEL_REGISTRY,
    canonicalize=_canonical_modality_name,
)


def get_imaging_model_class(model_name: str) -> type[ImagingModel]:
    """Return the registered imaging-model class for ``model_name``."""
    return _IMAGING_MODEL_REGISTRY.get(model_name, item_label="imaging_model")


def modality_uses_sample_environment_pattern(model_name: str) -> bool:
    """Whether ``model_name`` physically uses the optical substrate pattern."""
    return bool(
        getattr(get_imaging_model_class(model_name), "uses_sample_environment_pattern", False)
    )


def get_imaging_model(params: dict) -> ImagingModel:
    """Instantiate and return the imaging model specified by ``params``."""
    model_name = _canonical_modality_name(params.get("imaging_model", "bright_field"))
    return get_imaging_model_class(model_name)(params)


__all__ = [
    "LABEL_FREE_OPTICAL_MODALITIES",
    "CANONICAL_COHERENT_MODALITIES",
    "RELATIVE_REFERENCE_CONTRAST_MODALITIES",
    "SUPPORTED_MODALITIES",
    "MODALITY_ALIASES",
    "_IMAGING_MODEL_REGISTRY",
    "_MODEL_REGISTRY",
    "get_imaging_model_class",
    "modality_uses_sample_environment_pattern",
    "modality_uses_relative_reference_contrast",
    "get_imaging_model",
]
