"""Imaging model registry and modality capability helpers."""

from __future__ import annotations

from .base import ImagingModel
from config.runtime import ModalitySettings
from modality_registry import (
    CANONICAL_COHERENT_MODALITIES,
    LABEL_FREE_OPTICAL_MODALITIES,
    MODALITY_SPECS,
    RELATIVE_REFERENCE_CONTRAST_MODALITIES,
    SUPPORTED_MODALITIES,
    canonical_modality_name as _canonical_modality_name,
    modality_uses_relative_reference_contrast,
)
from registry_utils import ObjectRegistry
from .kohler import AnnularDarkFieldImagingModel, PartiallyCoherentBrightfieldImagingModel

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


_IMPLEMENTATION_CLASSES: dict[str, type[ImagingModel]] = {
    "imaging_models.kohler:PartiallyCoherentBrightfieldImagingModel": PartiallyCoherentBrightfieldImagingModel,
    "imaging_models.fluorescence_widefield:FluorescenceWidefieldImagingModel": FluorescenceWidefieldImagingModel,
    "imaging_models.fluorescence_tirf:TIRFFluorescenceImagingModel": TIRFFluorescenceImagingModel,
    "imaging_models.kohler:AnnularDarkFieldImagingModel": AnnularDarkFieldImagingModel,
    "imaging_models.zernike_phase:ZernikePhaseContrastImagingModel": ZernikePhaseContrastImagingModel,
    "imaging_models.dpc:DifferentialPhaseContrastImagingModel": DifferentialPhaseContrastImagingModel,
    "imaging_models.qpi:QuantitativePhaseImagingModel": QuantitativePhaseImagingModel,
    "imaging_models.off_axis_holography:OffAxisHolographyImagingModel": OffAxisHolographyImagingModel,
    "imaging_models.ricm:ReflectionInterferenceContrastImagingModel": ReflectionInterferenceContrastImagingModel,
    "imaging_models.interferometric:InterferometricImagingModel": InterferometricImagingModel,
    "imaging_models.tem:TransmissionElectronMicroscopyImagingModel": TransmissionElectronMicroscopyImagingModel,
    "imaging_models.sem:ScanningElectronMicroscopyImagingModel": ScanningElectronMicroscopyImagingModel,
    "imaging_models.coherent_brightfield:CoherentBrightfieldImagingModel": CoherentBrightfieldImagingModel,
    "imaging_models.coherent_darkfield:CoherentDarkFieldImagingModel": CoherentDarkFieldImagingModel,
}


def _build_model_registry() -> dict[str, type[ImagingModel]]:
    registry: dict[str, type[ImagingModel]] = {}
    for modality_id, spec in MODALITY_SPECS.items():
        try:
            registry[modality_id] = _IMPLEMENTATION_CLASSES[spec.implementation_class]
        except KeyError as exc:
            raise RuntimeError(
                f"No imaging-model implementation registered for {modality_id!r}: "
                f"{spec.implementation_class!r}."
            ) from exc
    return registry


_MODEL_REGISTRY: dict[str, type[ImagingModel]] = _build_model_registry()

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
    model_name = ModalitySettings.from_params(params).modality
    return get_imaging_model_class(model_name)(params)


__all__ = [
    "LABEL_FREE_OPTICAL_MODALITIES",
    "CANONICAL_COHERENT_MODALITIES",
    "RELATIVE_REFERENCE_CONTRAST_MODALITIES",
    "SUPPORTED_MODALITIES",
    "_IMAGING_MODEL_REGISTRY",
    "_MODEL_REGISTRY",
    "get_imaging_model_class",
    "modality_uses_sample_environment_pattern",
    "modality_uses_relative_reference_contrast",
    "get_imaging_model",
]
