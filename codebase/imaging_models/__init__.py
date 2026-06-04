"""Public imaging-model implementations and registry for Syniscopy."""

from __future__ import annotations

from .base import (
    ImagingModel,
    coherent_phase_from_reference,
    field_intensity,
    is_vectorial_field,
    reference_vector_for_scattered,
)

from .coherent_brightfield import CoherentBrightfieldImagingModel
from .coherent_darkfield import CoherentDarkFieldImagingModel
from .dpc import DifferentialPhaseContrastImagingModel
from .electron_constants import (
    electron_interaction_parameter_rad_per_V_nm,
    electron_wavelength_m,
    scherzer_defocus_m,
)
from .fluorescence_tirf import TIRFFluorescenceImagingModel
from .fluorescence_widefield import FluorescenceWidefieldImagingModel
from .interferometric import InterferometricImagingModel
from .kohler import AnnularDarkFieldImagingModel, PartiallyCoherentBrightfieldImagingModel
from .off_axis_holography import OffAxisHolographyImagingModel
from .qpi import QuantitativePhaseImagingModel
from .registry import (
    CANONICAL_COHERENT_MODALITIES,
    LABEL_FREE_OPTICAL_MODALITIES,
    RELATIVE_REFERENCE_CONTRAST_MODALITIES,
    SUPPORTED_MODALITIES,
    _MODEL_REGISTRY,
    get_imaging_model,
    get_imaging_model_class,
    modality_uses_relative_reference_contrast,
    modality_uses_sample_environment_pattern,
)
from .ricm import ReflectionInterferenceContrastImagingModel
from .sem import ScanningElectronMicroscopyImagingModel
from .tem import TransmissionElectronMicroscopyImagingModel
from .zernike_phase import ZernikePhaseContrastImagingModel

__all__ = [
    "LABEL_FREE_OPTICAL_MODALITIES",
    "CANONICAL_COHERENT_MODALITIES",
    "RELATIVE_REFERENCE_CONTRAST_MODALITIES",
    "SUPPORTED_MODALITIES",
    "_MODEL_REGISTRY",
    "ImagingModel",
    "coherent_phase_from_reference",
    "field_intensity",
    "is_vectorial_field",
    "reference_vector_for_scattered",
    "InterferometricImagingModel",
    "CoherentDarkFieldImagingModel",
    "CoherentBrightfieldImagingModel",
    "PartiallyCoherentBrightfieldImagingModel",
    "AnnularDarkFieldImagingModel",
    "ZernikePhaseContrastImagingModel",
    "DifferentialPhaseContrastImagingModel",
    "QuantitativePhaseImagingModel",
    "ReflectionInterferenceContrastImagingModel",
    "OffAxisHolographyImagingModel",
    "electron_wavelength_m",
    "electron_interaction_parameter_rad_per_V_nm",
    "scherzer_defocus_m",
    "TransmissionElectronMicroscopyImagingModel",
    "ScanningElectronMicroscopyImagingModel",
    "FluorescenceWidefieldImagingModel",
    "TIRFFluorescenceImagingModel",
    "get_imaging_model_class",
    "modality_uses_sample_environment_pattern",
    "modality_uses_relative_reference_contrast",
    "get_imaging_model",
]
