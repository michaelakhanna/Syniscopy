"""
Preset system for configuring Syniscopy simulations.

This module exposes one reusable configuration layer:

    - Instrument presets:
        Configure the simulation for a specific microscope / optical setup.
        These presets override optical and detector parameters such as
        wavelength, numerical aperture, pixel size, etc.

High-level dataset-generation wrappers can apply these instrument presets to
complete recipes, but dataset-specific parameter sweeps live outside this core
module.

Design notes
------------
- This module does *not* modify config.PARAMS in-place. All public functions
  that apply presets return new dictionaries.
- Instrument presets are named, reproducible microscope-configuration bundles.
  They touch instrument and detector metadata only.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable

from config import PARAMS


# ---------------------------------------------------------------------------
# Instrument preset definitions
# ---------------------------------------------------------------------------

INSTRUMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    # A 60x high-NA widefield-like configuration for reproducible public
    # examples and local dataset generation.
    "widefield_60x_high_na": {
        "image_size_pixels": 1024,
        "magnification": 60,
        "objective_model": "60x high-NA objective",
        "pixel_size_nm": 244.0,
        "wavelength_nm": 635.0,
        "bit_depth": 16,
        "numerical_aperture": 1.20,
        "refractive_index_medium": 1.33,
        "refractive_index_immersion": 1.518,
        "objective_focal_length_mm": 3.0,
    },
}


def _normalize_instrument_preset_name(preset_name: str) -> str:
    """
    Validate an instrument preset name.

    Args:
        preset_name (str): Canonical instrument preset key.

    Returns:
        str: Canonical instrument preset key used in INSTRUMENT_PRESETS.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    key = preset_name.strip()
    if key in INSTRUMENT_PRESETS:
        return key

    available = ", ".join(sorted(INSTRUMENT_PRESETS.keys()))
    raise ValueError(
        f"Unknown instrument preset '{preset_name}'. Use one canonical key: {available}"
    )


# ---------------------------------------------------------------------------
# Instrument preset public API
# ---------------------------------------------------------------------------

def get_instrument_preset_names() -> Iterable[str]:
    """Return a dictionary-view iterable of available instrument preset names."""
    return INSTRUMENT_PRESETS.keys()


def apply_instrument_preset(base_params: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Apply an instrument preset on top of a base parameter dictionary.

    This function does not modify the input dictionary in-place. Instead, it
    returns a deep copy of `base_params` with all key/value pairs from the
    specified instrument preset overlaid on top.

    The base parameter dictionary is typically a copy of config.PARAMS, but it
    can be any dictionary that follows the same structure.

    Args:
        base_params (Dict[str, Any]):
            The starting parameter dictionary to which the preset will be
            applied. This dictionary is not modified.
        preset_name (str):
            Canonical instrument preset key from get_instrument_preset_names().

    Returns:
        Dict[str, Any]: A new parameter dictionary with the instrument preset
        applied.

    Raises:
        TypeError: If `base_params` is not a dictionary.
        ValueError: If `preset_name` does not correspond to a known instrument
        preset.
    """
    if not isinstance(base_params, dict):
        raise TypeError("base_params must be a dictionary.")

    canonical = _normalize_instrument_preset_name(preset_name)

    params_copy = deepcopy(base_params)
    overrides = INSTRUMENT_PRESETS[canonical]

    # Overlay preset values onto the copied base parameters.
    for key, value in overrides.items():
        params_copy[key] = value

    return params_copy


def create_params_for_instrument(preset_name: str) -> Dict[str, Any]:
    """
    Convenience helper that creates a fresh parameter dictionary for a given
    instrument preset starting from the global config.PARAMS template.

    This is equivalent to:

        from copy import deepcopy
        from config import PARAMS
        params = apply_instrument_preset(deepcopy(PARAMS), preset_name)

    but packaged in a single function for clarity. The returned dictionary is
    independent of the global PARAMS and can be safely modified or passed to
    run_simulation without affecting other simulations.

    Args:
        preset_name (str): Canonical instrument preset key from
            get_instrument_preset_names().

    Returns:
        Dict[str, Any]: A new parameter dictionary configured for the specified
        instrument.
    """
    base = deepcopy(PARAMS)
    return apply_instrument_preset(base, preset_name)
