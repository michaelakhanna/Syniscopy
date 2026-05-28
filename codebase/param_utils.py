from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
import math

from config import PARAMS as BASE_PARAMS
from param_schema import PARAM_SCHEMA, ParamSpec


def _coerce_bool(value: Any) -> bool:
    """
    Convert a variety of truthy/falsey representations into a proper bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        return v in ("1", "true", "yes", "on", "y", "t")
    return bool(value)


def _validate_and_normalize_value(spec: ParamSpec, raw_value: Any) -> Any:
    """
    Validate and normalize a raw control value according to the parameter
    specification.

    This enforces:
      - type coercion (float/int/bool/enum)
      - min/max bounds for numeric types (if provided)
      - choices restriction for enums
    """
    ptype = spec["type"]

    if ptype == "float":
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{spec['key']} must be finite; got {raw_value!r}.")
        if "min" in spec and value < spec["min"]:
            raise ValueError(f"{spec['key']} must be >= {spec['min']}; got {raw_value!r}.")
        if "max" in spec and value > spec["max"]:
            raise ValueError(f"{spec['key']} must be <= {spec['max']}; got {raw_value!r}.")
        return value

    if ptype == "int":
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value is not None and not math.isfinite(numeric_value):
            raise ValueError(f"{spec['key']} must be finite; got {raw_value!r}.")
        value = int(raw_value)
        if "min" in spec and value < spec["min"]:
            raise ValueError(f"{spec['key']} must be >= {spec['min']}; got {raw_value!r}.")
        if "max" in spec and value > spec["max"]:
            raise ValueError(f"{spec['key']} must be <= {spec['max']}; got {raw_value!r}.")
        return value

    if ptype == "bool":
        return _coerce_bool(raw_value)

    if ptype == "enum":
        choices = spec.get("choices", [])
        if raw_value in choices:
            return raw_value
        # Try a case-insensitive match for strings if possible
        if isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            for c in choices:
                if isinstance(c, str) and c.strip().lower() == lowered:
                    return c
        raise ValueError(
            f"{spec['key']} must be one of {choices}; got {raw_value!r}."
        )

    # Unknown type; return raw value unchanged
    return raw_value


def get_default_control_values() -> Dict[str, Any]:
    """
    Return a dict of schema_key -> default control value.

    The default for each control is taken from, in order of precedence:
      1. BASE_PARAMS at the underlying PARAMS key, if present.
      2. The schema's 'default' field.

    Particle controls read from the first particle object's first component.
    """
    defaults: Dict[str, Any] = {}
    for schema_key, spec in PARAM_SCHEMA.items():
        base_key = spec["key"]

        if schema_key in ("particle_diameter_nm", "particle_material"):
            raw = _first_particle_control_default(schema_key, spec.get("default"))
        elif base_key in BASE_PARAMS:
            raw = BASE_PARAMS[base_key]
        else:
            raw = spec.get("default")

        defaults[schema_key] = raw
    return defaults


def build_params_from_controls(control_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a PARAMS-like dict from BASE_PARAMS and a set of control values.

    Parameters
    ----------
    control_values:
        Dict mapping schema keys (e.g., "particle_diameter_nm") to values
        provided by the user/UI.

    Behavior
    --------
    - Start from a deepcopy of BASE_PARAMS so the original config is untouched.
    - For each entry in PARAM_SCHEMA:
        * Determine a value to use:
            - If control_values contains the schema key, use that.
            - Else, if BASE_PARAMS already has the underlying PARAMS key,
              use BASE_PARAMS[key].
            - Else, fall back to the schema's "default".
        * Validate and normalize the value according to the spec["type"].
        * Write particle controls into the first particle object's first
          component; write other controls into the PARAMS dictionary under
          spec["key"].

    Returns
    -------
    dict:
        A full PARAMS-like dictionary ready to be passed into
        generate_single_frame_views or the main simulation pipeline.
    """
    params = deepcopy(BASE_PARAMS)

    for schema_key, spec in PARAM_SCHEMA.items():
        base_key = spec["key"]

        # Resolve the user value, normalize it, then write it to the correct
        # PARAMS location.
        if schema_key in control_values:
            raw_value = control_values[schema_key]
        elif schema_key in ("particle_diameter_nm", "particle_material"):
            raw_value = _first_particle_control_default(schema_key, spec.get("default"))
        elif base_key in params:
            raw_value = params[base_key]
        else:
            raw_value = spec.get("default")

        value = _validate_and_normalize_value(spec, raw_value)

        if schema_key == "particle_diameter_nm":
            component = _first_particle_component(params)
            component["diameter_nm"] = value
            params["particles"][0]["motion"]["hydrodynamic_diameter_nm"] = value
        elif schema_key == "particle_material":
            component = _first_particle_component(params)
            component["material"] = value
            component["refractive_index"] = None
        else:
            params[base_key] = value

    return params
def _first_particle_component(params: Dict[str, Any]) -> Dict[str, Any]:
    particles = params.get("particles")
    if not isinstance(particles, list):
        raise ValueError("PARAMS['particles'] must be a list of particle objects.")
    if not particles:
        raise ValueError("PARAMS['particles'] must contain at least one particle.")
    particle = particles[0]
    if not isinstance(particle, dict) or "motion" not in particle:
        raise ValueError("PARAMS['particles'][0] must include a motion object.")
    components = particle.get("components")
    if not isinstance(components, list):
        raise ValueError("PARAMS['particles'][0]['components'] must be a list.")
    if not components:
        raise ValueError("PARAMS['particles'][0]['components'] must contain at least one component.")
    return components[0]


def _first_particle_control_default(schema_key: str, fallback: Any) -> Any:
    try:
        component = _first_particle_component(deepcopy(BASE_PARAMS))
        if schema_key == "particle_diameter_nm":
            return component.get("diameter_nm", fallback)
        if schema_key == "particle_material":
            return component.get("material", fallback)
    except (KeyError, TypeError, ValueError):
        return fallback
    return fallback
