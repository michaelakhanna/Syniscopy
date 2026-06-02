from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict
import math

from config import PARAMS
from param_schema import PARAM_SCHEMA, ParamSpec


def _coerce_bool(value: Any) -> bool:
    """
    Convert a variety of truthy/falsey representations into a proper bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Boolean controls accept only 0/1 integers; got {value!r}.")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Boolean controls must be finite; got {value!r}.")
        if value in (0.0, 1.0):
            return bool(value)
        raise ValueError(f"Boolean controls accept only 0.0/1.0 floats; got {value!r}.")
    if isinstance(value, str):
        v = value.strip().lower()
        true_values = {"1", "true", "yes", "on", "y", "t"}
        false_values = {"0", "false", "no", "off", "n", "f"}
        if v in true_values:
            return True
        if v in false_values:
            return False
        raise ValueError(f"Boolean controls must be true/false values; got {value!r}.")
    raise ValueError(f"Boolean controls must be bools, 0/1, or true/false strings; got {value!r}.")


def _coerce_contract_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Contract truth flag accepts only 0/1 integers; got {value!r}.")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Contract truth flag must be finite; got {value!r}.")
        if value in (0.0, 1.0):
            return bool(value)
        raise ValueError(f"Contract truth flag accepts only 0.0/1.0 floats; got {value!r}.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"Contract truth flag must be true/false; got {value!r}.")
    raise ValueError(f"Contract truth flag must be bool, 0/1, or true/false string; got {value!r}.")


def _validate_and_normalize_value(spec: ParamSpec, raw_value: Any) -> Any:
    """
    Validate and normalize a raw control value according to the parameter
    specification.

    This enforces:
      - type coercion (float/int/bool/enum/string/json)
      - min/max bounds for numeric types (if provided)
      - choices restriction for enums
    """
    ptype = spec["type"]
    if raw_value is None:
        if spec.get("default") is None:
            return None
        raise ValueError(f"{spec['key']} cannot be None.")

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
        if isinstance(raw_value, bool):
            raise ValueError(f"{spec['key']} must be an integer, not a boolean; got {raw_value!r}.")
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{spec['key']} must be finite; got {raw_value!r}.")
        if not numeric_value.is_integer():
            raise ValueError(f"{spec['key']} must be an integer-valued input; got {raw_value!r}.")
        value = int(numeric_value)
        if "min" in spec and value < spec["min"]:
            raise ValueError(f"{spec['key']} must be >= {spec['min']}; got {raw_value!r}.")
        if "max" in spec and value > spec["max"]:
            raise ValueError(f"{spec['key']} must be <= {spec['max']}; got {raw_value!r}.")
        return value

    if ptype == "bool":
        return _coerce_bool(raw_value)

    if ptype == "string":
        if raw_value is None:
            return None
        if isinstance(raw_value, bool):
            raise ValueError(f"{spec['key']} must be a string path or scalar value; got {raw_value!r}.")
        return str(raw_value)

    if ptype == "json":
        if raw_value is None:
            return None
        if isinstance(raw_value, (dict, list, float, int, bool, str)):
            if isinstance(raw_value, str):
                try:
                    return json.loads(raw_value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{spec['key']} must be valid JSON for structured input; got {raw_value!r}.") from exc
            return raw_value
        raise ValueError(f"{spec['key']} must be structured JSON-compatible data; got {raw_value!r}.")

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

    raise ValueError(f"Unsupported PARAM_SCHEMA type {ptype!r} for {spec['key']!r}.")


def get_default_control_values() -> Dict[str, Any]:
    """
    Return a dict of schema_key -> default control value.

    The default for each control is taken from, in order of precedence:
      1. PARAMS at the underlying config key, if present.
      2. The schema's 'default' field.

    Particle controls read from the first particle object's first component.
    """
    defaults: Dict[str, Any] = {}
    for schema_key, spec in PARAM_SCHEMA.items():
        base_key = spec["key"]

        if schema_key in ("particle_diameter_nm", "particle_material"):
            raw = _first_particle_control_default(schema_key, spec.get("default"))
        elif base_key in PARAMS:
            raw = PARAMS[base_key]
        else:
            raw = spec.get("default")

        defaults[schema_key] = _validate_and_normalize_value(spec, raw)
    return defaults


def build_params_from_controls(control_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a PARAMS-like dict from config.PARAMS and a set of control values.

    Parameters
    ----------
    control_values:
        Dict mapping schema keys (e.g., "particle_diameter_nm") to values
        provided by the user/UI.

    Behavior
    --------
    - Start from a deepcopy of PARAMS so the original config is untouched.
    - For each entry in PARAM_SCHEMA:
        * Determine a value to use:
            - If control_values contains the schema key, use that.
            - Else, if PARAMS already has the underlying config key,
              use PARAMS[key].
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
    params = deepcopy(PARAMS)
    unknown_controls = sorted(set(control_values) - set(PARAM_SCHEMA))
    if unknown_controls:
        raise ValueError(f"Unknown control key(s): {unknown_controls!r}.")

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
            if component.get("material") != value:
                component["material_properties"] = None
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
        component = _first_particle_component(deepcopy(PARAMS))
        if schema_key == "particle_diameter_nm":
            return component.get("diameter_nm", fallback)
        if schema_key == "particle_material":
            return component.get("material", fallback)
    except (KeyError, TypeError, ValueError):
        return fallback
    return fallback
