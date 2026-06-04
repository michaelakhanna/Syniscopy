from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from config.runtime import internal_param_value, resolved_particles
from shared_constants import NONNEGATIVE_MATERIAL_PROPERTY_FIELDS, SE3_STATE_AXES


@dataclass(frozen=True)
class ParticleComponentSpec:
    """Static physical description of one spherical component."""

    shape: str
    offset_nm: tuple[float, float, float]
    diameter_nm: float
    material: str | None = None
    refractive_index: complex | None = None
    signal_multiplier: float = 1.0
    source_multiplier: float = 1.0
    material_properties: dict[str, Any] | None = None


_NONNEGATIVE_MATERIAL_FIELDS = NONNEGATIVE_MATERIAL_PROPERTY_FIELDS


@dataclass(frozen=True)
class ParticleMotionSpec:
    """Particle-level motion properties."""

    hydrodynamic_diameter_nm: float
    initial_position_nm: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ParticleSpec:
    """
    Canonical static particle object.

    Per-frame rendered quantities such as E_sca, source maps, contrast images,
    masks, and loss weights are not stored here. Those are render-time state
    derived from this physical object plus a trajectory frame.
    """

    name: str
    motion: ParticleMotionSpec
    components: tuple[ParticleComponentSpec, ...]
    signal_multiplier: float = 1.0
    source_multiplier: float = 1.0
    symmetry_class: str | None = None
    continuous_rotational_symmetry_dim: int | None = None
    singular_rotation_axes_body: tuple[str, ...] = ()

    @property
    def primary_component(self) -> ParticleComponentSpec:
        return self.components[0]

    @property
    def is_single_sphere(self) -> bool:
        if len(self.components) != 1:
            return False
        c = self.components[0]
        return c.shape.strip().lower() == "sphere" and all(
            abs(float(x)) <= 1e-12 for x in c.offset_nm
        )


def _coerce_optional_complex(value: Any) -> complex | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "real" in value or "imag" in value:
            out = complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
        else:
            out = complex(value)
    else:
        out = complex(value)
    if not np.isfinite(out.real) or not np.isfinite(out.imag):
        raise ValueError(f"refractive_index must have finite real/imag parts; got {value!r}.")
    return out


def _jsonable_complex(value: complex | None) -> Any:
    if value is None:
        return None
    return {"real": float(value.real), "imag": float(value.imag)}


def _coerce_vector3_nm(value: Any, *, field_name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{field_name} must be a length-3 [x, y, z] vector in nm.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must contain only finite values in nm.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _validate_material_properties(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be None or a material-properties dictionary.")
    out = deepcopy(value)
    for key in _NONNEGATIVE_MATERIAL_FIELDS:
        if key not in out or out[key] is None:
            continue
        numeric = float(out[key])
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{field_name}.{key} must be finite and non-negative; got {out[key]!r}.")
        out[key] = numeric
    for key in ("emission_peak_nm", "excitation_peak_nm"):
        if key not in out or out[key] is None:
            continue
        numeric = float(out[key])
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{field_name}.{key} must be finite and positive; got {out[key]!r}.")
        out[key] = numeric
    if "n_complex_visible" in out and out["n_complex_visible"] is not None:
        n_value = _coerce_optional_complex(out["n_complex_visible"])
        out["n_complex_visible"] = {"real": float(n_value.real), "imag": float(n_value.imag)}
    return out


def _fingerprint_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _fingerprint_json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _fingerprint_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _fingerprint_json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _particles_fingerprint(particles: Any) -> str:
    return json.dumps(_fingerprint_json_safe(particles), sort_keys=True, separators=(",", ":"))


def _build_component(
    raw_component: dict[str, Any],
    *,
    particle_index: int,
    component_index: int,
) -> ParticleComponentSpec:
    required_keys = {
        "shape",
        "offset_nm",
        "diameter_nm",
        "material",
        "refractive_index",
        "signal_multiplier",
        "source_multiplier",
        "material_properties",
    }
    missing = sorted(required_keys.difference(raw_component))
    if missing:
        raise ValueError(
            f"PARAMS['particles'][{particle_index}]['components'][{component_index}] "
            f"is missing required canonical keys: {missing}."
        )
    diameter_nm = float(raw_component["diameter_nm"])
    # NaN <= 0.0 is False, so a bare ``<= 0`` test does not reject NaN/inf
    # diameters and propagates them through to Mie scattering, the PSF FFT,
    # and the Brownian step size. Require finite + positive explicitly.
    if not np.isfinite(diameter_nm) or diameter_nm <= 0.0:
        raise ValueError(
            f"Particle {particle_index} component {component_index} diameter_nm "
            f"must be a finite positive number; got {diameter_nm!r}."
        )

    shape = str(raw_component["shape"]).strip().lower()
    if shape != "sphere":
        raise ValueError(
            f"Particle {particle_index} component {component_index} shape must be 'sphere'. "
            "Represent non-spherical particles as multiple spherical components."
        )

    signal_multiplier = float(raw_component["signal_multiplier"])
    if not np.isfinite(signal_multiplier) or signal_multiplier < 0.0:
        raise ValueError(
            f"Particle {particle_index} component {component_index} signal_multiplier "
            f"must be a finite non-negative number; got {signal_multiplier!r}."
        )
    source_multiplier = float(raw_component["source_multiplier"])
    if not np.isfinite(source_multiplier) or source_multiplier < 0.0:
        raise ValueError(
            f"Particle {particle_index} component {component_index} source_multiplier "
            f"must be a finite non-negative number; got {source_multiplier!r}."
        )

    return ParticleComponentSpec(
        shape=shape,
        offset_nm=_coerce_vector3_nm(
            raw_component["offset_nm"],
            field_name=f"PARAMS['particles'][{particle_index}]['components'][{component_index}].offset_nm",
        ),
        diameter_nm=diameter_nm,
        material=raw_component["material"],
        refractive_index=_coerce_optional_complex(raw_component["refractive_index"]),
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=_validate_material_properties(
            raw_component["material_properties"],
            field_name=(
                f"PARAMS['particles'][{particle_index}]['components']"
                f"[{component_index}].material_properties"
            ),
        ),
    )


def _coerce_optional_symmetry_dim(value: Any, *, particle_index: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"Particle {particle_index} continuous_rotational_symmetry_dim must be "
            f"an integer in [0, 3] or None; got {value!r}."
        )
    out = int(value)
    if out < 0 or out > 3:
        raise ValueError(
            f"Particle {particle_index} continuous_rotational_symmetry_dim must be "
            f"in [0, 3]; got {out}."
        )
    return out


def _coerce_singular_rotation_axes(value: Any, *, particle_index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        axes = (value,)
    else:
        axes = tuple(str(axis) for axis in value)
    allowed = set(SE3_STATE_AXES)
    invalid = [axis for axis in axes if axis not in allowed]
    if invalid:
        raise ValueError(
            f"Particle {particle_index} singular_rotation_axes_body contains "
            f"unsupported axis names: {invalid!r}."
        )
    return axes


def normalize_particle_specs(params: dict, *, mutate: bool = True) -> list[ParticleSpec]:
    """
    Parse PARAMS['particles'] into ParticleSpec objects.

    The core accepts a single particles list and deliberately does not read or
    synthesize parallel particle arrays.
    If ``mutate`` is true, the parsed canonical dictionaries replace
    ``params['particles']`` so downstream code sees normalized values.
    """

    particles = resolved_particles(params)
    if not isinstance(particles, list) or len(particles) == 0:
        raise ValueError("PARAMS['particles'] must be a non-empty list of particle objects.")

    specs: list[ParticleSpec] = []
    for p_idx, raw_particle in enumerate(particles):
        if not isinstance(raw_particle, dict):
            raise TypeError(f"PARAMS['particles'][{p_idx}] must be a dictionary.")

        required_particle_keys = {
            "name",
            "motion",
            "signal_multiplier",
            "source_multiplier",
            "components",
        }
        missing_particle_keys = sorted(required_particle_keys.difference(raw_particle))
        if missing_particle_keys:
            raise ValueError(
                f"PARAMS['particles'][{p_idx}] is missing required canonical keys: "
                f"{missing_particle_keys}."
            )

        name = str(raw_particle["name"])
        motion_raw = raw_particle["motion"]
        if not isinstance(motion_raw, dict):
            raise TypeError(f"PARAMS['particles'][{p_idx}]['motion'] must be a dictionary.")
        required_motion_keys = {"hydrodynamic_diameter_nm", "initial_position_nm"}
        missing_motion_keys = sorted(required_motion_keys.difference(motion_raw))
        if missing_motion_keys:
            raise ValueError(
                f"PARAMS['particles'][{p_idx}]['motion'] is missing required canonical keys: "
                f"{missing_motion_keys}."
            )

        components_raw = raw_particle["components"]
        if not isinstance(components_raw, list) or len(components_raw) == 0:
            raise ValueError(f"PARAMS['particles'][{p_idx}] must define at least one component.")

        components: list[ParticleComponentSpec] = []
        for c_idx, raw_component in enumerate(components_raw):
            if not isinstance(raw_component, dict):
                raise TypeError(f"Particle {p_idx} component {c_idx} must be a dictionary.")
            components.append(
                _build_component(
                    raw_component,
                    particle_index=p_idx,
                    component_index=c_idx,
                )
            )

        particle_signal_multiplier = float(raw_particle["signal_multiplier"])
        if not np.isfinite(particle_signal_multiplier) or particle_signal_multiplier < 0.0:
            raise ValueError(
                f"Particle {p_idx} signal_multiplier must be a finite non-negative "
                f"number; got {particle_signal_multiplier!r}."
            )
        particle_source_multiplier = float(raw_particle["source_multiplier"])
        if not np.isfinite(particle_source_multiplier) or particle_source_multiplier < 0.0:
            raise ValueError(
                f"Particle {p_idx} source_multiplier must be a finite non-negative "
                f"number; got {particle_source_multiplier!r}."
            )

        symmetry_class_raw = raw_particle.get("symmetry_class")
        symmetry_class = None if symmetry_class_raw is None else str(symmetry_class_raw)
        continuous_rotational_symmetry_dim = _coerce_optional_symmetry_dim(
            raw_particle.get("continuous_rotational_symmetry_dim"),
            particle_index=p_idx,
        )
        singular_rotation_axes_body = _coerce_singular_rotation_axes(
            raw_particle.get("singular_rotation_axes_body"),
            particle_index=p_idx,
        )

        hydrodynamic_raw = motion_raw["hydrodynamic_diameter_nm"]
        hydrodynamic_diameter_nm = float(hydrodynamic_raw)
        if not np.isfinite(hydrodynamic_diameter_nm) or hydrodynamic_diameter_nm <= 0.0:
            raise ValueError(
                f"Particle {p_idx} hydrodynamic_diameter_nm must be a finite "
                f"positive number; got {hydrodynamic_diameter_nm!r}."
            )

        initial_raw = motion_raw["initial_position_nm"]
        initial_position_nm = None if initial_raw is None else _coerce_vector3_nm(
            initial_raw,
            field_name=f"PARAMS['particles'][{p_idx}]['motion'].initial_position_nm",
        )

        specs.append(
            ParticleSpec(
                name=name,
                motion=ParticleMotionSpec(
                    hydrodynamic_diameter_nm=hydrodynamic_diameter_nm,
                    initial_position_nm=initial_position_nm,
                ),
                components=tuple(components),
                signal_multiplier=particle_signal_multiplier,
                source_multiplier=particle_source_multiplier,
                symmetry_class=symmetry_class,
                continuous_rotational_symmetry_dim=continuous_rotational_symmetry_dim,
                singular_rotation_axes_body=singular_rotation_axes_body,
            )
        )

    if mutate:
        normalized_particles = particle_specs_to_public_dicts(specs)
        params["particles"] = normalized_particles
        params["_particle_specs"] = specs
        params["_particle_specs_fingerprint"] = _particles_fingerprint(normalized_particles)
        params["_resolved_particles"] = normalized_particles
    return specs


def particle_specs_to_public_dicts(specs: list[ParticleSpec]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in specs:
        item = {
            "name": spec.name,
            "motion": {
                "hydrodynamic_diameter_nm": float(spec.motion.hydrodynamic_diameter_nm),
                "initial_position_nm": (
                    None if spec.motion.initial_position_nm is None else list(spec.motion.initial_position_nm)
                ),
            },
            "signal_multiplier": float(spec.signal_multiplier),
            "source_multiplier": float(spec.source_multiplier),
            "components": [
                {
                    "shape": component.shape,
                    "offset_nm": list(component.offset_nm),
                    "diameter_nm": float(component.diameter_nm),
                    "material": component.material,
                    "refractive_index": _jsonable_complex(component.refractive_index),
                    "signal_multiplier": float(component.signal_multiplier),
                    "source_multiplier": float(component.source_multiplier),
                    "material_properties": deepcopy(component.material_properties),
                }
                for component in spec.components
            ],
        }
        if spec.symmetry_class is not None:
            item["symmetry_class"] = spec.symmetry_class
        if spec.continuous_rotational_symmetry_dim is not None:
            item["continuous_rotational_symmetry_dim"] = int(
                spec.continuous_rotational_symmetry_dim
            )
        if spec.singular_rotation_axes_body:
            item["singular_rotation_axes_body"] = list(spec.singular_rotation_axes_body)
        out.append(item)
    return out


def get_particle_specs(params: dict) -> list[ParticleSpec]:
    particles = resolved_particles(params)
    current_fingerprint = _particles_fingerprint(particles)
    cached = internal_param_value(params, "_particle_specs")
    cached_fingerprint = internal_param_value(params, "_particle_specs_fingerprint")
    if isinstance(cached, list) and cached and cached_fingerprint == current_fingerprint:
        return cached
    return normalize_particle_specs(params, mutate=True)


def particle_count(params: dict) -> int:
    return len(get_particle_specs(params))


def hydrodynamic_diameters_nm(params: dict) -> np.ndarray:
    specs = get_particle_specs(params)
    return np.asarray([spec.motion.hydrodynamic_diameter_nm for spec in specs], dtype=float)


def initial_positions_from_specs_nm(params: dict) -> list[tuple[float, float, float] | None]:
    specs = get_particle_specs(params)
    return [spec.motion.initial_position_nm for spec in specs]
