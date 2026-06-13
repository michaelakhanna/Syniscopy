from __future__ import annotations
from configured_parameters import configured_assign, configured_value

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from material_volume_geometry import (
    MaterialVolumeGeometry,
    load_voxel_volume_file,
    voxelize_mesh_file,
)
from shared_constants import NONNEGATIVE_MATERIAL_PROPERTY_FIELDS, SE3_STATE_AXES
from simulation_runtime_state import runtime_state


@dataclass(frozen=True)
class ParticleComponentSpec:
    """Static physical description of one rigid primitive component."""

    shape: str
    offset_nm: tuple[float, float, float]
    diameter_nm: float
    axes_nm: tuple[float, float, float] | None = None
    length_nm: float | None = None
    material: str | None = None
    refractive_index: complex | None = None
    signal_multiplier: float = 1.0
    source_multiplier: float = 1.0
    material_properties: dict[str, Any] | None = None
    voxel_geometry: MaterialVolumeGeometry | None = None

    @property
    def shape_kind(self) -> str:
        return self.shape.strip().lower()

    @property
    def is_sphere(self) -> bool:
        return self.shape_kind == "sphere"

    @property
    def semi_axes_nm(self) -> tuple[float, float, float]:
        shape = self.shape_kind
        if shape == "sphere":
            r = 0.5 * float(self.diameter_nm)
            return (r, r, r)
        if shape == "ellipsoid":
            if self.axes_nm is None:
                raise ValueError("ellipsoid component requires axes_nm.")
            return tuple(0.5 * float(axis) for axis in self.axes_nm)
        if shape in {"cylinder", "spherocylinder"}:
            if self.length_nm is None:
                raise ValueError(f"{shape} component requires length_nm.")
            return (
                0.5 * float(self.length_nm),
                0.5 * float(self.diameter_nm),
                0.5 * float(self.diameter_nm),
            )
        if shape == "voxel_volume":
            raise ValueError("voxel_volume components do not have analytic semi_axes_nm.")
        raise ValueError(f"Unsupported particle component shape {self.shape!r}.")

    @property
    def volume_nm3(self) -> float:
        shape = self.shape_kind
        if shape == "sphere":
            r = 0.5 * float(self.diameter_nm)
            return float((4.0 / 3.0) * np.pi * r ** 3)
        if shape == "ellipsoid":
            a, b, c = self.semi_axes_nm
            return float((4.0 / 3.0) * np.pi * a * b * c)
        if shape == "cylinder":
            r = 0.5 * float(self.diameter_nm)
            return float(np.pi * r * r * float(self.length_nm))
        if shape == "spherocylinder":
            r = 0.5 * float(self.diameter_nm)
            cylinder_length = max(float(self.length_nm) - 2.0 * r, 0.0)
            return float(np.pi * r * r * cylinder_length + (4.0 / 3.0) * np.pi * r ** 3)
        if shape == "voxel_volume":
            if self.voxel_geometry is None:
                raise ValueError("voxel_volume component requires voxel_geometry.")
            return float(self.voxel_geometry.volume_nm3)
        raise ValueError(f"Unsupported particle component shape {self.shape!r}.")

    @property
    def bounding_radius_nm(self) -> float:
        shape = self.shape_kind
        if shape == "sphere":
            return 0.5 * float(self.diameter_nm)
        if shape == "ellipsoid":
            return max(self.semi_axes_nm)
        if shape == "cylinder":
            half_length, radius, _ = self.semi_axes_nm
            return float(np.hypot(half_length, radius))
        if shape == "spherocylinder":
            return 0.5 * float(self.length_nm)
        if shape == "voxel_volume":
            if self.voxel_geometry is None:
                raise ValueError("voxel_volume component requires voxel_geometry.")
            return float(self.voxel_geometry.bounding_radius_nm)
        raise ValueError(f"Unsupported particle component shape {self.shape!r}.")

    @property
    def source_normalization_length_nm(self) -> float:
        if self.shape_kind == "ellipsoid":
            return float(max(self.axes_nm or (self.diameter_nm,)))
        if self.shape_kind in {"cylinder", "spherocylinder"}:
            return float(max(float(self.length_nm), float(self.diameter_nm)))
        if self.shape_kind == "voxel_volume":
            if self.voxel_geometry is None:
                raise ValueError("voxel_volume component requires voxel_geometry.")
            return float(self.voxel_geometry.source_normalization_length_nm)
        return float(self.diameter_nm)

    @property
    def supports_exact_mie(self) -> bool:
        return self.is_sphere

    @property
    def geometry_dimensions_key(self) -> tuple[Any, ...]:
        """Stable geometry identity for renderer/scattering caches."""
        shape = self.shape_kind
        if shape == "sphere":
            return ("diameter_nm", float(self.diameter_nm))
        if shape == "ellipsoid":
            if self.axes_nm is None:
                raise ValueError("ellipsoid component requires axes_nm.")
            return ("axes_nm", tuple(float(v) for v in self.axes_nm))
        if shape in {"cylinder", "spherocylinder"}:
            if self.length_nm is None:
                raise ValueError(f"{shape} component requires length_nm.")
            return (
                "length_nm",
                float(self.length_nm),
                "diameter_nm",
                float(self.diameter_nm),
            )
        if shape == "voxel_volume":
            if self.voxel_geometry is None:
                raise ValueError("voxel_volume component requires voxel_geometry.")
            return (
                "voxel_size_nm",
                float(self.voxel_geometry.voxel_size_nm),
                "grid_shape",
                tuple(int(v) for v in self.voxel_geometry.grid_shape),
                "fingerprint",
                self.voxel_geometry.fingerprint,
            )
        raise ValueError(f"Unsupported particle component shape {self.shape!r}.")

    @property
    def source_geometry_key(self) -> tuple[Any, ...]:
        """Renderer identity for material-source modalities independent of optics."""
        return ("source_geometry", self.shape_kind, self.geometry_dimensions_key)

    @property
    def valid_optical_scattering_models(self) -> tuple[str, ...]:
        """Scattering models physically valid for this geometry shape."""
        if self.supports_exact_mie:
            return ("mie", "analytic_polarizability", "born_rayleigh_gans")
        if self.shape_kind in {"ellipsoid", "cylinder", "spherocylinder"}:
            return ("analytic_polarizability", "born_rayleigh_gans")
        if self.shape_kind == "voxel_volume":
            return ("born_rayleigh_gans",)
        return ()

    @property
    def default_optical_scattering_model(self) -> str:
        if self.supports_exact_mie:
            return "mie"
        valid = self.valid_optical_scattering_models
        if not valid:
            raise ValueError(
                f"Particle component shape {self.shape_kind!r} has no valid optical scattering model."
            )
        return valid[0]

    def supports_optical_scattering_model(self, model: str) -> bool:
        return str(model).strip().lower() in set(self.valid_optical_scattering_models)

    @property
    def is_axisymmetric_about_body_x(self) -> bool:
        shape = self.shape_kind
        if shape == "sphere":
            return True
        if shape in {"cylinder", "spherocylinder"}:
            return True
        if shape == "ellipsoid":
            axes = self.axes_nm or ()
            return len(axes) == 3 and np.isclose(float(axes[1]), float(axes[2]), rtol=1e-9, atol=1e-9)
        if shape == "voxel_volume":
            return False
        return False

    def axial_half_extent_nm(self, orientation_matrix: np.ndarray | None = None) -> float:
        """Return support half-width along world z for the oriented primitive."""
        shape = self.shape_kind
        if orientation_matrix is None:
            R = np.eye(3, dtype=float)
        else:
            R = np.asarray(orientation_matrix, dtype=float)
            if R.shape != (3, 3):
                raise ValueError("orientation_matrix must be 3x3.")
        z_local = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        if shape == "sphere":
            return 0.5 * float(self.diameter_nm)
        if shape == "ellipsoid":
            axes = np.asarray(self.semi_axes_nm, dtype=float)
            return float(np.sqrt(np.sum((axes * z_local) ** 2)))
        if shape == "cylinder":
            half_length = 0.5 * float(self.length_nm)
            radius = 0.5 * float(self.diameter_nm)
            radial = np.hypot(float(z_local[1]), float(z_local[2]))
            return float(abs(float(z_local[0])) * half_length + radius * radial)
        if shape == "spherocylinder":
            radius = 0.5 * float(self.diameter_nm)
            segment_half_length = 0.5 * max(float(self.length_nm) - float(self.diameter_nm), 0.0)
            return float(abs(float(z_local[0])) * segment_half_length + radius)
        if shape == "voxel_volume":
            if self.voxel_geometry is None:
                raise ValueError("voxel_volume component requires voxel_geometry.")
            return float(self.voxel_geometry.axial_half_extent_nm(R))
        raise ValueError(f"Unsupported particle component shape {self.shape!r}.")


_NONNEGATIVE_MATERIAL_FIELDS = NONNEGATIVE_MATERIAL_PROPERTY_FIELDS
_COMPONENT_SHAPES = {"sphere", "ellipsoid", "cylinder", "spherocylinder", "voxel_volume"}
_COMPONENT_PRODUCER_SHAPES = _COMPONENT_SHAPES | {"mesh", "voxel_file"}
_ROTATION_AXES = ("omega_x", "omega_y", "omega_z")


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
    geometry_symmetry_class: str | None = None
    geometry_continuous_rotational_stabilizer_dim: int | None = None
    geometry_singular_rotation_axes_body: tuple[str, ...] = ()

    @property
    def primary_component(self) -> ParticleComponentSpec:
        return self.components[0]

    @property
    def is_single_sphere(self) -> bool:
        if len(self.components) != 1:
            return False
        c = self.components[0]
        return c.is_sphere and all(
            abs(float(x)) <= 1e-12 for x in c.offset_nm
        )

    @property
    def is_orientation_invariant(self) -> bool:
        return self.geometry_continuous_rotational_stabilizer_dim == 3


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


def _coerce_positive_float(value: Any, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} must be provided as a finite positive number.")
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number; got {value!r}.")
    return out


def _coerce_positive_axes3_nm(value: Any, *, field_name: str) -> tuple[float, float, float]:
    axes = _coerce_vector3_nm(value, field_name=field_name)
    invalid = [axis for axis in axes if axis <= 0.0]
    if invalid:
        raise ValueError(f"{field_name} must contain finite positive axis diameters; got {value!r}.")
    return axes


def _equivalent_sphere_diameter_nm_for_axes(axes_nm: tuple[float, float, float]) -> float:
    axes = np.asarray(axes_nm, dtype=float)
    return float(np.prod(axes) ** (1.0 / 3.0))


def _assert_optional_diameter_matches(
    raw_component: dict[str, Any],
    *,
    expected_diameter_nm: float,
    particle_index: int,
    component_index: int,
) -> None:
    if "diameter_nm" not in raw_component or raw_component["diameter_nm"] is None:
        return
    supplied = _coerce_positive_float(
        raw_component["diameter_nm"],
        field_name=f"parameters['particles'][{particle_index}]['components'][{component_index}].diameter_nm",
    )
    if not np.isclose(supplied, expected_diameter_nm, rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"Particle {particle_index} component {component_index} supplies diameter_nm={supplied!r} "
            f"but its geometry derives equivalent diameter {expected_diameter_nm!r}. "
            "Do not maintain parallel size definitions; set the primitive dimensions only."
        )


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
    for key in ("atomic_number", "atomic_weight_g_mol"):
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
        "material",
        "refractive_index",
        "signal_multiplier",
        "source_multiplier",
        "material_properties",
    }
    missing = sorted(required_keys.difference(raw_component))
    if missing:
        raise ValueError(
            f"parameters['particles'][{particle_index}]['components'][{component_index}] "
            f"is missing required canonical keys: {missing}."
        )
    shape = str(raw_component["shape"]).strip().lower()
    if shape not in _COMPONENT_PRODUCER_SHAPES:
        raise ValueError(
            f"Particle {particle_index} component {component_index} shape must be one of "
            f"{sorted(_COMPONENT_PRODUCER_SHAPES)}; got {shape!r}."
        )

    axes_nm: tuple[float, float, float] | None = None
    length_nm: float | None = None
    voxel_geometry: MaterialVolumeGeometry | None = None
    diameter_nm: float
    if shape == "mesh":
        if "mesh_path" not in raw_component:
            raise ValueError(
                f"Particle {particle_index} component {component_index} mesh requires mesh_path."
            )
        if "mesh_voxel_size_nm" not in raw_component:
            raise ValueError(
                f"Particle {particle_index} component {component_index} mesh requires mesh_voxel_size_nm."
            )
        voxel_geometry = voxelize_mesh_file(
            raw_component["mesh_path"],
            voxel_size_nm=raw_component["mesh_voxel_size_nm"],
            mesh_scale_nm_per_unit=float(raw_component.get("mesh_scale_nm_per_unit", 1.0)),
        )
        shape = "voxel_volume"
        diameter_nm = float(voxel_geometry.equivalent_sphere_diameter_nm)
        _assert_optional_diameter_matches(
            raw_component,
            expected_diameter_nm=diameter_nm,
            particle_index=particle_index,
            component_index=component_index,
        )
    elif shape == "voxel_file":
        if "voxel_path" not in raw_component:
            raise ValueError(
                f"Particle {particle_index} component {component_index} voxel_file requires voxel_path."
            )
        if "voxel_size_nm" not in raw_component:
            raise ValueError(
                f"Particle {particle_index} component {component_index} voxel_file requires voxel_size_nm."
            )
        voxel_geometry = load_voxel_volume_file(
            raw_component["voxel_path"],
            voxel_size_nm=raw_component["voxel_size_nm"],
            voxel_array_key=raw_component.get("voxel_array_key"),
            occupancy_threshold=raw_component.get("voxel_occupancy_threshold"),
        )
        shape = "voxel_volume"
        diameter_nm = float(voxel_geometry.equivalent_sphere_diameter_nm)
        _assert_optional_diameter_matches(
            raw_component,
            expected_diameter_nm=diameter_nm,
            particle_index=particle_index,
            component_index=component_index,
        )
    elif shape == "voxel_volume":
        if "voxel_geometry" not in raw_component:
            raise ValueError(
                f"Particle {particle_index} component {component_index} voxel_volume requires voxel_geometry."
            )
        voxel_geometry = MaterialVolumeGeometry.from_public_dict(raw_component["voxel_geometry"])
        diameter_nm = float(voxel_geometry.equivalent_sphere_diameter_nm)
        _assert_optional_diameter_matches(
            raw_component,
            expected_diameter_nm=diameter_nm,
            particle_index=particle_index,
            component_index=component_index,
        )
        if "axes_nm" in raw_component and raw_component["axes_nm"] is not None:
            raise ValueError("voxel_volume components use voxel_geometry, not axes_nm.")
        if "length_nm" in raw_component and raw_component["length_nm"] is not None:
            raise ValueError("voxel_volume components use voxel_geometry, not length_nm.")
    elif shape == "sphere":
        if "axes_nm" in raw_component and raw_component["axes_nm"] is not None:
            raise ValueError("Sphere components use diameter_nm, not axes_nm.")
        if "length_nm" in raw_component and raw_component["length_nm"] is not None:
            raise ValueError("Sphere components use diameter_nm, not length_nm.")
        diameter_nm = _coerce_positive_float(
            raw_component.get("diameter_nm"),
            field_name=f"parameters['particles'][{particle_index}]['components'][{component_index}].diameter_nm",
        )
    elif shape == "ellipsoid":
        if "length_nm" in raw_component and raw_component["length_nm"] is not None:
            raise ValueError("Ellipsoid components use axes_nm, not length_nm.")
        if "axes_nm" not in raw_component:
            raise ValueError(
                f"Particle {particle_index} component {component_index} ellipsoid requires axes_nm."
            )
        axes_nm = _coerce_positive_axes3_nm(
            raw_component["axes_nm"],
            field_name=f"parameters['particles'][{particle_index}]['components'][{component_index}].axes_nm",
        )
        if np.allclose(axes_nm, axes_nm[0], rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"Particle {particle_index} component {component_index} has equal ellipsoid axes; "
                "use shape='sphere' for the exact isotropic primitive."
            )
        diameter_nm = _equivalent_sphere_diameter_nm_for_axes(axes_nm)
        _assert_optional_diameter_matches(
            raw_component,
            expected_diameter_nm=diameter_nm,
            particle_index=particle_index,
            component_index=component_index,
        )
    else:
        if "axes_nm" in raw_component and raw_component["axes_nm"] is not None:
            raise ValueError(f"{shape} components use length_nm and diameter_nm, not axes_nm.")
        diameter_nm = _coerce_positive_float(
            raw_component.get("diameter_nm"),
            field_name=f"parameters['particles'][{particle_index}]['components'][{component_index}].diameter_nm",
        )
        if "length_nm" not in raw_component:
            raise ValueError(
                f"Particle {particle_index} component {component_index} {shape} requires length_nm."
            )
        length_nm = _coerce_positive_float(
            raw_component["length_nm"],
            field_name=f"parameters['particles'][{particle_index}]['components'][{component_index}].length_nm",
        )
        if shape == "spherocylinder" and length_nm <= diameter_nm:
            raise ValueError(
                f"Particle {particle_index} component {component_index} spherocylinder length_nm "
                "must be greater than diameter_nm; use shape='sphere' for the equal-length limit."
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
            field_name=f"parameters['particles'][{particle_index}]['components'][{component_index}].offset_nm",
        ),
        diameter_nm=diameter_nm,
        axes_nm=axes_nm,
        length_nm=length_nm,
        material=raw_component["material"],
        refractive_index=_coerce_optional_complex(raw_component["refractive_index"]),
        signal_multiplier=signal_multiplier,
        source_multiplier=source_multiplier,
        material_properties=_validate_material_properties(
            raw_component["material_properties"],
            field_name=(
                f"parameters['particles'][{particle_index}]['components']"
                f"[{component_index}].material_properties"
            ),
        ),
        voxel_geometry=voxel_geometry,
    )


def _coerce_optional_geometry_stabilizer_dim(value: Any, *, particle_index: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"Particle {particle_index} geometry_continuous_rotational_stabilizer_dim must be "
            f"an integer in [0, 3] or None; got {value!r}."
        )
    out = int(value)
    if out < 0 or out > 3:
        raise ValueError(
            f"Particle {particle_index} geometry_continuous_rotational_stabilizer_dim must be "
            f"in [0, 3]; got {out}."
        )
    return out


def _coerce_geometry_singular_rotation_axes(value: Any, *, particle_index: int) -> tuple[str, ...]:
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
            f"Particle {particle_index} geometry_singular_rotation_axes_body contains "
            f"unsupported axis names: {invalid!r}."
        )
    return axes


def _derive_particle_symmetry(
    components: tuple[ParticleComponentSpec, ...],
) -> tuple[str, int, tuple[str, ...]]:
    if len(components) == 1:
        component = components[0]
        if component.is_sphere:
            return "sphere_so3", 3, _ROTATION_AXES
        if component.is_axisymmetric_about_body_x:
            return "axisymmetric_body_x", 1, ("omega_x",)
        return "asymmetric_rigid_geometry", 0, ()

    axisymmetric_about_x = True
    for component in components:
        offset = np.asarray(component.offset_nm, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            axisymmetric_about_x = False
            break
        if abs(float(offset[1])) > 1e-9 or abs(float(offset[2])) > 1e-9:
            axisymmetric_about_x = False
            break
        if not component.is_axisymmetric_about_body_x:
            axisymmetric_about_x = False
            break
    if axisymmetric_about_x:
        return "axisymmetric_body_x", 1, ("omega_x",)
    return "asymmetric_rigid_geometry", 0, ()


def _reject_conflicting_geometry_symmetry_metadata(
    raw_particle: dict[str, Any],
    *,
    particle_index: int,
    derived_geometry_symmetry_class: str,
    derived_geometry_stabilizer_dim: int,
    derived_geometry_singular_axes: tuple[str, ...],
) -> None:
    retired_keys = {
        "symmetry_class",
        "continuous_rotational_symmetry_dim",
        "singular_rotation_axes_body",
    }.intersection(raw_particle)
    if retired_keys:
        raise ValueError(
            f"Particle {particle_index} uses generic symmetry metadata {sorted(retired_keys)!r}. "
            "Symmetry is geometry-derived; use geometry_* metadata only in normalized "
            "particle records and do not declare it as an independent input."
        )

    if (
        "geometry_symmetry_class" in raw_particle
        and raw_particle["geometry_symmetry_class"] is not None
    ):
        supplied = str(raw_particle["geometry_symmetry_class"])
        if supplied != derived_geometry_symmetry_class:
            raise ValueError(
                f"Particle {particle_index} declares geometry_symmetry_class={supplied!r}, "
                f"but geometry derives {derived_geometry_symmetry_class!r}."
            )
    if (
        "geometry_continuous_rotational_stabilizer_dim" in raw_particle
        and raw_particle["geometry_continuous_rotational_stabilizer_dim"] is not None
    ):
        supplied_dim = _coerce_optional_geometry_stabilizer_dim(
            raw_particle["geometry_continuous_rotational_stabilizer_dim"],
            particle_index=particle_index,
        )
        if supplied_dim != derived_geometry_stabilizer_dim:
            raise ValueError(
                f"Particle {particle_index} declares "
                f"geometry_continuous_rotational_stabilizer_dim={supplied_dim!r}, "
                f"but geometry derives {derived_geometry_stabilizer_dim!r}."
            )
    if (
        "geometry_singular_rotation_axes_body" in raw_particle
        and raw_particle["geometry_singular_rotation_axes_body"] is not None
    ):
        supplied_axes = _coerce_geometry_singular_rotation_axes(
            raw_particle["geometry_singular_rotation_axes_body"],
            particle_index=particle_index,
        )
        if tuple(supplied_axes) != tuple(derived_geometry_singular_axes):
            raise ValueError(
                f"Particle {particle_index} declares "
                f"geometry_singular_rotation_axes_body={list(supplied_axes)!r}, "
                f"but geometry derives {list(derived_geometry_singular_axes)!r}."
            )


def normalize_particle_specs(params: dict, *, mutate: bool = True) -> list[ParticleSpec]:
    """
    Parse parameters['particles'] into ParticleSpec objects.

    The core accepts a single particles list and deliberately does not read or
    synthesize parallel particle arrays.
    If ``mutate`` is true, the parsed canonical dictionaries replace
    ``configured parameters['particles']`` so downstream code sees normalized values.
    """

    particles = mutable_particle_scene_from_params(params)

    specs: list[ParticleSpec] = []
    for p_idx, raw_particle in enumerate(particles):
        if not isinstance(raw_particle, dict):
            raise TypeError(f"parameters['particles'][{p_idx}] must be a dictionary.")

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
                f"parameters['particles'][{p_idx}] is missing required canonical keys: "
                f"{missing_particle_keys}."
            )

        name = str(raw_particle["name"])
        motion_raw = raw_particle["motion"]
        if not isinstance(motion_raw, dict):
            raise TypeError(f"parameters['particles'][{p_idx}]['motion'] must be a dictionary.")
        required_motion_keys = {"hydrodynamic_diameter_nm", "initial_position_nm"}
        missing_motion_keys = sorted(required_motion_keys.difference(motion_raw))
        if missing_motion_keys:
            raise ValueError(
                f"parameters['particles'][{p_idx}]['motion'] is missing required canonical keys: "
                f"{missing_motion_keys}."
            )

        components_raw = raw_particle["components"]
        if not isinstance(components_raw, list) or len(components_raw) == 0:
            raise ValueError(f"parameters['particles'][{p_idx}] must define at least one component.")

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

        (
            derived_geometry_symmetry_class,
            derived_geometry_stabilizer_dim,
            derived_geometry_singular_axes,
        ) = _derive_particle_symmetry(
            tuple(components)
        )
        _reject_conflicting_geometry_symmetry_metadata(
            raw_particle,
            particle_index=p_idx,
            derived_geometry_symmetry_class=derived_geometry_symmetry_class,
            derived_geometry_stabilizer_dim=derived_geometry_stabilizer_dim,
            derived_geometry_singular_axes=derived_geometry_singular_axes,
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
            field_name=f"parameters['particles'][{p_idx}]['motion'].initial_position_nm",
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
                geometry_symmetry_class=derived_geometry_symmetry_class,
                geometry_continuous_rotational_stabilizer_dim=derived_geometry_stabilizer_dim,
                geometry_singular_rotation_axes_body=derived_geometry_singular_axes,
            )
        )

    if mutate:
        normalized_particles = particle_specs_to_public_dicts(specs)
        configured_assign(params, 'particles', normalized_particles)
        state = runtime_state(params)
        state.particle_specs = specs
        state.particle_specs_fingerprint = _particles_fingerprint(normalized_particles)
    return specs


def particle_specs_to_public_dicts(specs: list[ParticleSpec]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in specs:
        component_dicts: list[dict[str, Any]] = []
        for component in spec.components:
            component_item = {
                "shape": component.shape,
                "offset_nm": list(component.offset_nm),
                "diameter_nm": float(component.diameter_nm),
                "material": component.material,
                "refractive_index": _jsonable_complex(component.refractive_index),
                "signal_multiplier": float(component.signal_multiplier),
                "source_multiplier": float(component.source_multiplier),
                "material_properties": deepcopy(component.material_properties),
            }
            if component.axes_nm is not None:
                component_item["axes_nm"] = list(component.axes_nm)
            if component.length_nm is not None:
                component_item["length_nm"] = float(component.length_nm)
            if component.voxel_geometry is not None:
                component_item["voxel_geometry"] = component.voxel_geometry.to_public_dict()
            component_dicts.append(component_item)
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
            "components": component_dicts,
        }
        item["geometry_symmetry_class"] = spec.geometry_symmetry_class
        item["geometry_continuous_rotational_stabilizer_dim"] = int(
            spec.geometry_continuous_rotational_stabilizer_dim or 0
        )
        if spec.geometry_singular_rotation_axes_body:
            item["geometry_singular_rotation_axes_body"] = list(
                spec.geometry_singular_rotation_axes_body
            )
        out.append(item)
    return out


def get_particle_specs(params: dict) -> list[ParticleSpec]:
    particles = mutable_particle_scene_from_params(params)
    current_fingerprint = _particles_fingerprint(particles)
    state = runtime_state(params)
    cached = state.particle_specs
    cached_fingerprint = state.particle_specs_fingerprint
    if cached is not None and cached_fingerprint == current_fingerprint:
        return cached
    return normalize_particle_specs(params, mutate=True)


def mutable_particle_scene_from_params(params: dict) -> list[dict[str, Any]]:
    """Return the editable public particle scene owned by particle_specs."""

    particles = configured_value(params, "particles")
    if not isinstance(particles, list) or len(particles) == 0:
        raise ValueError("parameters['particles'] must be a non-empty list of particle objects.")
    for p_idx, particle in enumerate(particles):
        if not isinstance(particle, dict):
            raise TypeError(f"parameters['particles'][{p_idx}] must be a dictionary.")
    return particles


def particle_count(params: dict) -> int:
    return len(get_particle_specs(params))


def hydrodynamic_diameters_nm(params: dict) -> np.ndarray:
    specs = get_particle_specs(params)
    return np.asarray([spec.motion.hydrodynamic_diameter_nm for spec in specs], dtype=float)


def excluded_volume_radii_nm(params: dict) -> np.ndarray:
    """
    Conservative particle excluded-volume radius in nm.

    Hydrodynamic diameter controls diffusion.  Collision and hard-wall
    exclusion need a geometric envelope instead, especially for composites
    whose rendered components can extend beyond the hydrodynamic sphere.
    """
    specs = get_particle_specs(params)
    radii: list[float] = []
    for spec in specs:
        radius_nm = 0.0
        for component in spec.components:
            offset = np.asarray(component.offset_nm, dtype=float)
            component_radius = component.bounding_radius_nm
            radius_nm = max(radius_nm, float(np.linalg.norm(offset)) + component_radius)
        if not np.isfinite(radius_nm) or radius_nm <= 0.0:
            radius_nm = 0.5 * float(spec.motion.hydrodynamic_diameter_nm)
        radii.append(radius_nm)
    return np.asarray(radii, dtype=float)


def initial_positions_from_specs_nm(params: dict) -> list[tuple[float, float, float] | None]:
    specs = get_particle_specs(params)
    return [spec.motion.initial_position_nm for spec in specs]
