"""Geometry and contrast stabilizer metadata for SE(3) rank diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

from shared_constants import SE3_STATE_AXES


_ROTATIONAL_DIMENSION = 3


def _coerce_stabilizer_dim(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer in [0, 3]; got {value!r}.")
    out = int(value)
    if out < 0 or out > _ROTATIONAL_DIMENSION:
        raise ValueError(f"{field_name} must be in [0, 3]; got {out}.")
    return out


def _coerce_axes(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        axes = (value,)
    else:
        axes = tuple(str(axis) for axis in value)
    invalid = [axis for axis in axes if axis not in set(SE3_STATE_AXES)]
    if invalid:
        raise ValueError(f"{field_name} contains unsupported axes: {invalid!r}.")
    return axes


@dataclass(frozen=True)
class GeometrySymmetrySpec:
    """Continuous stabilizer derived from the particle geometry itself."""

    geometry_symmetry_class: str
    geometry_continuous_rotational_stabilizer_dim: int
    geometry_singular_rotation_axes_body: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "geometry_continuous_rotational_stabilizer_dim",
            _coerce_stabilizer_dim(
                self.geometry_continuous_rotational_stabilizer_dim,
                field_name="geometry_continuous_rotational_stabilizer_dim",
            ),
        )
        object.__setattr__(
            self,
            "geometry_singular_rotation_axes_body",
            _coerce_axes(
                self.geometry_singular_rotation_axes_body,
                field_name="geometry_singular_rotation_axes_body",
            ),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "geometry_symmetry_class": self.geometry_symmetry_class,
            "geometry_continuous_rotational_stabilizer_dim": int(
                self.geometry_continuous_rotational_stabilizer_dim
            ),
            "geometry_singular_rotation_axes_body": list(
                self.geometry_singular_rotation_axes_body
            ),
            "geometry_symmetry_source": "derived_from_particle_geometry",
        }


@dataclass(frozen=True)
class ContrastSymmetrySpec:
    """
    Continuous stabilizer of the rendered contrast functional.

    If this is absent, Fisher rank diagnostics may still use the geometry
    stabilizer as a nullity lower bound, but they must not report an exact
    contrast-rank prediction.
    """

    contrast_symmetry_class: str
    contrast_continuous_rotational_stabilizer_dim: int
    contrast_singular_rotation_axes_body: tuple[str, ...] = ()
    contrast_stabilizer_source: str = "declared_contrast_functional"
    contrast_stabilizer_basis: str = "rendered_contrast_functional"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contrast_continuous_rotational_stabilizer_dim",
            _coerce_stabilizer_dim(
                self.contrast_continuous_rotational_stabilizer_dim,
                field_name="contrast_continuous_rotational_stabilizer_dim",
            ),
        )
        object.__setattr__(
            self,
            "contrast_singular_rotation_axes_body",
            _coerce_axes(
                self.contrast_singular_rotation_axes_body,
                field_name="contrast_singular_rotation_axes_body",
            ),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "contrast_symmetry_class": self.contrast_symmetry_class,
            "contrast_continuous_rotational_stabilizer_dim": int(
                self.contrast_continuous_rotational_stabilizer_dim
            ),
            "contrast_singular_rotation_axes_body": list(
                self.contrast_singular_rotation_axes_body
            ),
            "contrast_stabilizer_source": str(self.contrast_stabilizer_source),
            "contrast_stabilizer_basis": str(self.contrast_stabilizer_basis),
        }


def geometry_symmetry_from_particle_spec(particle_spec: Any) -> GeometrySymmetrySpec:
    """Build geometry-stabilizer metadata from a canonical ParticleSpec-like object."""

    return GeometrySymmetrySpec(
        geometry_symmetry_class=str(particle_spec.geometry_symmetry_class),
        geometry_continuous_rotational_stabilizer_dim=int(
            particle_spec.geometry_continuous_rotational_stabilizer_dim
        ),
        geometry_singular_rotation_axes_body=tuple(
            particle_spec.geometry_singular_rotation_axes_body
        ),
    )


def _require_no_retired_generic_symmetry(metadata: Mapping[str, Any]) -> None:
    retired = {
        "symmetry_class",
        "continuous_rotational_symmetry_dim",
        "singular_rotation_axes_body",
    }.intersection(metadata)
    if retired:
        raise ValueError(
            "Generic symmetry metadata is no longer accepted at the Fisher rank "
            f"boundary: {sorted(retired)!r}. Pass geometry_* stabilizer metadata "
            "and, when known, contrast_* stabilizer metadata."
        )


def normalize_se3_rank_symmetry_metadata(
    symmetry_metadata: Mapping[str, Any] | GeometrySymmetrySpec | ContrastSymmetrySpec | None,
) -> dict[str, Any]:
    """
    Normalize explicit geometry/contrast stabilizer metadata for SE(3) rank use.

    Geometry symmetry is a lower bound on contrast nullity. Contrast symmetry is
    the actual theorem input and is only available when supplied explicitly or
    inferred from observed Fisher rank by the comparison routine.
    """

    if symmetry_metadata is None:
        return {
            "geometry_stabilizer_available": False,
            "contrast_stabilizer_available": False,
        }
    if isinstance(symmetry_metadata, GeometrySymmetrySpec):
        return {
            **symmetry_metadata.to_metadata(),
            "geometry_stabilizer_available": True,
            "contrast_stabilizer_available": False,
        }
    if isinstance(symmetry_metadata, ContrastSymmetrySpec):
        return {
            **symmetry_metadata.to_metadata(),
            "geometry_stabilizer_available": False,
            "contrast_stabilizer_available": True,
        }
    metadata = dict(symmetry_metadata)
    _require_no_retired_generic_symmetry(metadata)

    out: dict[str, Any] = {}
    geometry_dim_raw = metadata.get("geometry_continuous_rotational_stabilizer_dim")
    if geometry_dim_raw is None:
        out["geometry_stabilizer_available"] = False
    else:
        geometry = GeometrySymmetrySpec(
            geometry_symmetry_class=str(
                metadata.get("geometry_symmetry_class", "geometry_stabilizer")
            ),
            geometry_continuous_rotational_stabilizer_dim=geometry_dim_raw,
            geometry_singular_rotation_axes_body=metadata.get(
                "geometry_singular_rotation_axes_body",
                (),
            ),
        )
        out.update(geometry.to_metadata())
        out["geometry_stabilizer_available"] = True

    contrast_dim_raw = metadata.get("contrast_continuous_rotational_stabilizer_dim")
    if contrast_dim_raw is None:
        out["contrast_stabilizer_available"] = False
    else:
        contrast = ContrastSymmetrySpec(
            contrast_symmetry_class=str(
                metadata.get("contrast_symmetry_class", "contrast_stabilizer")
            ),
            contrast_continuous_rotational_stabilizer_dim=contrast_dim_raw,
            contrast_singular_rotation_axes_body=metadata.get(
                "contrast_singular_rotation_axes_body",
                (),
            ),
            contrast_stabilizer_source=str(
                metadata.get("contrast_stabilizer_source", "declared_contrast_functional")
            ),
            contrast_stabilizer_basis=str(
                metadata.get("contrast_stabilizer_basis", "rendered_contrast_functional")
            ),
        )
        out.update(contrast.to_metadata())
        out["contrast_stabilizer_available"] = True

    return out


__all__ = [
    "ContrastSymmetrySpec",
    "GeometrySymmetrySpec",
    "geometry_symmetry_from_particle_spec",
    "normalize_se3_rank_symmetry_metadata",
]
