"""Material-resolved SEM source canvas objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from material_types import MaterialProperties


@dataclass(frozen=True)
class SEMMaterialChannelKey:
    """Physical material identity for one SEM source channel."""

    material_name: str
    atomic_number: float
    atomic_weight_g_mol: float
    density_g_cm3: float
    se_yield_coefficient: float

    @classmethod
    def from_material_properties(cls, material: MaterialProperties) -> "SEMMaterialChannelKey":
        if material is None:
            raise ValueError("SEM material source accumulation requires MaterialProperties.")
        fields = {
            "atomic_number": material.atomic_number,
            "atomic_weight_g_mol": material.atomic_weight_g_mol,
            "density_g_cm3": material.density_g_cm3,
            "se_yield_coefficient": material.se_yield_coefficient,
        }
        missing = [key for key, value in fields.items() if value is None]
        if missing:
            raise ValueError(
                f"Material {material.name!r} cannot be used for physical SEM source channels; "
                f"missing {missing}."
            )
        atomic_number = float(material.atomic_number)
        atomic_weight = float(material.atomic_weight_g_mol)
        density = float(material.density_g_cm3)
        se_yield = float(material.se_yield_coefficient)
        if (
            not np.isfinite(atomic_number)
            or not np.isfinite(atomic_weight)
            or not np.isfinite(density)
            or not np.isfinite(se_yield)
            or atomic_number <= 0.0
            or atomic_weight <= 0.0
            or density <= 0.0
            or se_yield < 0.0
        ):
            raise ValueError(
                f"Material {material.name!r} has invalid SEM source constants."
            )
        return cls(
            material_name=str(material.name),
            atomic_number=atomic_number,
            atomic_weight_g_mol=atomic_weight,
            density_g_cm3=density,
            se_yield_coefficient=se_yield,
        )


@dataclass
class SEMMaterialSourceCanvas:
    """SEM source image whose channels preserve material identity."""

    shape: tuple[int, ...]
    channels: dict[SEMMaterialChannelKey, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.shape)
        if len(shape) not in {2, 3} or any(v <= 0 for v in shape):
            raise ValueError(f"SEM source canvas shape must be 2D or 3D; got {self.shape!r}.")
        self.shape = shape
        for key, value in list(self.channels.items()):
            arr = np.asarray(value, dtype=float)
            if arr.shape != shape:
                raise ValueError(
                    f"SEM source channel {key!r} has shape {arr.shape}, expected {shape}."
                )
            self.channels[key] = arr

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def channel_for(self, material: MaterialProperties) -> np.ndarray:
        key = SEMMaterialChannelKey.from_material_properties(material)
        if key not in self.channels:
            self.channels[key] = np.zeros(self.shape, dtype=float)
        return self.channels[key]

    def normalize_exposure(self, num_subsamples: int) -> None:
        factor = float(num_subsamples)
        if factor <= 0.0:
            raise ValueError("num_subsamples must be positive when normalizing SEM source channels.")
        for key in tuple(self.channels):
            self.channels[key] = self.channels[key] / factor

    def crop(self, crop_start: int, crop_end: int) -> "SEMMaterialSourceCanvas":
        start = int(crop_start)
        end = int(crop_end)
        if self.ndim == 3:
            cropped = {
                key: value[:, start:end, start:end].copy()
                for key, value in self.channels.items()
            }
            shape = (self.shape[0], end - start, end - start)
        else:
            cropped = {
                key: value[start:end, start:end].copy()
                for key, value in self.channels.items()
            }
            shape = (end - start, end - start)
        return SEMMaterialSourceCanvas(shape=shape, channels=cropped)

    def scaled(self, gain: np.ndarray | float) -> "SEMMaterialSourceCanvas":
        gain_arr = np.asarray(gain, dtype=float)
        scaled = {
            key: np.asarray(value, dtype=float) * gain_arr
            for key, value in self.channels.items()
        }
        return SEMMaterialSourceCanvas(shape=self.shape, channels=scaled)

    def sum_array(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=float)
        for value in self.channels.values():
            out += np.asarray(value, dtype=float)
        return out

    def sum_projected(self) -> np.ndarray:
        summed = self.sum_array()
        if summed.ndim == 3:
            return np.sum(summed, axis=0)
        return summed

    def merged(self, others: Iterable["SEMMaterialSourceCanvas"]) -> "SEMMaterialSourceCanvas":
        merged = {key: value.copy() for key, value in self.channels.items()}
        for other in others:
            if other.shape != self.shape:
                raise ValueError(
                    f"Cannot merge SEM source canvases with shapes {self.shape} and {other.shape}."
                )
            for key, value in other.channels.items():
                if key not in merged:
                    merged[key] = np.zeros(self.shape, dtype=float)
                merged[key] += np.asarray(value, dtype=float)
        return SEMMaterialSourceCanvas(shape=self.shape, channels=merged)


def is_sem_material_source(value: object) -> bool:
    return isinstance(value, SEMMaterialSourceCanvas)


def source_like_normalize_exposure(source: object, num_subsamples: int) -> None:
    if isinstance(source, SEMMaterialSourceCanvas):
        source.normalize_exposure(num_subsamples)
    elif source is not None:
        source /= float(num_subsamples)


def source_like_crop(source: object, crop_start: int, crop_end: int) -> object:
    if source is None:
        return None
    if isinstance(source, SEMMaterialSourceCanvas):
        return source.crop(crop_start, crop_end)
    arr = np.asarray(source, dtype=float)
    if arr.ndim == 3:
        return arr[:, crop_start:crop_end, crop_start:crop_end]
    return arr[crop_start:crop_end, crop_start:crop_end]


def source_like_numeric_array(source: object) -> np.ndarray:
    if isinstance(source, SEMMaterialSourceCanvas):
        return source.sum_array()
    return np.asarray(source, dtype=float)


def source_like_projected_array(source: object) -> np.ndarray:
    if isinstance(source, SEMMaterialSourceCanvas):
        return source.sum_projected()
    arr = np.asarray(source, dtype=float)
    if arr.ndim == 3:
        return np.sum(arr, axis=0)
    return arr


def source_like_scaled(source: object, gain: np.ndarray | float) -> object:
    if isinstance(source, SEMMaterialSourceCanvas):
        return source.scaled(gain)
    return np.asarray(source, dtype=float) * np.asarray(gain, dtype=float)


def source_like_sum(sources: Iterable[object]) -> object:
    sources = [source for source in sources if source is not None]
    if not sources:
        raise ValueError("Cannot sum an empty SEM source list.")
    if all(isinstance(source, SEMMaterialSourceCanvas) for source in sources):
        first = sources[0]
        assert isinstance(first, SEMMaterialSourceCanvas)
        return first.merged(source for source in sources[1:] if isinstance(source, SEMMaterialSourceCanvas))
    if any(isinstance(source, SEMMaterialSourceCanvas) for source in sources):
        raise TypeError("Cannot mix material-resolved and numeric SEM source maps.")
    return np.sum([np.asarray(source, dtype=float) for source in sources], axis=0)


__all__ = [
    "SEMMaterialChannelKey",
    "SEMMaterialSourceCanvas",
    "is_sem_material_source",
    "source_like_crop",
    "source_like_normalize_exposure",
    "source_like_numeric_array",
    "source_like_projected_array",
    "source_like_scaled",
    "source_like_sum",
]
