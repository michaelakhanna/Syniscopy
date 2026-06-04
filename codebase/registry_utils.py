"""Small shared registry primitives for import-light lookup tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Mapping, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class NameRegistry:
    """Strict canonical-name registry with optional display labels."""

    supported: tuple[str, ...]
    display_names: Mapping[str, str] | None = None

    @staticmethod
    def normalize_key(value: object) -> str:
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    def canonical_name(self, value: object) -> str:
        key = self.normalize_key(value)
        return key

    def display_name(self, value: object) -> str:
        key = self.canonical_name(value)
        if self.display_names is not None and key in self.display_names:
            return self.display_names[key]
        return str(value).replace("_", " ")

    def contains(self, value: object) -> bool:
        return self.canonical_name(value) in self.supported


@dataclass(frozen=True)
class ObjectRegistry(Generic[T]):
    """Canonicalized lookup wrapper for registered implementation objects."""

    entries: Mapping[str, T]
    canonicalize: Callable[[object], str] = NameRegistry.normalize_key

    def key(self, value: object) -> str:
        return self.canonicalize(value)

    def get(self, value: object, *, item_label: str = "item") -> T:
        key = self.key(value)
        try:
            return self.entries[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {item_label} {value!r}. Supported values are: {list(self.entries.keys())}."
            ) from exc

    def contains(self, value: object) -> bool:
        return self.key(value) in self.entries


__all__ = ["NameRegistry", "ObjectRegistry"]
