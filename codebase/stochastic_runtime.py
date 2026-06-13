"""Central stochastic stream ownership for Syniscopy.

Reproducibility is a run-level concept, not a per-module convenience.  This
module is the only source file that should construct NumPy SeedSequence or
Generator objects directly.  Callers request named streams from a run seed,
cache key, or explicit entropy token.
"""

from __future__ import annotations
from configured_parameters import configured_optional

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np


_UINT32_MODULUS = 2**32


def _uint32(value: int) -> int:
    return int(value) % _UINT32_MODULUS


def _digest_words(parts: Iterable[Any], *, word_count: int = 4) -> list[int]:
    payload = repr(tuple(parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return [
        int.from_bytes(digest[i : i + 4], "big")
        for i in range(0, 4 * int(word_count), 4)
    ]


def unseeded_entropy_uint32(*, stream: str = "unseeded_entropy") -> int:
    """Return unseeded process entropy through the central stochastic owner."""

    entropy = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return _uint32(entropy ^ _digest_words((stream,), word_count=1)[0])


def unseeded_token(*, prefix: str, stream: str) -> str:
    """Return a per-run token for unseeded runtime caches."""

    return f"{prefix}:{unseeded_entropy_uint32(stream=stream)}"


def derive_seed(
    seed: int | None,
    *,
    stream: str,
    index: int = 0,
    entropy: int | None = None,
    bits: int = 31,
) -> int:
    """Derive a stable integer seed for a named stochastic stream."""

    if bits <= 0:
        raise ValueError(f"bits must be positive; got {bits!r}.")
    if seed is None:
        entropy_value = (
            unseeded_entropy_uint32(stream=f"{stream}:entropy")
            if entropy is None
            else _uint32(int(entropy))
        )
        parts = ("unseeded", entropy_value, str(stream), int(index))
    else:
        parts = ("seeded", int(seed), str(stream), int(index))
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**bits)


def rng_from_seed(
    seed: int | None,
    *,
    stream: str,
    index: int = 0,
    entropy: int | None = None,
) -> np.random.Generator:
    """Create a NumPy Generator for a named stream from an optional run seed."""

    if seed is None:
        seed_sequence = np.random.SeedSequence(
            [_uint32(entropy)] + _digest_words((stream, index), word_count=3)
            if entropy is not None
            else None
        )
    else:
        seed_sequence = np.random.SeedSequence(
            [_uint32(int(seed))] + _digest_words((stream, index), word_count=3)
        )
    return np.random.default_rng(seed_sequence)


def rng_from_seed_words(
    words: Iterable[int],
    *,
    stream: str,
    index: int = 0,
) -> np.random.Generator:
    """Create a stream from caller-owned deterministic seed words."""

    seed_words = [_uint32(int(word)) for word in words]
    seed_words.extend(_digest_words((stream, index), word_count=3))
    return np.random.default_rng(np.random.SeedSequence(seed_words))


def rng_from_cache_key(
    cache_key: Any,
    *,
    stream: str,
    index: int = 0,
) -> np.random.Generator:
    """Create a deterministic stream from a stable cache key."""

    return rng_from_seed_words(
        _digest_words((cache_key,), word_count=4),
        stream=stream,
        index=index,
    )


def spawn_child_rng(
    parent_rng: np.random.Generator,
    *,
    stream: str,
    index: int = 0,
) -> np.random.Generator:
    """Derive a child stream from an existing generator without exposing SeedSequence."""

    entropy = int(parent_rng.integers(0, _UINT32_MODULUS, dtype=np.uint32))
    return rng_from_seed(None, stream=stream, index=index, entropy=entropy)


@dataclass(frozen=True)
class StochasticRunContext:
    """Named stochastic-stream owner for one simulation/dataset run."""

    seed: int | None
    entropy: int | None = None
    label: str = "run"

    @classmethod
    def from_seed(cls, seed: int | None, *, label: str = "run") -> "StochasticRunContext":
        entropy = None if seed is not None else unseeded_entropy_uint32(stream=f"{label}:entropy")
        return cls(seed=None if seed is None else int(seed), entropy=entropy, label=str(label))

    @classmethod
    def from_params(cls, params: dict[str, Any], *, label: str = "run") -> "StochasticRunContext":
        seed = configured_optional(params, "random_seed")
        return cls.from_seed(None if seed is None else int(seed), label=label)

    def rng(self, stream: str, *, index: int = 0) -> np.random.Generator:
        return rng_from_seed(
            self.seed,
            stream=f"{self.label}:{stream}",
            index=index,
            entropy=self.entropy,
        )

    def seed_for(self, stream: str, *, index: int = 0, bits: int = 31) -> int:
        return derive_seed(
            self.seed,
            stream=f"{self.label}:{stream}",
            index=index,
            entropy=self.entropy,
            bits=bits,
        )

    def token(self, stream: str, *, prefix: str = "run") -> str:
        seed_value = self.seed_for(stream, bits=32)
        return f"{prefix}:{seed_value}"


__all__ = [
    "StochasticRunContext",
    "derive_seed",
    "rng_from_cache_key",
    "rng_from_seed",
    "rng_from_seed_words",
    "spawn_child_rng",
    "unseeded_entropy_uint32",
    "unseeded_token",
]
