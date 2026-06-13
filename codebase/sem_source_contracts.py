"""SEM source-representation capability and resolver contracts.

This module is intentionally independent of ``imaging_models`` so runtime
validation, schema checks, SEM renderers, and SEM transport backends all use the
same requested/effective source-basis policy.  The central invariant is that an
explicit user request for a z-y-x SEM source volume must either reach a backend
that natively consumes z-y-x material-depth source stacks or fail before any
scientific frame is rendered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SEM_SOURCE_REPRESENTATION_AUTO = "auto"
SEM_SOURCE_REPRESENTATION_PROJECTED = "projected"
SEM_SOURCE_REPRESENTATION_VOLUME = "volume"
SEM_SOURCE_REPRESENTATION_CHOICES = (
    SEM_SOURCE_REPRESENTATION_AUTO,
    SEM_SOURCE_REPRESENTATION_PROJECTED,
    SEM_SOURCE_REPRESENTATION_VOLUME,
)

SEM_BACKEND_CAPABILITY_PROJECTED_ONLY = "projected_source_only"
SEM_BACKEND_CAPABILITY_VOLUME_NATIVE = "native_volume_source"

SEM_BACKEND_SOURCE_CAPABILITIES: dict[str, str] = {
    "gaussian_probe_proxy": SEM_BACKEND_CAPABILITY_PROJECTED_ONLY,
    "interaction_volume_proxy": SEM_BACKEND_CAPABILITY_PROJECTED_ONLY,
    "syniscopy_transport_lite": SEM_BACKEND_CAPABILITY_PROJECTED_ONLY,
    "reference_kernel_table": SEM_BACKEND_CAPABILITY_PROJECTED_ONLY,
    "monte_carlo_transport": SEM_BACKEND_CAPABILITY_VOLUME_NATIVE,
    "monte_carlo_physical": SEM_BACKEND_CAPABILITY_VOLUME_NATIVE,
}
SEM_VOLUME_BACKENDS = frozenset(
    backend
    for backend, capability in SEM_BACKEND_SOURCE_CAPABILITIES.items()
    if capability == SEM_BACKEND_CAPABILITY_VOLUME_NATIVE
)


@dataclass(frozen=True)
class SEMSourceRepresentationResolution:
    """Resolved SEM source-basis contract crossing config, renderer, and backend.

    ``requested`` records the user/API value. ``effective`` is the numeric source
    basis actually allocated and consumed.  A non-volume backend may resolve
    ``auto`` to projected, but an explicit ``volume`` request is never silently
    projected because that would erase material-depth transport semantics.
    """

    requested: str
    effective: str
    backend_name: str
    backend_source_capability: str
    source_projection_policy: str
    requested_is_explicit: bool

    @property
    def backend_consumes_volume_source(self) -> bool:
        return self.effective == SEM_SOURCE_REPRESENTATION_VOLUME

    @property
    def request_satisfied(self) -> bool:
        return (not self.requested_is_explicit) or self.requested == self.effective

    @property
    def source_map_ndim(self) -> int:
        return 3 if self.backend_consumes_volume_source else 2

    @property
    def source_axis_order(self) -> str:
        return "zyx" if self.backend_consumes_volume_source else "yx"

    def metadata(self) -> dict[str, Any]:
        return {
            "sem_requested_source_representation": self.requested,
            "sem_effective_source_representation": self.effective,
            "sem_source_backend_capability": self.backend_source_capability,
            "sem_source_representation_resolution_mode": (
                "explicit" if self.requested_is_explicit else "auto"
            ),
            "source_representation_request_satisfied": self.request_satisfied,
            "source_projection_policy": self.source_projection_policy,
            "backend_consumes_volume_source": self.backend_consumes_volume_source,
            "source_map_ndim": self.source_map_ndim,
            "source_axis_order": self.source_axis_order,
        }


def normalize_sem_source_representation(value: Any) -> str:
    requested = str(value).strip().lower()
    if requested not in SEM_SOURCE_REPRESENTATION_CHOICES:
        raise ValueError(
            "parameters['sem_source_representation'] must be one of "
            f"{list(SEM_SOURCE_REPRESENTATION_CHOICES)}; got {value!r}."
        )
    return requested


def sem_backend_source_capability(backend_name: Any) -> str:
    backend = str(backend_name).strip().lower()
    try:
        return SEM_BACKEND_SOURCE_CAPABILITIES[backend]
    except KeyError as exc:
        supported = ", ".join(sorted(SEM_BACKEND_SOURCE_CAPABILITIES))
        raise ValueError(
            f"Unknown SEM backend {backend_name!r} for source-representation resolution. "
            f"Supported backends are: {supported}."
        ) from exc


def resolve_sem_source_representation(
    requested_value: Any,
    *,
    backend_name: Any,
) -> SEMSourceRepresentationResolution:
    """Resolve SEM requested source representation against backend capability.

    ``auto`` is a request for the backend's native source representation:
    projected-only backends stay projected, while native-volume Monte Carlo
    backends use z-y-x source volumes. Explicit ``volume`` is strict because
    silently projecting it would collapse different material-depth distributions
    into the same 2D map.
    """

    requested = normalize_sem_source_representation(requested_value)
    backend = str(backend_name).strip().lower()
    capability = sem_backend_source_capability(backend)
    volume_capable = capability == SEM_BACKEND_CAPABILITY_VOLUME_NATIVE

    if requested == SEM_SOURCE_REPRESENTATION_AUTO:
        effective = (
            SEM_SOURCE_REPRESENTATION_VOLUME
            if volume_capable
            else SEM_SOURCE_REPRESENTATION_PROJECTED
        )
        policy = (
            "auto_native_volume_transport"
            if volume_capable
            else "auto_projected_for_backend_capability"
        )
        return SEMSourceRepresentationResolution(
            requested=requested,
            effective=effective,
            backend_name=backend,
            backend_source_capability=capability,
            source_projection_policy=policy,
            requested_is_explicit=False,
        )

    if requested == SEM_SOURCE_REPRESENTATION_PROJECTED:
        return SEMSourceRepresentationResolution(
            requested=requested,
            effective=SEM_SOURCE_REPRESENTATION_PROJECTED,
            backend_name=backend,
            backend_source_capability=capability,
            source_projection_policy="user_selected_projected_source",
            requested_is_explicit=True,
        )

    if not volume_capable:
        raise ValueError(
            "parameters['sem_source_representation']='volume' requires a SEM backend "
            "with native z-y-x source-volume support. "
            f"parameters['sem_backend']={backend!r} has capability {capability!r}; "
            "use sem_source_representation='auto' or 'projected' for this backend, "
            "or select 'monte_carlo_transport'/'monte_carlo_physical'."
        )

    return SEMSourceRepresentationResolution(
        requested=SEM_SOURCE_REPRESENTATION_VOLUME,
        effective=SEM_SOURCE_REPRESENTATION_VOLUME,
        backend_name=backend,
        backend_source_capability=capability,
        source_projection_policy="backend_native_volume_transport",
        requested_is_explicit=True,
    )


__all__ = [
    "SEM_BACKEND_CAPABILITY_PROJECTED_ONLY",
    "SEM_BACKEND_CAPABILITY_VOLUME_NATIVE",
    "SEM_BACKEND_SOURCE_CAPABILITIES",
    "SEM_SOURCE_REPRESENTATION_AUTO",
    "SEM_SOURCE_REPRESENTATION_CHOICES",
    "SEM_SOURCE_REPRESENTATION_PROJECTED",
    "SEM_SOURCE_REPRESENTATION_VOLUME",
    "SEM_VOLUME_BACKENDS",
    "SEMSourceRepresentationResolution",
    "normalize_sem_source_representation",
    "resolve_sem_source_representation",
    "sem_backend_source_capability",
]
