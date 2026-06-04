"""Explicit SEM reference-kernel example payloads."""

from __future__ import annotations

from typing import Any

from ._metadata import SEM_REFERENCE_KERNEL_SCHEMA_VERSION


def example_sem_reference_kernel_payload() -> dict[str, Any]:
    return {
        "schema_version": SEM_REFERENCE_KERNEL_SCHEMA_VERSION,
        "validation_status": "physics_based_unvalidated",
        "kernel_rows": [
            {
                "source": 0.0,
                "yield": 0.0,
                "beam_energy_kV": 5.0,
                "source_depth_nm": 0.0,
                "takeoff_angle_deg": 45.0,
                "incident_angle_deg": 0.0,
                "geometry": "normal",
                "material": "default",
                "backscatter_yield": 0.0,
            },
            {
                "source": 1.0,
                "yield": 0.7,
                "beam_energy_kV": 5.0,
                "source_depth_nm": 0.0,
                "takeoff_angle_deg": 45.0,
                "incident_angle_deg": 0.0,
                "geometry": "normal",
                "material": "default",
                "backscatter_yield": 0.05,
            },
        ],
    }


__all__ = ["example_sem_reference_kernel_payload"]
