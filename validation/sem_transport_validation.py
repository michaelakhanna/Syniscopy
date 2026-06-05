#!/usr/bin/env python3
"""Validate physical SEM transport against published scalar anchors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from imaging_models.sem_backends.physical_transport import (  # noqa: E402
    kanaya_okayama_range_nm,
    normalize_sem_physical_elastic_model,
    reuter_backscatter_coefficient_20kev,
    simulate_sem_transport_observables,
)
from material_optical_catalog import sem_transport_material  # noqa: E402


VALIDATION_MATERIALS = ("carbon", "aluminum", "silicon", "copper", "silver", "gold")


def _relative_error(observed: float, expected: float) -> float:
    return abs(float(observed) - float(expected)) / max(abs(float(expected)), 1e-12)


def run_validation(
    *,
    histories: int,
    seed: int,
    max_steps: int,
    cutoff_keV: float,
    elastic_model: str,
    backscatter_tolerance: float,
    range_tolerance: float,
    energy_dependence_tolerance: float,
    check_energy_dependence: bool,
) -> bool:
    elastic_model = normalize_sem_physical_elastic_model(elastic_model)
    rows: list[tuple[str, bool, str]] = []
    backscatter_values: list[float] = []
    for material_name in VALIDATION_MATERIALS:
        material = sem_transport_material(material_name)
        observables, _, _, _, _ = simulate_sem_transport_observables(
            material=material,
            acceleration_keV=20.0,
            histories=histories,
            seed=seed,
            max_steps=max_steps,
            cutoff_keV=cutoff_keV,
            elastic_model=elastic_model,
        )
        eta_expected = reuter_backscatter_coefficient_20kev(material.atomic_number)
        eta_error = _relative_error(observables.backscatter_coefficient, eta_expected)
        range_expected = kanaya_okayama_range_nm(20.0, material)
        range_error = _relative_error(observables.max_penetration_depth_nm, range_expected)
        eta_ok = eta_error <= backscatter_tolerance
        range_ok = range_error <= range_tolerance
        active_ok = observables.active_histories_at_limit == 0
        ok = eta_ok and range_ok and active_ok
        backscatter_values.append(observables.backscatter_coefficient)
        rows.append(
            (
                material_name,
                ok,
                (
                    f"elastic_model={elastic_model} "
                    f"eta={observables.backscatter_coefficient:.4f} "
                    f"expected={eta_expected:.4f} rel_err={eta_error:.3f} "
                    f"range_nm={observables.max_penetration_depth_nm:.1f} "
                    f"KO_nm={range_expected:.1f} range_rel_err={range_error:.3f} "
                    f"p99_depth_nm={observables.p99_penetration_depth_nm:.1f} "
                    f"active_at_limit={observables.active_histories_at_limit}"
                ),
            )
        )

    monotonic_ok = all(
        later >= earlier for earlier, later in zip(backscatter_values, backscatter_values[1:])
    )
    energy_rows: list[tuple[str, bool, str]] = []
    if check_energy_dependence:
        for material_index, material_name in enumerate(VALIDATION_MATERIALS):
            material = sem_transport_material(material_name)
            eta_by_energy = []
            for energy_keV in (10.0, 20.0, 30.0):
                observables, _, _, _, _ = simulate_sem_transport_observables(
                    material=material,
                    acceleration_keV=energy_keV,
                    histories=histories,
                    seed=seed + 1000 * (material_index + 1) + int(energy_keV),
                    max_steps=max_steps,
                    cutoff_keV=cutoff_keV,
                    elastic_model=elastic_model,
                )
                eta_by_energy.append(observables.backscatter_coefficient)
            eta_mean = sum(eta_by_energy) / len(eta_by_energy)
            span = max(eta_by_energy) - min(eta_by_energy)
            rel_span = span / max(abs(eta_mean), 1e-12)
            ok = rel_span <= energy_dependence_tolerance
            energy_rows.append(
                (
                    material_name,
                    ok,
                    (
                        f"eta10={eta_by_energy[0]:.4f} "
                        f"eta20={eta_by_energy[1]:.4f} "
                        f"eta30={eta_by_energy[2]:.4f} "
                        f"rel_span={rel_span:.3f}"
                    ),
                )
            )
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    print(f"{'PASS' if monotonic_ok else 'FAIL'} monotonic_eta_vs_Z")
    for name, ok, detail in energy_rows:
        print(f"{'PASS' if ok else 'FAIL'} energy_dependence_{name}: {detail}")
    return (
        all(ok for _, ok, _ in rows)
        and monotonic_ok
        and all(ok for _, ok, _ in energy_rows)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--cutoff-kev", type=float, default=0.05)
    parser.add_argument("--elastic-model", default="screened_rutherford")
    parser.add_argument("--backscatter-tolerance", type=float, default=0.15)
    parser.add_argument("--range-tolerance", type=float, default=0.15)
    parser.add_argument("--energy-dependence-tolerance", type=float, default=0.15)
    parser.add_argument("--skip-energy-dependence", action="store_true")
    args = parser.parse_args()
    ok = run_validation(
        histories=args.histories,
        seed=args.seed,
        max_steps=args.max_steps,
        cutoff_keV=args.cutoff_kev,
        elastic_model=args.elastic_model,
        backscatter_tolerance=args.backscatter_tolerance,
        range_tolerance=args.range_tolerance,
        energy_dependence_tolerance=args.energy_dependence_tolerance,
        check_energy_dependence=not args.skip_energy_dependence,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
