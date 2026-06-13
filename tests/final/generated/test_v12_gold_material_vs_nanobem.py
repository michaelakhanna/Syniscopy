from __future__ import annotations

from pathlib import Path

import numpy as np


def _nanobem_gold_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repo_root = Path(__file__).resolve().parents[3]
    table_path = repo_root / "throwout" / "external" / "nanobem22" / "Material" / "gold.dat"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing nanobem gold table: {table_path}")

    wl_nm = []
    n_values = []
    k_values = []
    for raw_line in table_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            energy_ev = float(parts[0])
            n_val = float(parts[1])
            k_val = float(parts[2])
        except ValueError:
            continue
        wl_nm.append(1239.8419843320026 / energy_ev)
        n_values.append(n_val)
        k_values.append(k_val)

    if len(wl_nm) < 5:
        raise ValueError(f"Could not parse enough nanobem gold rows from {table_path}")

    order = np.argsort(wl_nm)
    return (
        np.asarray(wl_nm, dtype=float)[order],
        np.asarray(n_values, dtype=float)[order],
        np.asarray(k_values, dtype=float)[order],
    )


def test_gold_refractive_index_matches_nanobem_catalog_within_tolerance() -> None:
    from material_optical_catalog import lookup_refractive_index

    wavelengths_nm, n_ref, k_ref = _nanobem_gold_table()
    query_nm = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)

    n_query = np.interp(query_nm, wavelengths_nm, n_ref)
    k_query = np.interp(query_nm, wavelengths_nm, k_ref)

    max_abs_n = 0.0
    max_rel_k = 0.0
    for wl, ref_n, ref_k in zip(query_nm, n_query, k_query):
        syn = lookup_refractive_index("gold", float(wl))
        dn = abs(float(syn.real) - float(ref_n))
        dk = abs(float(syn.imag) - float(ref_k))
        dk_rel = dk / max(abs(ref_k), 1.0e-30)
        max_abs_n = max(max_abs_n, dn)
        max_rel_k = max(max_rel_k, dk_rel)

        assert np.isfinite(dn)
        assert np.isfinite(dk_rel)

    assert max_abs_n <= 0.20
    assert max_rel_k <= 0.12
