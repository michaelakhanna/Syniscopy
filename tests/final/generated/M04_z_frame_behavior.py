"""M04 [behavioral z-frame guard] Verify the source-z coordinate contract at RUNTIME.

The existing test_sem_source_z_frame_contract.py / test_tem_source_coordinate_contract.py
are STATIC string-greps: they prove the guard strings exist in the source, not that the
rendered physics actually uses the right z-frame. This file renders real frames and
asserts the physical consequence, so a regression that silently routes the wrong z-frame
(focus-relative defocus z where physical/material depth is required) is caught by behavior.

LAWS ASSERTED:
  TIRF-EVANESCENT : raising a particle's PHYSICAL height z by one penetration depth must
                    cut its total emitted signal by exp(-1). Proves TIRF excitation reads
                    physical interface height, not focus-relative defocus z. The total
                    (integrated) signal isolates the evanescent scalar from defocus, which
                    only redistributes an energy-conserving PSF.
  TEM-PROJECTED-Z : in projected (non-multislice) TEM the output must be IDENTICAL when the
                    particle's world z changes, because projected TEM declares projected_no_z
                    (z is not representable). A nonzero difference means world z is leaking
                    into a 2D projection.
  SOURCE-Z-BASIS  : each model's source_coordinate_contract(), called at runtime (not grepped),
                    must report a physical/material/projected source-density basis -- never
                    focus_relative -- for SEM volume and TEM source maps.

Run (you run it; ~1-2 min): python M04_z_frame_behavior.py
Writes _runs/M04/M04_results.json
"""
from __future__ import annotations
import os, json, copy
import numpy as np
from common import add_paths, banner, verdict
add_paths()

from calibration_profiles import native_params, CALIBRATION_PROFILES
from simulation import generate_single_frame_views
from modality_registry import canonical_modality_name

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_runs", "M04"); os.makedirs(OUT, exist_ok=True)

rows: list[dict] = []


def add(name, ok, detail):
    rows.append({"law": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'OK' if ok else 'FAIL'}] {name:16s} {detail}")


def _case(modality, **kw):
    canon = canonical_modality_name(modality)
    base = dict(CALIBRATION_PROFILES.get(canon, {"modality": canon}))
    base["modality"] = canon
    base.setdefault("diameter_nm", 60.0)
    base.update({"image_size_pixels": 48, "pixel_size_nm": 20.0,
                 "pupil_samples": 24, "vectorial_pupil_samples": 24,
                 "psf_oversampling_factor": 2})
    base.update(kw)
    return base


def _render_with_particle_z(case, z_nm):
    """Render one frame with the single particle's WORLD z set to z_nm."""
    params = native_params(case)
    pos = params["particles"][0]["motion"]["initial_position_nm"]
    pos[0] = float(pos[0]); pos[1] = float(pos[1]); pos[2] = float(z_nm)
    v = generate_single_frame_views(params)
    return np.asarray(v["contrast_frame"], dtype=float), params


banner("M04  runtime source-z coordinate-frame behavior")

# --- TIRF-EVANESCENT: physical height drives exp(-z/d) decay -----------------
try:
    case = _case("tirf_fluorescence")
    params0 = native_params(case)
    depth = float(params0.get("tirf_penetration_depth_nm", 120.0))
    # Compare two heights BOTH fully above the interface (z=d and z=2d). Using
    # z=0 as the baseline is wrong: a finite particle there straddles the
    # interface (its lower half sits below z=0 where there is no evanescent
    # field), so the ratio is not a clean exp(-1). With both heights above the
    # surface and a particle small vs d, the total-signal ratio is exp(-(2d-d)/d)
    # = exp(-1). The total (integral) also isolates the evanescent scalar from defocus.
    C1, _ = _render_with_particle_z(case, depth)
    C2, _ = _render_with_particle_z(case, 2.0 * depth)
    s1 = float(np.sum(np.abs(C1 - np.median(C1))))
    s2 = float(np.sum(np.abs(C2 - np.median(C2))))
    ratio = s2 / s1 if s1 > 0 else float("inf")
    expected = float(np.exp(-1.0))                         # 0.368
    ok = abs(ratio - expected) / expected < 0.20
    add("TIRF-EVANESCENT", ok,
        f"total signal ratio(z=2d)/(z=d) = {ratio:.3f}, expect exp(-1)={expected:.3f} "
        f"(physical-height evanescent decay; d={depth:.0f} nm)")
except Exception as e:
    add("TIRF-EVANESCENT", False, f"err {type(e).__name__}: {e}")

# --- TEM-PROJECTED-Z: projected TEM output invariant to world z --------------
try:
    # force projected (non-multislice) source if the knob exists; otherwise the
    # default profile is projected and the assertion still applies.
    case = _case("tem_phase_contrast")
    case.setdefault("tem_source_representation", "projected")
    C0, p0 = _render_with_particle_z(case, 0.0)
    C5, _ = _render_with_particle_z(case, 500.0)
    peak = float(np.max(np.abs(C0 - np.median(C0)))) + 1e-30
    drift = float(np.max(np.abs(C0 - C5)) / peak)
    multislice = bool(str(p0.get("tem_source_representation", "projected")).lower().startswith("volume")
                      or p0.get("tem_multislice_slices", 1) and int(p0.get("tem_multislice_slices", 1)) > 1)
    if multislice:
        add("TEM-PROJECTED-Z", True,
            f"SKIP: profile resolved to slice-resolved TEM (uses physical world z by design); drift={drift:.2e}")
    else:
        add("TEM-PROJECTED-Z", drift < 1e-9,
            f"max|C(z=0)-C(z=500nm)|/peak = {drift:.2e} (projected TEM must be z-invariant)")
except Exception as e:
    add("TEM-PROJECTED-Z", False, f"err {type(e).__name__}: {e}")

# --- SOURCE-Z-BASIS: runtime contract call, never focus_relative -------------
from imaging_models import get_imaging_model_class
from imaging_models.base import SOURCE_Z_BASIS_FOCUS_RELATIVE

for modality in ("sem_secondary_electron", "tem_phase_contrast"):
    try:
        canon = canonical_modality_name(modality)
        params = native_params(_case(modality))
        cls = get_imaging_model_class(canon)
        try:
            model = cls(params)
        except TypeError:
            model = cls.__new__(cls)  # contract methods that don't need full init
        basis = None
        contract = {}
        if hasattr(model, "source_coordinate_contract"):
            contract = model.source_coordinate_contract(params)
            basis = contract.get("source_density_z_basis") or contract.get("source_z_planes_basis")
        elif hasattr(model, "particle_source_z_basis"):
            basis = model.particle_source_z_basis(params)
        ok = basis is not None and basis != SOURCE_Z_BASIS_FOCUS_RELATIVE
        add(f"SOURCE-Z-BASIS[{canon}]", ok,
            f"runtime source_density_z_basis={basis!r} (must be physical/material/projected, "
            f"never {SOURCE_Z_BASIS_FOCUS_RELATIVE!r})")
    except Exception as e:
        add(f"SOURCE-Z-BASIS[{modality}]", False, f"err {type(e).__name__}: {e}")

all_ok = all(r["ok"] for r in rows)
json.dump(rows, open(os.path.join(OUT, "M04_results.json"), "w"), indent=2)
print(f"\nWROTE {os.path.join(OUT, 'M04_results.json')}")
print("Behavioral proof that physical/material z (not focus-relative defocus z) drives")
print("TIRF excitation and SEM/TEM source maps. A FAIL means a real z-frame leak.")
raise SystemExit(verdict(all_ok, "(physical z-frame drives TIRF/SEM/TEM source behavior)"))
