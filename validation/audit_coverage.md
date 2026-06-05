# Audit Coverage Map

This file maps the validation audit items in
`throwout/reports/syniscopy_validation_audit.md` to tracked validation commands.
The validation result is tolerance-based: reference value, observed value,
residual, tolerance, pass/fail.

## One-Command Gates

```bash
python3 validation/run_validation_suite.py --profile fast --json-output throwout/outputs/validation/fast.json
python3 validation/run_validation_suite.py --profile release --json-output throwout/outputs/validation/release.json
```

`fast` is the core-codebase gate. `release` adds lightweight external checks
when installed. Long stochastic SEM reference validation is intentionally not
part of `release`; run it with `--profile sem` or `--profile external`.

## Audit Items

| Audit ID | Surface | Validator | Current status |
|---|---|---|---|
| C1 | Mie angular amplitude | `run_validation_suite.py --profile release` / `validation/mie_crosscheck.py` | External check passes when `miepython` is installed. |
| C2 | Thin-film / Fresnel reflection | `run_validation_suite.py --profile release` / `validation/thinfilm_electron_validation.py` | Fresnel and reflectance-vs-`tmm` pass; complex phase convention is recorded separately. |
| C3 | Brownian trajectories | `run_validation_suite.py --profile fast` / `validation/trajectory_validation.py` | Direct one-step Brownian statistics and determinism pass. |
| C4 | Lateral Fisher / CRLB | `run_validation_suite.py --profile fast` / `validation/fisher_validation.py` | Gaussian closed-form CRLB/scaling checks pass. |
| C5 | Camera noise | `run_validation_suite.py --profile fast` / `validation/noise_validation.py` | Poisson/gain checks pass. |
| C6 | Electron constants | `run_validation_suite.py --profile fast` / `validation/thinfilm_electron_validation.py` | Published wavelength/Scherzer checks pass. |
| C7 | iSCAT contrast definition | `run_validation_suite.py --profile fast` / `validation/iscat_contrast_validation.py` | Contrast identity passes. |
| C8 | Scalar PSF Airy limit | `run_validation_suite.py --profile fast` / `validation/psf_airy_validation.py` | Low-NA Airy first-ring check passes. |
| C9 | Rotational diffusion/orientation | `run_validation_suite.py --profile fast` | Stokes-Einstein-Debye step scale, SO(3) orthogonality, determinant, and seeded determinism pass. |
| C10 | Metadata / manifest / packet fidelity | `run_validation_suite.py --profile fast` | Video/dataset manifest JSON and counterfactual packet save/load/validate round trips pass. |
| C11 | Vectorial Debye PSF | `validation/vectorial_debye_validation.py` via `fast` | Richards-Wolf rotation covariance, on-axis polarization, and low-NA Airy reduction pass. A high-NA external reference fixture remains optional. |
| C12 | Dark-field contrast | `validation/modality_equation_validation.py` via `fast` | Passes. |
| C13 | QPI phase | `validation/modality_equation_validation.py` via `fast` | Passes. |
| C14 | RICM interference | `validation/modality_equation_validation.py` via `fast` | Passes. |
| C15 | Off-axis holography | `validation/modality_equation_validation.py` via `fast` | Passes with detector/canvas-period convention explicit. |
| C16 | Fluorescence widefield | `validation/modality_equation_validation.py` via `fast` | Passes with current `fluorescence_emission_psf_sigma_px` key. |
| C17 | TIRF penetration depth | `validation/modality_equation_validation.py` via `fast` | Passes. |
| C18 | Zernike phase contrast | `validation/modality_equation_validation.py` via `fast` | Passes as documented visibility/phase-ring limit. |
| C19 | TEM CTF / TEM internal / abTEM fixture | `validation/modality_equation_validation.py`, `validation/tem_multislice_validation.py`, and `run_validation_suite.py --profile release --require-external` | CTF convention, internal multislice anchors, and the generated abTEM fixture comparison pass. |
| C20 | Kohler coherent limit / SEM property | `validation/modality_equation_validation.py` and SEM smoke via `fast` | Kohler bandlimited coherent limit passes; SEM smoke passes but full reference validation remains `diagnostic_only`. |

## Material 108-Sweep Finding

The old audit's "108 material x wavelength/diameter" note referred to a
now-stale comparison between top-level `materials.py` and
`material_optical_catalog.py`. In the current codebase, top-level
`codebase/materials.py` is gone and no core file imports it. The tracked
`material_single_source` check verifies that material optical/electron data now
resolve through `material_optical_catalog.py` only and that catalog lookups are
finite for the current material set.
