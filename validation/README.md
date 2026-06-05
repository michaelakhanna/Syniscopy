# Syniscopy Validation

This directory is a tracked verification layer for the scientific model. It is
not imported by the runtime simulator. Runtime modules expose metadata about the
selected model and its validation status; these scripts independently exercise
the implementation against analytic, published, or external-reference anchors.

## Status Vocabulary

- `validated`: the checked output passed the stated tolerance for the exact
  validation run and fixture.
- `diagnostic_only`: the model is physics based or internally checked, but the
  selected paper/runtime row is not promoted to reference-validated evidence.
- `external_artifact_required`: validation depends on an external package,
  fixture, reference data set, or long run that is not bundled into the fast
  local check.
- `unchecked`: no validation evidence has been supplied for that output.

## Fast Checks

These are intended to be safe pre-notebook checks. The suite prints pass/fail
rows and can write a JSON manifest with the measured residuals:

```bash
python3 -m compileall -q codebase validation scripts
python3 validation/run_validation_suite.py --profile fast --json-output throwout/outputs/validation/fast.json
python3 validation/run_validation_suite.py --profile release --json-output throwout/outputs/validation/release.json
python3 validation/tem_multislice_validation.py
python3 validation/noise_validation.py
python3 validation/fisher_validation.py
python3 validation/iscat_contrast_validation.py
python3 validation/psf_airy_validation.py
python3 validation/trajectory_validation.py
python3 validation/thinfilm_electron_validation.py
python3 validation/mie_crosscheck.py
python3 validation/flagship_validation.py
python3 validation/modality_equation_validation.py
python3 validation/vectorial_debye_validation.py
```

The pass/fail labels are tolerance gates, not bitwise-output claims. Each check
has a reference equation, published value, or external package and records the
numeric residual it measured.

The `fast` profile is the default core-codebase gate. It includes runtime
imports, canonical material-source checks, analytic physics checks, modality
contrast-equation checks, TEM internal anchors, and a SEM physical smoke check.
The `release` profile adds optional external-package checks when installed but
does not require long stochastic SEM reference validation.

The generated abTEM fixture is a small external-reference TEM comparison. Build
or refresh it once with:

```bash
python3 validation/run_abtem_reference.py
python3 validation/tem_multislice_validation.py --require-abtem
python3 validation/run_validation_suite.py --profile release --require-external --fail-on-skip --json-output validation/release_validation.json
```

The SEM transport script is heavier and stochastic. A small-history smoke run is
useful for import/API checks, but it is not the paper-grade tolerance run:

```bash
python3 validation/sem_transport_validation.py --histories 2000 --max-steps 512 --skip-energy-dependence
```

Use a larger history count and the default energy-dependence check for a serious
SEM validation sweep. Passing or failing that long sweep should be reported as a
validation artifact, not hidden inside runtime defaults.

```bash
python3 validation/run_validation_suite.py --profile sem --json-output throwout/outputs/validation/sem_full.json
```

Use the `external` profile when you want all optional external checks plus the
long SEM sweep in one run:

```bash
python3 validation/run_validation_suite.py --profile external --json-output throwout/outputs/validation/external_full.json
```

## Reference-Validated Claims

Do not mark a backend `reference_validated` only because a fast internal check
passes. For TEM multislice, the abTEM fixture comparison must pass and the
recorded reference hash/status must match the selected runtime metadata before
setting `tem_reference_status` to `reference_validated`. For SEM physical
transport, the full backscatter/range tolerance sweep must pass and be recorded
before promoting the backend beyond `diagnostic_only`.

See `validation/reference_ledger.md` for the equation/reference source behind
each check. Checks that do not import an external reference package must point
to a literature equation or documented analytic limit there.
