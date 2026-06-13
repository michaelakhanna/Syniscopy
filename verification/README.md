# Syniscopy Verification

Fresh verification harness for Syniscopy's core novelty claims:

- common-frame Fisher/CRLB math
- SE(3) rank deficits and fusion complementarity
- strict detected-quanta normalization
- supervision mask/drop-reason behavior
- emitted lab-report, mask, and matched-packet artifact consistency
- empirical Monte Carlo CRLB checks
- optional renderer stress tests for SEM/TEM voltage limits and frame coherence

This folder is intentionally separate from the older deleted `validation/`
suite. It does not import or depend on those files.

## Install

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r verification/requirements-verification.txt
```

## Run

Fast core checks:

```bash
python3 verification/run_verification_suite.py --profile quick
```

Full math checks, including Monte Carlo:

```bash
python3 verification/run_verification_suite.py --profile full --monte-carlo-samples 5000
```

Run only the nanorod construction and 360-degree FIM trace sweep:

```bash
python3 -m pytest verification/tests/test_nanorod_shape_rotation_invariance.py -q
```

Run only the visual/cross-output consistency tests from the latest checklist:

```bash
python3 -m pytest verification/tests/test_visual_output_consistency.py -q
```

Audit generated outputs from a lab Fisher report or dataset:

```bash
python3 verification/run_verification_suite.py \
  --profile artifacts \
  --lab-report lab_reports/my_setup \
  --mask-root outputs/syniscopy_masks \
  --packet-root datasets/syniscopy_dataset/matched_microscope_packets
```

Run adversarial live-renderer checks too:

```bash
python3 verification/run_verification_suite.py --profile adversarial --include-renderer
```

The renderer profile is expected to expose physics-proxy failures if SEM/TEM
voltage, dose, or source terms are treated as harmless display sliders.

## Direct Pytest

The runner is only a convenience wrapper. Direct pytest also works:

```bash
python3 -m pytest verification/tests -m "quick or full"
SYNISCOPY_VERIFY_LAB_REPORT=lab_reports/my_setup python3 -m pytest verification/tests -m artifacts
SYNISCOPY_VERIFY_RUN_RENDERER=1 python3 -m pytest verification/tests -m renderer
```
