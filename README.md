# Syniscopy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18617286.svg)](https://doi.org/10.5281/zenodo.18617286)

Syniscopy is a physics-based simulator for synthetic single-particle microscopy videos. It renders shared particle trajectories across multiple microscopy forward models and writes lossless frames, supervision masks, Cramér-Rao lower-bound diagnostics, and paired outputs for comparing the same particles across configured microscope candidates. A lab-facing Fisher report command turns a particle, microscope set, and detector-noise configuration into microscope rankings, Fisher matrices, Fisher-density maps, and fusion diagnostics.

## Citation

If you use Syniscopy, cite the archived software release:

```text
Khanna, Michael A. Syniscopy: Theoretical Foundations and Cross-Modality
Cramér-Rao Analysis for Synthetic Single-Particle Microscopy. Zenodo.
https://doi.org/10.5281/zenodo.18617286
```

The DOI `10.5281/zenodo.18617286` is the Zenodo all-versions DOI and resolves to
the latest archived version. Version-specific DOIs may be used when a workflow
needs to pin an exact release.

The live public repository is <https://github.com/michaelakhanna/syniscopy>.
Zenodo is the archival DOI record; GitHub hosts the codebase, starter notebooks,
release updates, and issue/usage workflow.

## Repository Layout

- `codebase/` contains the simulator, rendering, noise, supervision, and Cramér-Rao lower-bound modules.
- `recipes/` contains editable dataset recipes for local generation and Colab runs.
- `sam2_starter/` contains the public Segment Anything Model 2 training and inference starter notebooks.
- `paper/` contains the manuscript source and paper-facing figure/table artifacts.
- `supplemental/` contains numbered paper experiment notebooks and small configuration files.

Generated raw outputs and local build files should stay outside the public release tree.

## Local Setup

Syniscopy uses standard scientific Python packages. A minimal local environment
can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

The Segment Anything Model 2 starter notebooks install their own Google Colab dependencies at
runtime and clone Meta's external Segment Anything 2 repository separately.

## Lab Fisher Report

For a quick microscope-choice check, write an editable lab template and then run
the report command:

```bash
python codebase/lab_fisher_report.py --write-template lab_params.json
python codebase/lab_fisher_report.py --params-json lab_params.json --output lab_reports/my_setup
```

For a faster first check:

```bash
python codebase/lab_fisher_report.py --output lab_reports/smoke --modality-sweep bright_field,interferometric --image-size-pixels 96 --pupil-samples 96 --no-previews
```

Render a short trajectory and compute dynamic posteriors in one pass:

```bash
python codebase/lab_fisher_report.py --params-json lab_params.json --output lab_reports/sequence --num-frames 6 --dynamic-bayesian
```

List public names when editing a command:

```bash
python codebase/lab_fisher_report.py --list-modalities
python codebase/lab_fisher_report.py --list-instruments
```

The report writes:

- `report.md`: short ranked summary for the configured particle, pixel pitch, and detector-noise model.
- `microscope_ranking.csv`: per-microscope lateral Fisher matrices and Cramér-Rao bounds.
- `sequence_fisher_summary.csv`: per-frame and cumulative sequence CRLB rows.
- `fusion_crlb.csv`: best-k Fisher-fusion diagnostics; use `--include-full-fusion` to also write the full-library row when the microscope candidate list is longer than `--max-fusion-k`.
- `manifest.json`: requested, reported, and failed microscopes, with modality metadata.
- `params_base.json` and `params_resolved_by_microscope/`: run configuration records.
- `dynamic_microscope_summary.json`: optional dynamic Bayesian outputs when enabled.
- `previews/`: display-normalized first-frame contrast previews (from rendered frame sequences).
- `fisher_density/`: per-pixel lateral Fisher-density maps.

The default lab microscope set is the fixed-instrument modality sweep: it emits
one configured microscope candidate per optical, fluorescence, TEM, and SEM
contrast modality under the shared lab instrument settings.

See `docs/lab_fisher_workflow.md` for the full lab-facing workflow and
`examples/lab_fisher_params.json` for a small editable starting configuration.
For a full parameter-exposure audit (core/advanced/hidden + workflow/runtime
controls), see `docs/param_exposure_policy.md` and
`docs/param_exposure_matrix.json`.
For a read-only human summary, use `docs/param_exposure_summary.md`.

## Local Dataset Generation

List or export the editable public recipe. On macOS/Linux, `python3` is also
fine; on Windows PowerShell, `py -3` is usually the launcher.

```bash
python codebase/create_dataset.py --list-recipes
python codebase/create_dataset.py --write-params-template recipe_params.json
```

Generate a small local dataset:

```bash
python codebase/create_dataset.py --output datasets/syniscopy_dataset --num-videos 1 --seed 12345 --reset --verbose
```

Each generated video entry now has two public visual products by default:
`videos/video_XXXX.avi` is the background-subtracted contrast-analysis preview
used with the 8-bit training frame sequence, while
`videos/video_XXXX_raw_signal.avi` is a raw-camera signal preview with the
detector background left in place. Enable `save_raw_camera_frame_sequence` or
`save_raw_frame_views` when exact raw detector counts are needed.

## Segment Anything Model 2 Starter Package

Build the source ZIP expected by the Segment Anything Model 2 starter notebooks. The Python command
works on macOS, Linux, and Windows:

```bash
python sam2_starter/package_source_zip.py
```

The Bash wrapper `bash sam2_starter/package_source_zip.sh` is kept for
macOS/Linux users.

## Colab Supplemental Experiments

Build the source ZIP used by the supplemental notebooks. The Python command
works on macOS, Linux, and Windows:

```bash
python supplemental/package_experiments_for_colab.py
```

The Bash wrapper `bash supplemental/package_experiments_for_colab.sh` is kept
for macOS/Linux users.

This creates:

```text
supplemental/syniscopy_source.zip
```

The packaging script excludes local build outputs and macOS Finder artifacts
from `syniscopy_source.zip`; `.DS_Store` is also ignored by the repository.
Notebook output folders are created later by the notebooks when they write
generated data.

Upload the `supplemental/` folder itself to Google Drive as:

```text
MyDrive/supplemental
```

Then open and run the notebooks under `MyDrive/supplemental/`.
Generated notebook outputs are written inside the same uploaded folder at
`MyDrive/supplemental/outputs/`. The numbered supplemental notebooks write raw
outputs: individual images, arrays, CSV files, JSON manifests, and paper-facing
diagnostic tables. They do not assemble paper figures.
The numbered supplemental notebooks are CPU-oriented workflows and do not
require GPU acceleration.

## Paper Data

Run the numbered notebooks in `supplemental/` from the repository root or from
the uploaded Drive folder. They write raw outputs under matching
`supplemental/outputs/<notebook-id>/` folders. Paper-facing tables, figures, and
their provenance records live under `paper/figures/`.

```text
supplemental/E01.ipynb
supplemental/E02.ipynb
supplemental/E03.ipynb
supplemental/E04.ipynb
supplemental/E05.ipynb
```

The reusable Segment Anything Model 2 starter notebooks remain under
`sam2_starter/` as optional downstream workflow examples. They are not numbered
paper experiments and are not used as manuscript evidence.

## License And External Code

Syniscopy code and documentation are released under the MIT license; see
`LICENSE`. Third-party notices are summarized in `THIRD_PARTY_NOTICES.md`.
The Segment Anything Model 2 starter notebooks clone/download Meta's external
Segment Anything 2 code and checkpoints at runtime. Segment Anything Model 2 is
not bundled as Syniscopy code and remains under Meta's upstream licenses and
notices. Fine-tuned Segment Anything Model 2 checkpoints, if distributed, are
derived from upstream Segment Anything Model 2 weights and are not MIT-licensed
Syniscopy source code.
