# Lab Fisher Workflow

This workflow is for a lab that wants to ask a practical question:

> For this particle, pixel pitch, objective, wavelength, and detector-noise
> model, which configured modality profiles carry the most localization
> information?

It uses the same shared-scene Fisher/CRLB layer as the manuscript, but it is not
a paper-reproduction workflow.

## 1. Install

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

## 2. Start From A Parameter File

Use the bundled example:

```bash
python codebase/lab_fisher_report.py \
  --params-json examples/lab_fisher_params.json \
  --output lab_reports/example_setup \
  --no-previews
```

Or write a fresh editable template:

```bash
python codebase/lab_fisher_report.py --write-template lab_params.json
python codebase/lab_fisher_report.py --params-json lab_params.json --output lab_reports/my_setup
```

The most important fields to edit first are:

- `particles[0].components[0].diameter_nm`
- `particles[0].components[0].material`
- `pixel_size_nm`
- `wavelength_nm`
- `numerical_aperture`
- `refractive_index_medium`
- `refractive_index_immersion`
- `background_intensity`
- `read_noise_counts`
- `camera_gain_e_per_count`

For fluorescence checks, use a material with fluorescence metadata, such as
`fluorescent_polystyrene`, or provide `material_properties` with
`fluorophore_density`, `excitation_peak_nm`, and `emission_peak_nm`.

## 3. Pick Modalities

The default lab set focuses on optical and fluorescence paths:

```bash
python codebase/lab_fisher_report.py --params-json lab_params.json --output lab_reports/my_setup
```

List public names when editing commands:

```bash
python codebase/lab_fisher_report.py --list-modalities
python codebase/lab_fisher_report.py --list-instruments
```

For a smaller run:

```bash
python codebase/lab_fisher_report.py \
  --params-json lab_params.json \
  --modalities bright_field,interferometric,fluorescence_widefield \
  --output lab_reports/three_modalities
```

Include simplified electron-microscopy paths only when that comparison is
intentional:

```bash
python codebase/lab_fisher_report.py \
  --params-json lab_params.json \
  --include-electron \
  --output lab_reports/with_simplified_em
```

## 4. Inspect Outputs

Open `lab_reports/my_setup/report.md` first.

The report directory contains:

- `modality_ranking.csv`: lateral Fisher matrices and Cramér-Rao bounds.
- `fusion_crlb.csv`: best-k Fisher-fusion diagnostics. Add
  `--include-full-fusion` when a full-library algebraic fusion row is useful.
- `manifest.json`: requested, reported, and failed modalities.
- `params_base.json`: base configuration before per-modality imaging-model overrides.
- `params_resolved_by_modality/`: resolved configuration for each reported modality.
- `previews/`: display-normalized single-frame contrast previews.
- `fisher_density/`: per-pixel lateral Fisher-density maps and arrays.

The numbers are conditional on the configured forward model and noise model.
They are most useful for comparing candidate profiles under the same stated
assumptions, not for claiming native instrument performance without calibration.
Fusion rows additionally assume independent measurements of the same particle
state and zero cross-channel registration covariance.

## 5. What To Calibrate First

If the report will be used for a real experimental decision, start by matching:

- effective sample-plane pixel size;
- objective numerical aperture and immersion refractive index;
- wavelength or fluorescence excitation/emission settings;
- particle diameter and material/refractive index;
- background/reference count scale;
- camera gain and read-noise scale;
- any substrate or patterned-background assumptions.

If those quantities are unknown, run the report as a sensitivity check by
varying one parameter at a time and watching whether the modality ranking
changes.
