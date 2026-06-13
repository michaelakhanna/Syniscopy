# Parameter Exposure Policy (Core / Advanced / Workflow / Hidden)

This repository intentionally keeps the simulation runtime contract in `config.PARAMS` and separates

- **Core**: required/primary bench-facing controls for setup-and-decision workflows.
- **Advanced**: optional, assumption-rich, modality-specific, or calibration-heavy controls.
- **Workflow**: output framing, sequencing, file naming, and masking/output plumbing controls.
- **Hidden**: private/runtime bookkeeping and derived state that users should not set directly.

The policy below is generated from

- `config.PARAMS`
- `config.KNOWN_INTERNAL_PARAM_KEYS`
- `param_schema.PARAM_SCHEMA` (`group` values, where values containing `"Advanced"` are treated as advanced and `"Workflow"` as workflow)

## Interpretation rules

1. Any key in `config.KNOWN_INTERNAL_PARAM_KEYS` is **Hidden**.
2. Any key represented in `param_schema.PARAM_SCHEMA`:
   - group is `"Workflow"` or explicit workflow override -> **Workflow**
   - group contains `"Advanced"` → **Advanced**
   - otherwise → **Core**
3. Remaining `PARAMS` keys are classified by intent:
   - workflow/output/manifest knobs → **Workflow**
   - measurable bench setup knobs not in schema → **Core**
   - everything else → **Advanced**

## Guidance

- Expose **Core + Advanced + Workflow** knobs in lab-facing tooling unless a local use-case explicitly hides workflow output controls.
- Keep **Hidden** keys internal-only.
- For UI surfaces, place **Advanced** and **Workflow** controls behind explicit disclosure where appropriate.
- For any new high-level input, add it to this policy list and keep it measurable in a real lab workflow.

## What this check shows right now

- **Hidden**: only internal/instrumentation-only keys in
  `config.KNOWN_INTERNAL_PARAM_KEYS` are kept internal.
- **Core/Advanced/Workflow**: non-hidden knobs are either in
  `param_schema.PARAM_SCHEMA` (core, workflow, or advanced by group) or
  exposed as parameter overrides.
- **Workflow/Output knobs**: a small subset (paths, frame counts, preview/save flags,
  mask generation/output options) are intentionally workflow-facing.

Current matrix status:

- Hidden: 14
- Core: 137
- Advanced: 159
- Workflow: 31
- Decision (core+advanced): 296

The above counts are mirrored in `docs/param_exposure_summary.md`.

That leaves 0 user-facing parameters classified as non-measurable in
the code-path sense: every non-hidden quantity is either

- directly set in schema controls, or
- intentionally settable via `PARAMS`-style JSON overrides.

## Machine-readable artifact

- `docs/param_exposure_matrix.json`
- `docs/param_exposure_summary.md` (human-readable companion summary).
