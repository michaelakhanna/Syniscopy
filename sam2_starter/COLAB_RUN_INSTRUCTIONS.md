# Segment Anything Model 2 Starter Colab Instructions

These instructions are for users who want to fine-tune Segment Anything Model 2
(SAM2) on Syniscopy data or run inference with a fine-tuned checkpoint. This
starter is intentionally general: it is not hard-wired to any paper figure,
local sweep, or sample video.

## Files to Upload

From the repository root, rebuild the source zip and place it into the local
starter folder:

```bash
python sam2_starter/package_source_zip.py
```

On Windows PowerShell, use `py -3 sam2_starter/package_source_zip.py`. The Bash
wrapper remains available on macOS/Linux as
`bash sam2_starter/package_source_zip.sh`.

Upload the whole `sam2_starter/` folder to Google Drive. It should contain:

```text
syniscopy_codebase.zip
sam2training.ipynb
sam2inference.ipynb
```

The training notebook looks for the source zip at:

```text
MyDrive/sam2_starter/syniscopy_codebase.zip
```

Generated datasets, checkpoints, and inference outputs are written separately
under `MyDrive/Syniscopy/<MICROSCOPE_LABEL>/`.

## Training

Open `sam2training.ipynb` in Google Colab.

The notebook has three user-facing places:

1. **Setup label cell**: choose a temporary/default `MICROSCOPE_LABEL`.
2. **Optional single-frame preview cell**: edit the concrete microscope,
   particle, noise, and mask fields, run one frame, inspect raw/reference/
   contrast/final views, and copy the printed block if you like it.
3. **Final training condition cell**: paste the settings that will actually
   generate the full dataset and train Segment Anything Model 2.

For normal users, the final condition cell is the only required training
configuration:

```python
MICROSCOPE_LABEL = "default_configuration"
SUPERVISION_TARGET = "mask_supported"
DATASET_PRESET_NAME = "default"
DATASET_INSTRUMENT_PRESET = None
DATASET_NUM_VIDEOS = 20
DATASET_RANDOM_SEED = 12345
RESET_RAW_DATASET = False
RESET_TRAINING_CHECKPOINTS = False
DATASET_ONLY = False
DATASET_PARAM_OVERRIDES = {}
SAM2_VAL_FRACTION = 0.15
SAM2_VAL_RANDOM_SEED = 42
```

`MICROSCOPE_LABEL` is the Drive folder name for one trained configuration. Use
a different label for each microscope/configuration you want to keep.

`DATASET_PRESET_NAME` is normally `"default"` for the public starter. Change
the actual simulator settings in `DATASET_PARAM_OVERRIDES`. Particle changes
use canonical `particles` objects.

The preview step is optional. It calls Syniscopy's real single-frame viewer,
writes a complete default parameter surface JSON for reference, displays the
generated frame views, and prints a copy-paste block for the final condition
cell. You can skip it and run training directly if you already know your
parameters.

The notebook generates or reuses the full Syniscopy dataset, reads the
lossless PNG frame sequences recorded in `frame_sequence_dir` /
`training_frames_dir`, converts them to Segment Anything Model 2's
video-object-segmentation layout, splits videos into training and validation
lists, trains Segment Anything Model 2 with validation after each epoch,
and writes:

```text
MyDrive/Syniscopy/<MICROSCOPE_LABEL>/weights/final_checkpoint.pt
```

The exported checkpoint is selected from the best validation-loss checkpoint
when one is available. The training log folder keeps Segment Anything Model 2's
resume checkpoint (`checkpoint.pt`) and `checkpoint_best_val.pt`; numbered
epoch checkpoints are not retained, keeping Drive quota controlled while
preserving resume and best-model state.

Syniscopy `Ignore/` masks remove unsupported pixels from the loss. `LossWeight/`
modulates positive target pixels only; background pixels remain supervised so
large false-positive masks are penalized. Training records a loss-definition
version marker next to the checkpoint and stops if an existing checkpoint was
created under an incompatible loss contract.

## Optional CPU/GPU Split and Resume

You can save GPU time by generating the Syniscopy dataset on a CPU Colab
runtime, then reconnecting with a GPU runtime for Segment Anything Model 2
training.

CPU dataset run:

```python
MICROSCOPE_LABEL = "my_configuration"
RESET_RAW_DATASET = False
DATASET_ONLY = True
```

After the CPU run finishes, verify this file exists:

```text
MyDrive/Syniscopy/<MICROSCOPE_LABEL>/dataset/raw_syniscopy/dataset_manifest.json
```

GPU training run:

```python
MICROSCOPE_LABEL = "my_configuration"
RESET_RAW_DATASET = False
RESET_TRAINING_CHECKPOINTS = False
DATASET_ONLY = False
```

With `DATASET_ONLY = True`, the notebook generates/resumes the raw Syniscopy
dataset and stops before Segment Anything Model 2 setup. With
`DATASET_ONLY = False` and `RESET_RAW_DATASET = False`, the notebook checks the Drive dataset, skips
completed videos, regenerates only an incomplete video if needed, and then
continues to Segment Anything Model 2 training.

If Colab disconnects during dataset generation, rerun with
`RESET_RAW_DATASET = False`; completed videos are preserved. If Colab
disconnects during Segment Anything Model 2 training, rerun with the same
`MICROSCOPE_LABEL`, `RESET_RAW_DATASET = False`, and
`RESET_TRAINING_CHECKPOINTS = False`; the notebook keeps the Drive-backed
resume checkpoint under:

```text
MyDrive/Syniscopy/<MICROSCOPE_LABEL>/training_logs/checkpoints/
```

and Segment Anything Model 2 resumes model, optimizer, step, scaler, and dataset
state from `checkpoint.pt`. To start fresh intentionally, use a new `MICROSCOPE_LABEL` or
set `RESET_RAW_DATASET = True` and/or `RESET_TRAINING_CHECKPOINTS = True` for
that label.

## Inference

Open `sam2inference.ipynb` in Google Colab.

Edit the first setup cell so the label exactly matches a completed training
run:

```python
MICROSCOPE_LABEL = "default_configuration"
```

The notebook auto-loads:

```text
MyDrive/Syniscopy/<MICROSCOPE_LABEL>/weights/final_checkpoint.pt
```

Then upload the video directly into the Colab runtime when prompted. The
notebook extracts frames, displays the first extracted frame before the prompt
UI, then lets you add foreground/background point prompts and optional boxes.
Switch the point type before clicking the frame or entering X/Y coordinates.
Outputs are written under:

```text
MyDrive/Syniscopy/<MICROSCOPE_LABEL>/inference_outputs/
```

## Notes

The training notebook contains the Segment Anything Model 2 ignore/loss-weight
setup inline. Users do not upload any extra Python setup scripts.
