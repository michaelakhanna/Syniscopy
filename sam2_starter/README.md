# Segment Anything Model 2 Starter

Clean entry point for users who want to fine-tune or run Segment Anything Model
2 (SAM2) with synthetic Syniscopy data in Google Colab.

- `sam2training.ipynb`: Colab training notebook. It extracts the Syniscopy
  source zip, previews generated frame views, generates the synthetic dataset
  on Colab, reads Syniscopy's lossless PNG frame sequences, converts them to
  Segment Anything Model 2's video-object-segmentation layout, creates a video-level validation
  split, trains with validation after each epoch, and writes
  `weights/final_checkpoint.pt` from the best validation checkpoint for the
  selected configuration label.
  It can also stop after dataset generation with `DATASET_ONLY = True`, so a
  CPU runtime can generate data before a later GPU runtime resumes training.
  The notebook reads the Syniscopy supervision schema and lets the user select
  `mask_supported` or `mask_geometry`; it also exports
  `Ignore/` and `LossWeight/` maps alongside selected target (`GT/`) masks so the
  notebook's inline Segment Anything Model 2 setup keeps unsupported object pixels out of the loss.
  Background pixels remain supervised; `LossWeight/` only modulates positive
  target pixels, so broad false-positive masks are penalized.
- `sam2inference.ipynb`: Colab inference notebook. It selects a configuration
  label, auto-loads `weights/final_checkpoint.pt`, accepts a direct video
  upload into the runtime, displays the first extracted frame before prompt
  placement, supports foreground/background points and optional boxes, and
  writes overlay video, mask-only video, and per-frame mask PNGs.

The source zip that Colab users upload is a packaging artifact. Generate it
locally and place it into this starter folder with the cross-platform Python
packager:

```bash
python sam2_starter/package_source_zip.py
```

On Windows PowerShell, `py -3 sam2_starter/package_source_zip.py` is equivalent.
The Bash wrapper `bash sam2_starter/package_source_zip.sh` is kept for
macOS/Linux users.

Then upload the whole `sam2_starter/` folder to Google Drive. The required
Drive starter set is:

```text
syniscopy_codebase.zip
sam2training.ipynb
sam2inference.ipynb
```

The training notebook looks first for:

```text
MyDrive/sam2_starter/syniscopy_codebase.zip
```

and stores generated datasets/checkpoints under:

```text
MyDrive/Syniscopy/<MICROSCOPE_LABEL>/
```

The notebook records the Segment Anything Model 2 loss-definition version in checkpoint metadata and
will stop rather than resume a checkpoint created with an incompatible loss
contract. Use a new `MICROSCOPE_LABEL` or set `RESET_TRAINING_CHECKPOINTS=True`
when intentionally starting a fresh training run.

See `COLAB_RUN_INSTRUCTIONS.md` for the user-facing Colab flow.

## External Segment Anything Model 2 Assets

The starter notebooks clone Meta's external Segment Anything 2 repository and
download Segment Anything Model 2 checkpoints at runtime. Those files are not bundled as Syniscopy
code and remain under Meta's upstream licenses and notices. Fine-tuned
checkpoints derived from those weights are likewise not MIT-licensed Syniscopy
source code. Syniscopy supplies the synthetic dataset, auxiliary supervision
arrays, and wrapper notebooks only; see `../THIRD_PARTY_NOTICES.md` for the
release notice boundary.
