# Syniscopy Supplemental Reproducibility

Run these notebooks from the repository root or from the Google Drive Colab
upload folder. Upload this `supplemental/` folder itself to Drive as
`MyDrive/supplemental`. The numbered notebooks write outputs under matching
`outputs/<notebook-id>/` folders inside this folder. They output raw
files only: individual image files, arrays, CSV files, JSON
manifests, datasets, checkpoints, and inference masks/videos. They do not
assemble paper figures or write directly into `paper/figures/`; the paper
assembly scripts consume these outputs and write provenance manifests next to
the generated figure and table files.
The CPU notebooks do not require GPU acceleration. The synthetic-corpus
notebook writes both the raw Syniscopy dataset and the derived Segment Anything
Model 2 video-object-segmentation cache under its output folder. The Segment
Anything Model 2 training and transfer-inference notebooks need a GPU runtime;
the training notebook reads the generated Segment Anything Model 2 cache and
writes weights/logs under its output folder.

```text
E01.ipynb
E02.ipynb
E03.ipynb
E04.ipynb
E05.ipynb
E06.ipynb
E07.ipynb
E08.ipynb
E09.ipynb
```

The source ZIP needed in Colab is `syniscopy_source.zip`; rebuild it from the
repository root before uploading. Real-video transfer clips and checkpoints are
not Syniscopy code. The reviewed caustic-video selection is
`data/liverpool_caustic_50nm_review/`; raw 50 nm working inputs and the legacy
single-clip `data/liverpool_caustic_50nm/` folder are local inputs, not public
source contents. The public source release includes reviewed clip-window
manifests and audit metadata, not the reviewed clip AVI payload. After
downloading the DataCat `50nm/` folder, regenerate local reviewed clips with:

```bash
python supplemental/rebuild_liverpool_review_clips.py --raw-root /path/to/50nm
```

Generated transfer overlays, masks, and checkpoints are reproducible from the
notebooks and are not staged for public release by default. Paper figures
derived from the public caustic-video dataset are covered by the manuscript
citation and `THIRD_PARTY_NOTICES.md`.
The listed Segment Anything Model 2 paper workflow notebooks are separate from
the generic starter notebooks in `../sam2_starter/`.
The synthetic-corpus notebook creates the Segment Anything Model 2 video-object-segmentation cache and video-level validation split consumed
by Segment Anything Model 2 training. The training notebook checks that cache, trains Segment Anything Model 2, and exports the
best validation checkpoint to its weights folder when run. The public source release does not bundle
that `.pt` file unless a separate versioned release asset explicitly includes it with upstream Segment Anything Model 2
license notices and checksums.
Its loss wrapper keeps background pixels supervised, applies `LossWeight/`
only to positive target pixels, and records a loss-definition version so old
checkpoints are not resumed across incompatible training semantics.
The transfer-inference notebook runs the caustic-video transfer example with a fixed foreground
point-plus-box paper prompt and writes base Segment Anything Model 2 and fine-tuned outputs under
separate variant folders under its inference-output folder.

The public source release keeps source notebooks, notices, reviewed clip
windows, and manifests under `data/liverpool_caustic_50nm_review/`. Raw
DataCat downloads, stale single-clip transfer files, generated transfer outputs,
reviewed clip AVI payloads, and checkpoints stay outside the public source
tree. The full `outputs/` tree is reproducible from the numbered notebooks and
is not intended for GitHub or the default archival source/assets staging. Before
archiving any generated output folder as a separate release asset, check it with
`python scripts/verify_generated_release_assets.py <asset-folder>`.

Build the source ZIP with the cross-platform Python packager:

```bash
python supplemental/package_experiments_for_colab.py
```

On Windows PowerShell, `py -3 supplemental/package_experiments_for_colab.py`
is equivalent. The Bash wrapper `bash supplemental/package_experiments_for_colab.sh`
is kept for macOS/Linux users.

The packaging script excludes local build outputs and macOS Finder artifacts
from `syniscopy_source.zip`; `.DS_Store` is also ignored by the repository.

Upload `supplemental/` to Google Drive as `MyDrive/supplemental`.

The notebooks read bundled source from `syniscopy_source.zip` and write
generated data under `MyDrive/supplemental/outputs/`.

## External-Code Licenses

Syniscopy code and documentation are MIT licensed. The Segment Anything Model 2
notebooks clone/download Meta Segment Anything 2 code and checkpoints at
runtime; those external Segment Anything Model 2 assets, and any fine-tuned
checkpoint derived from them, remain under Meta's upstream licenses and notices.
