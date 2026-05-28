# Third-Party Notices

Syniscopy source code and documentation are released under the MIT license in
`LICENSE`. The notices below apply to external assets and upstream software
referenced by the notebooks or by optional release assets.

## University of Liverpool Caustic-Particle Video

- Source: Genevieve Schleyer, "Raw video files: Caustic signatures produced by
  gold nanoparticles"
- Data catalogue DOI: `10.17638/datacat.liverpool.ac.uk/2592`
- Source URL: `https://doi.org/10.17638/datacat.liverpool.ac.uk/2592`
- License: Creative Commons Attribution 4.0
- License URL: `https://creativecommons.org/licenses/by/4.0/`

The caustic-video transfer comparison uses the reviewed source windows listed
under `supplemental/data/liverpool_caustic_50nm_review/`. The raw DataCat
`50nm/` download is a local working input; only the reviewed clip windows,
review manifests, notices, and prompt/audit metadata are intended public inputs.
The source release does not bundle the regenerated reviewed clip AVI payload;
rebuild those clips locally from the DataCat download when rerunning the
transfer notebooks.

Any redistributed reviewed clip or paper figure derived from this dataset must
keep the DataCat attribution and CC BY 4.0 notice. No endorsement by Genevieve
Schleyer or the University of Liverpool is implied.

## Segment Anything Model 2

- Upstream repository: `https://github.com/facebookresearch/sam2`
- Notebook clone URL used by the public notebooks:
  `https://github.com/facebookresearch/segment-anything-2.git`
- Upstream license: Apache License 2.0
- License URL: `https://github.com/facebookresearch/sam2/blob/main/LICENSE`
- Default base checkpoint used by the Segment Anything Model 2 training and
  transfer-inference notebooks:
  `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`
- Default base checkpoint filename: `sam2.1_hiera_large.pt`
- Default config: `configs/sam2.1/sam2.1_hiera_l.yaml`

The Segment Anything Model 2 code and base checkpoints are downloaded at runtime
by the notebooks. They are not bundled as MIT-licensed Syniscopy source code.
Fine-tuned Segment Anything Model 2 checkpoints, if distributed, are derived
from upstream Segment Anything Model 2 weights and must carry the upstream
license notices and checksums for the distributed file.

## Generated Outputs

Notebook outputs under `supplemental/outputs/` are generated files, not source
files. Public GitHub-style source releases contain the runnable notebooks and
modules rather than generated notebook outputs.
If a versioned archival release includes generated CSVs, manuscript source
materials, reviewed clips, transfer overlays, or fine-tuned checkpoints, the
release manifest should list the files, checksums, provenance, and applicable
third-party notices.
