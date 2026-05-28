# Reviewed Liverpool Caustic Video Clips

This folder contains reviewed clip-window manifests and audit metadata for
19 reviewed clips derived from Genevieve
Schleyer's University of Liverpool DataCat dataset, "Raw video files: Caustic
signatures produced by gold nanoparticles."

- Source DOI: `10.17638/datacat.liverpool.ac.uk/2592`
- Source condition: reviewed DataCat `50nm/` videos selected for the caustic-video transfer comparison
- Reviewed clip manifest: `selected_clip_manifest.json`
- Review audit summary: `review_selection_audit.json`
- License for the derived clips: CC BY 4.0, `https://creativecommons.org/licenses/by/4.0/`

The raw DataCat `50nm/` download is a local working input and is not bundled in
this Syniscopy source tree. The reviewed clip AVI files are also not bundled by
default; regenerate them locally from the DataCat `50nm/` download with
`python supplemental/rebuild_liverpool_review_clips.py --raw-root /path/to/50nm`.
The manifest retains source-video identifiers, frame windows, prompt-frame
coordinates, and audit metadata needed by the caustic-video transfer notebooks.
No endorsement by Genevieve Schleyer or the University of Liverpool is implied.
