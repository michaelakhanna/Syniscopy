#!/usr/bin/env python3
"""Build the public Syniscopy release staging trees.

The repository keeps local/raw inputs and generated outputs next to the source
during development. This script constructs the smaller public payloads used for
GitHub, Zenodo, and arXiv from the same working tree while sanitizing reviewed
third-party caustic-video metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = REPO_ROOT / "release_uploads"

GITHUB_ROOT_FILES = [
    ".zenodo.json",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
]

GITHUB_SOURCE_DIRS = [
    ".github",
    "codebase",
    "docs",
    "examples",
    "recipes",
    "sam2_starter",
]

SCRIPT_FILES = [
    "scripts/build_public_release.py",
    "scripts/verify_generated_release_assets.py",
    "scripts/verify_release_tree.py",
]

SUPPLEMENTAL_SOURCE_FILES = [
    "supplemental/E01.ipynb",
    "supplemental/E02.ipynb",
    "supplemental/E03.ipynb",
    "supplemental/E04.ipynb",
    "supplemental/E05.ipynb",
    "supplemental/E06.ipynb",
    "supplemental/E07.ipynb",
    "supplemental/E07_training_condition_index.csv",
    "supplemental/E07_training_conditions.json",
    "supplemental/E08.ipynb",
    "supplemental/E09.ipynb",
    "supplemental/README.md",
    "supplemental/package_experiments_for_colab.py",
    "supplemental/package_experiments_for_colab.sh",
    "supplemental/rebuild_liverpool_review_clips.py",
]

ARXIV_FILES = [
    "paper/main.tex",
    "paper/main.bbl",
    "paper/references.bib",
]

ARXIV_DIRS = [
    "paper/sections",
    "paper/figures",
]

ARXIV_EXCLUDED_RELATIVE_FILES = {
    "paper/figures/artifact-provenance-manifest.json",
    "paper/figures/theoretical-diagnostics.json",
}

ARXIV_ALLOWED_SUFFIXES = {
    ".bbl",
    ".bib",
    ".png",
    ".tex",
}

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

LOCAL_REVIEW_ROOT = "throw" + "out"
LOCAL_SCRATCH_ROOT = "throw" + "_out"
LOCAL_HYPHEN_ROOT = "throw" + "-out"

SKIP_DIR_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    "__MACOSX",
    "__pycache__",
    "release_uploads",
    LOCAL_REVIEW_ROOT,
    LOCAL_SCRATCH_ROOT,
    LOCAL_SCRATCH_ROOT + "_2",
}

SKIP_FILE_NAMES = {
    ".DS_Store",
}

SKIP_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pyc",
    ".pyo",
    ".zip",
}

LOCAL_REVIEW_MARKERS = (
    LOCAL_REVIEW_ROOT + "/",
    LOCAL_SCRATCH_ROOT,
    LOCAL_HYPHEN_ROOT,
)

REVIEW_DATA_REL = Path("supplemental/data/liverpool_caustic_50nm_review")
LEGACY_REVIEW_DATA_REL = Path("supplemental/data/liverpool_caustic_50nm")
REVIEW_CLIP_PAYLOAD_DIRNAME = "clips"
REVIEW_CLIP_PAYLOAD_SUFFIXES = {".avi", ".mp4", ".mov"}


@dataclass
class BuildSummary:
    copied_files: int = 0
    copied_dirs: int = 0
    sanitized_json: int = 0
    sanitized_csv: int = 0
    zip_files: int = 0
    generated_notices: int = 0
    skipped_missing_optional: list[str] | None = None

    def __post_init__(self) -> None:
        if self.skipped_missing_optional is None:
            self.skipped_missing_optional = []


def relpath(path: Path) -> str:
    return path.as_posix()


def ensure_inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {path}") from exc
    return resolved


def should_skip_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIR_NAMES:
        return True
    if path.name in SKIP_FILE_NAMES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    path_posix = path.as_posix()
    if path_posix == relpath(LEGACY_REVIEW_DATA_REL) or path_posix.startswith(
        relpath(LEGACY_REVIEW_DATA_REL) + "/"
    ):
        return True
    if path_posix.startswith("supplemental/outputs/"):
        return True
    return False


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_release_cruft(path: Path) -> None:
    for finder_file in path.rglob(".DS_Store"):
        if finder_file.is_file():
            finder_file.unlink()


def copy_file(src_rel: str | Path, dst_root: Path, summary: BuildSummary) -> None:
    src_rel = Path(src_rel)
    if should_skip_path(src_rel):
        return
    src = ensure_inside_repo(REPO_ROOT / src_rel)
    if not src.exists():
        raise FileNotFoundError(f"required release source missing: {src_rel}")
    dst = dst_root / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    summary.copied_files += 1


def copy_optional_file(src_rel: str | Path, dst_root: Path, summary: BuildSummary) -> None:
    src_rel = Path(src_rel)
    src = REPO_ROOT / src_rel
    if not src.exists():
        summary.skipped_missing_optional.append(src_rel.as_posix())
        return
    copy_file(src_rel, dst_root, summary)


def copy_tree(src_rel: str | Path, dst_root: Path, summary: BuildSummary) -> None:
    src_rel = Path(src_rel)
    src = ensure_inside_repo(REPO_ROOT / src_rel)
    if not src.exists():
        raise FileNotFoundError(f"required release source directory missing: {src_rel}")
    for path in src.rglob("*"):
        rel = path.relative_to(REPO_ROOT)
        if should_skip_path(rel):
            continue
        if path.is_dir():
            (dst_root / rel).mkdir(parents=True, exist_ok=True)
            summary.copied_dirs += 1
        elif path.is_file():
            (dst_root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst_root / rel)
            summary.copied_files += 1


def has_local_review_marker(value: str) -> bool:
    return any(marker in value for marker in LOCAL_REVIEW_MARKERS)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"working_video", "review_state"}:
                continue
            clean[key] = sanitize_json_value(item)
        return clean
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, str) and has_local_review_marker(value):
        return "local_path_not_distributed"
    return value


def sanitize_text_value(value: str) -> str:
    return "local_path_not_distributed" if has_local_review_marker(value) else value


def write_review_notice(dst_dir: Path, summary: BuildSummary) -> None:
    manifest_path = REPO_ROOT / REVIEW_DATA_REL / "selected_clip_manifest.json"
    clip_count = 0
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        clips = data.get("clips", data) if isinstance(data, dict) else data
        clip_count = len(clips)
    text = f"""# Reviewed Liverpool Caustic Video Clips

This folder contains reviewed clip-window manifests and audit metadata for
{clip_count} reviewed clips derived from Genevieve
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
"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "NOTICE.md").write_text(text, encoding="utf-8")
    summary.generated_notices += 1


def copy_sanitized_review_data(dst_root: Path, summary: BuildSummary) -> None:
    src_root = ensure_inside_repo(REPO_ROOT / REVIEW_DATA_REL)
    if not src_root.exists():
        raise FileNotFoundError(f"reviewed caustic data folder missing: {REVIEW_DATA_REL}")
    dst_review = dst_root / REVIEW_DATA_REL
    for path in src_root.rglob("*"):
        rel = path.relative_to(src_root)
        dst = dst_review / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if should_skip_path(REVIEW_DATA_REL / rel):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if path.name == "NOTICE.md":
            continue
        if rel.parts[:1] == (REVIEW_CLIP_PAYLOAD_DIRNAME,) and suffix in REVIEW_CLIP_PAYLOAD_SUFFIXES:
            continue
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            sanitized = sanitize_json_value(data)
            dst.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary.sanitized_json += 1
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as src_fh:
                reader = csv.DictReader(src_fh)
                fieldnames = [
                    name
                    for name in (reader.fieldnames or [])
                    if name not in {"working_video", "review_state"}
                ]
                with dst.open("w", newline="", encoding="utf-8") as dst_fh:
                    writer = csv.DictWriter(dst_fh, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in reader:
                        writer.writerow({name: sanitize_text_value(row.get(name, "")) for name in fieldnames})
            summary.sanitized_csv += 1
        else:
            shutil.copy2(path, dst)
            summary.copied_files += 1
    write_review_notice(dst_review, summary)


def build_public_source_tree(dst_root: Path, summary: BuildSummary) -> None:
    clean_dir(dst_root)
    for src_rel in GITHUB_ROOT_FILES:
        copy_file(src_rel, dst_root, summary)
    for src_rel in GITHUB_SOURCE_DIRS:
        copy_tree(src_rel, dst_root, summary)
    copy_optional_file("paper/main.pdf", dst_root, summary)
    for src_rel in SCRIPT_FILES:
        copy_file(src_rel, dst_root, summary)
    for src_rel in SUPPLEMENTAL_SOURCE_FILES:
        copy_file(src_rel, dst_root, summary)
    copy_sanitized_review_data(dst_root, summary)


def build_arxiv_tree(dst_root: Path, summary: BuildSummary) -> None:
    clean_dir(dst_root)
    for src_rel in ARXIV_FILES:
        src = REPO_ROOT / src_rel
        if not src.exists():
            raise FileNotFoundError(f"required arXiv source missing: {src_rel}")
        out = dst_root / Path(src_rel).relative_to("paper")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        summary.copied_files += 1
    for src_rel in ARXIV_DIRS:
        src = REPO_ROOT / src_rel
        if src.exists():
            for path in src.rglob("*"):
                rel = path.relative_to(REPO_ROOT)
                if should_skip_path(rel):
                    continue
                if rel.as_posix() in ARXIV_EXCLUDED_RELATIVE_FILES:
                    continue
                if path.is_dir():
                    (dst_root / rel.relative_to("paper")).mkdir(parents=True, exist_ok=True)
                elif path.is_file():
                    if path.suffix.lower() not in ARXIV_ALLOWED_SUFFIXES:
                        continue
                    out = dst_root / rel.relative_to("paper")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, out)
                    summary.copied_files += 1
        else:
            raise FileNotFoundError(f"required arXiv source directory missing: {src_rel}")


def zip_directory(
    src_dir: Path,
    zip_path: Path,
    summary: BuildSummary,
    *,
    archive_root_name: str | None = None,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    if archive_root_name is None:
        archive_root_name = src_dir.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(src_dir).as_posix()
                arcname = rel if archive_root_name == "" else f"{archive_root_name}/{rel}"
                zf.write(path, arcname)
    summary.zip_files += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(root: Path, summary: BuildSummary) -> None:
    checksum_path = root / "SHA256SUMS.txt"
    rows = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != checksum_path.name:
            rows.append(f"{sha256_file(path)}  {path.name}")
    checksum_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    summary.copied_files += 1


def copy_github_assets(
    assets_root: Path,
    github_source_root: Path,
    version: str,
    summary: BuildSummary,
) -> None:
    clean_dir(assets_root)
    zip_directory(
        github_source_root,
        assets_root / f"syniscopy_public_source_{version}.zip",
        summary,
        archive_root_name=f"syniscopy_public_source_{version}",
    )
    paper_pdf = REPO_ROOT / "paper/main.pdf"
    if paper_pdf.exists():
        shutil.copy2(paper_pdf, assets_root / f"syniscopy_paper_{version}.pdf")
        summary.copied_files += 1
    else:
        summary.skipped_missing_optional.append("paper/main.pdf")
    write_sha256sums(assets_root, summary)


def copy_arxiv_upload_zip(
    release_root: Path,
    arxiv_root: Path,
    version: str,
    summary: BuildSummary,
) -> None:
    zip_directory(
        arxiv_root,
        release_root / f"syniscopy_arxiv_source_{version}.zip",
        summary,
        archive_root_name="",
    )


def copy_zenodo_assets(assets_root: Path, sanitized_source_root: Path, version: str, summary: BuildSummary) -> None:
    clean_dir(assets_root)
    paper_pdf = REPO_ROOT / "paper/main.pdf"
    if paper_pdf.exists():
        shutil.copy2(paper_pdf, assets_root / f"syniscopy_paper_{version}.pdf")
        summary.copied_files += 1
    else:
        summary.skipped_missing_optional.append("paper/main.pdf")

    review_src = sanitized_source_root / REVIEW_DATA_REL
    review_assets = assets_root / "liverpool_caustic_50nm_review"
    if review_src.exists():
        shutil.copytree(review_src, review_assets)
        summary.copied_dirs += 1
        zip_directory(review_assets, assets_root / "liverpool_caustic_50nm_review.zip", summary)
        for name in [
            "NOTICE.md",
            "selected_clip_manifest.json",
            "selected_clip_manifest.csv",
            "review_selection_audit.json",
        ]:
            src = review_assets / name
            if src.exists():
                shutil.copy2(src, assets_root / f"liverpool_caustic_50nm_review_{name}")
                summary.copied_files += 1


def run_check(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def verify_arxiv_tree(arxiv_root: Path) -> None:
    required = {
        "main.tex",
        "main.bbl",
        "references.bib",
        "sections",
        "figures",
    }
    failures = [
        f"missing required arXiv path: {name}"
        for name in sorted(required)
        if not (arxiv_root / name).exists()
    ]
    for path in arxiv_root.rglob("*"):
        rel = path.relative_to(arxiv_root).as_posix()
        if not path.is_file():
            continue
        if path.name == "main.pdf":
            failures.append("compiled PDF should not be included in the arXiv TeX source tree")
        if path.suffix.lower() not in ARXIV_ALLOWED_SUFFIXES:
            failures.append(f"unexpected arXiv file type: {rel}")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)
    print(f"arXiv tree check passed: {arxiv_root}")


def verify_release(release_root: Path, verify_assets: bool) -> None:
    run_check([sys.executable, "scripts/verify_release_tree.py", str(release_root / "github")])
    run_check([sys.executable, "scripts/verify_release_tree.py", str(release_root / "zenodo/source")])
    verify_arxiv_tree(release_root / "arxiv")
    if verify_assets:
        run_check([sys.executable, "scripts/verify_generated_release_assets.py", str(release_root / "github_assets")])
    if verify_assets:
        run_check([sys.executable, "scripts/verify_generated_release_assets.py", str(release_root / "zenodo/assets")])


def build_public_release(release_root: Path, version: str, verify: bool) -> BuildSummary:
    release_root = release_root.resolve()
    summary = BuildSummary()
    github_root = release_root / "github"
    github_assets_root = release_root / "github_assets"
    zenodo_source_root = release_root / "zenodo/source"
    zenodo_assets_root = release_root / "zenodo/assets"
    arxiv_root = release_root / "arxiv"

    build_public_source_tree(github_root, summary)
    copy_github_assets(github_assets_root, github_root, version, summary)
    clean_dir(zenodo_source_root)
    shutil.copytree(github_root, zenodo_source_root, dirs_exist_ok=True)
    build_arxiv_tree(arxiv_root, summary)
    copy_arxiv_upload_zip(release_root, arxiv_root, version, summary)
    copy_zenodo_assets(zenodo_assets_root, github_root, version, summary)
    remove_release_cruft(release_root)

    if verify:
        verify_release(release_root, verify_assets=True)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help="Release staging root to rebuild (default: release_uploads).",
    )
    parser.add_argument(
        "--version",
        default="v1.1.0",
        help="Version label used in generated asset filenames.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Build without running release hygiene verification.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Accepted for clarity; the staged release directories are always rebuilt cleanly.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_public_release(args.release_root, args.version, verify=not args.no_verify)
    print("public release staging rebuilt")
    print(f"  root: {args.release_root.resolve()}")
    print(f"  copied files: {summary.copied_files}")
    print(f"  copied dirs: {summary.copied_dirs}")
    print(f"  sanitized json: {summary.sanitized_json}")
    print(f"  sanitized csv: {summary.sanitized_csv}")
    print(f"  generated notices: {summary.generated_notices}")
    print(f"  zip files: {summary.zip_files}")
    if summary.skipped_missing_optional:
        print("  skipped missing optional paths:")
        for item in summary.skipped_missing_optional:
            print(f"    - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
