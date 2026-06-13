from __future__ import annotations

from pathlib import Path
import re


def _find_matlab_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    matlab_dir = repo_root / "throwout" / "external" / "iScatCRB" / "MATLAB"
    if not matlab_dir.is_dir():
        raise FileNotFoundError(f"Missing iScatCRB MATLAB directory: {matlab_dir}")
    return matlab_dir


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(line.split("%", 1)[0])
    return "\n".join(lines)


def test_iscatcrb_reference_gate_requires_full_matrix_candidates() -> None:
    matlab_dir = _find_matlab_dir()
    element_shortcuts = [
        re.compile(r"\bvar_[A-Za-z0-9_]*\s*=\s*1\s*/\s*FI\s*\(\s*\d+\s*,\s*\d+\s*\)", re.I),
        re.compile(r"\b1\s*/\s*FI\s*\(\s*\d+\s*,\s*\d+\s*\)", re.I),
        re.compile(r"\bFI\s*\(\s*\d+\s*,\s*\d+\s*\)\s*\^\s*-1\b", re.I),
    ]
    full_matrix_ops = [
        re.compile(r"\binv\s*\(\s*FI\s*\)", re.I),
        re.compile(r"\bpinv\s*\(\s*FI\s*\)", re.I),
        re.compile(r"\bFI\s*\\", re.I),
        re.compile(r"\\\s*FI\b", re.I),
    ]

    shortcut_hits = {}
    full_hits = {}
    total_rows = 0

    for path in sorted(matlab_dir.glob("*.m")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = _strip_comments(raw)
        total_rows += 1
        sc = sum(len(p.findall(text)) for p in element_shortcuts)
        fm = sum(len(p.findall(text)) for p in full_matrix_ops)
        if sc:
            shortcut_hits[path.name] = sc
        if fm:
            full_hits[path.name] = fm

    assert total_rows > 0
    assert bool(full_hits), "No full-matrix CRLB implementation pattern found in iScatCRB MATLAB sources."
    # Ensure there is some evidence that scalar shortcuts are distinguished from full-matrix paths.
    assert (
        bool(shortcut_hits) and any(k not in full_hits for k in shortcut_hits)
    ) or len(full_hits) > 1
