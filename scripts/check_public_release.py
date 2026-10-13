"""Reject common secrets and machine-local paths before a public release.

This intentionally scans only files that are normally publishable. Generated
artifacts, AppWorld data, vendored repositories, and scratch deployment files
are release inputs, not source files, and are excluded by .gitignore.
"""

from __future__ import annotations

import re
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SELF_PATH = Path(__file__).resolve()
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".pytest_tmp",
    ".ruff_cache",
    ".tmp",
    ".analysis",
    "artifacts",
    "data",
    "experiments",
    "appworld",
    "agentic-grpo-longhorizon",
    "shopping-grpo-longhorizon",
    "shopping-grpo-longhorizon-fresh",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private server absolute path", re.compile(r"/root/|[A-Za-z]:\\(?:Users|home)\\")),
    ("SSH endpoint", re.compile(r"(?:ssh|scp)\s+(?:root|[\w.-]+)@[^\s]+", re.I)),
    (
        "provider secret assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*"
            r"[\"'](?!<your|replace-me|empty|none|null|test)[A-Za-z0-9_./+=-]{16,}[\"']"
        ),
    ),
    ("password assignment", re.compile(r"(?i)\bpassword\s*[:=]\s*[\"'][^\"']{4,}[\"']")),
)


def _excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return bool(set(relative.parts) & EXCLUDED_PARTS) or (
        relative.name.startswith(".env") and relative.name != ".env.example"
    )


def main() -> int:
    findings: list[tuple[Path, int, str, str]] = []
    candidates: list[Path] = []
    for directory, names, filenames in os.walk(ROOT):
        current = Path(directory)
        names[:] = [name for name in names if name not in EXCLUDED_PARTS]
        for filename in filenames:
            candidates.append(current / filename)
    for path in sorted(candidates):
        if (
            not path.is_file()
            or path.resolve() == SELF_PATH
            or _excluded(path)
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append((path.relative_to(ROOT), line_number, label, line.strip()))

    if findings:
        print("Public release check failed:", file=sys.stderr)
        for path, line_number, label, line in findings:
            print(f"  {path}:{line_number}: {label}: {line[:180]}", file=sys.stderr)
        return 1
    print("Public release check passed: no obvious secrets or machine-local paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
