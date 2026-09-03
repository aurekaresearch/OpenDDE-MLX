#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Render a GitHub Release body from the matching CHANGELOG section.

python scripts/render_release_notes.py --version 0.1.0 --output NOTES.md
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^## (?P<version>\d+\.\d+\.\d+) \((?P<date>\d{4}-\d{2}-\d{2})\)$", re.M)
REPOSITORY = "https://github.com/aurekaresearch/OpenDDE-MLX"


def changelog_section(version: str, changelog: str | None = None) -> tuple[str, str]:
    """Return the ``(date, body)`` of one CHANGELOG section, or raise ``LookupError``."""
    text = changelog if changelog is not None else (ROOT / "CHANGELOG.md").read_text("utf-8")
    headings = list(HEADING.finditer(text))
    for index, match in enumerate(headings):
        if match.group("version") != version:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        return match.group("date"), text[match.end() : end].strip()
    raise LookupError(f"CHANGELOG.md has no section for {version}")


def render(version: str) -> str:
    """Build the release body for ``version``."""
    _, body = changelog_section(version)
    return (
        f"{body}\n\n"
        f"Install with `pip install opendde-mlx=={version}`.\n\n"
        f"Full changelog: {REPOSITORY}/blob/v{version}/CHANGELOG.md\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        notes = render(args.version)
    except LookupError as exc:
        raise SystemExit(f"error: {exc}") from None
    Path(args.output).write_text(notes, encoding="utf-8")
    print(notes, file=sys.stderr)


if __name__ == "__main__":
    main()
