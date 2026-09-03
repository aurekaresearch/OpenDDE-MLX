#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Validate that release metadata is consistent before publishing.

python scripts/check_release.py --tag v0.1.0
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_release_notes import changelog_section  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r'^__version__\s*=\s*"(?P<version>[^"]+)"\s*$', re.M)
STABLE = re.compile(r"^\d+\.\d+\.\d+$")
PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|XXX)\b", re.I)


def check(tag: str) -> list[str]:
    """Return every problem that blocks releasing ``tag``."""
    errors: list[str] = []
    match = VERSION.search((ROOT / "opendde" / "version.py").read_text("utf-8"))
    if match is None:
        return ["opendde/version.py has no __version__"]

    version = match.group("version")
    if not STABLE.fullmatch(version):
        errors.append(f"version is not a stable X.Y.Z release: {version}")
    if tag != f"v{version}":
        errors.append(f"tag {tag} does not match opendde/version.py ({version})")

    try:
        manifest = json.loads(
            (ROOT / "opendde" / "config" / "model_manifest.json").read_text("utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid model manifest: {exc}")
    else:
        if manifest.get("package") != {"name": "opendde-mlx", "version": version}:
            errors.append("model manifest package does not match opendde-mlx version")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    if pyproject["name"] != "opendde-mlx":
        errors.append(f"unexpected distribution name: {pyproject['name']}")
    if "version" in pyproject:
        errors.append("pyproject.toml pins a static version; it must stay dynamic")

    try:
        _, body = changelog_section(version)
    except LookupError as exc:
        errors.append(str(exc))
    else:
        if not body:
            errors.append(f"CHANGELOG.md section for {version} is empty")
        if found := PLACEHOLDER.search(body):
            errors.append(f"CHANGELOG.md section for {version} contains {found.group(0)!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.1.0")
    args = parser.parse_args()
    if errors := check(args.tag):
        raise SystemExit("Release check failed:\n" + "\n".join(f"  - {e}" for e in errors))
    print(f"Release metadata is consistent for {args.tag}")


if __name__ == "__main__":
    main()
