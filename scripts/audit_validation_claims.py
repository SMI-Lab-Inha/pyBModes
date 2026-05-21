# Copyright 2024-2026 Jae Hoon Seo
# Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit ``VALIDATION.md`` against the test suite to catch "claim ahead
of test" drift.

The validation matrix is the single structured source of truth for what
pyBmodes is validated against. Each row names a specific test file. This
script:

1. Parses every test-file link from ``VALIDATION.md`` (markdown links of
   the form ``[label](tests/...)``).
2. Asserts each linked path exists.
3. Asserts each linked path is one of (a) a single test file or (b) a
   directory glob like ``tests/fem/`` — and for case (a) that the file
   defines at least one ``def test_…`` method, so the row isn't pointing
   at an empty placeholder.

Run from the repo root:

    python scripts/audit_validation_claims.py

Exits 0 with a one-line OK summary if every row checks out, or non-zero
with a detailed per-row error report otherwise. Designed to live as
step 4.5 of ``docs/release_checklist.rst``.

The script does NOT execute any tests. It only checks file presence
and structural population — enough to catch a row that names a file
that doesn't exist, or a file that defines no test methods, both of
which would otherwise let the matrix silently advertise non-existent
coverage.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATION_MD = REPO_ROOT / "VALIDATION.md"

# Markdown link: [link-text](path). We restrict to links whose target
# starts with "tests/" — i.e. the test-file references inside the
# validation matrix. Other links (e.g. external citations, README
# cross-refs) are ignored.
_LINK_RE = re.compile(r"\]\((tests/[^)]+)\)")


def _collect_test_paths() -> list[pathlib.Path]:
    """Pull every distinct ``tests/...`` link target out of VALIDATION.md."""
    text = VALIDATION_MD.read_text(encoding="utf-8")
    seen: set[str] = set()
    out: list[pathlib.Path] = []
    for m in _LINK_RE.finditer(text):
        target = m.group(1).rstrip("/")
        if target in seen:
            continue
        seen.add(target)
        out.append(REPO_ROOT / target)
    return out


def _has_test_methods(path: pathlib.Path) -> bool:
    """True if ``path`` is a Python test file containing at least one
    ``def test_…`` method."""
    if not path.is_file() or path.suffix != ".py":
        return False
    text = path.read_text(encoding="utf-8")
    return bool(re.search(r"^\s*def\s+test_\w+", text, flags=re.MULTILINE))


def main() -> int:
    if not VALIDATION_MD.is_file():
        print(f"ERROR: {VALIDATION_MD} not found", file=sys.stderr)
        return 2

    paths = _collect_test_paths()
    if not paths:
        print("ERROR: no test-file links found in VALIDATION.md "
              "(at least one expected)", file=sys.stderr)
        return 2

    errors: list[str] = []
    for p in paths:
        rel = p.relative_to(REPO_ROOT).as_posix()
        if p.is_dir():
            # Directory globs like tests/fem/ — verify the directory
            # exists and contains at least one *_test.py / test_*.py
            # with a test method.
            candidates = list(p.glob("test_*.py")) + list(p.glob("*_test.py"))
            populated = [c for c in candidates if _has_test_methods(c)]
            if not populated:
                errors.append(
                    f"{rel}: directory exists but contains no populated "
                    "test_*.py files"
                )
        elif p.is_file():
            if not _has_test_methods(p):
                errors.append(
                    f"{rel}: file exists but defines no `def test_…` methods"
                )
        else:
            errors.append(f"{rel}: not found")

    if errors:
        print(f"FAIL: {len(errors)} VALIDATION.md row(s) reference "
              f"missing or empty test paths:\n", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"OK: every VALIDATION.md test-file reference exists and "
        f"contains at least one test method "
        f"({len(paths)} unique target path(s) checked)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
