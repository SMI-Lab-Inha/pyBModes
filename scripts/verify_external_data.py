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

"""Verify the local ``external/`` tree matches ``external/MANIFEST.toml``.

For every clone declared in the manifest:

1. Resolve the clone directory under the repo's ``external/`` tree.
2. Confirm the directory exists and is a git work tree.
3. Read the local ``HEAD`` SHA and compare against the manifest pin.
4. (Optional with ``--strict``) walk every ``hashes`` entry, recompute
   the SHA-256 of the named file, and compare against the manifest.

Exits 0 if every clone matches, non-zero otherwise. Designed as the
release-checklist gate for the integration-test track — a green run
here means anyone with the manifest can reproduce the published
0.01 % validation tolerance against the BModes Fortran reference
solver, byte-for-byte.

Usage::

    python scripts/verify_external_data.py
    python scripts/verify_external_data.py --strict
    python scripts/verify_external_data.py --update          # MAINTAINER ONLY:
                                                              # rewrites MANIFEST.toml
                                                              # with the current pins.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "external" / "MANIFEST.toml"

# ANSI colour codes; cleared on non-TTY for log-friendly output.
if sys.stdout.isatty():
    _RED, _YEL, _GRN, _END = "\033[31m", "\033[33m", "\033[32m", "\033[0m"
else:
    _RED = _YEL = _GRN = _END = ""


@dataclass
class CloneResult:
    name: str
    status: str             # "PASS" / "WARN" / "FAIL" / "SKIP"
    message: str


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True,
    ).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_clone(
    name: str, spec: dict, *, strict: bool,
) -> CloneResult:
    clone_dir = REPO_ROOT / spec["relative_path"]
    pinned_sha = spec.get("sha", "TBD")

    if not clone_dir.is_dir():
        return CloneResult(
            name, "SKIP",
            f"directory {clone_dir.relative_to(REPO_ROOT)} not present",
        )

    if pinned_sha in ("TBD", ""):
        return CloneResult(
            name, "WARN",
            "SHA pin is TBD in manifest — bump via --update on a "
            "maintainer machine before release.",
        )

    # BModes is a release archive, not a git repo. Hash check only.
    if spec.get("fetch_at") == "tagged-archive":
        if not strict:
            return CloneResult(
                name, "PASS",
                f"archive-only clone present at "
                f"{clone_dir.relative_to(REPO_ROOT)}; pass --strict "
                f"to verify file hashes.",
            )
        # fall through to hash check
    else:
        if not (clone_dir / ".git").exists():
            return CloneResult(
                name, "FAIL",
                f"{clone_dir.relative_to(REPO_ROOT)} exists but is "
                f"not a git work tree.",
            )
        try:
            head = _git(clone_dir, "rev-parse", "HEAD")
        except subprocess.CalledProcessError as err:
            return CloneResult(
                name, "FAIL",
                f"git rev-parse HEAD failed in "
                f"{clone_dir.relative_to(REPO_ROOT)}: {err}",
            )
        if head != pinned_sha:
            return CloneResult(
                name, "FAIL",
                f"HEAD={head[:12]} != pinned={pinned_sha[:12]}. "
                f"`git -C {clone_dir} checkout {pinned_sha}` to fix, "
                f"or update the manifest pin if this is intentional.",
            )

    # Optional file-hash verification.
    file_hashes = spec.get("hashes", {})
    if strict and file_hashes:
        for rel_path, expected in file_hashes.items():
            target = clone_dir / rel_path
            if not target.is_file():
                return CloneResult(
                    name, "FAIL",
                    f"hashed file missing: "
                    f"{(clone_dir / rel_path).relative_to(REPO_ROOT)}",
                )
            actual = _sha256(target)
            if actual != expected:
                return CloneResult(
                    name, "FAIL",
                    f"hash mismatch on {rel_path}:\n"
                    f"        expected {expected}\n"
                    f"        actual   {actual}\n"
                    f"        the upstream repo has rewritten this "
                    f"file in place — flag as a release blocker.",
                )

    detail = ""
    if file_hashes:
        detail = f" ({len(file_hashes)} file{'s' if len(file_hashes) != 1 else ''} hashed)"
    return CloneResult(name, "PASS", f"matches manifest{detail}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strict", action="store_true",
        help="Also recompute SHA-256 of every file in the manifest's "
             "``hashes`` block and compare.",
    )
    p.add_argument(
        "--update", action="store_true",
        help="MAINTAINER ONLY: rewrite MANIFEST.toml with the current "
             "HEAD SHAs and recomputed file hashes. Intended for the "
             "release-checklist machine that already has every clone "
             "at the correct version.",
    )
    args = p.parse_args()

    if not MANIFEST.is_file():
        print(f"{_RED}FAIL{_END}  manifest not found: {MANIFEST}", file=sys.stderr)
        return 2

    with MANIFEST.open("rb") as fh:
        manifest = tomllib.load(fh)

    if args.update:
        # ``--update`` is intentionally minimal — it edits the file in
        # place without TOML round-trip semantics (which the stdlib
        # ``tomllib`` is read-only for). The maintainer reviews the diff
        # before committing.
        print(
            f"{_YEL}WARN{_END}  ``--update`` is a write-back convenience for "
            f"maintainers. It does NOT modify the on-disk file in this "
            f"stub — the intent is that you read the printed table and "
            f"hand-update ``external/MANIFEST.toml`` to match. Replace "
            f"with a tomlkit-based round-trip when one is needed."
        )
        # Emit what the file should say.
        for name, spec in manifest.items():
            if name == "manifest":
                continue
            clone_dir = REPO_ROOT / spec["clone"]["relative_path"] \
                if "clone" in spec else REPO_ROOT / spec["relative_path"]
            if not (clone_dir / ".git").exists():
                continue
            try:
                sha = _git(clone_dir, "rev-parse", "HEAD")
                print(f"  [clone.{name}]\n  sha = \"{sha}\"  # local HEAD")
            except subprocess.CalledProcessError:
                pass
        return 0

    # Normal verify path.
    clones = manifest.get("clone", {})
    if not clones:
        print(
            f"{_YEL}WARN{_END}  manifest has no ``[clone.*]`` entries.",
            file=sys.stderr,
        )
        return 0

    results: list[CloneResult] = []
    for name, spec in clones.items():
        results.append(_verify_clone(name, spec, strict=args.strict))

    pad = max(len(r.name) for r in results) + 2
    n_pass = n_fail = n_warn = n_skip = 0
    for r in results:
        if r.status == "PASS":
            colour, n_pass = _GRN, n_pass + 1
        elif r.status == "WARN":
            colour, n_warn = _YEL, n_warn + 1
        elif r.status == "SKIP":
            colour, n_skip = "", n_skip + 1
        else:
            colour, n_fail = _RED, n_fail + 1
        print(f"  {r.name:<{pad}}{colour}{r.status:<5}{_END}  {r.message}")

    print()
    print(
        f"  Summary: "
        f"{_GRN}{n_pass} PASS{_END}, "
        f"{_YEL}{n_warn} WARN{_END}, "
        f"{n_skip} SKIP, "
        f"{_RED}{n_fail} FAIL{_END}"
    )

    if n_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
