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

"""Verify (and optionally clone) the ``external/`` tree against the manifest.

``external/MANIFEST.toml`` is the single source of truth for the upstream
reference decks the integration-test track consumes. Each ``[clone.<name>]``
block declares a repository, a pinned git SHA, the ``relative_path`` the
tests resolve it at, an ``optional`` flag, and (eventually) a ``hashes``
table of SHA-256 file pins.

Three modes
-----------

``--clone``
    Clone every *required* GitHub-hosted entry that is missing, checking
    out its pinned SHA (shallow fetch-by-commit). Optional entries are
    skipped unless ``--include-optional`` is given. This is the step the
    ``validation.yml`` CI workflow runs so the manifest — not a hand-kept
    list of clone steps — drives what gets fetched.

``--strict`` (verify, default mode)
    For every clone: confirm the directory exists, is a git work tree, and
    its ``HEAD`` matches the pinned SHA; then recompute the SHA-256 of every
    file in the ``hashes`` table and compare. Under ``--strict`` a *missing
    required* clone is a hard FAIL (not a silent SKIP) so a validation run
    can't look green while a pinned, non-optional entry was never checked.
    Optional entries (cross-comparison references, the non-redistributable
    BModes archive) stay SKIP-tolerant.

``--update``
    MAINTAINER ONLY. Rewrite ``MANIFEST.toml`` in place: re-pin every
    clone's ``sha`` to its local ``HEAD`` and recompute the ``hashes``
    table from each clone's declared ``hash_files`` list. Run this on a
    machine that already has every clone at the intended version. Pair with
    ``--dry-run`` to print the rewritten manifest to stdout without writing.

    A declared ``hash_files`` path that can't be resolved (typo, or an
    absent clone) **aborts the update and writes nothing** — a mistyped
    load-bearing path must not silently leave a stale ``hashes`` table in
    place. Pass ``--allow-missing-hashes`` to downgrade that to a warning
    and write only the computable subset.

    Hashes are computed over line-ending-normalized content (CRLF -> LF),
    so ``--update`` may be run on any platform: a Windows checkout and a
    fresh Linux CI checkout of the same text deck yield the same hash.

Exit codes: 0 if every entry is acceptable (PASS / WARN / SKIP), 1 if any
entry FAILs. Designed as the release-checklist gate for the integration
track — a green ``--strict`` run means anyone with the manifest can
reproduce the published 0.01 % validation tolerance.

Usage::

    python scripts/verify_external_data.py
    python scripts/verify_external_data.py --strict
    python scripts/verify_external_data.py --clone
    python scripts/verify_external_data.py --clone --include-optional
    python scripts/verify_external_data.py --update            # MAINTAINER
    python scripts/verify_external_data.py --update --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
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

_GIT_URL_PREFIXES = ("http://", "https://", "git@", "ssh://")


@dataclass
class CloneResult:
    name: str
    status: str             # "PASS" / "WARN" / "FAIL" / "SKIP"
    message: str


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True,
    ).strip()


def _run_git(*args: str) -> None:
    """Run a git command, surfacing combined output on failure."""
    subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _sha256(path: Path) -> str:
    """SHA-256 of *path* with line endings normalized to LF.

    Hashing line-ending-normalized content (CRLF -> LF) makes the pin
    reproducible across platforms: a Windows checkout (git ``autocrlf``)
    and a fresh Linux CI checkout of the same text deck produce the same
    hash, so ``--strict`` passes in both — the pin captures *content*,
    not the incidental EOL of the checkout. ``hash_files`` entries are
    text decks (.yaml / .dat); a binary file must not be declared as a
    ``hash_files`` entry (normalization would corrupt its hash).
    """
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _is_git_source(repo: str) -> bool:
    return repo.startswith(_GIT_URL_PREFIXES)


def _is_optional(spec: dict) -> bool:
    return bool(spec.get("optional", False))


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def _verify_clone(
    name: str, spec: dict, *, strict: bool,
) -> CloneResult:
    clone_dir = REPO_ROOT / spec["relative_path"]
    rel = clone_dir.relative_to(REPO_ROOT)
    pinned_sha = spec.get("sha", "TBD")

    if not clone_dir.is_dir():
        if _is_optional(spec):
            return CloneResult(name, "SKIP", f"optional clone {rel} not present")
        # Required clone missing. Under --strict this is a release blocker:
        # a green validation artifact must not hide an unchecked pinned
        # entry. Without --strict keep it a SKIP for local-dev convenience.
        if strict:
            return CloneResult(
                name, "FAIL",
                f"required clone {rel} is MISSING — clone it "
                f"(`verify_external_data.py --clone`) or mark "
                f"`optional = true` in the manifest if it is not "
                f"CI-required.",
            )
        return CloneResult(
            name, "SKIP",
            f"required clone {rel} not present (run --clone to fetch, "
            f"or --strict to make this a hard failure)",
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
                f"archive-only clone present at {rel}; pass --strict "
                f"to verify file hashes.",
            )
        # Under --strict an archive-only entry has no git HEAD to check, so
        # content hash pins are the ONLY thing that verifies it. An empty
        # ``hashes`` table means the present clone was never actually
        # checked — report WARN rather than the misleading PASS the
        # fall-through would otherwise produce. Declare ``hash_files`` and
        # run --update on a maintainer machine to pin the hashes.
        if not spec.get("hashes"):
            return CloneResult(
                name, "WARN",
                f"archive-only clone at {rel} is present but UNVERIFIED — "
                f"no content hashes pinned (tagged-archive entries have no "
                f"git HEAD to check). Declare ``hash_files`` and run "
                f"`verify_external_data.py --update` on a maintainer "
                f"machine to pin them.",
            )
        # fall through to hash check
    else:
        if not (clone_dir / ".git").exists():
            return CloneResult(
                name, "FAIL",
                f"{rel} exists but is not a git work tree.",
            )
        try:
            head = _git(clone_dir, "rev-parse", "HEAD")
        except subprocess.CalledProcessError as err:
            return CloneResult(
                name, "FAIL",
                f"git rev-parse HEAD failed in {rel}: {err}",
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
                    f"hashed file missing: {(clone_dir / rel_path).relative_to(REPO_ROOT)}",
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


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------
def _shallow_fetch_checkout(repo: str, sha: str, clone_dir: Path) -> None:
    """Fetch only the pinned commit (depth 1) and detach onto it.

    GitHub serves a reachable commit by SHA, so a depth-1 fetch of the
    exact pin is far cheaper than a full ``git clone`` of large repos
    (WISDEM, the IEA RWTs). Falls back to a full fetch if the server
    rejects a by-SHA request.
    """
    _run_git("init", "-q", str(clone_dir))
    _run_git("-C", str(clone_dir), "remote", "add", "origin", repo)
    try:
        _run_git("-C", str(clone_dir), "fetch", "-q", "--depth", "1", "origin", sha)
        _run_git("-C", str(clone_dir), "checkout", "-q", "FETCH_HEAD")
    except subprocess.CalledProcessError:
        _run_git("-C", str(clone_dir), "fetch", "-q", "origin")
        _run_git("-C", str(clone_dir), "checkout", "-q", sha)


def _clone_one(
    name: str, spec: dict, *, include_optional: bool,
) -> CloneResult:
    repo = spec.get("repo", "")
    clone_dir = REPO_ROOT / spec["relative_path"]
    rel = clone_dir.relative_to(REPO_ROOT)

    if spec.get("fetch_at") == "tagged-archive" or not _is_git_source(repo):
        return CloneResult(
            name, "SKIP",
            f"not a git clone source ({repo[:48]}) — fetch manually.",
        )
    if _is_optional(spec) and not include_optional:
        return CloneResult(
            name, "SKIP", "optional — pass --include-optional to clone.",
        )
    if clone_dir.is_dir():
        return CloneResult(name, "PASS", f"already present at {rel}")

    sha = spec.get("sha", "")
    if sha in ("", "TBD"):
        return CloneResult(name, "FAIL", "no SHA pin to clone at.")

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        _shallow_fetch_checkout(repo, sha, clone_dir)
    except subprocess.CalledProcessError as err:
        # Remove the partial clone so a re-run starts clean.
        shutil.rmtree(clone_dir, ignore_errors=True)
        return CloneResult(name, "FAIL", f"clone failed: {err}")
    return CloneResult(name, "PASS", f"cloned {repo} @ {sha[:12]}")


# ---------------------------------------------------------------------------
# Update (manifest write-back)
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(r"^\s*\[clone\.([^\]]+)\]\s*$")


def _render_inline_hashes(hashes: dict[str, str]) -> str:
    if not hashes:
        return "{ }"
    items = ", ".join(f'"{k}" = "{v}"' for k, v in hashes.items())
    return "{ " + items + " }"


def _rewrite_manifest_text(
    text: str,
    sha_updates: dict[str, str],
    hash_updates: dict[str, dict[str, str]],
) -> str:
    """Return ``text`` with each clone's ``sha`` / ``hashes`` line rewritten.

    Comment-preserving: only the value side of the matched key lines is
    replaced. Trailing ``# …`` comments on the ``sha`` line are kept.
    Implemented with stdlib only (no tomlkit dependency) by editing the
    two well-known key lines within each ``[clone.<name>]`` block.
    """
    out: list[str] = []
    current: str | None = None
    for line in text.splitlines(keepends=True):
        section = _SECTION_RE.match(line)
        if section:
            current = section.group(1)
            out.append(line)
            continue
        if current is not None and "=" in line:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            key = stripped.split("=", 1)[0].strip()
            if key == "sha" and current in sha_updates:
                comment = ""
                hash_idx = line.find("#")
                if hash_idx != -1:
                    comment = "  " + line[hash_idx:].rstrip("\n")
                out.append(f'{indent}sha = "{sha_updates[current]}"{comment}\n')
                continue
            if key == "hashes" and current in hash_updates:
                out.append(
                    f"{indent}hashes = {_render_inline_hashes(hash_updates[current])}\n"
                )
                continue
        out.append(line)
    return "".join(out)


def _collect_updates(
    clones: dict,
) -> tuple[dict[str, str], dict[str, dict[str, str]], list[tuple[str, str]]]:
    """Return ``(sha_updates, hash_updates, missing)``.

    ``missing`` is the list of ``(clone_name, declared_path)`` pairs whose
    ``hash_files`` entry could not be resolved (typo'd path or absent
    clone). The caller decides whether that is fatal — see ``_do_update``.
    """
    sha_updates: dict[str, str] = {}
    hash_updates: dict[str, dict[str, str]] = {}
    missing: list[tuple[str, str]] = []
    for name, spec in clones.items():
        clone_dir = REPO_ROOT / spec["relative_path"]
        if spec.get("fetch_at") != "tagged-archive" and (clone_dir / ".git").exists():
            try:
                sha_updates[name] = _git(clone_dir, "rev-parse", "HEAD")
            except subprocess.CalledProcessError:
                print(f"{_YEL}WARN{_END}  {name}: could not read HEAD; sha left as-is.")
        declared = spec.get("hash_files", [])
        computed: dict[str, str] = {}
        for rel in declared:
            target = clone_dir / rel
            if target.is_file():
                computed[rel] = _sha256(target)
            else:
                missing.append((name, rel))
        # Record every clone that DECLARES hash_files — even when the
        # computable set is empty — so its ``hashes`` table is rewritten
        # to reflect exactly what was hashable. Omitting an empty result
        # would leave a stale ``hashes`` pin in place (the "write the
        # computable subset" contract must also clear an emptied set).
        # Clones with no hash_files are left untouched.
        if declared:
            hash_updates[name] = computed
    return sha_updates, hash_updates, missing


def _do_update(clones: dict, *, dry_run: bool, allow_missing: bool) -> int:
    sha_updates, hash_updates, missing = _collect_updates(clones)

    # Fail loud on a declared-but-unresolvable hash_files path. A silent
    # success here could leave a stale (or empty) ``hashes`` table behind
    # after a maintainer mistypes a load-bearing path — exactly the
    # release-grade failure mode we want to refuse. ``--allow-missing-
    # hashes`` is the explicit escape hatch (writes the computable subset).
    if missing:
        for name, rel in missing:
            print(f"{_YEL}WARN{_END}  {name}: declared hash_files not found: {rel}")
        if not allow_missing:
            print(
                f"{_RED}FAIL{_END}  --update aborted: {len(missing)} declared "
                f"hash_files path(s) could not be resolved (see WARN lines). "
                f"Fix the path(s) in MANIFEST.toml, ensure the clone is "
                f"present at its pinned SHA, or pass --allow-missing-hashes "
                f"to write only the computable subset. Nothing written.",
                file=sys.stderr,
            )
            return 1

    text = MANIFEST.read_text(encoding="utf-8")
    new_text = _rewrite_manifest_text(text, sha_updates, hash_updates)
    n_sha = len(sha_updates)
    n_hash = sum(len(v) for v in hash_updates.values())
    if dry_run:
        # The manifest carries non-ASCII characters; write UTF-8 bytes
        # directly so a Windows cp1252 stdout doesn't choke. Falls back
        # to a text write when stdout has no binary buffer (e.g. under
        # a test capture).
        buf = getattr(sys.stdout, "buffer", None)
        if buf is not None:
            buf.write(new_text.encode("utf-8"))
            buf.flush()
        else:
            sys.stdout.write(new_text)
        print(
            f"\n{_YEL}--dry-run{_END}  would re-pin {n_sha} SHA(s) and "
            f"{n_hash} file hash(es); nothing written.",
            file=sys.stderr,
        )
        return 0
    if new_text == text:
        print(f"{_GRN}OK{_END}  manifest already matches local clones; nothing to write.")
        return 0
    MANIFEST.write_text(new_text, encoding="utf-8")
    print(
        f"{_GRN}WROTE{_END}  {MANIFEST.relative_to(REPO_ROOT)} — "
        f"re-pinned {n_sha} SHA(s) and {n_hash} file hash(es). "
        f"Review the diff before committing."
    )
    return 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_table(results: list[CloneResult]) -> int:
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
    return 1 if n_fail else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strict", action="store_true",
        help="Recompute SHA-256 of every manifest ``hashes`` file AND treat "
             "a missing required (non-optional) clone as FAIL, not SKIP.",
    )
    p.add_argument(
        "--clone", action="store_true",
        help="Clone every missing required GitHub-hosted entry at its "
             "pinned SHA (shallow). Use --include-optional to clone the "
             "optional cross-comparison references too.",
    )
    p.add_argument(
        "--include-optional", action="store_true",
        help="With --clone, also fetch entries marked ``optional = true``.",
    )
    p.add_argument(
        "--update", action="store_true",
        help="MAINTAINER ONLY: rewrite MANIFEST.toml with current HEAD SHAs "
             "and recomputed ``hash_files`` hashes.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="With --update, print the rewritten manifest to stdout instead "
             "of writing it.",
    )
    p.add_argument(
        "--allow-missing-hashes", action="store_true",
        help="With --update, downgrade an unresolvable ``hash_files`` path "
             "from a hard failure to a warning and write the computable "
             "subset (default: --update fails loud and writes nothing).",
    )
    args = p.parse_args()

    if not MANIFEST.is_file():
        print(f"{_RED}FAIL{_END}  manifest not found: {MANIFEST}", file=sys.stderr)
        return 2

    with MANIFEST.open("rb") as fh:
        manifest = tomllib.load(fh)

    clones = manifest.get("clone", {})
    if not clones:
        print(
            f"{_YEL}WARN{_END}  manifest has no ``[clone.*]`` entries.",
            file=sys.stderr,
        )
        return 0

    if args.update:
        return _do_update(
            clones, dry_run=args.dry_run, allow_missing=args.allow_missing_hashes,
        )

    if args.clone:
        results = [
            _clone_one(name, spec, include_optional=args.include_optional)
            for name, spec in clones.items()
        ]
        return _print_table(results)

    # Normal verify path.
    results = [_verify_clone(name, spec, strict=args.strict) for name, spec in clones.items()]
    return _print_table(results)


if __name__ == "__main__":
    sys.exit(main())
