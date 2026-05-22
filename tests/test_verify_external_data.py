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

"""Unit tests for ``scripts/verify_external_data.py``.

Self-contained: exercise the manifest write-back rewriter and the
required/optional SKIP-vs-FAIL policy with synthetic inputs only — no
``external/`` clones required, so these run in the default suite.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tomllib

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_external_data.py"
)
_spec = importlib.util.spec_from_file_location("verify_external_data", _SCRIPT)
assert _spec is not None and _spec.loader is not None
ved = importlib.util.module_from_spec(_spec)
# Register before exec so the @dataclass decorator can resolve the module
# via sys.modules[cls.__module__] during class processing.
sys.modules[_spec.name] = ved
_spec.loader.exec_module(ved)


_SAMPLE_MANIFEST = """\
[manifest]
schema_version = 1

[clone.alpha]
repo = "https://example.com/a.git"
sha = "OLDSHA"  # pinned 1.8.1
relative_path = "external/a"
hashes = { }

[clone.beta]
repo = "https://example.com/b.git"
sha = "BETAOLD"
relative_path = "external/b"
hashes = { "x.txt" = "deadbeef" }
"""


# ---------------------------------------------------------------------------
# _sha256 — line-ending-normalized so pins reproduce cross-platform
# ---------------------------------------------------------------------------
def test_sha256_is_line_ending_normalized(tmp_path):
    lf = tmp_path / "lf.dat"
    crlf = tmp_path / "crlf.dat"
    lf.write_bytes(b"alpha\nbeta\ngamma\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    # A CRLF (Windows) checkout and an LF (Linux CI) checkout of the same
    # content must hash identically, so --strict passes in both.
    assert ved._sha256(lf) == ved._sha256(crlf)


def test_sha256_still_detects_content_change(tmp_path):
    a = tmp_path / "a.dat"
    b = tmp_path / "b.dat"
    a.write_bytes(b"alpha\nbeta\n")
    b.write_bytes(b"alpha\nDELTA\n")
    assert ved._sha256(a) != ved._sha256(b)


# ---------------------------------------------------------------------------
# _render_inline_hashes
# ---------------------------------------------------------------------------
def test_render_inline_hashes_empty():
    assert ved._render_inline_hashes({}) == "{ }"


def test_render_inline_hashes_populated():
    rendered = ved._render_inline_hashes({"a.yaml": "abc", "b.dat": "def"})
    assert rendered == '{ "a.yaml" = "abc", "b.dat" = "def" }'


# ---------------------------------------------------------------------------
# _rewrite_manifest_text
# ---------------------------------------------------------------------------
def test_rewrite_updates_sha_and_hashes_preserving_comment():
    out = ved._rewrite_manifest_text(
        _SAMPLE_MANIFEST,
        sha_updates={"alpha": "NEWSHA"},
        hash_updates={"alpha": {"f.yaml": "abc123"}},
    )
    # sha re-pinned, trailing comment preserved.
    assert 'sha = "NEWSHA"  # pinned 1.8.1' in out
    # hashes inline table populated.
    assert 'hashes = { "f.yaml" = "abc123" }' in out
    # beta block untouched.
    assert 'sha = "BETAOLD"' in out
    assert 'hashes = { "x.txt" = "deadbeef" }' in out
    # [manifest] header key untouched.
    assert "schema_version = 1" in out


def test_rewrite_output_is_valid_toml():
    out = ved._rewrite_manifest_text(
        _SAMPLE_MANIFEST,
        sha_updates={"alpha": "NEWSHA", "beta": "BETANEW"},
        hash_updates={"alpha": {"f.yaml": "abc123"}},
    )
    parsed = tomllib.loads(out)
    assert parsed["clone"]["alpha"]["sha"] == "NEWSHA"
    assert parsed["clone"]["alpha"]["hashes"] == {"f.yaml": "abc123"}
    assert parsed["clone"]["beta"]["sha"] == "BETANEW"
    # beta hashes left as-is (not in hash_updates).
    assert parsed["clone"]["beta"]["hashes"] == {"x.txt": "deadbeef"}


def test_rewrite_noop_when_no_updates():
    out = ved._rewrite_manifest_text(_SAMPLE_MANIFEST, sha_updates={}, hash_updates={})
    assert out == _SAMPLE_MANIFEST


# ---------------------------------------------------------------------------
# _verify_clone — missing required vs optional SKIP/FAIL policy
# ---------------------------------------------------------------------------
_MISSING = {"relative_path": "external/__pybmodes_does_not_exist__", "sha": "abc123"}


def test_missing_required_under_strict_is_fail():
    result = ved._verify_clone("req", dict(_MISSING), strict=True)
    assert result.status == "FAIL"


def test_missing_required_without_strict_is_skip():
    result = ved._verify_clone("req", dict(_MISSING), strict=False)
    assert result.status == "SKIP"


def test_missing_optional_is_skip_even_under_strict():
    spec = dict(_MISSING, optional=True)
    assert ved._verify_clone("opt", spec, strict=True).status == "SKIP"
    assert ved._verify_clone("opt", spec, strict=False).status == "SKIP"


# ---------------------------------------------------------------------------
# --update fail-loud on missing declared hash_files
# ---------------------------------------------------------------------------
_GHOST = {
    "relative_path": "external/__pybmodes_does_not_exist__",
    "hash_files": ["a.yaml", "deep/b.dat"],
}


def test_collect_updates_reports_missing_hash_files():
    sha_updates, hash_updates, missing = ved._collect_updates({"ghost": dict(_GHOST)})
    # A clone that DECLARES hash_files is recorded even when nothing was
    # computable, so its table is rewritten (to ``{ }``) rather than left
    # stale — see test_update_clears_stale_hashes_when_all_missing.
    assert hash_updates == {"ghost": {}}
    assert ("ghost", "a.yaml") in missing
    assert ("ghost", "deep/b.dat") in missing


def test_collect_updates_leaves_clones_without_hash_files_untouched():
    # No declared hash_files → the clone must not appear in hash_updates,
    # so its existing ``hashes`` table is left alone.
    _, hash_updates, missing = ved._collect_updates(
        {"plain": {"relative_path": "external/__nope__"}}
    )
    assert hash_updates == {}
    assert missing == []


def test_update_clears_stale_hashes_when_all_missing():
    # Regression (static review): --update must not leave a stale hashes
    # pin behind when a declared clone has no computable files. The table
    # is rewritten to ``{ }``, dropping the obsolete entry.
    sample = (
        '[clone.ghost]\n'
        'relative_path = "external/__nope__"\n'
        'hash_files = ["a.yaml"]\n'
        'hashes = { "old.dat" = "deadbeef" }\n'
    )
    sha_updates, hash_updates, _ = ved._collect_updates(
        {"ghost": {"relative_path": "external/__nope__", "hash_files": ["a.yaml"]}}
    )
    assert hash_updates == {"ghost": {}}
    out = ved._rewrite_manifest_text(sample, sha_updates, hash_updates)
    assert "hashes = { }" in out
    assert "deadbeef" not in out


def test_update_fails_loud_on_missing_hash_files(tmp_path, monkeypatch):
    fake = tmp_path / "MANIFEST.toml"
    fake.write_text(_SAMPLE_MANIFEST, encoding="utf-8")
    monkeypatch.setattr(ved, "MANIFEST", fake)
    rc = ved._do_update({"ghost": dict(_GHOST)}, dry_run=False, allow_missing=False)
    assert rc == 1
    # Aborted before writing — manifest untouched.
    assert fake.read_text(encoding="utf-8") == _SAMPLE_MANIFEST


def test_update_allows_missing_with_flag(tmp_path, monkeypatch):
    fake = tmp_path / "MANIFEST.toml"
    fake.write_text(_SAMPLE_MANIFEST, encoding="utf-8")
    monkeypatch.setattr(ved, "MANIFEST", fake)
    # Escape hatch: the unresolvable path is a warning, the update proceeds
    # (nothing computable here, so it's a no-op write) and exits 0.
    rc = ved._do_update({"ghost": dict(_GHOST)}, dry_run=False, allow_missing=True)
    assert rc == 0


# ---------------------------------------------------------------------------
# tagged-archive entries: no git HEAD, so --strict must verify content
# hashes — a present archive with empty hashes is UNVERIFIED, not PASS.
# ---------------------------------------------------------------------------
def _archive_spec(**over) -> dict:
    spec = {
        "relative_path": "external/arc",
        "sha": "v3.00.00",
        "fetch_at": "tagged-archive",
        "optional": True,
        "hashes": {},
    }
    spec.update(over)
    return spec


def test_tagged_archive_present_without_hashes_warns_under_strict(tmp_path, monkeypatch):
    """Regression (static review): a present tagged-archive clone with no
    content hash pins must WARN under --strict — it has no git HEAD to
    check, so an empty ``hashes`` table means it was never verified. The
    old code returned a misleading PASS."""
    monkeypatch.setattr(ved, "REPO_ROOT", tmp_path)
    (tmp_path / "external" / "arc").mkdir(parents=True)
    spec = _archive_spec()
    assert ved._verify_clone("arc", spec, strict=True).status == "WARN"
    # Without --strict it stays a lenient PASS (archive present).
    assert ved._verify_clone("arc", spec, strict=False).status == "PASS"


def test_tagged_archive_with_matching_hash_passes_under_strict(tmp_path, monkeypatch):
    """A tagged-archive entry whose pinned content hash matches the file
    PASSes under --strict (the hash check is the verification)."""
    monkeypatch.setattr(ved, "REPO_ROOT", tmp_path)
    d = tmp_path / "external" / "arc"
    d.mkdir(parents=True)
    (d / "x.txt").write_bytes(b"hello\n")
    spec = _archive_spec(hashes={"x.txt": ved._sha256(d / "x.txt")})
    assert ved._verify_clone("arc", spec, strict=True).status == "PASS"


def test_tagged_archive_with_mismatched_hash_fails_under_strict(tmp_path, monkeypatch):
    """A tagged-archive entry whose pinned hash does NOT match FAILs."""
    monkeypatch.setattr(ved, "REPO_ROOT", tmp_path)
    d = tmp_path / "external" / "arc"
    d.mkdir(parents=True)
    (d / "x.txt").write_bytes(b"hello\n")
    spec = _archive_spec(hashes={"x.txt": "0" * 64})
    assert ved._verify_clone("arc", spec, strict=True).status == "FAIL"
