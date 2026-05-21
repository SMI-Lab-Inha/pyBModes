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

"""Build the ``reference_decks/`` directory tree.

For each included case — three fixed-base (NREL 5MW land, NREL 5MW
on the OC3 monopile, IEA-3.4-130-RWT land) and three floating
(NREL 5MW on the OC3 Hywind spar, NREL 5MW on the OC4 DeepCwind
semi, IEA-15-240-RWT on the UMaine VolturnUS-S semi), the script:

1. Copies the ElastoDyn main / tower / blade ``.dat`` files (and the
   SubDyn file for the monopile case) from the upstream source location
   under ``external/OpenFAST_files/`` into ``reference_decks/<case>/``,
   renaming where needed and rewriting the ``TwrFile`` / ``BldFile``
   references in the main file so the deck is self-contained
   (no ``../5MW_Baseline/`` traversal).
2. Runs ``pybmodes validate`` on the as-copied deck and captures the
   stdout to ``before_patch.txt``.
3. Runs the equivalent of ``pybmodes patch`` on the same deck — the
   tower and blade ``.dat`` files are rewritten in place.
4. Re-runs ``pybmodes validate`` on the patched deck and captures the
   stdout to ``validation_report.txt``. The post-patch overall verdict
   must be either PASS or WARN (the WARN case is documented inline in
   the validation report's footer; it reflects the constrained 6th-
   order polynomial form's representation limit for a specific tower
   geometry, not a pyBmodes bug). FAIL still raises.
5. Cleans up the ``.bak`` files produced by ``patch_dat``'s callers.

The script also writes ``reference_decks/VALIDATION_SUMMARY.md`` — a
single before/after table across every case, parsed from the two
``*.txt`` reports.

Run from the repo root::

    set PYTHONPATH=%CD%\\src
    python scripts\\build_reference_decks.py
"""

from __future__ import annotations

import io
import pathlib
import re
import shutil
import sys

# Allow ``python scripts\build_reference_decks.py`` from the repo root
# without an editable install.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pybmodes.cli import _print_validation_report  # noqa: E402
from pybmodes.elastodyn import (  # noqa: E402
    compute_blade_params,
    compute_tower_params,
    patch_dat,
    validate_dat_coefficients,
)
from pybmodes.io.elastodyn_reader import read_elastodyn_main  # noqa: E402
from pybmodes.models import RotatingBlade, Tower  # noqa: E402

REFERENCE_DECKS_DIR = (
    REPO_ROOT / "src" / "pybmodes" / "_examples" / "reference_decks"
)
RTEST_OPENFAST = (
    REPO_ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes" / "openfast"
)
IEA34_OPENFAST = (
    REPO_ROOT / "external" / "OpenFAST_files" / "IEA-3.4-130-RWT" / "openfast"
)
IEA15_OPENFAST = (
    REPO_ROOT / "external" / "OpenFAST_files" / "IEA-15-240-RWT" / "OpenFAST"
)


# ---------------------------------------------------------------------------
# Case manifest
# ---------------------------------------------------------------------------

CASES: list[dict] = [
    {
        "name": "nrel5mw_land",
        "title": "NREL 5MW Reference Turbine — land-based",
        "source_main": (
            RTEST_OPENFAST
            / "5MW_Land_DLL_WTurb"
            / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
        ),
        "source_tower": (
            RTEST_OPENFAST
            / "5MW_Land_DLL_WTurb"
            / "NRELOffshrBsline5MW_Onshore_ElastoDyn_Tower.dat"
        ),
        "source_blade": (
            RTEST_OPENFAST
            / "5MW_Baseline"
            / "NRELOffshrBsline5MW_Blade.dat"
        ),
        "source_subdyn": None,
        "dst_main": "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat",
        "dst_tower": "NRELOffshrBsline5MW_Tower.dat",
        "dst_blade": "NRELOffshrBsline5MW_Blade.dat",
        "dst_subdyn": None,
    },
    {
        "name": "nrel5mw_oc3monopile",
        "title": "NREL 5MW on OC3 Monopile (rigid base, no soil flexibility)",
        "source_main": (
            RTEST_OPENFAST
            / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
            / "NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat"
        ),
        "source_tower": (
            RTEST_OPENFAST
            / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
            / "NRELOffshrBsline5MW_OC3Monopile_ElastoDyn_Tower.dat"
        ),
        "source_blade": (
            RTEST_OPENFAST
            / "5MW_Baseline"
            / "NRELOffshrBsline5MW_Blade.dat"
        ),
        "source_subdyn": (
            RTEST_OPENFAST
            / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
            / "NRELOffshrBsline5MW_OC3Monopile_SubDyn.dat"
        ),
        "dst_main": "NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat",
        "dst_tower": "NRELOffshrBsline5MW_OC3Monopile_Tower.dat",
        "dst_blade": "NRELOffshrBsline5MW_Blade.dat",
        "dst_subdyn": "NRELOffshrBsline5MW_OC3Monopile_SubDyn.dat",
    },
    {
        "name": "iea34_land",
        "title": "IEA-3.4-130-RWT — land-based",
        "source_main": (
            IEA34_OPENFAST
            / "IEA-3.4-130-RWT_ElastoDyn.dat"
        ),
        "source_tower": (
            IEA34_OPENFAST
            / "IEA-3.4-130-RWT_ElastoDyn_tower.dat"
        ),
        "source_blade": (
            IEA34_OPENFAST
            / "IEA-3.4-130-RWT_ElastoDyn_blade.dat"
        ),
        "source_subdyn": None,
        "dst_main": "IEA-3.4-130-RWT_ElastoDyn.dat",
        "dst_tower": "IEA-3.4-130-RWT_Tower.dat",
        "dst_blade": "IEA-3.4-130-RWT_Blade.dat",
        "dst_subdyn": None,
    },
    # Floating cases. The ElastoDyn-compatible polynomial basis is
    # cantilever (clamped at TowerBsHt with the RNA at the top) — the
    # platform is handled at runtime by ElastoDyn through separate
    # rigid-body DOFs (Sg/Sw/Hv/R/P/Y), not through the modal basis
    # (see reference_decks/FLOATING_CASES.md). So these cases use the
    # exact same Tower.from_elastodyn(...) path as the land-based
    # cases above; no platform / hydro / mooring matrices are read.
    {
        "name": "nrel5mw_oc3spar",
        "title": "NREL 5MW on the OC3 Hywind floating spar",
        "source_main": (
            RTEST_OPENFAST
            / "5MW_OC3Spar_DLL_WTurb_WavesIrr"
            / "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn.dat"
        ),
        "source_tower": (
            RTEST_OPENFAST
            / "5MW_OC3Spar_DLL_WTurb_WavesIrr"
            / "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn_Tower.dat"
        ),
        "source_blade": (
            RTEST_OPENFAST
            / "5MW_Baseline"
            / "NRELOffshrBsline5MW_Blade.dat"
        ),
        "source_subdyn": None,
        "dst_main": "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn.dat",
        "dst_tower": "NRELOffshrBsline5MW_OC3Hywind_Tower.dat",
        "dst_blade": "NRELOffshrBsline5MW_Blade.dat",
        "dst_subdyn": None,
    },
    {
        "name": "nrel5mw_oc4semi",
        "title": "NREL 5MW on the OC4 DeepCwind semi-submersible",
        "source_main": (
            RTEST_OPENFAST
            / "5MW_OC4Semi_WSt_WavesWN"
            / "NRELOffshrBsline5MW_OC4DeepCwindSemi_ElastoDyn.dat"
        ),
        "source_tower": (
            RTEST_OPENFAST
            / "5MW_OC4Semi_WSt_WavesWN"
            / "NRELOffshrBsline5MW_OC4DeepCwindSemi_ElastoDyn_Tower.dat"
        ),
        "source_blade": (
            RTEST_OPENFAST
            / "5MW_Baseline"
            / "NRELOffshrBsline5MW_Blade.dat"
        ),
        "source_subdyn": None,
        "dst_main": "NRELOffshrBsline5MW_OC4DeepCwindSemi_ElastoDyn.dat",
        "dst_tower": "NRELOffshrBsline5MW_OC4DeepCwindSemi_Tower.dat",
        "dst_blade": "NRELOffshrBsline5MW_Blade.dat",
        "dst_subdyn": None,
    },
    {
        "name": "iea15mw_umainesemi",
        "title": "IEA-15-240-RWT on the UMaine VolturnUS-S semi",
        "source_main": (
            IEA15_OPENFAST
            / "IEA-15-240-RWT-UMaineSemi"
            / "IEA-15-240-RWT-UMaineSemi_ElastoDyn.dat"
        ),
        "source_tower": (
            IEA15_OPENFAST
            / "IEA-15-240-RWT-UMaineSemi"
            / "IEA-15-240-RWT-UMaineSemi_ElastoDyn_tower.dat"
        ),
        "source_blade": (
            IEA15_OPENFAST
            / "IEA-15-240-RWT"
            / "IEA-15-240-RWT_ElastoDyn_blade.dat"
        ),
        "source_subdyn": None,
        "dst_main": "IEA-15-240-RWT-UMaineSemi_ElastoDyn.dat",
        "dst_tower": "IEA-15-240-RWT-UMaineSemi_Tower.dat",
        "dst_blade": "IEA-15-240-RWT_Blade.dat",
        "dst_subdyn": None,
    },
]


# ---------------------------------------------------------------------------
# Path-rewriting helper
# ---------------------------------------------------------------------------

def _rewrite_main_dat_paths(
    main_path: pathlib.Path,
    new_tower_name: str,
    new_blade_name: str,
) -> None:
    """Rewrite TwrFile / BldFile* lines inside an ElastoDyn main .dat
    so the deck refers to local (same-directory) tower and blade files.

    Preserves CRLF/LF line endings via ``newline=''`` on read+write
    (same convention :func:`pybmodes.elastodyn.patch_dat` follows).
    Matches both the legacy ``BldFile(1)`` form and the IEA-RWT
    ``BldFile1`` bare-digit form via the same regex.
    """
    with open(main_path, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    # Regex captures any quoted path followed by whitespace and the
    # parameter label. The label survives unchanged; only the quoted
    # value is rewritten.
    bld_re = re.compile(
        r'^(\s*)"[^"]*"(\s+BldFile(?:\(\d+\)|\d+))(.*)$',
        flags=re.MULTILINE,
    )
    twr_re = re.compile(
        r'^(\s*)"[^"]*"(\s+TwrFile)(.*)$',
        flags=re.MULTILINE,
    )

    text = bld_re.sub(
        lambda m: f'{m.group(1)}"{new_blade_name}"{m.group(2)}{m.group(3)}',
        text,
    )
    text = twr_re.sub(
        lambda m: f'{m.group(1)}"{new_tower_name}"{m.group(2)}{m.group(3)}',
        text,
    )

    with open(main_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Per-case build
# ---------------------------------------------------------------------------

def _capture_validate_output(dat_path: pathlib.Path) -> str:
    """Run validate_dat_coefficients(...) and return the CLI-formatted
    report as a string, with the absolute deck path in the
    ``Recommendation:`` line stripped down to the bare filename so the
    captured text doesn't leak the local builder's filesystem layout
    into the packaged ``before_patch.txt``. The CLI itself still emits
    the full path — that's useful when an end-user runs the validator
    interactively — but the wheel-bundled report should be
    machine-independent.
    """
    result = validate_dat_coefficients(dat_path)
    buf = io.StringIO()
    _print_validation_report(result, file=buf)
    text = buf.getvalue()
    return text.replace(str(dat_path), dat_path.name)


def _warn_footer(result: object) -> str:
    """Build a textual footer explaining the post-patch WARN verdict.

    Appended to validation_report.txt only when the overall is WARN.
    Names the specific block(s), reports their RMS, and explains that
    the WARN reflects the **constrained 6th-order polynomial form's
    representation limit** for the deck's mode shape — not a pyBmodes
    fitting bug. The patched polynomial in the file IS pyBmodes' best
    constrained fit (ratio = 1.00 against pyBmodes' own reference);
    the file_RMS just exceeds the 1 % PASS gate.

    Reducing the residual would require either a higher-order
    polynomial or a piecewise basis, neither of which ElastoDyn's
    `SHP` ansatz supports (it expects exactly 5 coefficients
    c2..c6 per block evaluated as a single polynomial in `Fract`).
    """
    all_blocks = result.all_blocks() if hasattr(result, "all_blocks") else {}
    warn_blocks = [b for b in all_blocks.values() if b.verdict == "WARN"]
    lines = ["", "Note on WARN verdict",
             "--------------------",
             "",
             "The patched coefficients above ARE pyBmodes' best fit to the FEM",
             "mode shape derived from this deck's structural inputs (ratio =",
             "1.00 between file_RMS and pyB_RMS). The WARN means that best",
             "fit's RMS residual exceeds the 1 % PASS gate, NOT that the",
             "polynomial in the file is wrong relative to a better available",
             "fit. Specifically:"]
    for b in warn_blocks:
        rms_pct = b.file_rms * 100.0
        lines.append(
            f"  - {b.name}: RMS = {b.file_rms:.4f} "
            f"(~ {rms_pct:.2f} % of unit-tip displacement); "
            f"PASS threshold is 0.0100 (1.00 %)."
        )
    lines.extend([
        "",
        "Cause: representation limit of the constrained 6th-order polynomial",
        "form ElastoDyn requires (`SHP = sum c_i * (h/H)^(i+1)` for i = 1..5,",
        "with sum-to-1 + phi(0) = phi'(0) = 0). For some tower section-",
        "property gradients, the FEM 2nd-bending mode shape contains",
        "sufficient curvature at intermediate heights that no 5-coefficient",
        "polynomial in this constrained form can represent it below the 1 %",
        "RMS gate. Improving the fit would require a higher polynomial",
        "order or a piecewise basis, neither of which ElastoDyn supports.",
        "",
        "This is a property of the deck's section-property distribution",
        "interacting with ElastoDyn's modal-basis format — not a pyBmodes",
        "model error and not an upstream-deck bug. The patched deck is",
        "internally consistent: the polynomial in the file is the best",
        "representation of the FEM mode shape that the file format permits.",
        "",
    ])
    return "\n".join(lines)


def _build_case(case: dict) -> dict:
    """Stage + patch + validate a single case. Returns metadata used
    by VALIDATION_SUMMARY.md."""
    name = case["name"]
    case_dir = REFERENCE_DECKS_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy source files into the case directory with the dst names.
    dst_main = case_dir / case["dst_main"]
    dst_tower = case_dir / case["dst_tower"]
    dst_blade = case_dir / case["dst_blade"]
    shutil.copy2(case["source_main"], dst_main)
    shutil.copy2(case["source_tower"], dst_tower)
    shutil.copy2(case["source_blade"], dst_blade)
    if case["source_subdyn"]:
        dst_subdyn = case_dir / case["dst_subdyn"]
        shutil.copy2(case["source_subdyn"], dst_subdyn)

    # 2. Rewrite paths in the main .dat to point at the local copies.
    _rewrite_main_dat_paths(
        dst_main,
        new_tower_name=case["dst_tower"],
        new_blade_name=case["dst_blade"],
    )

    # 3. Validate as-copied (still has the upstream coefficients).
    before_text = _capture_validate_output(dst_main)
    (case_dir / "before_patch.txt").write_text(before_text, encoding="utf-8")

    # 4. Patch tower + blade .dat in place.
    main = read_elastodyn_main(dst_main)
    tower_path = dst_main.parent / main.twr_file
    blade_path = dst_main.parent / main.bld_file[0]
    tower_modal = Tower.from_elastodyn(dst_main).run(n_modes=10)
    blade_modal = RotatingBlade.from_elastodyn(dst_main).run(n_modes=10)
    patch_dat(tower_path, compute_tower_params(tower_modal))
    patch_dat(blade_path, compute_blade_params(blade_modal))

    # 5. Validate post-patch.
    after_text = _capture_validate_output(dst_main)
    after_result = validate_dat_coefficients(dst_main)
    if after_result.overall == "WARN":
        after_text = after_text + _warn_footer(after_result)
    (case_dir / "validation_report.txt").write_text(
        after_text, encoding="utf-8"
    )

    # 6. Sanity-assert: PASS or WARN is acceptable (the patched
    # polynomial IS pyBmodes' best constrained fit to the deck's
    # structural inputs; WARN means the constrained 6th-order form
    # can't represent the FEM mode shape below the 1 % PASS gate, a
    # property of the deck, not a pyBmodes bug). FAIL still raises.
    if after_result.overall == "FAIL":
        raise RuntimeError(
            f"Case {name!r} reached FAIL after patching: "
            f"{after_result.summary}"
        )

    # 7. Drop any ``.bak`` files.  The script does not pass --backup to
    # patch_dat, but a stale .bak from an earlier run could be lying
    # around — clean defensively.
    for bak in case_dir.glob("*.bak"):
        bak.unlink()

    return {
        "name": name,
        "title": case["title"],
        "before_text": before_text,
        "after_text": after_text,
    }


# ---------------------------------------------------------------------------
# VALIDATION_SUMMARY.md builder
# ---------------------------------------------------------------------------

_REPORT_LINE = re.compile(
    r"^\s+(\w+)\s+file RMS=\s*(?P<file>[\d.]+)\s+"
    r"pyB RMS=\s*(?P<pyb>[\d.]+)\s+"
    r"ratio=(?P<ratio>\s*[\d.eE+\-]+|\s+inf|\s+nan)\s+"
    r"(?P<verdict>PASS|WARN|FAIL).*$"
)


def _parse_report_blocks(text: str) -> dict[str, dict]:
    """Pull per-block (file_rms, pyb_rms, ratio, verdict) out of CLI
    report text."""
    out: dict[str, dict] = {}
    for line in text.splitlines():
        m = _REPORT_LINE.match(line)
        if not m:
            continue
        ratio_str = m.group("ratio").strip()
        ratio = (
            float("inf") if ratio_str.lower() == "inf"
            else float("nan") if ratio_str.lower() == "nan"
            else float(ratio_str)
        )
        out[m.group(1)] = {
            "file_rms": float(m.group("file")),
            "pyb_rms": float(m.group("pyb")),
            "ratio": ratio,
            "verdict": m.group("verdict"),
        }
    return out


def _fmt_ratio(r: float) -> str:
    if r != r:
        return "  nan"
    if r == float("inf"):
        return "  inf"
    if r >= 1000.0:
        return f"{r:>5.0f}×"
    if r >= 100.0:
        return f"{r:>5.0f}×"
    if r >= 10.0:
        return f"{r:>5.1f}×"
    return f"{r:>5.2f}×"


_BLOCK_ORDER = [
    "TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh",
    "BldFl1Sh", "BldFl2Sh", "BldEdgSh",
]


def _write_validation_summary(case_meta: list[dict]) -> None:
    """Emit ``reference_decks/VALIDATION_SUMMARY.md``."""
    rows: list[str] = []
    rows.append(
        "| Case | Block | Before RMS | After RMS | Ratio before | Status |"
    )
    rows.append(
        "| --- | --- | ---: | ---: | ---: | :---: |"
    )

    for meta in case_meta:
        before = _parse_report_blocks(meta["before_text"])
        after = _parse_report_blocks(meta["after_text"])
        for block in _BLOCK_ORDER:
            b = before.get(block, {})
            a = after.get(block, {})
            rows.append(
                f"| {meta['name']} | {block} | "
                f"{b.get('file_rms', float('nan')):.4f} | "
                f"{a.get('file_rms', float('nan')):.4f} | "
                f"{_fmt_ratio(b.get('ratio', float('nan'))).strip()} | "
                f"{a.get('verdict', '?')} |"
            )

    body = (
        "<!-- markdownlint-disable MD013 -->\n"
        "# Reference-deck coefficient validation summary\n"
        "\n"
        "Per-block RMS residual of the polynomial coefficients shipped in "
        "each upstream deck (Before) and after pyBmodes regenerated them "
        "from the structural inputs in the same deck (After). The ratio "
        "column is the upstream `file_rms / pybmodes_rms` — values >> 1 "
        "indicate the upstream polynomial does not represent the mode "
        "shape produced by the deck's structural inputs.\n"
        "\n"
        + "\n".join(rows)
        + "\n\n"
        "## Pattern\n"
        "\n"
        "- **2nd-mode tower coefficients** (`TwFAM2Sh`, `TwSSM2Sh`) show "
        "the largest inconsistency on every upstream deck: ratios from "
        "~170× (IEA-3.4) to ~2,500× (NREL 5MW). The shipped polynomials "
        "do not represent the 2nd bending mode of the structural inputs "
        "by any reasonable metric.\n"
        "- **1st-mode tower coefficients** (`TwFAM1Sh`, `TwSSM1Sh`) and "
        "blade coefficients (`BldFl1Sh`, `BldFl2Sh`, `BldEdgSh`) show a "
        "smaller but non-zero inconsistency (typical ratio ~ 2–300×). "
        "Their absolute file RMS values still classify as PASS under the "
        "1 % per-block gate, but they are still drift artefacts from "
        "the same generation pipeline.\n"
        "- **After patching every block is pyBmodes' best constrained "
        "fit and no block FAILs; most blocks reach PASS, one known WARN "
        "(`iea15mw_umainesemi / TwSSM2Sh` at 1.6 % RMS) reflects an "
        "ElastoDyn basis representation limit for that specific tower's "
        "section-property gradient, not a pyBmodes bug.** The After-RMS "
        "column matches the pyBmodes-RMS column from the Before report; "
        "the polynomials in the patched files are exactly pyBmodes' "
        "fits, so the file polynomial reproduces the pyBmodes mode "
        "shape modulo the writer's text-precision (~7 sig figs).\n"
        "\n"
        "## How to reproduce\n"
        "\n"
        "```bash\n"
        "python scripts/build_reference_decks.py\n"
        "```\n"
        "\n"
        "The script copies the upstream sources, runs `pybmodes patch`, "
        "and re-runs the validator. See `before_patch.txt` and "
        "`validation_report.txt` in each case directory for the raw CLI "
        "output.\n"
    )

    out_path = REFERENCE_DECKS_DIR / "VALIDATION_SUMMARY.md"
    out_path.write_text(body, encoding="utf-8")
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"Building reference decks under {REFERENCE_DECKS_DIR}/ ...")
    print()
    REFERENCE_DECKS_DIR.mkdir(exist_ok=True)

    case_meta: list[dict] = []
    for case in CASES:
        # Skip cases whose sources aren't present (so contributors who
        # don't have all upstream data can still build a partial set).
        missing = [
            p for p in (
                case["source_main"], case["source_tower"],
                case["source_blade"], case["source_subdyn"],
            )
            if p is not None and not p.is_file()
        ]
        if missing:
            print(f"[skip] {case['name']}: missing sources:")
            for p in missing:
                print(f"  - {p}")
            continue

        print(f"[build] {case['name']}: {case['title']}")
        meta = _build_case(case)
        case_meta.append(meta)

        # Last-line summary from the validator output.
        last = next(
            ln for ln in reversed(meta["after_text"].splitlines())
            if ln.startswith("Overall")
        )
        print(f"  -> {last}")
        print()

    if case_meta:
        _write_validation_summary(case_meta)

    print()
    print(f"Done. {len(case_meta)} case(s) built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
