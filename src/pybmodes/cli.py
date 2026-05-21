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

"""Command-line interface for pyBmodes.

Exposes seven subcommands:

* ``pybmodes validate <main.dat>`` — coefficient-consistency report for
  an OpenFAST ElastoDyn deck. Compares the polynomial blocks shipped in
  the deck against pyBmodes' own fits to the FEM mode shapes produced
  by the deck's structural inputs.
* ``pybmodes patch <main.dat> [--backup]`` — regenerate the polynomial
  blocks in the deck's tower and blade ``.dat`` files in place from the
  pyBmodes fits. Optional ``--backup`` saves a ``.bak`` copy of each
  modified file first.
* ``pybmodes campbell <input> --rated-rpm R --max-rpm M [--orders 1,2,3,6,9]
  [--out PATH]`` — sweep a blade across rotor speeds 0..max_rpm and emit a
  Campbell diagram (PNG by default) plus a per-step CSV summary. Accepts
  either a ``.bmi`` deck or an ElastoDyn main ``.dat``.
* ``pybmodes batch <root> [--validate --patch --out OUT]`` — walk a
  directory tree for ElastoDyn main decks, run validate / patch per
  deck, write a per-deck report and a summary CSV.
* ``pybmodes report <main.dat> [--format md|html|csv] [--campbell]`` —
  one-shot bundled report covering modal solve, coefficient validation,
  and an optional Campbell sweep.
* ``pybmodes windio <ontology.yaml | RWT-dir> [--format md|html|csv]
  [--campbell] [--water-depth M]`` — the one-click WISDEM/WindIO
  entry point. Reads a WindIO ontology ``.yaml`` (or scans an RWT
  directory for one), auto-discovers any companion
  HydroDyn/MoorDyn/ElastoDyn decks scoped to that turbine root, and
  solves the composite-layup blade + tubular tower + (for a
  ``floating_platform``) the coupled platform rigid-body modes, then
  emits the bundled report (+ optional Campbell PNG/CSV). With the
  companion decks present the floating platform is the industry-grade
  deck-backed coupled model; without them it degrades to a
  ``UserWarning``-labelled screening preview.
* ``pybmodes examples --copy DIR [--kind all|samples|decks]`` — vendor
  ``sample_inputs/`` and/or ``reference_decks/`` from the bundled
  ``pybmodes._examples`` package into a user-supplied directory, so
  wheel-installed users can seed a working tree without keeping the
  full repo checkout around.

The script entry point is wired up in ``pyproject.toml`` as
``pybmodes = "pybmodes.cli:main"``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> int:
    """Thin CLI wrapper — delegates to
    :func:`pybmodes.workflows.run_validate` and translates the
    typed result into stdout / stderr + exit code."""
    from pybmodes.workflows import run_validate

    try:
        result = run_validate(args.dat_file)
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    for line in result.messages:
        print(line)
    for line in result.errors:
        print(line, file=sys.stderr)
    return result.exit_code


def _cmd_patch(args: argparse.Namespace) -> int:
    """Thin CLI wrapper — translates argparse flags into a call to
    :func:`pybmodes.workflows.run_patch` and renders the typed result.

    The two output-destination aliases (``--output-dir`` and ``--output``)
    are reconciled here, since the underlying workflow takes a single
    ``output_dir`` keyword: identical values fold to one, different
    values are rejected as ambiguous user error (exit code 2).
    """
    from pybmodes.workflows import run_patch

    if (
        args.output is not None
        and args.output_dir is not None
        and pathlib.Path(args.output) != pathlib.Path(args.output_dir)
    ):
        print(
            "error: --output and --output-dir were given different "
            f"paths ({args.output!r} vs {args.output_dir!r}); they are "
            "aliases — pass only one (or the same value)",
            file=sys.stderr,
        )
        return 2

    output_target = args.output_dir or args.output

    try:
        result = run_patch(
            args.dat_file,
            n_modes=args.n_modes,
            backup=args.backup,
            output_dir=output_target,
            dry_run=args.dry_run,
            diff=args.diff,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # Argument-validation error (e.g. output_dir + dry_run combo).
        # Rephrase using the CLI flag names the user actually typed.
        msg = str(exc).replace("output_dir", "--output / --output-dir")
        msg = msg.replace("dry_run", "--dry-run").replace("diff", "--diff")
        print(f"error: {msg}", file=sys.stderr)
        return 2

    for line in result.messages:
        print(line)
    for line in result.errors:
        print(line, file=sys.stderr)
    return result.exit_code


def _cmd_campbell(args: argparse.Namespace) -> int:
    """Run a rotor-speed sweep and write a Campbell diagram + CSV."""
    import numpy as np

    from pybmodes.campbell import campbell_sweep, plot_campbell

    src = pathlib.Path(args.input).resolve()
    if not src.is_file():
        print(f"error: file not found: {src}", file=sys.stderr)
        return 2

    try:
        orders = [int(x) for x in args.orders.split(",") if x.strip()]
    except ValueError:
        print(f"error: --orders must be a comma-separated list of integers; "
              f"got {args.orders!r}", file=sys.stderr)
        return 2
    if not orders:
        print("error: --orders must list at least one integer", file=sys.stderr)
        return 2

    if args.max_rpm <= 0.0:
        print(f"error: --max-rpm must be > 0; got {args.max_rpm}", file=sys.stderr)
        return 2
    if args.n_steps < 2:
        print(f"error: --n-steps must be >= 2; got {args.n_steps}", file=sys.stderr)
        return 2

    rpm = np.linspace(0.0, args.max_rpm, args.n_steps)
    tower_input = pathlib.Path(args.tower).resolve() if args.tower else None
    print(f"Campbell sweep: {src.name}")
    print(f"  rpm grid       : 0..{args.max_rpm} ({args.n_steps} points)")
    print(f"  blade modes    : {args.n_blade_modes}")
    print(f"  tower modes    : {args.n_tower_modes}")
    if tower_input is not None:
        print(f"  tower override : {tower_input}")
    result = campbell_sweep(
        src,
        rpm,
        n_blade_modes=args.n_blade_modes,
        n_tower_modes=args.n_tower_modes,
        tower_input=tower_input,
    )

    out_path = pathlib.Path(args.out).resolve() if args.out else \
        src.with_name(src.stem + "_campbell.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ``CampbellResult.to_csv()`` instead of a hand-rolled
    # ``np.savetxt`` so the CLI's CSV output carries the per-step MAC
    # tracking-confidence columns alongside the frequencies (the
    # canonical schema). Hand-rolling here used to drop those columns.
    csv_path = out_path.with_suffix(".csv")
    result.to_csv(csv_path)
    print(f"  wrote {csv_path}")

    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        pass

    fig = plot_campbell(result, excitation_orders=orders, rated_rpm=args.rated_rpm)
    fig.savefig(out_path)
    print(f"  wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Batch subcommand — validate / patch every ElastoDyn deck under a root
# ---------------------------------------------------------------------------

def _cmd_batch(args: argparse.Namespace) -> int:
    """Thin CLI wrapper — delegates to
    :func:`pybmodes.workflows.run_batch` and renders the typed result."""
    from pybmodes.workflows import run_batch

    try:
        result = run_batch(
            args.root,
            args.out,
            kind=args.kind,
            validate=args.validate,
            patch=args.patch,
            n_modes=args.n_modes,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for line in result.messages:
        print(line)
    for line in result.errors:
        print(line, file=sys.stderr)
    return result.exit_code


# ---------------------------------------------------------------------------
# Examples subcommand — vendor bundled sample inputs / reference decks
# ---------------------------------------------------------------------------

_EXAMPLE_BUNDLES = {
    # bundle-name -> (sub-directory under pybmodes/_examples/,
    #                 human description)
    "samples": ("sample_inputs",
                "analytical-reference cases + 7 RWT samples"),
    "decks":   ("reference_decks",
                "6 patched ElastoDyn decks (land + monopile + floating)"),
}


def _resolve_examples_root() -> pathlib.Path:
    """Locate ``pybmodes/_examples/`` on the installed package.

    Both wheel-installed and source-installed users find the bundle
    tree alongside the imported ``pybmodes`` package — wheel users
    via the data ``setuptools.package-data`` vendored it as, source
    users via the literal ``src/pybmodes/_examples/`` directory.
    """
    import pybmodes
    pkg_dir = pathlib.Path(pybmodes.__file__).resolve().parent
    return pkg_dir / "_examples"


def _cmd_examples(args: argparse.Namespace) -> int:
    """Thin CLI wrapper — delegates to
    :func:`pybmodes.workflows.run_examples_copy` and translates the
    typed result into stdout / stderr + exit code."""
    from pybmodes.workflows import run_examples_copy

    result = run_examples_copy(args.copy, kind=args.kind, force=args.force)
    for line in result.messages:
        print(line)
    for line in result.errors:
        print(line, file=sys.stderr)
    return result.exit_code


# ---------------------------------------------------------------------------
# Report subcommand — bundled per-deck analysis report
# ---------------------------------------------------------------------------

def _cmd_report(args: argparse.Namespace) -> int:
    """Thin CLI wrapper — delegates to
    :func:`pybmodes.workflows.run_report` and renders the typed result."""
    from pybmodes.workflows import run_report

    try:
        result = run_report(
            args.dat_file,
            args.out,
            n_modes=args.n_modes,
            format=args.format,
            validate=args.validate,
            campbell=args.campbell,
            max_rpm=args.max_rpm,
            n_steps=args.n_steps,
            n_blade_modes=args.n_blade_modes,
            n_tower_modes=args.n_tower_modes,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for line in result.messages:
        print(line)
    for line in result.errors:
        print(line, file=sys.stderr)
    return result.exit_code


def _discover_windio_inputs(path: pathlib.Path) -> dict:
    """Resolve a WindIO ``.yaml`` and any companion OpenFAST decks.

    ``path`` may be the ontology ``.yaml`` itself or an RWT directory
    (the ``IEA-*-RWT`` layout). Companion HydroDyn / MoorDyn /
    ElastoDyn-main decks are auto-discovered so the floating platform
    uses the **industry-grade** deck-fallback by default (see
    :meth:`pybmodes.models.Tower.from_windio_floating`). Returns a
    dict with ``yaml`` and optional ``hydrodyn`` / ``moordyn`` /
    ``elastodyn`` paths (``None`` when absent → that leg drops to the
    labelled screening preview)."""
    path = pathlib.Path(path)
    if path.is_file():
        yaml_path = path
    elif path.is_dir():
        cands = sorted(
            p for p in path.rglob("*.yaml")
            if "components:" in p.read_text(errors="ignore")[:4000]
            and "OpenFAST" not in str(p) and "openfast" not in str(p)
        )
        if not cands:
            raise FileNotFoundError(
                f"no WindIO ontology .yaml found under {path}"
            )
        yaml_path = cands[0]
    else:
        raise FileNotFoundError(f"WindIO input not found: {path}")

    # Auto-discovery is scoped to a bona-fide *turbine root*: the
    # directory the user passed, or the nearest ancestor (≤ 4 levels
    # up from the yaml) that owns an ``OpenFAST``/``openfast`` tree.
    # A bare yaml sitting in some scratch / user directory yields NO
    # decks (→ the labelled screening preview) — we must never
    # recursively scan an arbitrary parent (it could be a huge user
    # profile, and would wrongly pull a different turbine's / r-test's
    # decks anyway).
    turbine_root: pathlib.Path | None = None
    if path.is_dir():
        turbine_root = path
    else:
        for anc in list(yaml_path.parents)[:4]:
            if (anc / "OpenFAST").is_dir() or (anc / "openfast").is_dir():
                turbine_root = anc
                break

    if turbine_root is None:
        return {"yaml": yaml_path, "hydrodyn": None,
                "moordyn": None, "elastodyn": None}

    # Prefer decks for the configuration matching the ontology's
    # floating-ness (a VolturnUS-S floating yaml wants the UMaineSemi
    # decks, not the Monopile ones). NB: ``floating_platform:`` sits
    # *after* the large blade/tower blocks, so the whole file must be
    # scanned — a head-only check mis-detects every real RWT yaml.
    floating = "floating_platform:" in yaml_path.read_text(
        errors="ignore")
    pref = (("semi", "spar", "umaine", "volturn", "floating", "hywind")
            if floating
            else ("monopile", "land", "onshore", "fixed", "tower"))

    def _rglob_safe(root: pathlib.Path, pattern: str):
        """``rglob`` that tolerates directories vanishing mid-scan
        (Windows temp / cache churn) and unreadable subtrees."""
        out: list[pathlib.Path] = []
        try:
            for p in root.rglob(pattern):
                out.append(p)
        except (FileNotFoundError, PermissionError, OSError):
            pass
        return out

    def _find(pattern: str,
              exclude: tuple[str, ...] = ()) -> pathlib.Path | None:
        hits = [
            p for p in _rglob_safe(turbine_root, pattern)
            if not any(x in p.name.lower() for x in exclude)
            and "r-test" not in p.parts
        ]
        if not hits:
            return None
        preferred = [
            p for p in hits
            if any(t in str(p).lower() for t in pref)
        ]
        pool = preferred or hits
        return sorted(pool, key=lambda p: len(str(p)))[0]

    return {
        "yaml": yaml_path,
        "hydrodyn": _find("*HydroDyn*.dat"),
        "moordyn": _find("*MoorDyn*.dat"),
        # ElastoDyn *main* deck only (exclude _tower / _blade files).
        "elastodyn": _find("*ElastoDyn.dat", exclude=("tower", "blade")),
    }


def _cmd_windio(args: argparse.Namespace) -> int:
    """One-click WindIO: discover the ontology + companion decks,
    solve tower + blade + (floating) coupled platform, optionally
    sweep a Campbell diagram, and emit a bundled report (+ Campbell
    PNG/CSV). The companion decks make the floating platform
    industry-grade by default; without them it is a labelled
    screening preview."""
    import numpy as np

    from pybmodes.io.windio import _dup_anchor_loader, _require_yaml
    from pybmodes.models import RotatingBlade, Tower
    from pybmodes.report import generate_report

    try:
        inp = _discover_windio_inputs(pathlib.Path(args.input).resolve())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    yaml_path = inp["yaml"]
    print(f"windio: ontology {yaml_path}")
    for k in ("hydrodyn", "moordyn", "elastodyn"):
        tag = inp[k].name if inp[k] else "— (screening preview)"
        print(f"  companion {k:9s}: {tag}")

    yaml = _require_yaml()
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))
    comps = doc.get("components", {})
    is_floating = "floating_platform" in comps

    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- blade -----------------------------------------------------------
    blade_params = None
    if "blade" in comps:
        print("  solving blade (composite reduction)…")
        try:
            from pybmodes.elastodyn import compute_blade_params
            bl = RotatingBlade.from_windio(yaml_path)
            blade_modal = bl.run(n_modes=args.n_modes, check_model=False)
            blade_params = compute_blade_params(blade_modal)
        except Exception as exc:  # noqa: BLE001
            print(f"  blade skipped: {type(exc).__name__}: {exc}")

    # --- tower / coupled platform ---------------------------------------
    if is_floating:
        tier = ("industry-grade (deck-backed)"
                if all(inp[k] for k in
                       ("hydrodyn", "moordyn", "elastodyn"))
                else "SCREENING preview (missing decks)")
        print(f"  solving coupled floating tower+platform [{tier}]…")
        model = Tower.from_windio_floating(
            yaml_path,
            water_depth=args.water_depth,
            hydrodyn_dat=inp["hydrodyn"],
            moordyn_dat=inp["moordyn"],
            elastodyn_dat=inp["elastodyn"],
        )
    else:
        print("  solving tower (cantilever)…")
        model = Tower.from_windio(yaml_path)
    modal = model.run(n_modes=args.n_modes, check_model=False)

    # --- Campbell (reuses the validated rotor-speed sweep on the
    #     discovered ElastoDyn deck when present) ---------------------
    campbell = None
    if args.campbell and inp["elastodyn"] is not None:
        from pybmodes.campbell import campbell_sweep
        print(f"  Campbell sweep 0–{args.max_rpm} rpm "
              f"(via {inp['elastodyn'].name})…")
        campbell = campbell_sweep(
            inp["elastodyn"],
            np.linspace(0.0, args.max_rpm, args.n_steps),
            n_blade_modes=args.n_blade_modes,
            n_tower_modes=args.n_tower_modes,
        )
        if args.format != "csv":
            try:
                from pybmodes.campbell import plot_campbell

                # For a floating turbine overlay the 6 platform
                # rigid-body modes (rotor-speed-independent) from the
                # already-solved coupled model, named off
                # ModalResult.mode_labels — the BModes-cross-validated
                # surge/sway/heave/roll/pitch/yaw set.
                plat = None
                if is_floating and modal.mode_labels is not None:
                    plat = [
                        (lbl, float(modal.frequencies[i]))
                        for i, lbl in enumerate(modal.mode_labels)
                        if lbl is not None
                    ]
                    plat = plat or None
                png = out.with_suffix(".campbell.png")
                _cfig = plot_campbell(
                    campbell,
                    platform_modes=plat,
                    log_freq=plat is not None,
                )
                _cfig.savefig(png, dpi=120)
                import matplotlib.pyplot as _plt
                _plt.close(_cfig)        # don't leak figures in batch
                print(f"  wrote {png}")
            except Exception as exc:  # noqa: BLE001
                print(f"  campbell plot skipped: {exc}")
        csv = out.with_suffix(".campbell.csv")
        campbell.to_csv(csv)
        print(f"  wrote {csv}")
    elif args.campbell:
        print("  Campbell skipped: no companion ElastoDyn deck "
              "(the rotor-speed sweep needs the blade rotor schedule)")

    # Environmental-loading frequency-placement diagram for floating
    # cases: wind / wave spectra + 1P/3P bands vs the tower fore-aft
    # and side-side natural frequencies. Driven off the Campbell sweep
    # (it supplies both the rotor-speed range and the rotor-speed-
    # independent tower bending frequencies) so no site rpm / rated
    # data is fabricated.
    if is_floating and campbell is not None and args.format != "csv":
        try:
            from pybmodes.plots import plot_environmental_spectra

            lbls = [str(x).lower() for x in campbell.labels]
            f0 = np.asarray(campbell.frequencies)[0]

            def _pick(*keys: str) -> float | None:
                for i, lb in enumerate(lbls):
                    if all(k in lb for k in keys):
                        return float(f0[i])
                return None

            # Distinguish "not found" (None) from a valid 0.0 — never
            # let `or` swallow a legitimate zero.
            _fa = _pick("tower", "fa")
            fa = _fa if _fa is not None else _pick("tower", "fore")
            _ss = _pick("tower", "ss")
            ss = _ss if _ss is not None else _pick("tower", "side")

            # Operating-rpm range for the 1P/3P bands. Without an
            # explicit --min-rpm the band would start at DC and
            # visually overstate resonance overlap, so the title flags
            # it as a screening envelope and the design band uses the
            # given (min, max). Screening-grade environmental defaults
            # (IEC class-I turbulence scale; representative sea state)
            # — callers wanting site-specific inputs use
            # pybmodes.plots.plot_environmental_spectra directly.
            rpm_lo = float(getattr(args, "min_rpm", 0.0) or 0.0)
            rpm_hi = float(args.max_rpm)
            rated = getattr(args, "rated_rpm", None)
            if rated is not None:
                # --rated-rpm visibly shapes the figure: the 1P/3P
                # *design* band is the true operating range
                # (cut-in -> rated); the *constraint* band is the
                # wider allowable window out to --max-rpm.
                rpm_design = (rpm_lo, float(rated))
                rpm_constraint = (rpm_lo, rpm_hi)
                title = ("Environmental loading vs tower frequency "
                         f"placement (operating {rpm_lo:g}–{rated:g} "
                         f"rpm, rated {rated:g})")
            else:
                rpm_design = (rpm_lo, rpm_hi)
                rpm_constraint = None
                title = (
                    "Environmental loading vs tower frequency placement"
                    if rpm_lo > 0.0 else
                    "Environmental loading vs tower frequency placement "
                    "(SCREENING envelope — no operating rpm range "
                    "given; pass --min-rpm / --rated-rpm)"
                )
            fig = plot_environmental_spectra(
                tower_fa_hz=fa,
                tower_ss_hz=ss,
                rpm_design=rpm_design,
                rpm_constraint=rpm_constraint,
                wind={"mean_speed": 11.0, "length_scale": 340.2},
                wave={"hs": 6.0, "tp": 10.0},
                title=title,
            )
            spng = out.with_suffix(".spectra.png")
            fig.savefig(spng, dpi=120)
            import matplotlib.pyplot as _plt
            _plt.close(fig)          # don't leak figures in batch runs
            print(f"  wrote {spng}")
        except Exception as exc:  # noqa: BLE001
            print(f"  spectra plot skipped: {exc}")

    generate_report(
        modal, out, format=args.format, model=model,
        blade_params=blade_params, campbell=campbell,
        source_file=yaml_path,
    )
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pybmodes",
        description=(
            "pyBmodes — pure-Python finite-element library for "
            "wind-turbine blade and tower modal analysis."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate",
        help="validate ElastoDyn polynomial coefficients vs structural "
             "inputs",
    )
    p_validate.add_argument(
        "dat_file",
        help="path to the ElastoDyn main .dat file",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_patch = sub.add_parser(
        "patch",
        help="regenerate ElastoDyn polynomial coefficients from "
             "structural inputs (writes tower and blade .dat files; "
             "supports --dry-run / --diff / --output-dir for "
             "review-before-write workflows)",
    )
    p_patch.add_argument(
        "dat_file",
        help="path to the ElastoDyn main .dat file",
    )
    p_patch.add_argument(
        "--backup",
        action="store_true",
        help="save .bak copies of the tower and blade .dat files before "
             "patching in place; ignored when --dry-run, --diff, or "
             "--output-dir is set",
    )
    p_patch.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="number of FEM modes to extract before fitting (default: 10)",
    )
    # --dry-run and --diff both mean "don't write anywhere"; allowing
    # them together is harmless (--diff implies dry-run; --dry-run
    # alone prints just the summary). --output-dir is incompatible
    # with both — they describe different output destinations.
    p_patch.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the patched coefficients and print a per-block "
             "change summary; no files are modified",
    )
    p_patch.add_argument(
        "--diff",
        action="store_true",
        help="print a unified diff of the proposed tower + blade "
             "changes; implies --dry-run (no files are modified)",
    )
    p_patch.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="write the patched tower and blade .dat files into this "
             "directory instead of modifying the originals in place; "
             "the source files are left untouched",
    )
    p_patch.add_argument(
        "--output",
        type=str,
        default=None,
        help="alias for --output-dir; takes a directory path and writes "
             "the patched tower and blade .dat files there with their "
             "original filenames preserved",
    )
    p_patch.set_defaults(func=_cmd_patch)

    p_camp = sub.add_parser(
        "campbell",
        help="sweep a blade or tower across rotor speeds and emit a Campbell "
             "diagram (PNG) plus a CSV summary",
    )
    p_camp.add_argument(
        "input",
        help="path to a .bmi deck or an ElastoDyn main .dat file",
    )
    p_camp.add_argument(
        "--rated-rpm",
        type=float,
        default=None,
        help="operating rotor speed (rpm); drawn as a vertical reference line",
    )
    p_camp.add_argument(
        "--max-rpm",
        type=float,
        required=True,
        help="upper end of the rotor-speed sweep (rpm)",
    )
    p_camp.add_argument(
        "--n-steps",
        type=int,
        default=16,
        help="number of rotor-speed points in the sweep, including 0 and "
             "max-rpm (default: 16)",
    )
    p_camp.add_argument(
        "--orders",
        type=str,
        default="1,2,3,6,9",
        help="comma-separated per-rev excitation orders to overlay "
             "(default: 1,2,3,6,9)",
    )
    p_camp.add_argument(
        "--n-blade-modes",
        type=int,
        default=4,
        help="number of blade modes to track across the sweep (default: 4 — "
             "1st/2nd flap and 1st/2nd edge)",
    )
    p_camp.add_argument(
        "--n-tower-modes",
        type=int,
        default=4,
        help="number of tower modes to overlay as horizontal lines (default: 4 — "
             "1st/2nd FA and 1st/2nd SS); set to 0 to suppress",
    )
    p_camp.add_argument(
        "--tower",
        type=str,
        default=None,
        help="optional tower .bmi file; overrides the deck-supplied tower when the "
             "primary input is an ElastoDyn .dat, or pairs with a blade-only .bmi",
    )
    p_camp.add_argument(
        "--out",
        type=str,
        default=None,
        help="output PNG path (default: <input>_campbell.png alongside the input)",
    )
    p_camp.set_defaults(func=_cmd_campbell)

    p_batch = sub.add_parser(
        "batch",
        help="walk a directory of ElastoDyn decks, optionally validate "
             "and / or patch each, and write a summary CSV",
    )
    p_batch.add_argument(
        "root",
        help="directory to walk (recursively) for ElastoDyn main .dat files",
    )
    p_batch.add_argument(
        "--kind",
        type=str,
        default="elastodyn",
        choices=["elastodyn"],
        help="deck flavour to scan for (default: elastodyn; only kind "
             "currently supported)",
    )
    p_batch.add_argument(
        "--out",
        type=str,
        default="./reports/",
        help="directory to write per-deck validation reports and the "
             "summary CSV (default: ./reports/)",
    )
    p_batch.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="number of FEM modes to extract per deck when patching "
             "(default: 10)",
    )
    p_batch.add_argument(
        "--validate",
        action="store_true",
        help="emit a per-deck validation-report .txt under --out; the "
             "summary CSV is always written regardless of this flag",
    )
    p_batch.add_argument(
        "--patch",
        action="store_true",
        help="regenerate the polynomial coefficient blocks in each "
             "deck's tower and blade .dat files (in place). When "
             "combined with --validate, also writes a "
             "<deck>_validate_after.txt report alongside the "
             "before-patch one. Use with care — patching is in-place.",
    )
    p_batch.set_defaults(func=_cmd_batch)

    p_report = sub.add_parser(
        "report",
        help="run modal solve + validation + optional Campbell sweep on "
             "one ElastoDyn deck and emit a single Markdown / HTML / CSV "
             "report",
    )
    p_report.add_argument(
        "dat_file",
        help="path to the ElastoDyn main .dat file",
    )
    p_report.add_argument(
        "--format",
        type=str,
        default="md",
        choices=["md", "html", "csv"],
        help="report output format (default: md)",
    )
    p_report.add_argument(
        "--out",
        type=str,
        default=None,
        help="output report path (default: <dat_file>_report.<format> "
             "alongside the input)",
    )
    p_report.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="number of FEM modes to extract (default: 10)",
    )
    p_report.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="include coefficient-validation verdict in the report "
             "(default: on)",
    )
    p_report.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="skip coefficient validation (faster; useful for blade-only "
             "or sanity-check runs)",
    )
    p_report.add_argument(
        "--campbell",
        action="store_true",
        help="also run a rotor-speed Campbell sweep and include the "
             "first / last frequencies per mode in the report",
    )
    p_report.add_argument(
        "--max-rpm",
        type=float,
        default=15.0,
        help="upper end of the Campbell sweep when --campbell is set "
             "(default: 15.0 rpm)",
    )
    p_report.add_argument(
        "--n-steps",
        type=int,
        default=16,
        help="number of rotor-speed points in the Campbell sweep "
             "(default: 16)",
    )
    p_report.add_argument(
        "--n-blade-modes",
        type=int,
        default=4,
        help="number of blade modes to track in the Campbell sweep "
             "(default: 4)",
    )
    p_report.add_argument(
        "--n-tower-modes",
        type=int,
        default=4,
        help="number of tower modes in the Campbell sweep (default: 4)",
    )

    def _default_report_out(args: argparse.Namespace) -> argparse.Namespace:
        """argparse can't compute the default ``--out`` from ``dat_file``
        directly because the two are different arguments. We patch it
        in by inspecting ``args`` after parsing."""
        if args.out is None:
            args.out = str(
                pathlib.Path(args.dat_file).with_suffix("")
                .with_name(pathlib.Path(args.dat_file).stem + f"_report.{args.format}")
            )
        return args

    p_report.set_defaults(
        func=lambda a: _cmd_report(_default_report_out(a)),
    )

    # -----------------------------------------------------------------
    # windio — one-click WindIO ontology → tower + blade + (floating)
    # coupled platform + Campbell + bundled report
    # -----------------------------------------------------------------
    p_windio = sub.add_parser(
        "windio",
        help="one-click: a WindIO ontology .yaml (or an RWT directory) "
             "→ tower + blade + (floating) coupled-platform modes + "
             "optional Campbell + a bundled report. Companion "
             "HydroDyn/MoorDyn/ElastoDyn decks are auto-discovered so "
             "a floating platform is industry-grade by default; "
             "without them it is a labelled screening preview.",
    )
    p_windio.add_argument(
        "input",
        help="WindIO ontology .yaml, or an RWT directory to search",
    )
    p_windio.add_argument(
        "--out", type=str, default=None,
        help="report path (default: <yaml-stem>_windio_report.<format> "
             "in the CWD)",
    )
    p_windio.add_argument(
        "--format", type=str, default="md",
        choices=["md", "html", "csv"],
        help="report format (default: md)",
    )
    p_windio.add_argument(
        "--n-modes", type=int, default=12,
        help="FEM modes to extract (default: 12)",
    )
    p_windio.add_argument(
        "--water-depth", type=float, default=None,
        help="site water depth (m) — only needed for the yaml-only "
             "floating screening preview when no MoorDyn deck is found",
    )
    p_windio.add_argument(
        "--campbell", action="store_true",
        help="also run a rotor-speed Campbell sweep (uses the "
             "discovered companion ElastoDyn deck)",
    )
    p_windio.add_argument(
        "--max-rpm", type=float, default=12.0,
        help="Campbell sweep upper rpm (default: 12.0)",
    )
    p_windio.add_argument(
        "--min-rpm", type=float, default=0.0,
        help="lower end of the operating rotor-speed range used for "
             "the 1P/3P design bands on the environmental-spectra "
             "figure (default: 0.0 — the figure is then labelled a "
             "SCREENING envelope since no real cut-in is given)",
    )
    p_windio.add_argument(
        "--rated-rpm", type=float, default=None,
        help="rated rotor speed (rpm); when given, marks the "
             "operating range as specified on the environmental-"
             "spectra figure",
    )
    p_windio.add_argument(
        "--n-steps", type=int, default=16,
        help="Campbell rotor-speed points (default: 16)",
    )
    p_windio.add_argument(
        "--n-blade-modes", type=int, default=4,
        help="blade modes tracked in the Campbell sweep (default: 4)",
    )
    p_windio.add_argument(
        "--n-tower-modes", type=int, default=4,
        help="tower modes in the Campbell sweep (default: 4)",
    )

    def _default_windio_out(a: argparse.Namespace) -> argparse.Namespace:
        if a.out is None:
            stem = pathlib.Path(a.input).name
            if pathlib.Path(a.input).is_file():
                stem = pathlib.Path(a.input).stem
            a.out = f"{stem}_windio_report.{a.format}"
        return a

    p_windio.set_defaults(
        func=lambda a: _cmd_windio(_default_windio_out(a)),
    )

    # -----------------------------------------------------------------
    # examples — vendor bundled sample inputs / reference decks
    # -----------------------------------------------------------------
    p_examples = sub.add_parser(
        "examples",
        help="copy bundled sample inputs / reference decks into a "
             "user-supplied directory",
    )
    p_examples.add_argument(
        "--copy",
        required=True,
        metavar="DIR",
        help="destination directory (created if missing)",
    )
    p_examples.add_argument(
        "--kind",
        choices=["all", "samples", "decks"],
        default="all",
        help=(
            "which bundle to copy: "
            "'samples' = sample_inputs/ (analytical references + "
            "RWT samples), 'decks' = reference_decks/ (6 patched "
            "ElastoDyn decks), 'all' = both (default)"
        ),
    )
    p_examples.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing destination subdirectories",
    )
    p_examples.set_defaults(func=_cmd_examples)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
