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

"""``pybmodes windio`` workflow as a typed library function.

One-click WISDEM / WindIO ontology entry point:

1. Resolve the ontology ``.yaml`` (or a turbine-root directory) and
   discover companion OpenFAST decks scoped to that root.
2. Solve the composite-layup blade.
3. Solve the tubular tower (fixed cantilever) **or** the coupled
   floating tower + platform (industry-grade when the decks are
   present, screening preview otherwise).
4. Optionally run a Campbell sweep against the discovered ElastoDyn
   deck and overlay the platform rigid-body modes.
5. Optionally emit an environmental-loading frequency-placement plot
   (floating cases).
6. Render a bundled report (MD / HTML / CSV).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pybmodes.workflows._base import WorkflowResult

if TYPE_CHECKING:
    from pybmodes.campbell import CampbellResult
    from pybmodes.elastodyn.params import BladeElastoDynParams
    from pybmodes.models.result import ModalResult

WindIOFormat = Literal["md", "html", "csv"]
SkipPolicy = Literal["warn", "fail-on-data", "fail"]

# Internal classification of each skip site by what was lost:
#
# - ``"data"`` — a computational result the workflow would normally
#   produce is now missing (e.g. blade composite reduction failed →
#   the report's blade-frequencies section is incomplete).
# - ``"presentation"`` — the underlying data was computed but the
#   *figure* rendering failed (matplotlib backend issue / log-axis
#   floor / etc.). The CSV / data still exists; only the PNG is
#   missing.
# - ``"input"`` — the user's invocation requested an output that
#   needs a companion input that wasn't discovered (e.g.
#   ``campbell=True`` with no ElastoDyn deck). Not a code failure;
#   informational.
#
# The ``on_skip`` policy on :func:`run_windio` consults this table to
# decide whether each skip toggles ``exit_code = 1``.
_SKIP_KIND: dict[str, str] = {
    "blade": "data",
    "campbell": "input",
    "campbell_plot": "presentation",
    "spectra": "presentation",
}


def _skip_fails_under(skip_name: str, policy: SkipPolicy) -> bool:
    """Return ``True`` if a skip with the given name should toggle
    ``exit_code = 1`` under the chosen ``on_skip`` policy. Unknown
    skip names default to the strictest interpretation (treated as a
    data skip) so new failure modes fail-loud by default rather than
    silently downgrade to warning."""
    if policy == "warn":
        return False
    if policy == "fail":
        return True
    # "fail-on-data" — only computational skips fail; presentation +
    # input skips warn.
    return _SKIP_KIND.get(skip_name, "data") == "data"


@dataclass
class WindioDiscovery:
    """Resolved WindIO inputs (ontology + companion decks).

    Returned by :func:`discover_windio_inputs`. ``hydrodyn`` /
    ``moordyn`` / ``elastodyn`` are ``None`` when the companion deck
    was not auto-discovered under the turbine root — a fully-``None``
    triple keeps a floating analysis at "screening preview" rather
    than industry-grade.
    """

    yaml: pathlib.Path
    hydrodyn: pathlib.Path | None = None
    moordyn: pathlib.Path | None = None
    elastodyn: pathlib.Path | None = None


@dataclass
class WindioResult(WorkflowResult):
    """Result of :func:`run_windio`.

    Attributes
    ----------
    yaml : pathlib.Path | None
        The ontology ``.yaml`` actually loaded.
    discovery : WindioDiscovery | None
        Resolved companion-deck paths (hydrodyn / moordyn / elastodyn
        or ``None`` for each leg).
    is_floating : bool
        Whether the ontology declares a ``floating_platform``
        component.
    model
        The constructed :class:`~pybmodes.models.Tower` (cantilever or
        coupled-floating).
    modal : ModalResult | None
        The tower-side modal-solve result.
    blade_params : BladeElastoDynParams | None
        Composite-blade fit (``None`` when blade extraction was
        skipped, e.g. ontology has no ``blade`` component or the
        reduction raised).
    campbell : CampbellResult | None
        Campbell sweep result; ``None`` when ``campbell=False`` or
        the rotor-speed sweep was skipped (no companion ElastoDyn
        deck).
    report_path, campbell_png_path, campbell_csv_path, spectra_png_path
        Resolved paths of every artefact written. ``None`` for plots
        that were skipped (matplotlib unavailable, CSV-only format,
        rendering raised).
    """

    yaml: pathlib.Path | None = None
    discovery: WindioDiscovery | None = None
    is_floating: bool = False
    model: object | None = None
    modal: ModalResult | None = None
    blade_params: BladeElastoDynParams | None = None
    campbell: CampbellResult | None = None
    report_path: pathlib.Path | None = None
    campbell_png_path: pathlib.Path | None = None
    campbell_csv_path: pathlib.Path | None = None
    spectra_png_path: pathlib.Path | None = None
    skipped: list[str] = field(default_factory=list)
    # Completeness stamp shown in the report's Model summary section and
    # available to callers. ``"complete"`` (full fidelity, nothing
    # skipped), ``"screening"`` (floating with the seakeeping decks
    # missing — reduced fidelity by design), or ``"partial"`` (something
    # the workflow normally produces was skipped).
    report_status: str = "complete"


def _load_windio_doc(path: pathlib.Path) -> dict | None:
    """Parse ``path`` as a WindIO ontology document, or return ``None``.

    Returns the parsed mapping only when the file is a *bona-fide*
    WindIO ontology: it parses as YAML, the top level is a mapping, and
    it carries a ``components`` mapping. A parse error, a non-mapping
    document (e.g. a list-only config), or a yaml without ``components``
    all return ``None``. This replaces the previous substring scan for
    ``"components:"`` / ``"floating_platform:"``, which picked the wrong
    file (any yaml that merely *mentioned* the word) and missed valid
    ontologies whose key sat past the scanned byte window.
    """
    from pybmodes.io.windio import _dup_anchor_loader, _require_yaml

    try:
        yaml = _require_yaml()
        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    if not isinstance(doc.get("components"), dict):
        return None
    return doc


def _doc_is_floating(doc: dict | None) -> bool:
    """True when a parsed WindIO document declares a floating platform."""
    if not isinstance(doc, dict):
        return False
    comps = doc.get("components")
    return isinstance(comps, dict) and "floating_platform" in comps


def discover_windio_inputs(
    path: str | pathlib.Path,
) -> WindioDiscovery:
    """Resolve a WindIO ``.yaml`` plus any companion OpenFAST decks.

    ``path`` may be the ontology ``.yaml`` itself or an RWT directory
    (the ``IEA-*-RWT`` layout). Companion HydroDyn / MoorDyn /
    ElastoDyn-main decks are auto-discovered so the floating platform
    uses the **industry-grade** deck-fallback by default (see
    :meth:`pybmodes.models.Tower.from_windio_floating`).

    Auto-discovery is scoped to a bona-fide *turbine root*: the
    directory the user passed, or the nearest ancestor (≤ 4 levels
    up from the yaml) that owns an ``OpenFAST`` / ``openfast`` tree.
    A bare yaml in some scratch directory yields no decks (→ the
    labelled screening preview). Candidate ontologies are confirmed by
    a structured YAML parse (:func:`_load_windio_doc`), not a substring
    scan, so a non-WindIO yaml that merely mentions ``components`` is
    never selected.
    """
    path = pathlib.Path(path)
    if path.is_file():
        yaml_path = path
    elif path.is_dir():
        cands = sorted(
            p for p in path.rglob("*.yaml")
            if "OpenFAST" not in str(p) and "openfast" not in str(p)
            and _load_windio_doc(p) is not None
        )
        if not cands:
            raise FileNotFoundError(
                f"no WindIO ontology .yaml found under {path}"
            )
        yaml_path = cands[0]
    else:
        raise FileNotFoundError(f"WindIO input not found: {path}")

    turbine_root: pathlib.Path | None = None
    if path.is_dir():
        turbine_root = path
    else:
        for anc in list(yaml_path.parents)[:4]:
            if (anc / "OpenFAST").is_dir() or (anc / "openfast").is_dir():
                turbine_root = anc
                break

    if turbine_root is None:
        return WindioDiscovery(yaml=yaml_path)

    floating = _doc_is_floating(_load_windio_doc(yaml_path))
    pref = (
        ("semi", "spar", "umaine", "volturn", "floating", "hywind")
        if floating
        else ("monopile", "land", "onshore", "fixed", "tower")
    )

    def _rglob_safe(root: pathlib.Path, pattern: str) -> list[pathlib.Path]:
        out: list[pathlib.Path] = []
        try:
            for p in root.rglob(pattern):
                out.append(p)
        except (FileNotFoundError, PermissionError, OSError):
            pass
        return out

    def _find(
        pattern: str,
        exclude: tuple[str, ...] = (),
    ) -> pathlib.Path | None:
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

    return WindioDiscovery(
        yaml=yaml_path,
        hydrodyn=_find("*HydroDyn*.dat"),
        moordyn=_find("*MoorDyn*.dat"),
        elastodyn=_find("*ElastoDyn.dat", exclude=("tower", "blade")),
    )


def run_windio(
    input_path: str | pathlib.Path,
    *,
    out_path: str | pathlib.Path | None = None,
    format: WindIOFormat = "md",
    n_modes: int = 12,
    water_depth: float | None = None,
    campbell: bool = False,
    max_rpm: float = 12.0,
    min_rpm: float = 0.0,
    rated_rpm: float | None = None,
    n_steps: int = 16,
    n_blade_modes: int = 4,
    n_tower_modes: int = 4,
    on_skip: SkipPolicy = "fail-on-data",
) -> WindioResult:
    """One-click WindIO ontology workflow.

    Library entry point for :command:`pybmodes windio`. Resolves the
    ontology (and any companion OpenFAST decks scoped to the turbine
    root), solves the blade + tower (or coupled floating tower +
    platform), optionally runs a Campbell sweep against the
    discovered ElastoDyn deck, and writes a bundled report.

    Parameters
    ----------
    input_path : str or pathlib.Path
        WindIO ontology ``.yaml``, or an RWT directory to discover it
        in.
    out_path : str, pathlib.Path, or None
        Destination report file. ``None`` →
        ``<yaml-stem>_windio_report.<format>`` in the current
        directory.
    format : {"md", "html", "csv"}, default "md"
        Report format.
    n_modes : int, default 12
        Number of FEM modes to extract.
    water_depth : float or None
        Site water depth (m); only used by the yaml-only floating
        screening preview when no MoorDyn deck is found.
    campbell : bool, default False
        Run a rotor-speed Campbell sweep against the discovered
        companion ElastoDyn deck. Skipped (with a message) if no
        ElastoDyn deck was discovered; the ``on_skip`` policy below
        controls whether that counts as a failure.
    max_rpm, min_rpm, rated_rpm
        Rotor-speed sweep bounds + (optional) rated rpm overlay on
        the environmental-spectra plot for floating cases.
    n_steps, n_blade_modes, n_tower_modes : int
        Campbell-sweep parameters.
    on_skip : {"warn", "fail-on-data", "fail"}, default ``"fail-on-data"``
        How to handle workflow skips. Three classes of skip exist
        internally:

        * **data** — a computational result is missing
          (blade composite reduction raised). Under ``"fail-on-data"``
          (the new default in 1.8.0) and ``"fail"`` these toggle
          ``exit_code = 1`` so library callers / scripted automation
          notice the missing engineering output instead of silently
          publishing an incomplete report.
        * **presentation** — the data was computed but figure
          rendering failed (Campbell plot, environmental-spectra
          plot). Under ``"warn"`` and ``"fail-on-data"`` these only
          warn (the CSV / data is still on disk); under ``"fail"``
          they toggle ``exit_code = 1``.
        * **input** — an output was requested but its companion
          input wasn't discovered (e.g. ``campbell=True`` with no
          ElastoDyn deck under the turbine root). Under ``"warn"``
          and ``"fail-on-data"`` warns; under ``"fail"`` fails.

        Pass ``"warn"`` to recover the pre-1.8.0 permissive
        behaviour (every skip just messages, exit_code stays 0).
        ``WindioResult.skipped`` lists every skip regardless of
        policy.

    Returns
    -------
    WindioResult
        Carries the loaded yaml path, the auto-discovery result, the
        solved model + modal result, optional blade fit, optional
        Campbell sweep, and every written-artefact path. ``exit_code``
        is ``0`` on success, ``1`` when a skip toggled the failure
        gate via ``on_skip``.

    Raises
    ------
    FileNotFoundError
        When ``input_path`` does not resolve to a yaml or a directory
        containing one.
    """
    import numpy as np

    from pybmodes.io.windio import _dup_anchor_loader, _require_yaml
    from pybmodes.models import RotatingBlade, Tower
    from pybmodes.report import generate_report

    discovery = discover_windio_inputs(input_path)
    yaml_path = discovery.yaml

    messages: list[str] = []
    messages.append(f"windio: ontology {yaml_path}")
    for k, val in (
        ("hydrodyn", discovery.hydrodyn),
        ("moordyn", discovery.moordyn),
        ("elastodyn", discovery.elastodyn),
    ):
        tag = val.name if val else "— (screening preview)"
        messages.append(f"  companion {k:9s}: {tag}")

    yaml = _require_yaml()
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))
    comps = doc.get("components", {})
    is_floating = "floating_platform" in comps

    if out_path is None:
        out = pathlib.Path.cwd() / (
            f"{yaml_path.stem}_windio_report.{format}"
        )
    else:
        out = pathlib.Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []

    blade_params = None
    if "blade" in comps:
        messages.append("  solving blade (composite reduction)…")
        try:
            from pybmodes.elastodyn import compute_blade_params
            bl = RotatingBlade.from_windio(yaml_path)
            blade_modal = bl.run(n_modes=n_modes, check_model=False)
            blade_params = compute_blade_params(blade_modal)
        except Exception as exc:
            messages.append(
                f"  blade skipped: {type(exc).__name__}: {exc}"
            )
            skipped.append("blade")

    if is_floating:
        tier = (
            "industry-grade (deck-backed)"
            if all([discovery.hydrodyn, discovery.moordyn, discovery.elastodyn])
            else "SCREENING preview (missing decks)"
        )
        messages.append(
            f"  solving coupled floating tower+platform [{tier}]…"
        )
        model = Tower.from_windio_floating(
            yaml_path,
            water_depth=water_depth,
            hydrodyn_dat=discovery.hydrodyn,
            moordyn_dat=discovery.moordyn,
            elastodyn_dat=discovery.elastodyn,
        )
    else:
        messages.append("  solving tower (cantilever)…")
        model = Tower.from_windio(yaml_path)
    modal = model.run(n_modes=n_modes, check_model=False)

    campbell_result = None
    campbell_png: pathlib.Path | None = None
    campbell_csv: pathlib.Path | None = None
    if campbell and discovery.elastodyn is not None:
        from pybmodes.campbell import campbell_sweep
        messages.append(
            f"  Campbell sweep 0–{max_rpm} rpm "
            f"(via {discovery.elastodyn.name})…"
        )
        campbell_result = campbell_sweep(
            discovery.elastodyn,
            np.linspace(0.0, max_rpm, n_steps),
            n_blade_modes=n_blade_modes,
            n_tower_modes=n_tower_modes,
        )
        if format != "csv":
            try:
                from pybmodes.campbell import plot_campbell

                plat = None
                if is_floating and modal.mode_labels is not None:
                    plat = [
                        (lbl, float(modal.frequencies[i]))
                        for i, lbl in enumerate(modal.mode_labels)
                        if lbl is not None
                    ]
                    plat = plat or None
                campbell_png = out.with_suffix(".campbell.png")
                fig = plot_campbell(
                    campbell_result,
                    platform_modes=plat,
                    log_freq=plat is not None,
                )
                fig.savefig(campbell_png, dpi=120)
                import matplotlib.pyplot as plt
                plt.close(fig)
                messages.append(f"  wrote {campbell_png}")
            except Exception as exc:
                messages.append(f"  campbell plot skipped: {exc}")
                skipped.append("campbell_plot")
                campbell_png = None
        campbell_csv = out.with_suffix(".campbell.csv")
        campbell_result.to_csv(campbell_csv)
        messages.append(f"  wrote {campbell_csv}")
    elif campbell:
        messages.append(
            "  Campbell skipped: no companion ElastoDyn deck "
            "(the rotor-speed sweep needs the blade rotor schedule)"
        )
        skipped.append("campbell")

    spectra_png: pathlib.Path | None = None
    if is_floating and campbell_result is not None and format != "csv":
        try:
            from pybmodes.plots import plot_environmental_spectra

            lbls = [str(x).lower() for x in campbell_result.labels]
            f0 = np.asarray(campbell_result.frequencies)[0]

            def _pick(*keys: str) -> float | None:
                for i, lb in enumerate(lbls):
                    if all(k in lb for k in keys):
                        return float(f0[i])
                return None

            _fa = _pick("tower", "fa")
            fa = _fa if _fa is not None else _pick("tower", "fore")
            _ss = _pick("tower", "ss")
            ss = _ss if _ss is not None else _pick("tower", "side")

            rpm_lo = float(min_rpm)
            rpm_hi = float(max_rpm)
            if rated_rpm is not None:
                rpm_design = (rpm_lo, float(rated_rpm))
                rpm_constraint: tuple[float, float] | None = (
                    rpm_lo, rpm_hi
                )
                title = (
                    "Environmental loading vs tower frequency "
                    f"placement (operating {rpm_lo:g}–{rated_rpm:g} "
                    f"rpm, rated {rated_rpm:g})"
                )
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
            spectra_png = out.with_suffix(".spectra.png")
            fig.savefig(spectra_png, dpi=120)
            import matplotlib.pyplot as plt
            plt.close(fig)
            messages.append(f"  wrote {spectra_png}")
        except Exception as exc:
            messages.append(f"  spectra plot skipped: {exc}")
            skipped.append("spectra")
            spectra_png = None

    # Stamp the report's completeness so a reader can tell at a glance
    # whether it is the full analysis. A data skip (e.g. blade reduction
    # failed) is "partial"; a floating run without the seakeeping decks
    # is a known-reduced-fidelity "screening" preview; otherwise
    # "complete". A data skip outranks screening (missing output is more
    # severe than a deliberately reduced-fidelity model).
    data_skipped = any(_SKIP_KIND.get(s, "data") == "data" for s in skipped)
    screening = is_floating and not all(
        [discovery.hydrodyn, discovery.moordyn, discovery.elastodyn]
    )
    if data_skipped:
        report_status = "partial"
    elif screening:
        report_status = "screening"
    elif skipped:
        report_status = "partial"
    else:
        report_status = "complete"

    generate_report(
        modal, out, format=format, model=model,
        blade_params=blade_params, campbell=campbell_result,
        source_file=yaml_path, status=report_status,
    )
    messages.append(f"wrote {out} [{report_status}]")

    # Apply the on_skip policy: classify each accumulated skip and
    # toggle exit_code accordingly. The report is always written first
    # so callers in strict mode can still inspect the partial artefact
    # on disk + the structured ``skipped`` field for triage.
    errors: list[str] = []
    failing = [s for s in skipped if _skip_fails_under(s, on_skip)]
    if failing:
        kinds = [
            f"{name} ({_SKIP_KIND.get(name, 'data')})" for name in failing
        ]
        errors.append(
            f"on_skip={on_skip!r}: {len(failing)} skip(s) toggled "
            f"failure: {', '.join(kinds)}. The partial report at "
            f"{out} reflects the available results; rerun with "
            f"on_skip='warn' to recover the pre-1.8.0 permissive "
            f"behaviour."
        )
    exit_code = 1 if failing else 0

    return WindioResult(
        exit_code=exit_code,
        messages=messages,
        errors=errors,
        yaml=yaml_path,
        discovery=discovery,
        is_floating=is_floating,
        model=model,
        modal=modal,
        blade_params=blade_params,
        campbell=campbell_result,
        report_path=out,
        campbell_png_path=campbell_png,
        campbell_csv_path=campbell_csv,
        spectra_png_path=spectra_png,
        skipped=skipped,
        report_status=report_status,
    )
