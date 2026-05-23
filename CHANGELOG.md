<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.14.0] — 2026-05-23

An engineering-hardening pass that makes the library fail closed on
non-physical input, surface what it could not model, and report the
numerical health of every solve. One behaviour change (ERROR-severity
pre-solve checks now raise by default), everything else additive.

### Changed

- **Pre-solve ERROR findings now fail closed.** ``Tower.run`` and
  ``RotatingBlade.run`` previously routed every ``check_model`` finding,
  including ERROR-severity ones (NaN section properties, non-positive
  mass, a malformed support matrix), through ``UserWarning`` and then
  fed the eigensolver the non-physical input anyway. They now raise
  ``pybmodes.checks.ModelValidationError`` (a ``ValueError`` subclass)
  on any ERROR finding. The new ``on_error`` keyword controls this.
  ``on_error="raise"`` is the default. Pass ``on_error="warn"`` to
  restore the pre-1.14.0 warn-and-continue behaviour, or
  ``check_model=False`` to skip the checks entirely. WARN findings still
  emit as ``UserWarning`` unchanged. This only affects models that were
  already producing meaningless output, so it is a safety hardening
  rather than a feature removal.

### Added

- **``ModalResult.diagnostics`` (``SolverDiagnostics``).** Every solve
  now carries a numerical-health record. It names the path taken
  (``dense_symmetric`` / ``dense_general`` / ``sparse_shift_invert``),
  whether the sparse path silently fell back to dense (and why), the
  mode-count guarantee (requested vs returned), the per-mode
  backward-error residuals ``||K x - λ M x|| / ||K x||``, and a
  mass-matrix conditioning estimate. The general path also emits a
  ``RuntimeWarning`` when it recovers fewer valid modes than requested
  rather than letting a downstream broadcast fail opaquely. Telemetry
  about the run, so it is excluded from equality and not serialised.
- **``ModalResult.ignored_physics``.** A tuple naming any
  parsed-but-not-modelled physics the solve dropped, so a result is
  honest about its fidelity rather than silently approximate.
  Distributed hydrodynamic added mass (``distr_m``) is read from the BMI
  but not yet assembled into the mass matrix, so a model carrying it now
  reports ``"distributed added mass (distr_m)"`` here (and the bundled
  report shows it). Persisted when non-empty.
- **Report completeness stamp.** ``generate_report`` accepts a
  ``status`` argument rendered at the top of the Model summary section,
  and ``run_windio`` sets it to ``"complete"``, ``"partial"`` (a result
  the workflow normally produces was skipped) or ``"screening"`` (a
  floating run without the seakeeping decks). ``WindioResult`` carries
  the same value as ``report_status``.

### Fixed

- **WindIO input discovery is now a structured parse, not a text scan.**
  ``discover_windio_inputs`` chose its ontology yaml and detected a
  floating platform by searching the file text for ``"components:"`` and
  ``"floating_platform:"``. A yaml that merely mentioned those words was
  wrongly accepted and a valid ontology whose key sat past the scanned
  window was missed. It now parses each candidate as YAML and checks the
  structure (a ``components`` mapping, a ``floating_platform`` key).
- **BMI parser errors are now a first-class diagnostic.** The
  line-oriented ``.bmi`` reader raised bare ``ValueError`` / ``IndexError``
  / ``EOFError`` with no file or line context on a truncated array, an
  empty value line or an early end of file. It now raises
  ``pybmodes.io.errors.BMIParseError`` carrying the file, the 1-based
  line, the offending line text and the section it sits in.
  ``BMIParseError`` subclasses ``ValueError`` so existing
  ``except ValueError`` callers are unaffected.

### Internal

- The static-typing ratchet gained ``pybmodes.checks``,
  ``pybmodes.coords``, ``pybmodes.io.errors`` and
  ``pybmodes.workflows._base`` (verified clean under the strict mypy
  flags).
- An enforced coverage floor (``fail_under = 85``) now gates the test
  run, replacing the previous informational-only coverage report.
- ``pybmodes.campbell`` no longer re-exports its private helpers at the
  package root. The supported surface is ``CampbellResult`` /
  ``campbell_sweep`` / ``plot_campbell``; internal helpers are imported
  from their sub-modules.

## [1.13.1] — 2026-05-23

A Campbell-diagram labelling fix for floating turbines, fuller bending-mode
names on both engineering figures, and a docs-update section. No numerical
change to any model. This is figure and label text only.

### Fixed

- **Campbell diagram mislabelled floating-platform rigid-body modes as
  flexible tower bending modes.** On a floating turbine the Campbell sweep
  could name a low-frequency rigid-body mode (surge or sway at roughly
  0.01 Hz) `"1st tower FA"` or `"1st tower SS"`, the same name a roughly
  0.5 Hz flexible bending mode carries. So a user reading "1st tower
  fore-aft" off the diagram (for example to feed
  `plot_environmental_spectra`) got the platform frequency instead of the
  bending frequency. The mode-shape plots and reports happened to read a
  cleaner eigenbasis. There were three root causes, all fixed.

  - **Near-degenerate rigid-body pairs were not recognised as
    degenerate.** A real floater's surge/sway (and roll/pitch) pair is
    rarely exactly degenerate, because a slightly asymmetric mooring or
    hull splits it by a fraction of a percent. The eigensolver still
    returns it in an arbitrary rotated or mixed basis, and that mix varies
    run to run. The platform classifier's degeneracy window
    (`_DEGENERATE_FREQ_RTOL`) was a strict `1e-4`, so a sub-percent split
    fell through un-aligned, the mixed pair dropped below the dominance
    gate, and it was left unnamed in one solve while a sister solve named
    it (the mode-shape plot versus the Campbell sweep). The window is
    widened to `2e-2`, so a near-degenerate pair is rotated onto its
    platform axes and named deterministically, and the report, mode-shape
    plot and Campbell now agree. Over-widening is self-protecting, because
    the rotation is accepted only when it cleanly separates the pair onto
    two different dominant DOFs, so a genuinely distinct close pair is
    left as-is.
  - The Campbell tower path solved only `n_tower_modes` modes, so when
    `n_tower_modes` was below 6 (the default is 4) the classifier's
    rigid-block assignment was truncated and could leave surge or sway
    unnamed. The floating tower is now always classified over the full
    six-mode rigid block, then sliced back to `n_tower_modes`, so the
    Campbell labels match a direct `Tower(...).run().mode_labels`
    regardless of how many tower modes are requested.
  - The Campbell fallback that names modes the classifier still leaves
    `None` had no rigid-versus-flexible distinction. A `None` mode inside
    the rigid-body block is now labelled as a coupled platform mode (and
    drawn in the red **Platform** family on the diagram), never as
    flexible `"Nth tower FA/SS"`. Only a `None` mode beyond the rigid
    block gets the participation-derived flexible name. Verified across
    all eleven bundled reference turbines (land, monopile and floating).

### Changed

- **Bending-mode names spelled out in full on both engineering figures.**
  The Campbell diagram and `plot_environmental_spectra` now label modes
  `"flapwise bending"`, `"edgewise bending"`, `"fore-aft bending"` and
  `"side-to-side bending"`. The environmental diagram's tower lines were
  also unified from `"Side-Side"` to `"Side-to-Side"`. This is figure text
  only, and `CampbellResult.labels` keeps the terse `"1st flap"` and
  `"1st tower FA"` tokens for CSV and API stability.

### Docs

- Installation guide gained an **"Updating to a new release"** section,
  including how to upgrade a conda-environment install (it is a `pip`
  operation inside the env, and `conda update` does not apply). Fixed a
  stale `furo` reference in the optional-extras table (the docs theme is
  `sphinx-rtd-theme`).

## [1.13.0] — 2026-05-23

Off-axis floating support (#100), a clearer Campbell diagram (#57), and a
docs-build fix. Additive and backward-compatible; no numerical change to any
existing model (the new platform fields default to the previous behaviour).

### Added

- **`PlatformSupport.ref_x` / `ref_y` — off-axis hydrodynamics & mooring
  (#100).** The rigid-arm transform previously applied the horizontal arm only
  to the structural inertia (`cm_pform_x` / `cm_pform_y`); the added-mass,
  hydrostatic and mooring matrices were assumed referenced on the tower axis.
  `ref_x` / `ref_y` (the `PtfmRefxt` / `PtfmRefyt` horizontal position of the
  hydro/mooring reference relative to the tower axis) now carry `hydro_M` /
  `hydro_K` / `mooring_K` horizontally to the tower base as well, so a
  genuinely off-axis floater (e.g. a tower on an off-centre column) can be
  modelled consistently. Settable on a hand-built `PlatformSupport` and
  round-tripped through the `.bmi` text format (an off-axis `ref_msl_xyz`
  line, mirroring `cm_pform_xyz`); ElastoDyn decks reference the platform on
  the tower axis (there is no `PtfmRefxt` / `PtfmRefyt` field), so the deck
  ingestion path is unaffected. Defaults are `0.0` — every standard on-axis
  deck is byte-identical. The `check_model` large-CM-offset warning stands
  down when `ref_x` / `ref_y` are set (intentional off-axis modelling).

- **`PlatformSupport.tower_base_z` — intuitive `draft` alias (#100).** A
  positive-up accessor for the tower-base elevation above MSL
  (`tower_base_z == -draft`). `draft` keeps its BModes-inherited signed
  (negative-up) spelling for format fidelity, but reads as a naval-architecture
  misnomer; `tower_base_z` lets you set "tower base 15 m above MSL" as `15`.

### Changed

- **Campbell diagram: right-margin frequency labels + per-line dashing
  (#57).** Every structural mode's name + frequency now sits in a clean,
  de-overlapped column down the right edge (with a thin leader from each line's
  right end), instead of scattered inline along the curves — much easier to
  read for a FOWT whose modes cluster. Same-family lines get distinct dashes
  within their colour band so adjacent lines can be told apart. Family colours,
  the four-key legend, per-rev rays and the operating-speed shading are
  unchanged.

### Fixed

- **Documentation build.** Switched the Sphinx theme from `furo` (which was
  failing to provision on the docs builder) to `sphinx_rtd_theme` (the Read the
  Docs theme), and added a Read the Docs status badge to the README.
- **README documentation links.** Repointed the documentation table and inline
  links from the (not-yet-imported, 404ing) Read the Docs URLs to the
  GitHub-rendered source under `docs/`, so they resolve today; added the new
  conventions guide to the table and dropped the stale "if the URL 404s" note.
  Also added an *Updating to a new release* section to the install guide
  (`pip install --upgrade pybmodes`, version check, pinning, source-checkout
  refresh, and the conda-environment upgrade path).

## [1.12.1] — 2026-05-23

Documentation release — clarifies the reference-frame conventions that were
tripping users up. No code-behaviour change.

### Added

- **Coordinate-systems / conventions documentation.** A new
  `docs/conventions.rst` guide precisely defines — for *every* model type
  (land, monopile, floating, blade) — the single origin (tower base), axis
  directions, the 6-DOF order, the boundary conditions (`hub_conn`), the
  tower-top/RNA frame, and (for floaters) the MSL vertical datum with the
  exact sign of `draft` / `cm_pform` / `ref_msl`, the horizontal
  `cm_pform_x`/`cm_pform_y` arm (inertia-only) and the static-equilibrium
  assumption — with a worked OC3 Hywind example and a common-pitfalls list.
  The `PlatformSupport`, `TipMassProps` and `BMIFile` dataclasses gained full
  field-level docstrings carrying the same conventions. Notably documents that
  the `draft` field is the **signed tower-base elevation relative to MSL
  (negative = above)** — a name inherited from the BModes `.bmi` format, *not*
  the naval-architecture draft.

## [1.12.0] — 2026-05-23

Domain-aware input validation (#102): construction-layer guards that catch the
mechanical / civil-structural mistakes a non-specialist makes (unit errors,
implausible geometry), plus two fixes for review findings on the 1.10.1 /
1.11.0 floating guards. Additive and backward-compatible; no numerical change
to any existing model.

### Added

- **Construction-layer plausibility guards in `tubular_section_props` (#102).**
  Every geometry path (`Tower.from_geometry`, `from_windio`,
  `from_windio_with_monopile`) now emits a `UserWarning` for physically
  implausible *raw* inputs — caught at the door, before a meaningless solve:
  - **Material modulus** `E` outside `[1e9, 1e12]` Pa (catches the classic
    "E in GPa/MPa instead of Pa" unit error).
  - **Material density** `ρ` outside `[100, 25000]` kg/m³ (catches t/m³ or
    g/cm³).
  - **Diameter-to-thickness** `D/t` outside a broad `[5, 10000]` band — a
    gross unit/geometry sanity (a ×1000 D-vs-t unit mismatch, or a near-solid
    / sub-mm wall). Deliberately loose: real towers span D/t ≈ 56–1096
    (the high end is the IEA-15 VolturnUS-S *floating* tower, whose thin
    upper wall is legitimate), so this is not a fixed-tower shell-buckling
    check.
  - **Taper direction** — outer diameter growing base→top (reversed station
    ordering).
  Material checks run on the user's raw `E`/`ρ`, the only safe place — derived
  `SectionProperties` carry convention-dependent placeholders (ElastoDyn
  towers are axially rigid, so their `axial_stff` is not the physical E·A).
  All bands verified silent on every WindIO reference turbine, fixed and
  floating.

### Fixed

- **Floating-readiness `check_model` gates no longer fire on fixed-bottom
  decks.** The 1.11.0 floating gates (added mass / restoring / platform
  inertia / CM offset) keyed only on the presence of a `PlatformSupport`, so
  they emitted spurious WARN/ERROR findings on fixed-bottom **monopile** decks
  that carry an all-zero `PlatformSupport` block by layout convention (the
  bundled `02`/`04`/`05`/`06` samples — 2 ERRORs + 2 WARNs each). They now
  gate on `hub_conn == 2` (the genuine free-free floating path), matching where
  the solver actually assembles the platform DOFs. (Codex review, P1.)

- **`classify_platform_modes` no longer raises `IndexError` on a truncated
  `frequencies` array.** When `frequencies` has fewer entries than the
  rigid-body block (e.g. an externally-built mode subset), the degeneracy
  alignment is skipped (the global assignment still labels the modes) instead
  of indexing past the array end. (Codex review, P2.)

## [1.11.0] — 2026-05-23

A minor feature release: a WindIO monopile + tower constructor (#92) and a
suite of **floating-model readiness guards** in `check_model` that catch the
seakeeping mistakes a non-specialist makes when treating a WindIO `.yaml`
(geometry + material only) as sufficient for a floating system — the input
errors behind #95. Additive and backward-compatible; no numerical change to
any existing model.

### Added

- **Floating-model readiness guards in `check_model` (#95).** A WindIO `.yaml`
  carries geometry + material — enough for a *land* tower, but **not** for a
  floating system, whose rigid-body behaviour is set by hydrodynamics (added
  mass + hydrostatic restoring) and mooring that live in companion HydroDyn /
  MoorDyn decks, not the yaml. A seakeeping-naive user can silently produce a
  meaningless floating result; these gates make that loud. For any
  `hub_conn = 2` `PlatformSupport` model, `check_model` now additionally
  checks:
  - **No hydrodynamic added mass** (`hydro_M` / `A_inf` all zero) — WARN; this
    biases every rigid-body frequency high (commonly 10–30 %) and is the most
    common omission.
  - **No restoring at all** (`hydro_K` *and* `mooring_K` both zero) — WARN; the
    rigid-body modes collapse to ~0 Hz (a free body, not a station-kept
    floater).
  - **Non-physical platform inertia** (non-positive `mass_pform` or a
    non-positive `i_matrix` diagonal) — ERROR.
  Each message names the fix and points at the validated deck path
  (`Tower.from_windio_floating(yaml, hydrodyn_dat=…, moordyn_dat=…,
  elastodyn_dat=…)`). Verified to stay silent on every validated floating model
  (OC3 Hywind spar, IEA-15 UMaineSemi, the WindIO Morison screening tier).

- **`check_model` gate for an implausibly large horizontal platform CM offset
  (#95).** `cm_pform_x` / `cm_pform_y` on a `PlatformSupport` are the CM offset
  *from the tower axis*; a value comparable to the platform's own size is
  almost always a coordinate-origin offset leaking into the field, which
  injects spurious surge/sway↔yaw coupling, shifts the rigid-body frequencies,
  and mislabels the modes. The new check warns (WARN) when the horizontal
  offset magnitude exceeds the platform's yaw radius of gyration `√(I_yaw/m)`,
  turning a previously silent input error into an actionable message at the
  call site. New `CheckOptions.platform_cm_offset_gyradius_factor` (default
  `1.0`) tunes the threshold; the `PlatformSupport.cm_pform_x` / `cm_pform_y`
  docs now state explicitly that they are tower-axis-relative (not a global
  coordinate). Surfaced the diagnosis on #95: the reported "yaw on the first
  mode" was this input error (a ~39 m CM offset on the IEA-15 VolturnUS-S),
  not a classifier regression — with `cm_pform_x ≈ 0` the rigid-body modes
  label cleanly.

- **`Tower.from_windio_with_monopile(yaml, …)` — WindIO monopile + tower
  splice (#92).** `from_windio` reduces a *single* tube; there was no WindIO
  path that joined a monopile and tower into one fixed-bottom system. The new
  classmethod reduces the `monopile` and `tower` components separately (each
  keeps its own wall schedule and steel grade) and splices them bottom-to-top
  at the transition piece — the elevation where the monopile top meets the
  tower base, taken from each component's `reference_axis.z` — into a single
  cantilever clamped at the mudline (`hub_conn = 1`), with the RNA lumped at
  the tower top via `tip_mass`. It is the WindIO analog of
  `Tower.from_elastodyn_with_subdyn` (the ElastoDyn + SubDyn splice).
  Supports the same `tip_mass` (TipMassProps or bare-float kg) and per-segment
  `n_nodes` mesh refinement as `from_windio`; raises `ValueError` if the two
  segments do not meet at a common transition-piece elevation. Backed by
  `pybmodes.io.windio.read_windio_monopile_tower` (with `WindIOMonopileTower`)
  and validated end-to-end on the IEA-15-240-RWT monopile ontology.

  This is the **rigid fixed-base** path: the pile is clamped at the mudline
  with no soil flexibility, matching `from_elastodyn_with_subdyn` and the
  bundled monopile samples — so the estimate is conservative (softer) for a
  pile whose embedded length would otherwise be laterally supported by soil.
  Distributed soil springs (a Winkler `distr_k` / `hub_conn = 3` foundation,
  e.g. the NREL closed-form soil-stiffness model) and Morison hydrodynamics
  are out of scope here and tracked as a separate follow-up.

- **`pybmodes.io.windio.WindIOTubular` now carries the absolute base/top
  elevations** (`z_base`, `z_top`) parsed from `reference_axis.z`, in addition
  to the existing normalised grid and `flexible_length`. Additive, defaulted
  fields; existing callers are unaffected.

## [1.10.1] — 2026-05-23

A labelling-only bug-fix release: the floating-platform rigid-body mode
classifier (`ModalResult.mode_labels`) mislabelled asymmetric platforms.
Frequencies and mode shapes are unchanged on every deck.

### Fixed

- **Asymmetric-platform rigid-body mode labels (`ModalResult.mode_labels`).**
  The floating-platform rigid-mode classifier
  (`pybmodes.fem.platform_modes.classify_platform_modes`) named each of the
  six lowest modes by a *greedy* per-mode argmax over mass-weighted
  base-node modal kinetic energy, dropping any DOF name already taken by an
  earlier mode. On an asymmetric platform (horizontal CM offset or
  off-diagonal mooring/hydro coupling) a genuine surge mode carrying a small
  parasitic high-inertia rotation reads, mass-weighted, as yaw-dominated —
  so it stole `yaw`, the true yaw mode then hit the "already used" branch
  and was left `None`, and the mislabel cascaded onto the neighbouring modes
  (reported as "surge classified as yaw, sway as surge, third mode
  unclassified"). The labels are now assigned by a **global Hungarian
  optimal matching** (`scipy.optimize.linear_sum_assignment`, already used
  in `pybmodes.mac`) over the 6 rigid candidates × 6 platform DOFs, so the
  mode that best expresses each DOF wins that label and no DOF is ever named
  twice. The conservative dominance threshold (0.6) still gates each
  assignment, so a genuinely coupled / rotated pair whose energy splits
  across DOFs stays `None` rather than mislabelled. (#93)

- **Deterministic rigid-body labels for symmetric platforms (degenerate
  pairs).** On a (bi)symmetric platform surge≈sway and roll≈pitch share an
  eigenvalue, so the eigensolver may return any rotation of that 2-D
  eigenspace — a basis that depends on BLAS thread ordering (the same
  non-determinism hazard fixed for the FA/SS *tower* pair in 1.10.0). A
  45°-mixed degenerate pair reads 50 / 50 across two DOFs and would fall
  below the dominance threshold, so the modes could be silently left
  `None`. `classify_platform_modes` now takes the modal `frequencies` and
  rotates each frequency-degenerate rigid pair back onto its platform axes
  before labelling (the rigid-body analog of `_rotate_degenerate_pairs` in
  `pybmodes.elastodyn.params`), accepting the rotation only when it cleanly
  separates the pair into two distinct dominant DOFs. The result no longer
  depends on the arbitrary eigensolver basis. Asymmetric platforms break the
  degeneracy, so this step is a no-op there.

Both changes are confined to mode **labelling**; frequencies and mode
shapes are byte-for-byte unchanged, and symmetric decks (OC3 Hywind, IEA-15
UMaineSemi, …) — whose rigid modes are ~98 % single-DOF — keep their
existing labels. The new `frequencies` argument on
`classify_platform_modes` is optional and keyword-only-friendly (defaults to
`None`, which skips the degeneracy resolution), so the change is
backward-compatible.

## [1.10.0] — 2026-05-22

Static-review follow-up on top of 1.9.0: a deterministic fix for the
2nd-tower FA/SS degenerate-pair resolution (the only numerical change —
affected symmetric-tower polynomial coefficients move toward the
reproducible result; frequencies unchanged), the `n_nodes` mesh-refinement
keyword on the WindIO floating constructor, and validation-integrity /
reference-accuracy hardening.

### Changed

- **`verify_external_data.py --strict` no longer reports a bare PASS for an
  unverified `tagged-archive` entry.** Such entries (the BModes clone) have
  no git HEAD to check, so content hash pins are their only verification;
  a present archive with an empty `hashes` table now reports **WARN**
  ("present but UNVERIFIED") instead of PASS. The BModes manifest entry
  gained a `hash_files` list (the Test03 / Test04 decks) so a maintainer
  `--update` can pin real EOL-normalized content hashes and promote it to
  a genuine PASS.

- **`ModalResult.from_json` now refuses an unsupported `schema_version`.**
  The reader checks the embedded `schema_version` (written as `"1"` by
  `to_json`) and raises `ValueError` on any other value rather than
  silently parsing a future-schema file under the v1 layout — closing the
  gap between the documented serialisation contract ("older loaders refuse
  a higher version") and the code. A payload with no `schema_version` key
  is still accepted as `"1"` (lenient for any pre-field files). Files
  written by current pyBmodes are `"1"` and unaffected.

### Fixed

- **Deterministic degenerate-pair resolution (2nd tower FA/SS).** The
  symmetric-tower FA/SS degenerate-pair resolver accepted its rotation on a
  *twist-inclusive* purity metric while only aligning the bending (flap +
  lag) DOFs. For a clean-but-slightly-twisted 2nd-tower-bending pair that
  metric sat right on the 0.99 accept threshold, so floating-point / BLAS-
  thread-ordering noise could flip the gate, leave the mixed eigenvectors
  for the classifier, and swap the 2nd FA/SS family members — producing
  non-reproducible `TwFAM2Sh` / `TwSSM2Sh` polynomial coefficients (up to
  ~2× off, e.g. the NREL 5MW land `TwSSM2Sh` round-trip). The accept gate
  now keys on a twist-free bending-purity metric (`_bending_purity`);
  torsion contamination is still rejected, deterministically, by the
  existing torsion-energy gate in `_select_tower_family`. The coefficient
  validator now also measures `file_rms` / `pybmodes_rms` against the same
  rotated shape the fit used, so the per-block ratio/verdict is meaningful
  for degenerate pairs. Tower **frequencies** are unaffected (rotation is a
  basis change within the degenerate eigenspace); only the per-block
  polynomial coefficients on affected symmetric towers change — toward the
  reproducible single-thread result.

- **Citation links point at canonical sources.** `VALIDATION.md` and
  `docs/theory.rst` now link Bir (2010) and Jonkman (2007) to their
  `docs.nrel.gov` PDFs and Bir (2009) to its AIAA DOI (`10.2514/6.2009-1035`)
  instead of indirect OSTI biblio records, and add the BModes User Guide
  (Bir 2005, NREL/TP-500-39133). The "BModes is not on GitHub" claim in
  `VALIDATION.md`, `external/MANIFEST.toml`, and the historical changelog /
  release-notes is corrected — the BModes Fortran source + CertTest decks
  are on GitHub (`old-NWTC/BModes`); only the patched BModes_JJ binary
  behind the CS_Monopile / OC3Hywind reference outputs is distribution-
  restricted.

- **Internal code-quality cleanups (no behaviour change).** Removed dead
  helpers (`subdyn_reader._strip_comment` no-op, `elastodyn.writer._MISSING`
  unused sentinel), merged two byte-identical `bmi._read_platform_inertia_*`
  readers into one, precompiled the `patch_dat` coefficient regex once per
  parameter (was rebuilt per line × parameter), vectorised the solver's
  per-column L2 normalisation, and replaced a hand-rolled mooring step clamp
  with `np.clip`.

### Added

- **`Tower.from_windio_floating(..., n_nodes=N)`** (issue #58) — the
  WindIO floating constructor now takes the same `n_nodes` mesh-
  refinement keyword as `Tower.from_windio` / `from_geometry`,
  re-gridding the tower beam onto `N` evenly-spaced stations (geometry
  linearly interpolated, closed-form tube properties recomputed
  exactly); the platform assembly is untouched. Completes the uniform
  `n_nodes` surface across the WindIO/geometry constructors. The deck /
  BMI readers are intentionally left without it — re-sampling their
  tabulated section properties would silently smooth deliberately
  stepped decks. Additive, non-breaking (default `None` keeps the
  WindIO grid).

## [1.9.0] — 2026-05-22

Static-review follow-up on top of 1.8.1. No numerical change; the only
public-API change is one new opt-in keyword (``allow_legacy_pickle``).
Security, release-integrity, validation-coverage, and source-quality
items:

### Security

- **Legacy NPZ pickle loading is refused by default.**
  ``ModalResult.load`` / ``CampbellResult.load`` open archives with
  ``allow_pickle=False``; on encountering a legacy pre-1.0 ``__meta__``
  stored as a pickled object array they now **raise** instead of
  silently reaching for ``allow_pickle=True`` — object-array unpickling
  can execute arbitrary code, and ``SECURITY.md`` puts NPZ
  deserialisation in scope. Pass the new keyword-only
  ``allow_legacy_pickle=True`` to opt in for a file you trust (it still
  warns — never silent). Archives written by current pyBmodes are
  pickle-free and unaffected.

### Changed

- **PyPI publishing is now gated on external validation.** A new
  ``validation-gate`` job in ``.github/workflows/publish.yml`` queries
  the Actions API and refuses to publish unless the *Validation
  (external data)* workflow has a **successful run on the exact commit
  being tagged**. The release ordering is now: run validation on the
  release commit → confirm green → push the ``vX.Y.Z`` tag. Closes the
  gap where a tag push could publish after build/smoke alone without
  the integration-tolerance workflow ever having run on that commit.
- **The validation workflow now exercises every required manifest
  clone, and ``--strict`` fails on a missing one.** Cloning in
  ``validation.yml`` is manifest-driven via
  ``verify_external_data.py --clone`` (no hardcoded URL/SHA list), so
  CI now fetches OpenFAST r-test, IEA-3.4 / IEA-10 / IEA-15 / IEA-22,
  and WISDEM — previously only r-test, IEA-3.4, and IEA-15. Manifest
  entries gained an ``optional`` flag (MoorPy / RAFT / BModes); under
  ``--strict`` a missing **required** clone is now a hard FAIL instead
  of a silent SKIP, so a verifier report can't look acceptable while a
  pinned non-optional entry went unchecked. README + ``VALIDATION.md``
  coverage text updated to match.
- **Expanded the ruff ruleset beyond ``E/F/W/I``.** Added ``UP`` (py-
  upgrade), ``B`` (bugbear), ``C4`` (comprehensions), ``SIM``, ``PIE``,
  and ``RUF`` with a documented ``ignore`` list (the ambiguous-unicode
  rules fire on legitimate engineering glyphs; a few stylistic rules
  are deliberately deferred). Safe autofixes applied across the tree.
- **Raised the global mypy floor and roughly tripled the strict-typed
  surface.** New package-wide flags ``warn_redundant_casts``,
  ``strict_equality``, ``no_implicit_optional``, ``extra_checks``
  (``warn_unused_ignores`` is enabled per-module only, not globally —
  it is environment-fragile for loose modules that import opt-in,
  stub-less third-party packages); the per-module strict-override list
  grew from 10 to 29 modules (every pure-logic / typed-dataclass module
  that passes ``--disallow-untyped-defs --warn-return-any``; parsers
  and matplotlib helpers stay loose).
- **Neutral-provenance scrub.** Replaced the remaining non-neutral
  provenance wording in ``validation.yml``, ``CHANGELOG.md``, and a
  test comment with the project's neutral vocabulary, per
  ``CONTRIBUTING.md`` §3.

### Added

- **File-content hash verification is now ACTIVE** (was the last
  audit-hardening gap). All six required clones (r-test, IEA-3.4 / 10 /
  15 / 22, WISDEM) declare ``hash_files`` for their load-bearing decks
  and ship **populated ``hashes``**; ``--strict`` re-checks them on
  every run on top of commit-SHA pinning. Hashes are computed over
  **line-ending-normalized content (CRLF→LF)**, so they reproduce
  identically on a Windows dev checkout and a fresh Linux CI checkout —
  no LF-only population dance, and local + CI ``--strict`` both pass.
- **Real ``verify_external_data.py --update`` write-back** (replacing
  the previous stub): re-pins each clone ``sha`` to local ``HEAD`` and
  recomputes the ``hashes`` table from per-clone ``hash_files``. A
  declared ``hash_files`` path that can't be resolved (typo / absent
  clone) **aborts the update and writes nothing** so a mistake can't
  leave a stale ``hashes`` table behind; ``--allow-missing-hashes`` is
  the explicit escape hatch, and an emptied computable set clears the
  table to ``{ }`` rather than leaving an obsolete pin. New ``--clone``
  / ``--include-optional`` / ``--dry-run`` / ``--allow-missing-hashes``
  flags; ``tests/test_verify_external_data.py`` covers the rewriter,
  the required/optional SKIP-vs-FAIL policy, the fail-loud path, the
  stale-clearing path, and EOL-normalized hashing.

### Fixed

- **``Tower.from_windio_floating`` now honours a caller-supplied
  ``rna_tip`` in the screening tier** (issue #83). When no companion
  ElastoDyn deck and no injected ``platform_support`` were given, the
  screening path unconditionally reset ``rna_tip`` to a zero tower-top
  lump, silently dropping the argument; it now uses the passed value as
  the default (a discovered ElastoDyn deck still overrides it with the
  deck-derived RNA, unchanged). The injected-platform branch already
  behaved correctly. Regression test in ``tests/test_windio_floating.py``.
- **Documentation consistency.** ``VALIDATION.md``'s closing section
  no longer says integration coverage is "developer-local … not
  CI-gated" — it now splits the **CI-gated public required set**
  (r-test + IEA-3.4 / 10 / 15 / 22 + WISDEM, enforced by
  ``validation.yml``) from the **maintainer-local** BModes archive and
  optional cross-comparisons, matching the *Enforcement* section.
  ``release.yml``'s header points maintainers at ``validation.yml`` as
  the external-data gate instead of "run it locally". ``installation.rst``
  leads with the source install and marks *From PyPI* as "after the
  first release" (mirroring the README pre-release note), and drops
  ``linkify-it-py`` from the ``[docs]`` extra table (not a declared
  dependency; linkify is disabled in ``conf.py``). The mypy bullet
  above no longer lists ``warn_unused_ignores`` as a global flag.
  ``docs/release_checklist.rst``'s manifest note now states the public
  required set is CI-gated (only BModes + optional clones stay
  maintainer-local), matching ``VALIDATION.md``. The
  ``_serialize`` ``_metadata_to_npz_value`` docstring now describes the
  refuse-by-default pickle contract.
- **Reference-turbine sample count corrected from "seven" to
  "eleven".** The bundled ``reference_turbines/`` library ships 11
  sub-cases — 6 fixed-base (NREL 5MW land + OC3 monopile, IEA-3.4 land,
  IEA-10 / 15 / 22 monopiles) and 5 floating (NREL 5MW OC3 Hywind +
  OC4 DeepCwind, IEA-15 UMaineSemi, IEA-22 Semi, UPSCALE-25). The "seven"
  count was stale in ``data_sources.rst``, ``_examples/__init__.py``,
  both ``sample_inputs`` READMEs, ``cli.py``, and
  ``workflows/examples.py``.
- **sdist now ships the reproducibility artefacts its docs reference.**
  ``MANIFEST.in`` adds ``scripts/*.py`` and re-includes
  ``external/MANIFEST.toml`` (the pinned-commit + content-hash
  contract) after the ``prune external``, so an sdist consumer can
  follow the validation workflow the shipped docs describe.
- **Stale metadata / links.** ``CITATION.cff`` version bumped
  ``1.7.0`` → ``1.8.1`` to match ``pyproject.toml``; the
  ``CONTRIBUTING.md`` install link points at ``docs/installation.rst``
  (was ``.md``); ``.gitattributes`` and ``.pre-commit-config.yaml`` now
  state one coherent line-ending policy (tracked decks LF; gitignored
  ``external/`` native; validation hashing normalized).
- **Validation matrix matched to the pickle hardening.** The
  ``VALIDATION.md`` NPZ row no longer claims a "warned
  ``allow_pickle=True`` fallback"; it now states legacy object metadata
  is refused by default and loads only via ``allow_legacy_pickle=True``.
- **Walkthrough notebooks are release-ready.** Removed the "fill these
  in" placeholder from the IEA-15 UMaineSemi notebook's *Key findings*
  (the concrete findings were already written below it); added
  ``tests/test_notebooks.py`` coverage for the third tracked notebook,
  ``cases/iea15_volturnus_windio_walkthrough.ipynb`` (friendly-error +
  integration paths, mirroring the UMaineSemi contract); and declared
  all three notebooks **source-only** (committed without executed
  outputs; CI executes them headlessly) with a note in each notebook
  and a reconciled release-checklist notebook-smoke step.
- **Citations are citation-grade.** Every reference in ``VALIDATION.md``
  and ``docs/theory.rst`` now carries a DOI, a ``docs.nrel.gov`` PDF
  link, or an OSTI.GOV record — and the textbooks (Blevins; Karnovsky &
  Lebed) are explicitly marked "no DOI" rather than left bare.

## [1.8.1] — 2026-05-21

Static-review follow-up patch on top of 1.8.0. No public-API
change; no numerical change; no behaviour change. Three
governance / reproducibility items addressed:

### Changed

- **Scrubbed review/tool attribution from tracked content** (eleven
  sites across ``CHANGELOG.md``,
  ``.github/workflows/validation.yml``,
  ``src/pybmodes/workflows/batch.py``,
  ``tests/test_io_errors.py``, ``tests/test_workflows.py``, and
  ``.github/ISSUE_TEMPLATE/code_review.yml``). The
  ``CONTRIBUTING.md`` neutral-provenance rule existed but the
  Phase 4 round had quietly re-introduced non-neutral references
  through the per-finding follow-up commits — a governance self-
  contradiction caught by the post-1.8.0 static review. All
  references now use neutral provenance wording plus the PR number
  for audit trail.
- **Populated the external-data manifest's per-clone SHA pins.**
  ``external/MANIFEST.toml`` carried ``sha = "TBD"`` for all eight
  upstream clones in 1.8.0 — ``validation.yml`` therefore cloned
  upstream HEAD with ``--depth=1``, making the workflow
  reproducible-as-process but not pinned to exact upstream commits.
  Every clone now has a real commit SHA (read from the maintainer's
  local ``external/`` clones at the 1.8.1 release-prep snapshot), and
  ``validation.yml`` checks out each pinned SHA explicitly after
  the clone. The workflow is now reproducible from the tagged commit,
  pinned to immutable upstream commits. File-hash pins (the manifest's ``hashes`` tables) remain
  empty pending the ``verify_external_data.py --update`` rewrite —
  that's a follow-up maintainer action separate from this patch.
- **Tightened the validation-workflow coverage claim** in
  ``README.md`` and ``VALIDATION.md``. The 1.8.0 text said the
  workflow validates "the IEA-Task-37 reference turbines" plural,
  but it only actually clones IEA-3.4-130-RWT and IEA-15-240-RWT;
  the broader claim implied IEA-10 / IEA-22 / WISDEM / MoorPy /
  RAFT were also exercised. Both files now name the three
  upstreams the workflow actually clones (OpenFAST r-test +
  IEA-3.4 + IEA-15) and call out the remaining manifest clones as
  reproducibility-pinned but not CI-exercised.

## [1.8.0] — 2026-05-21

Multi-phase architecture refactor + project-infrastructure round +
static-review follow-up. No public-API change; one CLI default
change (``pybmodes windio --on-skip`` defaults to ``fail-on-data``
instead of silently warning — see Phase 4 below); no numerical
change. Four threads landed:

- **Phase 1** — re-license (MIT → Apache 2.0), Sphinx documentation
  site on Read the Docs, governance + CI hardening, strict mypy /
  ruff ratchet, unified ``ParseError`` hierarchy, centralised
  numerical options dataclasses.
- **Phase 2** — every ``pybmodes`` CLI subcommand is now a typed
  library entry point under :mod:`pybmodes.workflows`; ``cli.py``
  shrinks from ~1500 to ~520 lines and is purely
  argparse + delegation. Seven new workflow functions and seven
  result dataclasses; thirty-nine direct workflow tests.
- **Phase 3** — three biggest source files broken up by concern:
  ``campbell.py`` (1301 LOC → 7-file ``campbell/`` sub-package),
  ``mooring.py`` (1202 LOC → 6-file ``mooring/`` sub-package),
  ``models/tower.py`` private helpers extracted to
  ``models/_shared.py`` (cross-model) and ``models/_platform.py``
  (tower-specific platform-scalar parsers). Every public name
  is preserved; existing imports keep working through the
  ``__init__.py`` re-export layer.
- **Phase 4** — static-review hardening: ``pybmodes batch
  --patch`` gains the same dry-run / backup / output-dir safety
  contract that ``pybmodes patch`` has had since 1.0 (now with
  ``backup=True`` default for tree-wide mutation); ``run_windio``
  gains an ``on_skip`` policy parameter that flips
  computational-skip failures from silent ``exit_code=0`` to
  ``exit_code=1`` by default; a new ``validation.yml`` GitHub
  Actions workflow enforces the published 0.01 % integration-test
  tolerance in CI by cloning the upstream OpenFAST / IEA-Task-37
  repositories on the fly, so the validation claim no longer
  depends on maintainer-local state.

### Added

- **``pybmodes.workflows.*`` typed-result library API** (Phase 2 PRs
  B1–B3). Seven entry-points, one per CLI subcommand, each returning
  a :class:`WorkflowResult` subclass with ``exit_code`` /
  ``messages`` / ``errors`` plus the workflow's typed payload:

  - :func:`run_validate` → :class:`ValidateResult`
  - :func:`run_examples_copy` → :class:`ExamplesResult`
  - :func:`run_patch` → :class:`PatchResult` (five mutually-
    supportive output modes: default in-place, ``backup``,
    ``output_dir``, ``dry_run``, ``diff``)
  - :func:`run_report` → :class:`ReportResult`
  - :func:`run_batch` → :class:`BatchResult`
  - :func:`run_campbell` → :class:`CampbellWorkflowResult`
  - :func:`run_windio` → :class:`WindioResult` plus
    :func:`discover_windio_inputs` returning a
    :class:`WindioDiscovery` dataclass.

  Lets notebooks and external scripts run the same flows the CLI
  uses without ``subprocess`` (the original 0.x workflows-via-stdout
  contract is preserved via the slim CLI wrappers).

- **Sphinx documentation site** (``docs/``) hosted on Read the Docs
  (``.readthedocs.yaml``). Flat reStructuredText tree — 13 pages:
  ``index``, ``installation``, ``quickstart``, ``theory``,
  ``data_sources``, ``units``, ``limitations``, ``validation``,
  ``api``, ``api_contract``, ``changelog``, ``contributing``,
  ``release_checklist``. ``CHANGELOG.md`` / ``VALIDATION.md`` /
  ``CONTRIBUTING.md`` are pulled in via ``.. include::`` so the site
  never drifts from the source. Furo theme, autodoc with numpy-style
  napoleon, intersphinx into numpy / scipy / matplotlib, copy-button.
  MyST-parser is registered only to back the ``.. include::`` of the
  root-level Markdown files; new docs/ pages should be ``.rst``.
  ``docs/Makefile`` + ``docs/make.bat`` for local builds. New
  ``[docs]`` extra in ``pyproject.toml`` pinned to the Sphinx 8.x
  line until the ``sphinx-autodoc-typehints`` 3.x deprecation
  backlog clears.
- **README.md condensed to ~ 100 lines** (down from ~ 890). The
  detailed feature tour, code examples, case-study tables, public-API
  enumeration, development workflow, and compatibility policy now
  live in the corresponding ``docs/*.rst`` pages; the README's job
  is to point users at the right doc page in five seconds.
- **Governance files** (industry-standard set):
  ``CONTRIBUTING.md`` (explicit-path-staging rule, pre-commit, nox
  sessions, PR checklist), ``SECURITY.md`` (vulnerability scope +
  private reporting channel), ``CODE_OF_CONDUCT.md`` (Contributor
  Covenant v2.1, maintainer as enforcement contact),
  ``CITATION.cff`` (Zenodo / "Cite this repository" widget,
  authored under SMI Lab, Inha University).
- **Pre-commit hooks** (``.pre-commit-config.yaml``): ruff fix,
  trailing-whitespace, end-of-file-fixer, check-yaml, check-toml,
  check-merge-conflict, check-added-large-files, mixed-line-ending,
  and codespell. Same ruff scope as CI (``src tests scripts``).
- **``MANIFEST.in``** so the sdist includes ``LICENSE``, ``README``,
  ``CHANGELOG``, ``VALIDATION``, ``CITATION``, ``CONTRIBUTING``,
  ``CODE_OF_CONDUCT``, ``SECURITY``, the docs source, and the
  bundled examples — with ``external/`` and ``docs/_build/``
  excluded.
- **``noxfile.py``** with seven sessions: ``lint`` / ``type`` /
  ``tests`` / ``integration`` / ``docs`` / ``build`` /
  ``audit_validation``. Default session set is ``lint + type +
  tests + docs``.
- **New CI jobs in ``.github/workflows/ci.yml``**: a ``docs`` job
  that builds the Sphinx site and uploads it as a workflow
  artifact, and a ``sdist`` job that builds the source
  distribution and verifies it installs from source on a clean
  Python (complements the existing ``wheel-smoke`` job).
- **Release-readiness workflow**
  (``.github/workflows/release.yml``, manually triggered via
  ``workflow_dispatch``). Runs lint, type-check, default pytest,
  validation-matrix audit, sample verifier, Sphinx docs build,
  sdist, wheel, and cross-install version sanity; uploads the
  dist and docs as artifacts. Pre-tag gate that goes beyond the
  per-PR CI.
- **Unified :class:`pybmodes.io.ParseError` hierarchy** (Phase 1 PR
  A3). Six per-format subclasses
  (``BMIParseError``, ``ElastoDynParseError``, ``SubDynParseError``,
  ``WAMITParseError``, ``MoorDynParseError``, ``WindIOParseError``)
  let library callers catch parse errors precisely. Twenty-nine
  regression tests pin hashability and ``__eq__`` semantics
  (static review caught the original ``@dataclass`` shadowing of
  ``__hash__`` post-merge — see PR #68 follow-up).
- **Centralised numerical options dataclasses** (Phase 1 PR A1):
  :class:`pybmodes.options.SolverOptions`,
  :class:`pybmodes.options.FitOptions`,
  :class:`pybmodes.options.CheckOptions`. ``frozen=True`` dataclasses
  that surface every magic number from ``fem/solver``,
  ``elastodyn/params``, and ``checks`` as named, defaulted
  parameters.

### Changed

- **Re-licensed MIT → Apache 2.0.** ``LICENSE`` is the Apache 2.0
  text with copyright ``2024-2026 Jae Hoon Seo, Marine Structural
  Mechanics and Integrity Lab (SMI Lab), Inha University``;
  ``pyproject.toml`` classifier and README badge updated; the
  bundled ``_examples/`` and ``reference_decks/`` README provenance
  notes re-state the new licence. The book-citation reference to
  "MIT Press" in ``mooring.py`` is unrelated to project licensing
  and is preserved.
- **Upstream-data directories relocated to ``external/``** (issue
  raised in static review). ``docs/{BModes, OpenFAST_files, MoorPy,
  RAFT, references}`` → ``external/{...}``; the ``docs/`` tree is
  now reserved for the Sphinx documentation site. ``.gitignore``,
  local development notes, README, RELEASE_CHECKLIST,
  every case-study ``run.py``, every visualisation script, the
  reference-turbine ``build.py``, and 30 tracked files in total
  point at ``external/`` after the rewrite. Integration tests
  unchanged behaviourally; they skip when the data is absent.
- **``docs/RELEASE_CHECKLIST.md`` moved to
  ``https://pybmodes.readthedocs.io/en/latest/release_checklist.html``** to slot into the new Sphinx
  developer-guide toctree.
- **Strict mypy expanded** (Phase 1 PR A2). Five additional modules
  joined the strict-typing override list:
  ``pybmodes.models.tower``, ``pybmodes.models.blade``,
  ``pybmodes.campbell``, ``pybmodes.mac``, ``pybmodes.io.bmi``,
  ``pybmodes.options``.

### Refactored

- **CLI subcommands extracted to typed workflow functions** (Phase 2
  PRs B1–B3, merged in #69, #71, #72; #70 is the static-review
  P3 follow-up to B1). ``cli.py`` shrinks from ~1500 LOC to ~520
  LOC and is purely argparse + delegation. Every subcommand
  (validate / examples / patch / report / batch / campbell /
  windio) is now library-callable with a typed result. CLI flag /
  exit-code contract preserved byte-for-byte; the existing CLI
  smoke tests (``test_validate.py``, ``test_examples_cli.py``,
  ``test_batch.py``, ``test_report.py``, ``test_campbell.py``,
  ``test_windio_cli.py``) pass unchanged. Static-review P3 fix on
  PR B1: :func:`pybmodes.workflows.run_examples_copy` now
  preserves the accumulated skipped-bundle warning when a later
  bundle hits a destination conflict; regression test in
  ``test_workflows.py`` pins the contract.
- **``pybmodes.campbell`` split into a sub-package** (Phase 3 PR
  C1, merged in #73). The 1301-LOC monolith becomes seven files —
  ``result.py`` (271 LOC, :class:`CampbellResult` dataclass + NPZ
  / CSV serialization), ``_models.py`` (210 LOC, input
  dispatcher), ``_classify.py`` (119 LOC, mode-naming
  heuristics), ``_mac.py`` (90 LOC, MAC + Hungarian assignment),
  ``_sweep.py`` (347 LOC, rotor-speed sweep drivers + public
  ``campbell_sweep``), ``_plot.py`` (393 LOC, engineering-report
  diagram). ``__init__.py`` re-exports every public and underscore-
  private name existing tests import by name; no test edits
  required.
- **``pybmodes.mooring`` split into a sub-package** (Phase 3 PR
  C2, merged in #74). 1202-LOC monolith → six files —
  ``types.py`` (248 LOC, :class:`LineType` / :class:`Point` /
  :class:`Line` + :meth:`Line.solve_static`), ``system.py`` (608
  LOC, :class:`MooringSystem` + two ``from_*`` parsers),
  ``_catenary.py`` (113 LOC, residual + analytical Jacobian for
  Jonkman 2007 B-1/B-2 + B-7/B-8), ``_rotation.py`` (41 LOC,
  3-2-1 Euler primitive), ``_moordyn_parser.py`` (255 LOC,
  MoorDyn ``.dat`` section / row tokenisers). Public API
  (:class:`LineType`, :class:`Point`, :class:`Line`,
  :class:`MooringSystem`) byte-identical via ``__init__.py``
  re-exports.
- **``models/tower.py`` private helpers extracted** (Phase 3 PR
  C3, merged in #75). ``_run_validation_and_warn`` →
  ``models/_shared.py`` (cross-model; ``RotatingBlade.from_elastodyn``
  was already reaching across, now imports directly).
  ``_scan_platform_fields`` + ``_platform_inertia_matrix`` →
  ``models/_platform.py`` (tower-specific). ``tower.py`` ends
  at 968 LOC (down from 1083) with all three names re-exported
  for back-compat. The :class:`Tower` class stayed together;
  a planned mixin-based split was rejected by Plan-agent review
  as architecturally hollow.

Phase 4 — static-review hardening
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **``run_batch`` safety contract** (Phase 4 PR D1, merged in #77).
  ``run_batch(..., patch=True)`` previously called ``patch_dat``
  directly on every discovered side-deck. The per-deck loop now
  delegates to ``run_patch`` so the compute-before-write split +
  write-mode selection are reused unchanged. Three new parameters
  on ``run_batch`` plumbed through ``--dry-run`` / ``--backup`` /
  ``--no-backup`` / ``--output-dir`` on the CLI: ``dry_run``
  (default ``False``), ``backup`` (**default ``True`` — new in
  1.8.0**, since ``batch`` mutates a tree surfaced by recursive
  discovery, not a single hand-typed filename — single-deck
  ``run_patch`` keeps ``backup=False`` default), ``output_dir``
  (per-deck destination namespaces on the deck's *relative path
  under root* to avoid stem collisions between sibling sub-trees,
  fix from PR #77 static-review pass). Addresses static-review
  finding P1-3.
- **``run_windio`` ``on_skip`` policy** (Phase 4 PR D2, merged in
  #78). Three policies: ``"warn"`` (legacy permissive),
  ``"fail-on-data"`` (**new default** — computational skips toggle
  ``exit_code=1``; presentation + input skips warn), ``"fail"``
  (strict — any skip fails). Each internal skip site is classified
  as ``data`` (blade composite reduction), ``presentation``
  (Campbell / spectra plot rendering), or ``input`` (Campbell
  requested without a discovered ElastoDyn deck). CLI plumbing:
  ``pybmodes windio --on-skip {warn,fail-on-data,fail}``.
  **Behaviour change**: callers / automation that previously relied
  on ``run_windio`` returning ``exit_code=0`` on blade-extraction
  failure now see ``exit_code=1`` by default. Pass
  ``--on-skip warn`` (or ``on_skip="warn"`` library-side) for the
  pre-1.8.0 permissive default. Addresses static-review finding
  P1-2.
- **Validation workflow** (Phase 4 PR D3, merged in #79). New
  ``.github/workflows/validation.yml`` (``workflow_dispatch`` +
  weekly Monday cron) clones the upstream OpenFAST r-test + IEA-
  Task-37 reference-turbine repositories from public GitHub and
  runs ``pytest -m integration --tb=short`` hard-fail (no exit-5
  tolerance, unlike the per-PR ``ci.yml``). The verifier-report
  artifact is uploaded with 90-day retention. The new
  ``[![Validation](...)]`` badge on README + VALIDATION.md links to
  the workflow; the *Enforcement* paragraph in VALIDATION.md names
  the CI-validated coverage explicitly (NREL r-test family +
  IEA-Task-37 RWTs) and calls out the BModes CertTest 03/04 gap
  (the BModes CertTest decks are government-funded reference data not
  bundled in CI, so those cases stay maintainer-local enforcement).
  Addresses static-review finding P1-1.

### Fixed

- **Blade Campbell labels for operating-only sweeps.** In
  ``plot_campbell`` (issue #54 static-review follow-up), blade-mode
  labels were positioned using a comb scaled to ``xmax = rpm.max()``,
  which parked early-comb positions left of the blade curve for any
  sweep whose ``omega_rpm`` doesn't start at 0 (e.g.
  ``[6.9, 12.1]``). ``np.interp`` then silently clamped to
  ``curve[0]``, so the bracketed Hz was misleading too. The comb
  fraction is now scaled to the *curve domain*
  (``rpm.min() + frac × (rpm.max() - rpm.min())``) for blade
  labels; tower / platform labels (which sit on full-axis
  ``axhline``s) keep the original full-axis comb. Backward-compatible
  on ``rpm.min() == 0``. Regression test ``test_plot_campbell_blade_
  label_anchored_on_curve_for_positive_rpm`` pins the new behaviour.
- **``pybmodes.io.ParseError`` hashability** (PR #68 static-review
  follow-up). The original ``@dataclass`` decorator on the
  ``ParseError`` base shadowed ``__hash__`` to ``None``, breaking
  ``set(exceptions)`` callers. Switched to ``@dataclass(eq=False)``
  with 29 regression tests in ``test_io_errors.py``.

## [1.7.0] — 2026-05-18

### Added

- **``plot_campbell`` ``operating_rpm`` and ``freq_max`` (issue #54).**
  ``operating_rpm=(lo, hi)`` shades
  the operating rotor-speed window grey (outside stays white) and
  draws a ``↔ Operating Speed Range`` marker. ``freq_max`` sets the
  frequency-axis top; ``None`` (default) auto-caps just above the
  highest *structural* mode so the modes of interest fill the figure
  while the steep per-rev rays run off the top (standard Campbell-
  report framing). Both additive, keyword-only.

### Changed

- **``plot_campbell`` redesigned to an engineering-report
  convention (issue #54).** A multi-round, sample-driven redesign
  approved by the requester:

  - **Legend carries only the four family keys** — *Blades* (green),
    *Tower* (black), *Platform* (red), *Blade Passing* (blue) — in
    the **upper-left**; never a per-rev or per-mode entry.
  - Structural modes are coloured by family and **named inline along
    their lines** at staggered, white-backed positions (no
    right-margin column, no arrows), spelled out with the frequency
    in brackets — ``1st flapwise (0.68 Hz)``, ``1st Fore-Aft
    (0.48 Hz)``, ``surge/sway (0.008 Hz)``. The spelling-out
    (flap→flapwise, edge→edgewise, FA→Fore-Aft, SS→Side-to-Side) is
    figure-only; ``CampbellResult.labels`` keeps the terse tokens for
    CSV / API (serialisation contract unchanged).
  - Near-degenerate symmetric platform pairs are merged
    (``surge/sway``); the clustered low-frequency platform modes get
    **distinct red line styles** so they stay distinguishable, and a
    bunched sub-0.1 Hz cluster is decluttered in both x and y with
    thin leaders.
  - Per-rev rays are uniform blue with inline ``nP`` tags
    (white-backed, off the lines); the **default
    ``excitation_orders`` is now ``[1, 3, 6, 9]``** (was
    ``[1, 2, 3, 6, 9]``).

  No public-API signature break (the never-released interim
  ``resonance_margin`` knob from development was dropped before
  release); figure *appearance* changes substantially by design.

## [1.6.0] — 2026-05-18

### Added

- **``Tower.from_windio`` / ``Tower.from_geometry`` gain ``n_nodes``
  and ``tip_mass`` (issue #35).**

  - ``n_nodes`` — request an FE mesh of *N* evenly-spaced stations:
    the tower geometry (outer diameter / wall thickness) is linearly
    interpolated onto a uniform span and the closed-form tube
    reduction is recomputed **exactly** at each refined station (the
    geometry is interpolated, not the derived properties), so a finer
    mesh resolves the higher tower-bending mode shapes. Validated
    self-convergent and bias-free — on a uniform tube the refined
    result is identical to the native grid and matches
    Euler-Bernoulli; on a taper ``n_nodes=200`` vs ``400`` agree to
    < 0.2 % (1st) / < 2 % (4th). The WindIO *blade* path already had
    the analogous ``n_span``. ``None`` keeps the supplied grid (a
    uniform resample linearly smooths a deliberately *stepped*
    geometry — omit ``n_nodes`` to preserve steps).
  - ``tip_mass`` — a tower-top RNA lump as a
    :class:`pybmodes.io.bmi.TipMassProps` **or a bare float** (RNA
    mass in kg; offsets/inertia default to zero — the common case).
    Replaces the type-fragile ``tower._bmi.tip_mass = mass``
    workaround and mirrors :meth:`from_windio_floating`'s
    ``rna_tip``. ``from_geometry`` already accepted ``TipMassProps``;
    the bare-float convenience now works there and on
    ``from_windio``.

  Both are additive keywords — existing call sites are unchanged.

## [1.5.2] — 2026-05-18

### Added

- **``campbell_sweep`` accepts already-loaded models (issue #51).**
  ``input_path`` (and
  ``tower_input``) now take a path **or** a constructed
  :class:`~pybmodes.models.RotatingBlade` /
  :class:`~pybmodes.models.Tower` from *any* constructor —
  ``__init__``, ``from_elastodyn``, ``from_windio``,
  ``from_windio_floating``, … The model is used verbatim with **no
  disk re-read**, so there is a single point of load-in
  (``blade = RotatingBlade(...); campbell_sweep(blade, ω,
  tower_input=tower)``) and — crucially — a ``from_windio`` /
  ``from_elastodyn`` model, whose pre-built section properties no
  path can re-read, can finally be swept (previously ``.bmi`` /
  ElastoDyn ``.dat`` only). Routed to blade/tower by ``beam_type`` so
  either may be passed as the primary argument. Fully backward
  compatible — existing path / string calls are unchanged
  (non-breaking overload; the 1.x public signature is preserved).

### Fixed

- **Modern ``elastic_properties`` named flap/edge inertia is no
  longer silently swapped (issue #50 follow-up — static
  review).** The parametrised ``inertia_matrix`` principal
  moments were taken as the closed-form ``½(i_f+i_e) ∓ rad``
  eigenvalues, which for ``i_cp`` absent / zero reduces to
  ``min``/``max`` — so a schema-labelled blade with ``i_flap >
  i_edge`` had its rotary inertias transposed (``flp_iner`` ↔
  ``edge_iner``), shifting modal results. The 2×2 ``[[i_flap, i_cp],
  [i_cp, i_edge]]`` is now eigen-decomposed and assigned to the
  *named* axes by principal-axis alignment — the same label-
  preserving rule the full-6×6 path uses — so a diagonal
  (``i_cp = 0``) input passes through with its schema labels intact
  and a rotated one keeps flap/edge attached to the correct axis.

## [1.5.1] — 2026-05-18

### Fixed

- **WindIO published blade stiffness is now decoupled at the elastic
  / shear centre and principal elastic axes (issue #50).** The 1.5.0 published-properties
  path read the raw reference-axis 6×6 *diagonal*
  (``K44→EI_flap``, ``K55→EI_edge``, ``K66→GJ``). A WindIO /
  BeamDyn sectional 6×6 is expressed about the blade reference axis,
  **not** the elastic centre, and is generally **not** aligned with
  the principal elastic axes — so for any offset / pre-twisted blade
  the raw diagonal carries the axial↔bending (``K34``/``K35``) and
  bending↔bending (``K45``) coupling and is *not* the physical
  ``EI_flap``/``EI_edge``. New module
  :mod:`pybmodes.io._precomp.decouple` performs the standard
  rigid-offset congruence reduction (tension centre → principal-axis
  eigen-decomposition → shear centre for ``GJ``; Bauchau / the DNV
  Bladed *WindIO to Bladed* construction), implemented clean-room in
  numpy (the project's independence stance — no vendored reference
  code). Both WindIO dialects already ship the **full** 6×6 (modern
  ``stiffness_matrix`` carries the upper triangle ``K11``..``K66``;
  ``elastic_properties_mb`` the 21-element flatten), so the coupling
  is now honoured rather than discarded. The companion-BeamDyn
  validation oracle is decoupled the same way (apples-to-apples):
  decoupled-WindIO reproduces decoupled-BeamDyn to mass/EA < 3 %,
  EI < 5 %, GJ < 8 % on IEA-15, materially tighter and more correct
  than the 1.5.0 raw-diagonal copy where the section is offset.

### Changed

- **Default WindIO published-blade numerics shift for offset /
  pre-twisted blades.** Turbines whose published 6×6 has non-trivial
  axial/bending/twist coupling (every realistic blade) now yield the
  *decoupled* principal ``EI``/``GJ`` instead of the raw diagonal —
  *by design*, this is the issue #50 correctness fix. Magnitude is
  the difference between the reference-axis diagonal and the
  principal-elastic-axis value (can be tens of percent on coupled
  sections). The PreComp reduction path already did its own
  principal-axis reduction and is unchanged.

## [1.5.0] — 2026-05-18

### Added

- **WindIO blades now prefer the file's *published* distributed beam
  properties over the PreComp reduction (issue #48).** When a WindIO blade carries an
  ``elastic_properties`` block (modern, named ``K11``..``K66`` /
  ``mass`` / ``i_flap`` / ``i_edge`` — e.g. IEA-15) **or** an
  ``elastic_properties_mb.six_x_six`` block (the 21-element
  upper-triangular 6×6 flatten — e.g. IEA-22 / IEA-10),
  ``RotatingBlade.from_windio`` / ``windio_blade_section_props`` now
  use those distributed properties directly so pyBmodes matches the
  reference model exactly, instead of re-deriving them from the
  layup. New keyword ``elastic`` (default ``"auto"``): ``"auto"`` =
  published when present else PreComp; ``"precomp"`` = always reduce
  the layup (the pre-1.5 behaviour); ``"file"`` = require the
  published block. The decoupled-beam mapping is
  ``K33→EA, K44→EI_flap, K55→EI_edge, K66→GJ``;
  ``M11→mass/length, M44/M55→flap/edge inertia`` (off-diagonal
  coupling stays intentionally unmodelled — the documented
  diagonal-beam limitation; the point is to reproduce the *reference
  diagonal* exactly, not re-derive it).

- **Campbell / mode-shape plotting fixes (issue #47).** Three defects in the floating-case
  plots are fixed and two new opt-in knobs added:

  - ``plot_campbell(..., log_freq=True)`` no longer drops the per-rev
    excitation rays (1P / 2P / …). They were sampled on a two-point
    ``[0, rpm_max]`` grid, which is a straight line only on a linear
    axis — on a log-frequency axis the ``rpm = 0`` endpoint is
    ``log(0)`` and the segment collapsed. The rays are now sampled on
    a dense grid (and started just above zero on a log axis) so they
    render as the correct curve at either scale.
  - ``campbell_sweep`` now carries the FEM's own BModes-cross-
    validated platform-mode classification
    (``ModalResult.mode_labels``) through ``CampbellResult.labels``
    for a coupled floating tower: the six lowest tower columns come
    out named ``surge`` / ``sway`` / ``heave`` / ``roll`` / ``pitch``
    / ``yaw`` instead of the meaningless participation-derived
    ``"1st tower FA"`` …, and the flexible bending modes keep a
    *bending-only* ordinal (the first real bending mode is
    ``"1st tower FA"`` even with six rigid modes ahead of it).
    ``plot_campbell`` auto-draws those columns in the navy platform
    family with a period label — you no longer have to pass
    ``platform_modes`` by hand for the coupled case (it is still
    honoured, and merged/deduplicated, for the screening path).
  - ``plot_mode_shapes`` mode-line colours no longer collide: the
    styled ``apply_style`` palette is 7 colours, so 8+ modes wrapped
    the cycle and mode 8 reused mode 1's hue. Once the mode count
    exceeds the palette a perceptually-ordered continuous colormap is
    sampled so every mode is a distinct hue. New ``colors=`` kwarg
    (a colormap name or explicit list) on ``plot_mode_shapes`` /
    ``bir_mode_shape_plot`` / ``bir_mode_shape_subplot`` for full
    control.
  - ``plot_campbell`` draws the floating-platform 6-DOF rigid-body
    modes (surge / sway / heave / roll / pitch / yaw) as **legend
    entries**, each with its own colour and line style and a
    frequency + period label, instead of crowded right-margin
    annotations — so on a FOWT the six modes (≈ 0.008–0.12 Hz, all
    squeezed into the bottom sliver of a multi-Hz axis) are
    individually identifiable. The per-rev rays (1P / 2P / 3P / 6P /
    9P) remain in the legend. The remaining tower-bending right-margin
    labels are decluttered (spread apart with a thin leader to their
    unmoved line) so they can't stack either.
  - ``plot_environmental_spectra`` accepts ``rpm_constraint=False`` to
    suppress the wider placement-envelope band (and its legend
    entries) and draw only the operating *design* band — for callers
    with a fixed operating range but no separate constraint window.
    ``None`` still auto-draws the ±15 % band (unchanged default).

### Fixed

- **WindIO blade static-review hardening.**

  - **Blade twist units are now auto-detected.** ``np.degrees`` was
    applied to ``outer_shape*.twist`` unconditionally. That is correct
    for radian-convention windIO files (IEA-3.4: root ≈ 0.349 rad) but
    turned a degree-convention file's 15.6° root twist (IEA-15-240-RWT
    ships degrees) into ≈ 894°. ``_twist_to_degrees`` now decides by
    magnitude (a physical blade twist never exceeds ~2 rad), so both
    conventions yield the correct structural twist. Previously
    uncaught because the unit tests all used zero twist.
  - **A present-but-unparseable published elastic block no longer
    silently degrades.** ``_read_blade_elastic`` now distinguishes
    "absent" (silent PreComp fallback — correct) from "present but
    malformed / schema-drifted" (recorded on
    ``WindIOBlade.elastic_parse_error``). ``elastic="auto"`` emits a
    ``UserWarning`` naming the parse problem before falling back;
    ``elastic="file"`` raises — so a typo can't hide behind a
    plausible lower-fidelity result.
  - **Single-airfoil blades parse.** A constant-profile blade with one
    airfoil definition hit ``len(af_grid) - 2 = -1`` indexing; an
    empty schedule failed obscurely. Both are handled (reuse the one
    profile / clear ``ValueError``).
- **``outfitting_factor`` docs now match the implementation.**
  ``Tower.from_geometry`` and ``tubular_section_props`` documented
  scaling "mass density and rotary inertia"; the implementation
  deliberately scales only the distributed mass density (rotary
  inertia is a structural section property and stays unscaled). The
  docstrings now say so — behaviour unchanged.

### Changed

- **``plot_mode_shapes`` now shares one amplitude scale per mode
  (issue #47).** Previously each panel was normalised to its own
  peak, so a 95 %-flap / 5 %-lag mode looked full-height in *both*
  the flap and lag panels and you could not tell FA from SS (or a
  rigid platform DOF from a real bending mode). The new default
  ``normalize="mode"`` scales both panels by the single peak
  ``|displacement|`` across flap *and* lag, so the dominant direction
  reaches ±1 and the minor one stays proportionally small. Pass
  ``normalize="component"`` to reproduce pre-1.5 figures exactly.
- **Default WindIO-blade numerics shift for turbines that ship
  published elastic properties.** With the new ``elastic="auto"``
  default, IEA-15 (and any blade with an ``elastic_properties`` /
  ``elastic_properties_mb`` block) now uses the published stiffness /
  inertia rather than the PreComp diagonal reduction — *by design*,
  to minimise deltas to the reference model. Pass
  ``elastic="precomp"`` to recover the exact pre-1.5 behaviour. The
  PreComp reduction itself is unchanged, and its BeamDyn-class
  validation (``tests/test_windio_blade.py::
  test_windio_blade_vs_beamdyn_precomp_class``) is pinned to
  ``elastic="precomp"`` so it still validates the layup path.

### Notes

- Honest ecosystem-drift observation: IEA-15's published
  ``elastic_properties`` reproduces its companion BeamDyn 6×6
  diagonal tightly (mass / EA median < 3 %, EI / GJ < 5 % — verified
  in ``test_windio_blade_published_matches_beamdyn_far_tighter``),
  but **IEA-10's published ``elastic_properties_mb`` diverges ~50 %
  from its own companion BeamDyn deck** — the same
  reference-blocks-not-mutually-consistent pattern documented in
  ``cases/ECOSYSTEM_FINDING.md``. ``elastic="auto"`` treats the
  WindIO ontology as the canonical reference (issue #48's intent);
  use ``elastic="precomp"`` if you specifically want the
  layup-derived properties.

## [1.4.8] — 2026-05-17

### Changed

- **`CampbellResult._validate` consolidated to a single uniform
  shape contract — the special-cased empty-sweep exemption is
  removed.** That ad-hoc branch had leaked three successive
  edge-case findings (missing `omega_rpm`/`mac` checks, then `.size`
  vs `.shape`, then the same on more arrays). The structural fix:
  derive `(n_steps, n_modes)` from a (now always 2-D) `frequencies`
  and apply the *same* per-array checks unconditionally. A genuinely
  empty sweep is just the `n_steps == n_modes == 0` instance — it
  satisfies every check vacuously (canonical shapes `frequencies
  (0,0)`, `omega_rpm (0,)`, `participation (0,0,3)`, `mac (0,0)`),
  while any malformed zero-size variant fails the ordinary check it
  violates. The `mac_to_previous` rule is likewise tightened to
  "exactly the `(0,0)` default **or** `(n_steps, n_modes)`" (a
  `(2,0)`/`(0,2)` size-0 array is no longer accepted as "unset").
  Behaviour-equivalent for all well-formed results; the whole class
  of finding is now closed by design rather than by per-review
  patches.

## [1.4.7] — 2026-05-17

### Fixed

- **`CampbellResult` empty-sweep exemption now compares expected
  empty *shapes*, not just `.size`** (static-review follow-up — a
  real refinement of the 1.4.6 fix). A zero-*size* but wrong-*shape*
  array (`omega_rpm=(0,2)`, `mac_to_previous=(2,0)`,
  `participation=(2,0,3)`) is `.size == 0` yet implies steps/modes;
  the exemption now requires the canonical empty shapes exactly —
  `frequencies` ∈ {`(0,)`,`(0,0)`}, `omega_rpm` `(0,)`,
  `participation` `(0,0,3)`, `mac_to_previous` `(0,0)`. Regression
  test extended with the wrong-shape variants.

## [1.4.6] — 2026-05-17

### Fixed

- **`CampbellResult` empty-sweep exemption now also requires empty
  `omega_rpm` and `mac_to_previous`** (static-review follow-up — a
  genuine gap in the 1.4.5 tightening, not stale). A `(0, 0)` frequencies
  array with stray rotor-speed or MAC rows but no frequency rows
  previously passed validation and could be saved/loaded as an
  inconsistent archive; it is now rejected. Regression-tested.

## [1.4.5] — 2026-05-17

Third hardening round (static-review follow-up): six edge-case
invariant-tightening fixes. All fail-loud; no public name removed;
every fix has a regression test.

The two static-review comments in this round were **stale** — they
reviewed superseded commit diffs. Both were already fixed in the
released code: the injected-platform `radius + draft` length bug
(fixed 1.4.3 → released 1.4.4) and the shared-span check (already a
separate NPZ-only `_validate_shared_span_for_npz`, *not* called by
the JSON paths, with `tests/test_serialize.py` asserting JSON
round-trips per-mode span grids). No code change for those.

### Fixed

- **`CampbellResult._validate` no longer exempts a zero-size *shaped*
  array.** Only a genuinely empty sweep (no modes, no steps, no
  labels/participation/counts) is exempt; a `(0, 3)` array — size 0
  but implying 3 modes — is rejected instead of smuggling unvalidated
  metadata through.
- **Negative mode counts rejected.** `n_blade_modes` / `n_tower_modes`
  must be ≥ 0 (e.g. `-1 + 5 == 4` no longer passes).
- **`mac_to_previous` allows `NaN` but rejects `inf`.** `NaN` is the
  documented "not meaningful" sentinel; `±inf` is not and is now
  caught.
- **`participation` is validated as physical energy fractions** —
  non-negative, and each row sums to 1 (or to 0, the documented
  null-mode sentinel mirroring the `mac` NaN one). Negative entries
  or any other row sum is rejected. Applies to both `ModalResult` and
  `CampbellResult`.
- **Validated `distr_k` array is the one used.** The pipeline now
  divides the coerced float `ndarray` (not the raw
  `PlatformSupport.distr_k`), so a valid Python-list injection no
  longer passes validation and then fails on `list / scalar`.
- **Spectrum helpers reject a non-finite frequency input.**
  `kaimal_spectrum` / `jonswap_spectrum` now raise on `NaN` / `inf`
  in `f` (parameters were already finite-guarded).

## [1.4.4] — 2026-05-17

Second hardening round from a follow-up static-review pass, which
confirmed the 1.4.3 fixes landed. All
additive / fail-loud; no public name removed; every fix has a
regression test. (1.4.3 was a merged intermediate tag; its content
is included here — 1.4.4 is the published release.)

### Fixed

- **Result validators now reject non-finite scientific data.**
  `ModalResult._validate_lengths` and `CampbellResult._validate`
  raise on `NaN` / `inf` in the physical arrays (frequencies,
  mode-shape displacements/slopes/twist/span, participation,
  `omega_rpm`). `CampbellResult.mac_to_previous` stays exempt — `NaN`
  there is the documented "not meaningful" sentinel.
- **`ModalResult.to_json` emits standards-compliant JSON.**
  `json.dumps(..., allow_nan=False)` — a non-finite value now raises
  rather than writing the non-standard `NaN` / `Infinity` literals
  that strict JSON parsers reject (the finite guard above already
  fires first; this is the last line).
- **`ModalResult.save` validates a shared span grid.** Equal-length
  but *different* per-mode `span_loc` arrays used to silently reload
  every mode onto `shapes[0]`'s grid; this is now rejected.
- **Injected `PlatformSupport.distr_k` is fully validated.** Beyond
  the 1.4.3 sort check: matching `distr_k_z` / `distr_k` lengths,
  finite values, and non-negative stiffness — a hand-built support
  can no longer poison the FEM matrices late.
- **`plot_environmental_spectra` integer/finite edge cases.**
  `n_points` must be an integer ≥ 2 (no silent `int(2.9) → 2`
  truncation; `NaN`/`inf` raise the intended `ValueError`);
  `harmonics` likewise rejects non-integer / non-finite entries.
- **`pybmodes windio --rated-rpm` now visibly shapes the figure** —
  the 1P/3P *design* band is the operating range `cut-in → rated`
  inside a wider *constraint* band to `--max-rpm`, and the title
  states it; previously the flag only toggled a title word. CLI
  spectra / Campbell figures are closed after saving (no figure
  accumulation in batch runs).
- **`read_out` / `BModeOutParseError` are re-exported from
  `pybmodes.io`** and listed in the public API, matching the README
  prominence of `read_out(path, strict=True)`.

## [1.4.3] — 2026-05-17

Hardening release from two independent static-review passes. All
additive / fail-loud; no public name removed.

### Fixed

- **Injected-platform tower length was wrong for a non-zero floater
  draft (regression introduced in 1.4.2; magnitude: all modal
  frequencies shifted).** The FEM beam length is
  `radius + draft − hub_rad` (`make_params`); the deck path cancels
  this via `radius = tower_top`, but the new
  `from_windio_floating(..., platform_support=…)` branch passed
  `radius = flexible_length`, so a supplied platform with e.g.
  `draft = −20 m` modelled a tower 20 m too short and shifted every
  frequency. Now passes `radius = flexible_length − draft` (mirrors
  the deck path). A draft-sweep structural-invariant regression test
  pins `radius + draft == flexible_length`. *Only 1.4.2 is affected;*
  the deck-backed and screening tiers were always correct.
- **Loaders now validate on ingest, not only on export.**
  `ModalResult.load` / `from_json` and `CampbellResult.load` run the
  full schema check before returning; `ModalResult.load` also rejects
  ragged per-mode arrays with a clear message instead of an opaque
  `IndexError`. A corrupt / hand-edited archive fails loudly at load,
  not silently in downstream plotting/export.
- **`_validate_lengths` now requires 1-D `frequencies`** — a 2-D
  array with the same total size as `len(shapes)` previously slipped
  through the size-only check.
- **Strict `.out` parsing is now genuinely fail-loud.** A mode header
  with zero data rows raises under `strict=True` even when a later
  block parses (it used to vanish silently); the non-finite error now
  reports the *offending row's* line number, not the next header /
  EOF.
- **Environmental functions reject NaN / inf / physically invalid
  inputs.** `kaimal_spectrum` (sign/finite-guards `mean_speed`,
  `length_scale`, `sigma`, `turbulence_intensity`),
  `jonswap_spectrum` (`hs`, `tp`, finite `gamma`), and
  `plot_environmental_spectra` (`freq_max`, `n_points ≥ 2`, RPM
  bands, tower frequencies) now raise instead of producing a
  misleading figure.
- **`distr_k_z` monotonicity is enforced** before the
  distributed-soil-stiffness `np.interp`, which would otherwise
  silently return wrong stiffness for unsorted coordinates.
- **`pybmodes windio` floating spectra**: added `--min-rpm` /
  `--rated-rpm`; without an operating range the 1P/3P bands no longer
  silently start at DC — the figure is explicitly titled a
  *SCREENING envelope*. The tower-frequency pick distinguishes
  "not found" from a valid `0.0` (no longer uses `or`).
- Stale `_serialize` docstring corrected to match the
  `allow_pickle=False` security model; `src/pybmodes/plots/`
  environmental module now counts toward coverage.

### Notes

Each fix carries a regression test (corrupt NPZ/JSON, empty strict
mode block, line-accurate non-finite context, NaN/inf spectra, bad
plot inputs, draft-invariant beam length). Gates: ruff + mypy
(58 files) clean; default pytest 693 passed / 1 skipped /
121 deselected; integration 121 passed; validation-claims audit OK.

## [1.4.2] — 2026-05-17

### Added

- **Separately-designed floater input
  (`Tower.from_windio_floating(..., platform_support=...)`, issue #35).**
  The tower
  geometry comes from the WindIO ontology while the floating platform
  is supplied *verbatim* as a `PlatformSupport` (its own `A_inf` /
  `C_hst` / `mooring_K` / 6×6 inertia / `draft` / `ref_msl`) — the
  realistic workflow where the floater is designed separately (a
  frequency-domain tool / WAMIT export / published 6×6 set) and the
  rotor + tower come from the ontology. Nothing about the
  substructure is read from the yaml or any deck; it feeds the *same*
  BModes-JJ-validated free-free `PlatformSupport` FEM (the one that
  reproduces OC3 Hywind to ≈ 0.0003 %), so the path adds no new
  numerics — a regression test asserts byte-equivalence to the
  documented manual BMI recipe. Mutually exclusive with the companion
  decks (clear `ValueError`, not a silent precedence rule); optional
  `rna_tip` for the tower-top RNA lump; no screening warning (the
  caller owns the platform fidelity). `PlatformSupport` and
  `TipMassProps` are now exported from `pybmodes.io`.
- **6-DOF floating-platform rigid-body modes on the Campbell diagram**
  (issue #39).
  `pybmodes.campbell.plot_campbell` gains `platform_modes=[(dof, f),
  …]` and `log_freq=` (both default-off — the diagram is byte-
  identical without them). The six platform rigid-body modes
  (surge / sway / heave / roll / pitch / yaw) are drawn as
  rotor-speed-independent horizontal references — dotted navy,
  distinct from the tower dashed-grey lines — with right-margin
  labels carrying both frequency (Hz) and period (s), since the
  natural period is the design-relevant quantity for a floater;
  near-degenerate pairs (surge ≈ sway, roll ≈ pitch on a symmetric
  platform) merge into one label. The optional log-frequency axis
  lets the ~0.007–0.05 Hz platform modes and the ~0.3–5 Hz
  tower/blade modes read on one figure. The `pybmodes windio`
  floating path wires this automatically, sourcing the frequencies
  and DOF names from the already-BModes-cross-validated coupled solve
  (`ModalResult.mode_labels`); the plot adds no new numerics.

## [1.4.1] — 2026-05-17

### Added

- **Environmental-loading frequency-placement diagram**
  (`pybmodes.plots.plot_environmental_spectra`). The soft-stiff /
  frequency-separation figure used in reference-turbine design
  reports: normalised power spectral density versus frequency
  overlaying the Kaimal wind turbulence spectrum, the JONSWAP wave
  spectrum, the 1P / 3P rotor-excitation *design* and *constraint*
  bands, and the tower 1st fore-aft / side-side natural frequencies
  as vertical reference lines. The two closed forms
  (`kaimal_spectrum`, `jonswap_spectrum`) are exported and
  independently unit-tested against their analytic properties (the
  Kaimal low-frequency plateau + monotonicity; the JONSWAP peak at
  `1/Tp` and the `m0 = Hs²/16` significant-wave-height identity, the
  latter exact by construction). `pybmodes windio` auto-emits the
  figure for a floating turbine off the Campbell sweep (which supplies
  both the rotor-speed range and the rotor-speed-independent tower
  frequencies, so no site rpm / sea-state data is fabricated).
- **Opt-in strict `.out` parsing.** `read_out(path, strict=True)`
  raises `BModeOutParseError` — carrying the source file, the 1-based
  line number, and the mode context — on a short data row, a
  non-numeric or non-finite value, a duplicate mode number, or a file
  that yields no modes. The default (`strict=False`) stays tolerant,
  so the semver-frozen 1.x parser contract is unchanged; validation
  and cross-solver-comparison workflows opt in.

### Changed

- **Internal development-process scaffolding removed from source
  comments and docstrings** (sub-phase / review-pass labels and
  pointers to the local-only operations file). Format names and
  scientific citations are retained; behaviour is unchanged
  (comments / docstrings only). UK English throughout.

### Fixed

Post-1.4.0 code-review-pass hardening (all additive / behaviour-
preserving for well-formed input; no public name change):

- **NPZ load no longer enables pickle on the common path.**
  `ModalResult.load` / `CampbellResult.load` now open archives with
  `allow_pickle=False`. Modern archives have been pickle-free since
  pre-1.0 (`np.str_` `__meta__`); only a legacy `dtype=object`
  `__meta__` (pre-1.0 saves) now takes an explicit, `UserWarning`-
  announced `allow_pickle=True` fallback for that one member, instead
  of every load silently enabling pickle. Shared helper
  `pybmodes.io._serialize._read_npz_meta`.
- **`check_model` n_modes guard now uses the FEM's exact solvable
  DOF count.** It previously estimated `6 × n_nodes`, which
  *under*counts the true `n_free_dof` (the element carries 9 DOFs per
  global node) and raised a false `ERROR` for valid `n_modes` in the
  `(6·n_nodes, n_free_dof]` window. Now calls
  `pybmodes.fem.boundary.n_free_dof(nselt, hub_conn)` — exact for
  every boundary condition.
- **`pybmodes patch` rejects contradictory `--output` /
  `--output-dir`.** The two are aliases; giving them *different*
  paths now exits 2 with a clear message instead of silently
  honouring one and dropping the other. Single-flag and equal-value
  invocations are unchanged (the locked CLI contract only gains a
  rejection for genuinely-ambiguous input); the check now runs before
  any deck I/O.
- **Result dataclasses validate their documented array schema before
  export.** `ModalResult._validate_lengths` now also asserts
  `participation` is `(n_modes, 3)`; new `CampbellResult._validate`
  (called from `save` / `to_csv`) asserts the
  `omega_rpm` / `frequencies` / `labels` / `participation` /
  `mac_to_previous` / `n_blade+n_tower` consistency contract — so a
  malformed result can't be written to an archive or CSV that loads
  back inconsistent.
- **`ModalResult.to_json` drops the `json.dumps(default=str)`
  catch-all.** The payload is constructed entirely from JSON-native
  types; a non-native object reaching the encoder is a regression and
  now raises `TypeError` loudly rather than being silently
  stringified into an un-round-trippable blob. (The reviewed "
  participation serialised as a string" concern was a false positive
  — lines build a list comprehension, not a generator, and the JSON
  round-trip test exercises non-`None` participation.)

## [1.4.0] — 2026-05-17

### Added

- **One-click WISDEM/WindIO FOWT pipeline (issue #35).** A WindIO
  ontology `.yaml` (or an RWT directory) now goes end-to-end —
  composite-layup blade + tubular tower + (for a floating platform)
  the coupled platform rigid-body modes + an optional Campbell
  diagram + a bundled report — in a single command. New optional
  `[windio]` extra (PyYAML); the runtime core stays `numpy + scipy`
  only (same opt-in stance as `[plots]`), and an absent extra raises
  a friendly install hint rather than a bare `ModuleNotFoundError`.
  - **`pybmodes windio <ontology.yaml | RWT-dir>`** — the new
    seventh CLI subcommand. `_discover_windio_inputs(path)` resolves
    the ontology and auto-discovers companion HydroDyn / MoorDyn /
    ElastoDyn decks **scoped to the turbine root** (the nearest
    ancestor ≤ 4 levels up with an `OpenFAST/` child). A bare yaml in
    a scratch directory yields no decks — it never recursively scans
    an arbitrary parent, and never picks another turbine's decks.
    Flags: `--out --format {md,html,csv} --n-modes --water-depth
    --campbell --max-rpm --n-steps --n-blade-modes --n-tower-modes`.
  - **`RotatingBlade.from_windio(yaml_path, *, component='blade',
    n_span=30, rot_rpm=0.0, n_perim=300)`** — composite blade beam
    properties via a PreComp-class thin-wall multi-cell Bredt–Batho
    classical-lamination-theory reduction of the layup (new
    `pybmodes.io._precomp` sub-package + `pybmodes.io.windio_blade`),
    **not** a deck shortcut. Validated against each turbine's own
    WISDEM-PreComp-generated BeamDyn 6×6 across IEA-3.4 / 10 / 15 /
    22 (mass / EA PreComp-class; GJ / EI diagonal-reduction
    approximate — documented limitation, see `VALIDATION.md`).
    Resolves both WindIO key dialects plus WISDEM's parametric
    `fixed:` / `width` / `midpoint` layer forms (IEA-3.4 / IEA-10).
  - **`Tower.from_windio_floating(yaml_path, *, component_tower=
    'tower', water_depth=None, hydrodyn_dat=None, moordyn_dat=None,
    elastodyn_dat=None, rho=1025.0, g=9.80665)`** — the coupled FOWT
    constructor, **two-tier by design**. With the companion HydroDyn,
    MoorDyn, and ElastoDyn decks present (auto-discovered by the CLI,
    or passed explicitly) it builds the full deck-backed coupled
    model — byte-identical to the BModes-JJ-validated
    `from_elastodyn_with_mooring` path except the tower is the
    machine-exact WindIO one; **all six platform rigid-body modes +
    1st tower bending land at 0.0–0.3 %** vs that reference
    (reference grade). Without the decks it degrades to a WindIO-yaml
    member-Morison hydro + catenary-mooring **screening preview** and
    emits one `UserWarning` explicitly naming it
    `SCREENING-fidelity (NOT industry-grade)`.
  - **`MooringSystem.from_windio_mooring(floating, *, depth,
    moordyn_fallback=None, rho, g)`** — reuses the existing Jonkman
    elastic-catenary engine; line properties resolve explicit yaml →
    MoorDyn deck-fallback → studless-chain regression (with a
    `UserWarning` for the last).
  - **`pybmodes.io.windio_floating`** — `read_windio_floating`,
    `hydrostatic_restoring` (WAMIT/`.hst` buoyancy + waterplane
    convention), `added_mass` (Morison strip + RAFT `Ca_End`
    end-cap), `rigid_body_inertia`, `WindIOFloating`. Cross-validated
    against the IEA-15 UMaine VolturnUS-S potential-flow WAMIT `.hst`
    (heave 0.8 %, roll/pitch 1.6 %).
- `cases/iea15_volturnus_windio_walkthrough.ipynb` — an end-to-end
  Jupyter walkthrough of the one-click pipeline and the individual
  `from_windio*` constructors with engineering-paper-styled plots
  (mode shapes, Campbell, MAC). Data-dependent (upstream IEA-15 tree
  under gitignored `docs/`), so it lives under `cases/` rather than
  the contractually-synthetic `notebooks/`.

### Changed

- README, `VALIDATION.md`, `cases/ECOSYSTEM_FINDING.md`,
  and the `pybmodes.__init__` / `pybmodes.cli` docstrings document
  the WindIO one-click surface, the two-tier fidelity contract, and
  the new validation cluster. `VALIDATION.md` records the
  worst-observed margins for every new case; the structural-blocks
  counterpoint in `ECOSYSTEM_FINDING.md` is sharpened by the
  machine-exact IEA-15 WindIO geometry round-trip.

### Notes

- No `master` merge accompanies this release — 1.4.0 ships on the
  long-running `dev` branch by project decision (issue #35). The
  semver-frozen 1.x public surface is only **added to** (new
  constructors, a new CLI subcommand, new optional modules behind
  the `[windio]` extra); nothing on the existing frozen list is
  renamed or removed.

## [1.3.1] — 2026-05-14

### Fixed

- **Restored the `ModalResult` positional constructor ABI broken in
  1.3.0.** `ModalResult` is part of the semver-frozen 1.x public
  surface. 1.3.0 inserted the new `mode_labels` field *before*
  `metadata`, which shifted the generated dataclass `__init__`
  signature: an existing caller using the documented positional form
  `ModalResult(frequencies, shapes, participation, fit_residuals,
  metadata)` would have its `metadata` dict silently bound to
  `mode_labels` (leaving `metadata` unset, then either tripping
  `_validate_lengths()` or serialising bogus labels). `mode_labels`
  is now the **last** field — purely appended after `metadata` — so
  the pre-1.3.0 positional signature is byte-for-byte preserved and
  `mode_labels` remains keyword-constructible as before. A
  field-order guard comment and
  `tests/test_serialize.py::test_modal_result_positional_constructor_abi`
  pin this against recurrence. Surfaced in post-release review of
  #32.

## [1.3.0] — 2026-05-14

### Added

- **Floating-platform rigid-body modes are now named** (surge / sway
  / heave / roll / pitch / yaw). New optional
  `ModalResult.mode_labels` (one entry per mode, parallel to
  `shapes` / `frequencies`; `None` for a non-floating model). For a
  free-free floating model (`hub_conn = 2` with a `PlatformSupport`)
  a solve-time classifier
  (`pybmodes.fem.platform_modes.classify_platform_modes`) names the
  six platform rigid-body modes from the tower-base motion in the
  global eigenvector, weighted by the platform 6×6 inertia (the
  metric that makes a translation amplitude comparable to a
  rotation). Deliberately conservative: a flexible tower mode, a
  strongly coupled / eigensolver-rotated pair, or a duplicate
  dominant DOF is left `None` rather than mislabelled; only the
  lowest six modes are rigid-body candidates (a real floater's
  rigid-body periods sit far below the first tower-bending mode).
  Cantilever / monopile models keep `mode_labels = None`.
  - **Surfaced where it's useful.** `report.generate_report` adds a
    *Platform DOF* column to the mode-classification section (omitted
    for non-floating decks → existing reports unchanged);
    `plots.plot_mode_shapes` appends the DOF name to the legend;
    `cases/iea15_umainesemi_walkthrough.ipynb` now reads the labels
    off `mode_labels` instead of a hand-typed DOF-order list — which
    also fixed a latent error there (IEA-15 UMaine's modal order is
    surge/sway/**yaw**/roll/pitch/**heave**, not the textbook order
    the notebook previously assumed).
  - `mode_labels` round-trips through the NPZ and JSON serialisers.
  - Closes #31 (the v1.3.0 commit message references "#30" — a
    transcription slip; the resolved issue is #31, *Classification
    of modes*). Validation: `tests/test_platform_mode_labels.py`
    (default suite — bundled samples + classifier unit +
    serialization); `tests/test_floating_samples.py` integration
    r-tests run the classifier on three real upstream decks straight
    from their OpenFAST files (`from_elastodyn_with_mooring`) —
    IEA-15 UMaineSemi, IEA-22 Semi, NREL-5MW OC4 DeepCwind — and
    `tests/test_asymmetric_platform.py` covers the hand-authored
    `.bmi` route (symmetric and asymmetric — the hand-authored .bmi workflow).

### Fixed

- **`ModalResult.save` silently wrote a corrupt archive when
  `len(frequencies) != len(shapes)`.** The consistency check was
  gated on `mode_numbers.size` being non-zero, so a result with
  frequencies but no shapes skipped it and round-tripped mismatched.
  Validation is now a shared `_validate_lengths()` enforced by
  **both** `save` and `to_json` (the latter previously had no check
  at all), covering `frequencies` / `shapes` / `mode_labels` /
  `participation` lengths; only the fully-empty failed-solve case is
  exempt.
- **`.npz` archives are now loadable with `allow_pickle=False`.**
  `fit_residual_keys` was still written as an object array (pickle-
  backed), so any result carrying `fit_residuals` produced a pickled
  member despite `__meta__` already being pickle-free. It (and the
  new `mode_labels`) now use fixed-width Unicode arrays — every
  archive member is Unicode/numeric, restoring the
  `allow_pickle=False` invariant the serialiser module documents.
  Pinned by `tests/test_serialize.py::`
  `test_modal_result_npz_loads_without_pickle`.

## [1.2.2] — 2026-05-14

### Fixed

- **Incomplete `cm_pform` horizontal-offset pair now rejected.** The
  1.2.1 same-line extension treats the horizontal CM offsets as an
  `(x, y)` pair, but the parser accepted a leading numeric run of 2
  (`<cm_pform> <cm_pform_x>  cm_pform : …`, the `y` omitted) and
  silently defaulted `cm_pform_y = 0.0`, turning a malformed
  hand-authored line into a plausible-but-wrong platform geometry
  instead of an input error. `_read_cm_pform_line` now requires the
  run to be exactly 1 (symmetric) or 3 (asymmetric) and raises a
  `ValueError` naming the count otherwise — consistent with the
  parser's "raise on malformed, never silently default" stance.
  Valid 1- or 3-value lines (every bundled sample, the OC3 cert deck,
  correctly hand-authored asymmetric decks) are unaffected.
  Surfaced in post-merge review of #28. Test:
  `tests/test_asymmetric_platform.py::`
  `test_incomplete_cm_pform_offset_pair_rejected`.

## [1.2.1] — 2026-05-14

### Added

- **Hand-authored asymmetric `.bmi` support** — the v1.2.0 horizontal
  platform-CM capability now reaches the `.bmi` text format, not just
  `Tower.from_elastodyn_with_mooring`. The `cm_pform` line accepts an
  optional pair of trailing numbers —
  `<cm_pform> [<cm_pform_x> <cm_pform_y>]  cm_pform : <comment>` —
  read as the leading numeric run (the label word terminates it).
  This is **zero new lines**: every pre-1.2.1 deck (the canonical
  `OC3Hywind.bmi`, all bundled samples, hand-authored fixtures) has a
  single leading number followed by the label, so it parses
  identically with `cm_pform_x = cm_pform_y = 0`. The build.py writer
  emits the two extra numbers **only when non-zero**, so every
  symmetric bundled sample regenerates byte-identically (verified: no
  content diff across all 11 samples + 6 reference decks). Closes the
  remaining half of #22 — the requester drives pyBmodes from
  hand-authored `.bmi` files (no OpenFAST), which the v1.2.0
  in-memory-only path did not cover.
  - Validation (extends `tests/test_asymmetric_platform.py`, default
    suite): a symmetric platform still emits the legacy single-value
    line; a hand-authored asymmetric `.bmi` round-trips
    (`emit → read_bmi`) preserving the offsets; and the parsed offset
    reaches the solver end-to-end (spectrum shifts, `n_modes`-stable).

## [1.2.0] — 2026-05-14

### Added

- **Asymmetric floating-substructure support — horizontal platform-CM
  offset (`PtfmCMxt` / `PtfmCMyt`).** Through 1.1.x the platform
  rigid-arm transform in `pybmodes.fem.nondim` carried only a
  *vertical* lever (`cm_pform − draft`); a floating platform whose
  centre of mass is offset horizontally from the tower axis (an
  asymmetric semi / barge) had its surge↔yaw, sway↔yaw and
  heave↔bending-slope couplings under-modelled. The transform is now
  the full 3-D rigid-body kinematic transfer
  `G = [[I₃, −skew(r)], [0, I₃]]` for the complete arm
  `r = (PtfmCMxt, PtfmCMyt, cm_pform − draft)`, so `Gᵀ M G` produces
  the translation↔rotation coupling **and** the full 3-D
  parallel-axis rotational block automatically.
  - `Tower.from_elastodyn_with_mooring(...)` now reads `PtfmCMxt` /
    `PtfmCMyt` from the ElastoDyn deck and applies them (previously
    scanned but discarded).
  - New optional `PlatformSupport.cm_pform_x` / `cm_pform_y` fields
    (default `0.0`). Adding defaulted dataclass fields is non-breaking
    under semver; the `.bmi` text format is unchanged, so hand-authored
    decks and every bundled sample are byte-identical and a
    hand-authored asymmetric `.bmi` is a possible future extension.
  - **Strict superset:** when `rx = ry = 0` the transform is
    byte-identical to the pre-1.2.0 vertical-only form, so every
    axisymmetric spar / symmetric semi (OC3 Hywind, the IEA-15 /
    IEA-22 / OC4 / UPSCALE samples) is numerically unchanged — the
    OC3 Hywind cert test still matches BModes JJ at 0.0003 %.
  - Validation: `tests/test_asymmetric_platform.py` (default suite,
    self-contained) pins (1) the `rx=ry=0` byte-identical guarantee,
    (2) the rigid-body kinematic structure of the 3-D transform,
    (3) a non-circular closed-form point-mass spatial-inertia
    (parallel-axis) transfer through the actual transform, and
    (4) end-to-end wiring + `n_modes`-stability on the bundled
    sample-09 tower. Closes the asymmetric-systems request in #22.

## [1.1.2] — 2026-05-14

### Fixed

- **Physical floating `axial_stff` was scaled by the `AdjTwMa`
  mass-tuning knob.** The `physical=True` section-property path
  derives `axial_stff = E·A` from the section mass density via
  `E·A = (ρ·A)·(E/ρ)`. It used the *adjusted* density
  `mass_den = TMassDen · AdjTwMa`, so a deck tuning tower mass through
  `AdjTwMa` (a mass-only calibration knob — stiffness tuning is
  `AdjFASt` / `AdjSSSt`) silently scaled the synthesised axial
  stiffness too, which could re-soften/stiffen the axial DOF and
  reintroduce the very conditioning collapse the physical path exists
  to prevent. `axial_stff` now uses the *structural* (un-adjusted)
  `TMassDen`, since `AdjTwMa` inflates effective mass without changing
  cross-sectional area; the adjusted density still feeds the mass
  matrix. The bundled `floating_with_mooring` samples now serialise
  the adapter's already-correct `SectionProperties` verbatim
  (single source of truth) instead of re-deriving them. Material on
  IFE UPSCALE 25 MW (its tower deck sets `AdjTwMa = 1.012`, so its
  bundled axial column was 1.2 % too stiff); the other reference
  decks are `AdjTwMa = 1` and are numerically unchanged. Surfaced in
  post-merge review of the v1.1.1 / `#25` work.
- **`Tower.from_elastodyn_with_mooring` carried the same ill-conditioned
  axial proxy the bundled samples did (v1.1.1).** The v1.1.1 fix
  repaired `build.py`'s sample emitter, but the in-memory
  ElastoDyn→pyBmodes adapter (`_stack_tower_section_props`) still
  synthesised `axial_stff = 1e6·EI` (~5e6× too stiff for a real steel
  tower) for *every* tower path. Harmless for the clamped-base
  cantilever / monopile constructors (base axial + torsion DOFs
  locked, out of band — the validated cert frequency targets are
  unaffected and unchanged), but a user driving their own asymmetric
  spar / semi deck through `Tower.from_elastodyn_with_mooring(...)`
  still hit the conditioning collapse: the soft platform rigid-body
  modes drifted with the requested mode count instead of resolving to
  the physical surge / sway / heave / roll / pitch / yaw spread.

  The free-base floating path now threads `physical_sec_props=True`
  through `to_pybmodes_tower`, so `_stack_tower_section_props` emits
  the same exact homogeneous-steel material identities the bundled
  floating samples use (`axial_stff = mass_den·E/ρ`,
  `flp_iner = flp_stff·ρ/E`, `tor_stff = EI/(1+ν)`). After the fix
  the IEA-15 UMaine VolturnUS-S deck solved via
  `from_elastodyn_with_mooring` is `n_modes`-stable with a physically
  distinct rigid-body spectrum, matching the bundled sample. The
  cantilever / monopile path keeps the proxy (`physical=False`
  default); `test_5mw_tower_frequency_target` (0.3324 Hz) and
  `test_iea34_tower_frequency_sanity` are byte-for-byte unchanged.

  Regression: `tests/test_floating_samples.py::`
  `test_from_elastodyn_with_mooring_spectrum_is_nmodes_stable`
  (integration) pins the in-memory path, complementing the
  default-suite `test_floating_samples_spectra` bundled-BMI gate.

## [1.1.1] — 2026-05-14

### Fixed

- **Bundled floating reference-turbine samples emitted with
  ill-conditioned section properties.** `build.py::_emit_tower_sec_props`
  synthesised `axial_stff = 1e6·EI` (~5e6× too stiff for a real
  steel tower), `tor_stff = 100·EI`, and a near-zero rotary-inertia
  floor. Those cantilever proxies are harmless for the clamped-base
  land / monopile samples (base axial + torsion DOFs locked, out of
  band) but for the FREE-base floating samples (`hub_conn = 2`) they
  wrecked the conditioning of the global matrices. On the
  OC3-Hywind-style asymmetric platform (which routes through the
  general `scipy.linalg.eig` path) the soft rigid-body modes
  collapsed into a single degenerate value whose magnitude *drifted
  with the requested mode count* (≈ 0.11 Hz at `n_modes=9` →
  0.07 Hz at `n_modes=15`), while the tower-bending pair stayed
  roughly right — so the build-time "1st-FA > 0.3 Hz" check and the
  PlatformSupport round-trip test both missed it. Pre-existing since
  the OC3 Hywind sample (sub-case 07) was first generated; the
  `test_certtest_oc3hywind` cert test validates the *solver* against
  the canonical `OC3Hywind.bmi`, never the bundled sample, so it
  never caught this.

  The floating section-property emitter now uses exact
  homogeneous-steel material identities — `axial_stff = mass_den·E/ρ`,
  `flp_iner = edge_iner = flp_stff·ρ/E`, `tor_stff = flp_stff/(1+ν)`
  (thin-wall circular tube, ν = 0.3, ρ = 8500 kg/m³ effective) — all
  of which reproduce the canonical OC3 Hywind section table to the
  printed digits. Post-fix the bundled OC3 Hywind sample reproduces
  the BModes JJ reference spectrum (Jonkman 2010, NREL/TP-500-47535)
  to within ~ 0.2 % across the first nine modes and is `n_modes`-
  stable; the IEA-15 / IEA-22 / OC4 / UPSCALE semi samples gain a
  cleaner surge≈sway degeneracy too. The clamped-base land /
  monopile samples and the cert tests are unchanged (the proxy path
  is retained, gated behind the new `physical=` flag).

  New self-contained regression `tests/test_floating_samples_spectra.py`
  pins both invariants the old code violated: `n_modes`-stability
  for every bundled floating sample, and the OC3 Hywind sample vs the
  BModes JJ reference spectrum at 0.5 % tolerance. Runs in the
  default suite (no external data).

## [1.1.0] — 2026-05-14

### Added

- **Four new floating reference-turbine samples** under
  `src/pybmodes/_examples/sample_inputs/reference_turbines/`:
  - **08 NREL 5MW on the OC4 DeepCwind floating semi-submersible**
    (Robertson 2014, NREL/TP-5000-60601). 1st-FA bending ≈ 0.45 Hz.
  - **09 IEA-15-240-RWT on the UMaine VolturnUS-S floating semi**
    (Allen 2020, NREL/TP-5000-76773). 1st-FA bending ≈ 0.53 Hz —
    matches the v1.1 redesigned tower target within engineering
    tolerance. Same physics as the
    `cases/iea15_umainesemi_walkthrough.ipynb` end-to-end notebook.
  - **10 IEA-22-280-RWT on the IEA Wind Task 55 semi** (Bortolotti
    2024, technical report in preparation). 1st-FA bending ≈ 0.34 Hz.
  - **11 IFE UPSCALE 25 MW (CentralTower) on a floating semi**
    (Sandua-Fernández 2023). 1st-FA bending ≈ 0.44 Hz.

  Each sample carries `<id>_tower.bmi` (free-free tower with full
  6×6 `PlatformSupport` block — hydro added-mass / hydrostatic
  restoring / mooring stiffness / platform inertia all assembled
  programmatically from the upstream OpenFAST ElastoDyn + HydroDyn +
  MoorDyn decks via `Tower.from_elastodyn_with_mooring`) plus
  `<id>_blade.bmi` and per-sample `README.md`. The previously-shipped
  OC3 Hywind spar sample (sub-case 07) was the only pre-1.1 floating
  sample; floating coverage now spans 5 platform types (1 spar +
  4 semis) across 4 RWT designations (NREL 5MW, IEA-15, IEA-22,
  UPSCALE 25MW).

  Together with the existing 7 fixed-base samples (01-07) this
  brings the bundled reference-turbine library to **11 samples**.
  Closes the *Planned for 1.1+ — Additional floating reference-
  turbine samples* item from the 1.0 release notes.
- **`tests/test_parser_negative_paths.py` — comprehensive parser
  audit.** 28 tests covering every parser entry point against the
  rubric `{well-formed, truncated count/table, bad numeric token,
  non-finite token, cross-reference mismatch, path normalization}`.
  Parsers covered: BMI, section-properties, SubDyn, WAMIT,
  HydroDyn, ElastoDyn, MoorDyn, `.out`. Most negative behaviours
  raise `ValueError` with file + row context; the WAMIT and `.out`
  parsers deliberately tolerate header-like rows that don't fit
  the numeric schema (documented per-parser). The audit was the
  rubric proposed at the end of pass 4 — one-shot gate replacing
  ongoing whack-a-mole reviews. Pass-5 static review.

### Fixed (fifth post-1.0 static-review pass)

- **BMI `_LineReader.read_ary(n)` silently truncated.** A line with
  fewer than `n` tokens used to slice down to whatever was present;
  callers like `el_loc=np.array([...])` then failed downstream with
  shape / broadcast errors that didn't name the source file or row.
  Now raises `ValueError` with the BMI line number, expected and
  actual token counts, and the offending line text. Common cause
  is a wrapped line or a missing element-count scalar earlier in
  the deck. Pass-5 static review.
- **SubDyn `MEMBERS` table raised bare `IndexError` on short
  rows.** A member row with fewer than 5 columns indexed off the
  end of the parsed row list. Now raises `ValueError` naming the
  row index, source file, and the offending row text. Mirrors the
  `_lookup_joint` error-message style from pass 2. Pass-5 static
  review.
- **WAMIT `.1` / `.hst` parsers silently accepted non-finite
  numeric entries.** A stray `nan` / `inf` in an `A(i,j)` /
  `C(i,j)` cell propagated into the `PlatformSupport` matrices
  with no diagnostic. Now uses a two-tier `_parse_fortran_float` /
  `_parse_fortran_float_lenient` split: the lenient parser is
  inside the "is this a schema-matching row?" try block (where
  `ValueError` continues to mean "skip this header / comment
  line"), and an explicit `_require_finite` call outside the try
  raises with line + `A(i,j)` / `C(i,j)` context when an
  otherwise-schema-matching row carries a non-finite value. The
  strict-finite `_parse_fortran_float` continues to be used for
  HydroDyn's one-shot scalar reads (`WAMITULEN` etc.). Pass-5
  static review.
- **ElastoDyn `_parse_float` silently accepted non-finite values.**
  Pass 4 tightened the BMI and section-properties parsers but
  missed the ElastoDyn-specific copy in
  `pybmodes.io._elastodyn.lex._parse_float`. The pass-5 audit
  itself caught this — `TipRad = inf` parsed cleanly and produced
  a non-physical model. Now rejects non-finite results consistent
  with the other parsers.

### Fixed (fourth post-1.0 static-review pass)

- **`check_model` silently passed models with NaN / Inf section
  properties.** Every downstream comparison (`mass_den <= 0`,
  `np.diff(span) <= 0`, stiffness-jump ratios) returns False on NaN,
  so a section-properties table with `nan` / `inf` could slip into
  the eigensolver and produce NaN frequencies with no upstream
  diagnostic. New `_check_section_properties_finite` runs FIRST and
  fires an ERROR-severity `ModelWarning` naming the field and first
  offending index. Pass-4 static review.
- **`bmi._parse_float`, `sec_props._parse_fortran_float`, and the
  MoorDyn LINE TYPES / POINTS strict-parse paths accepted `nan` /
  `inf`.** A stray non-finite literal in any numeric field silently
  produced a non-physical model. All three reject non-finite values
  with a clear `ValueError`; `sec_props` uses a two-stage parse
  (loose first, then `_is_finite` check) so trailing notes after
  the data table still break the loop cleanly while a numeric
  nan/inf raises with row + column context. Pass-4 static review.
- **MoorDyn OPTIONS silently swallowed malformed `WtrDpth` / `rhoW`
  / `g` values** via `try / except: pass`. `rhoW` directly feeds
  the wet-weight formula `w = (m_air - rho_w · A) · g`, so a typo
  silently shifted every mooring stiffness. The three recognised
  keys now route through `_parse_finite_option` which raises;
  unknown keys remain permissive. Pass-4 static review.
- **ElastoDyn tower / blade distributed-property tables truncated
  silently** when a row was short or contained a non-numeric token
  — the loop broke and downstream consumers got fewer stations than
  the file's declared `NTwInpSt` / `NBlInpSt`. The parsers now
  cross-check the parsed row count against the declared count and
  raise `ValueError` with the gap named. Pass-4 static review.
- **`pybmodes.fitting.fit_mode_shape` lacked input validation.**
  Empty arrays raised `IndexError` on `y[-1]`; shape mismatch raised
  broadcasting errors; non-finite inputs produced NaN coefficients;
  non-monotonic span produced a silently degenerate fit. Now
  validates 1-D shape, matching lengths, ≥ 2 stations, all-finite
  values, and strictly-increasing span_loc up front; bad inputs
  raise `ValueError` with actionable messages. Pass-4 static
  review.
- **`ElastoDynMain.compute_rot_mass` ignored `AdjBlMs`.** The blade
  adapter `to_pybmodes_blade` already applies the scalar; the
  user-facing method on the dataclass didn't, so a caller using
  `compute_rot_mass` directly on a deck with `AdjBlMs ≠ 1` got an
  under- / over-reported rotor mass. Now multiplies through.
  Pass-4 static review.
- **`_serialize._metadata_to_npz_value` stored metadata as
  `dtype=object`** (pickle-backed) even though the module docstring
  promised pickle-free loading. Switched to `dtype=np.str_` so the
  archive loads cleanly under `np.load(..., allow_pickle=False)`.
  Files written by older pyBmodes versions still load via the
  `allow_pickle=True` kwarg `ModalResult.load` / `CampbellResult.load`
  continue to pass. Pass-4 static review.

### Fixed (third post-1.0 static-review pass)

- **`[notebook]` optional extra was incomplete.** The notebook test
  installed `nbclient` / `nbformat` / `ipykernel` but the notebooks
  themselves import `matplotlib.pyplot` and `pybmodes.plots`, so
  `pip install -e ".[dev,notebook]"` followed by `pytest` produced a
  `ModuleNotFoundError` from inside `nbclient` rather than the
  documented clean SKIP. CI installed `[dev,plots,notebook]` together
  so this never showed in CI but bit contributors who took the
  documented `[notebook]` extra at face value. The extra now also
  carries `matplotlib>=3.7`. Pass-3 static review.
- **`cases/*/run.py` docstrings — `\s` invalid-escape `SyntaxWarning`.**
  The earlier `D:\repos\...` → `%CD%\src` scrub was scoped to
  ruff's coverage (`src/ tests/ scripts/`), so `cases/` slipped
  through with a single backslash. Python 3.12 emits a
  `SyntaxWarning` for this; running the script under `-W error` (or
  in a future Python where invalid escapes become hard errors)
  fails before the script can start. All five affected files
  (`bir_2010_floating`, `bir_2010_land_tower`, `bir_2010_monopile`,
  `iea3mw_land`, `nrel5mw_land`) now use the double-escaped
  `%CD%\\src` form. Pass-3 static review.

### Added

- **`tests/test_cases_compile_clean.py` — ratchet test.** Compiles
  every `cases/*/run.py` (plus a `compileall` walk over the whole
  `cases/` tree) with `SyntaxWarning` promoted to an error, so the
  W605 invalid-escape regression class can't slip back in. The
  `cases/` tree is deliberately outside ruff's scope (the case
  studies are exploratory and shouldn't bear full lint
  conformance), so this targeted compile-clean check is the right
  granularity. Pass-3 static review.

### Fixed (second post-1.0 static-review pass)

- **`MooringSystem.from_moordyn` — silent malformed-row drops.** The
  LINE TYPES / POINTS / LINES section parsers used to `continue` on
  rows that failed `len(parts) < N` or per-column `float` / `int`
  parsing, which turned a typo into an incomplete mooring model with
  no diagnostic. The parsers now raise `ValueError` with the section
  name, the offending row text, and the source file path on rows
  that look like data but fail strict parsing. Rows that look like
  column-name or units headers are still skipped — `_looks_like_header_row`
  identifies them by checking whether every first-four token parses
  as numeric (data) or whether the entire row is parenthesised
  (units). Pass-2 static review.
- **`MooringSystem._split_sections` — hardcoded 2-row header
  assumption.** The splitter previously skipped exactly two rows
  after every section divider. A MoorDyn variant shipped with only
  one header row (column names but no units line) had its first
  data row silently eaten. The splitter now inspects each post-
  divider row and only skips it if `_looks_like_header_row` flags
  it; the moment a data-looking row appears, the inspect-and-skip
  loop stops. Handles 0 / 1 / 2 header rows. Pass-2 static review.
- **SubDyn adapter — bare `StopIteration` on missing reaction /
  interface joint IDs.** A SubDyn deck whose `BASE REACTION` or
  `INTERFACE JOINTS` block referenced a joint ID absent from
  `STRUCTURE JOINTS` produced an uninformative `StopIteration`
  from the `next(...)` generator expression. Now routed through a
  `_lookup_joint(subdyn, joint_id, role)` helper that raises
  `ValueError` naming the missing ID, the role ("reaction" or
  "interface"), the source file, and the known joint IDs — matches
  the existing `_circ_prop_for` error-message style. Pass-2 static
  review.

### Changed

- **`cases/ECOSYSTEM_FINDING.md` — refreshed OC3 spar footnote.**
  The footnote on the polynomial-comparison table said pyBmodes had
  no `from_elastodyn` path for floating decks because parsing
  HydroDyn + MoorDyn into a 6 × 6 PlatformSupport was "out of scope".
  Stale since `Tower.from_elastodyn_with_mooring` landed before 1.0.
  The case-study table continues to use the BMI deck (the
  cantilever `hub_conn = 1` basis is the only one ElastoDyn's `SHP`
  ansatz can represent) — the footnote now spells out *why* rather
  than implying the path doesn't exist. Pass-2 static review.
- **`src/pybmodes/_examples/sample_inputs/README.md` — broken
  `../../external/BModes` link.** The relative link to
  `../../external/BModes/docs/examples/` didn't resolve from the
  packaged-wheel location, and the target is gitignored under the
  Independence stance anyway. Replaced with plain text that names
  the path and explains the local-only / upstream-clone story
  inline. Pass-2 static review.
- **`notebooks/walkthrough.ipynb` — closing-section "sparse eigsh
  on the roadmap" claim refreshed.** The sparse shift-invert
  `scipy.sparse.linalg.eigsh(sigma=0, mode='normal')` path landed in
  `pybmodes.fem.solver` before 1.0 and activates automatically on
  symmetric problems with a small mode-count request. The closing
  text now describes that and points at
  `scripts/benchmark_sparse_solver.py` for the timing study. Pass-2
  static review.

### Added

- **`wheel-smoke` CI job.** New matrix job (Python 3.11 + 3.12) that
  builds the wheel via `python -m build`, installs it into a fresh
  `python -m venv`, asserts `importlib.metadata.version("pybmodes")`
  matches the `[project] version` line in `pyproject.toml`,
  exercises `pybmodes examples --copy` end-to-end, and runs the
  sample verifier against the installed wheel. Catches packaging
  regressions the editable-install `test` job can't see: missing
  `package_data` entries, wheel-build failures, `sys.path` leaks
  from the source tree, and version drift between `pyproject.toml`
  and the dist-info metadata.
- **`tests/test_notebooks.py`.** Three tests covering the two
  bundled walkthrough notebooks:
  - `notebooks/walkthrough.ipynb` (synthetic, default suite) —
    executes every cell via `nbclient` and asserts no
    `CellExecutionError`. Was previously not in CI; a refactor
    that silently broke a cell would ship without warning (and
    one did — see *Fixed*).
  - `cases/iea15_umainesemi_walkthrough.ipynb` — split into two
    paths. The default-suite test asserts that with the upstream
    OpenFAST decks absent, the first code cell raises a friendly
    `FileNotFoundError` carrying the documented "Clone the
    upstream IEA-15-240-RWT" hint. Previously this was just a
    design contract; now it's a tested invariant. The
    `@integration`-marked counterpart executes every cell when
    the upstream data IS present.
- **New `[notebook]` optional extra** in `pyproject.toml` —
  `nbclient` / `nbformat` / `ipykernel`, test-time-only deps for
  the headless notebook execution path above. CI installs
  `.[dev,plots,notebook]` so the notebook tests run in the default
  suite.

### Fixed

- **`notebooks/walkthrough.ipynb` — missing `PALETTE` and `OI`
  references.** The `plot_mode_shapes_paper` and
  `plot_fit_quality_paper` helpers in the setup cell referenced
  `PALETTE[1:]` and `OI['verm']` / `OI['blue']` / `OI['green']` but
  never imported / defined them — silent `NameError` since the
  apply_style refactor in commit `00ff4c7`. The new notebook-
  execution CI step caught this on the first run. Setup cell now
  imports `from pybmodes.plots.style import PALETTE, apply_style`
  and defines `OI = {'verm': PALETTE[1], 'blue': PALETTE[2], 'green': PALETTE[3]}`.

- **Default-suite tests for three previously integration-only modules.**
  `tests/test_coords.py` covers the `pybmodes.coords` 6-DOF naming
  contract (`DOF_NAMES` ↔ `DOF_INDEX` agreement). `tests/test_elastodyn_writer.py`
  exercises the ElastoDyn writer's parse → emit → re-parse fixed point
  against the bundled NREL 5MW reference deck under
  `src/pybmodes/_examples/reference_decks/nrel5mw_land/` (no upstream-
  data dependency). `tests/test_subdyn_reader.py` exercises the SubDyn
  parser and the `SubDynCircProp` derived properties against a
  synthetic snippet emitted to `tmp_path`. Coverage on
  `src/pybmodes/coords.py` rose from 0 → 100 %, `io/_elastodyn/writer.py`
  from 6 → 82 %, and `io/subdyn_reader.py` from 0 → 71 % in the default
  pytest run.

### Changed

- **IEA-15 UMaineSemi walkthrough relocated `notebooks/ → cases/`.** The
  walkthrough at `notebooks/iea15_umainesemi_walkthrough.ipynb` depends
  on upstream OpenFAST decks under `external/OpenFAST_files/` (gitignored
  per the Independence stance), so a fresh clone got a notebook that
  errored on the first cell. The `notebooks/` directory is contractually
  self-contained (`notebooks/walkthrough.ipynb` runs entirely on
  inline synthetic cases); data-dependent walkthroughs belong under
  `cases/` alongside the existing `run.py` case studies. Moved to
  `cases/iea15_umainesemi_walkthrough.ipynb` and the `sys.path`
  prologue rewritten to walk up from CWD looking for `src/pybmodes`
  so the notebook works regardless of where Jupyter launches.

## [1.0.0] — 2026-05-13

This is the stable 1.x baseline. Semver-protected public API
enumerated in [`src/pybmodes/__init__.py`](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/src/pybmodes/__init__.py)
and the *Public API* section of [`README.md`](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/README.md). The
following constructors / dataclasses / functions are now frozen
across 1.x minor releases: every name in the *Public API* list,
including the floating-platform additions originally documented as
"Provisional API" in 0.4.0 — `pybmodes.mooring` (`LineType`,
`Point`, `Line`, `MooringSystem`), `pybmodes.io` (`HydroDynReader`,
`WamitReader`, `WamitData`), and `Tower.from_elastodyn_with_mooring`.

### Highlights for 1.0

- **Validated FEM core** — 1.0 ships frequency-accuracy validation
  on six BModes-JJ reference decks (Test01–04 land / tension-wire,
  CS_Monopile, OC3 Hywind) at ≤ 0.01 % cert tolerance plus the full
  closed-form analytical regression suite (Euler-Bernoulli
  cantilever, cantilever + tip mass, Wright 1982 / Bir 2009
  rotating uniform blade, Bir 2010 Table 5 rotating + tip mass,
  Bir 2009 Eq. 8 pinned-free cable). The matrix is enumerated in
  [`VALIDATION.md`](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/VALIDATION.md).
- **ElastoDyn-deck adapters** — `Tower.from_elastodyn` /
  `Tower.from_elastodyn_with_subdyn` / `Tower.from_elastodyn_with_mooring`
  / `RotatingBlade.from_elastodyn` cover the land + monopile +
  floating configurations.
- **OpenFAST polynomial-coefficient workflows** — six CLI
  subcommands (`validate` / `patch` / `campbell` / `batch` /
  `report` / `examples`) plus `pybmodes.elastodyn` Python API for
  programmatic use. Six patched reference decks ship under
  `src/pybmodes/_examples/reference_decks/` (3 fixed + 3 floating).
- **Quasi-static mooring linearisation** — `pybmodes.mooring`
  parses MoorDyn v1 / v2 and produces a 6 × 6 stiffness matrix
  reproducing the OC3 Hywind surge stiffness to better than 0.01 %
  vs Jonkman 2010 Table 5-1 (41,180 N/m).
- **WAMIT output reader** — `pybmodes.io.WamitReader` extracts
  `A_inf` / `A_0` / `C_hst` from a HydroDyn-pointed WAMIT `.1` /
  `.hst` pair, validated against the IEA-15-240-RWT-UMaineSemi
  upstream files at 1 % tolerance.
- **Bundled examples ship inside the wheel** — 4 analytical-
  reference BMIs, 7 RWT samples, 6 patched ElastoDyn decks. The
  `pybmodes examples --copy <dir>` CLI vendors them out from any
  install (source, editable, or wheel).
- **CI required on master** — branch-protection ruleset gates merges
  on `test (3.11)` + `test (3.12)` green; the merge model went
  through a one-time conversion to PR-required flow in 0.4.x.

### Fixed

- **`WamitReader` — upper-triangle-only WAMIT outputs are now mirrored.** Some WAMIT runs write only the upper triangle of a symmetric matrix (`A_inf`, `A_0`, `C_hst`); the parser previously assigned only `C[i, j]` per row and left the transpose at zero, silently losing half of the off-diagonal coupling for those files. The parsers for `.1` and `.hst` now mirror non-zero entries into the corresponding `[j, i]` slot after reading, preserving explicit zeros from fully-written matrices. Pre-1.0 review (pass 2).
- **`WamitReader` / `HydroDynReader` — Fortran D-exponent notation.** WAMIT and HydroDyn output writers occasionally emit Fortran-style `1.234D+02` instead of `1.234E+02`. The parsers previously used `float(value)` directly and silently dropped rows with `D` / `d` exponents (`ValueError` swallowed in the row loop). A shared `_parse_fortran_float` helper now normalises both forms across `pybmodes.io.wamit_reader` and `pybmodes.models.tower._scan_platform_fields`. Pre-1.0 review (pass 2).
- **`_scan_platform_fields` — raise on malformed critical scalars.** A typo or unsupported numeric token in `PtfmMass` / `PtfmRIner` / `PtfmPIner` / `PtfmYIner` previously fell through to the default `0.0`, producing a physically meaningless floating model with no hard failure. Those four scalars now raise `ValueError` on parse failure (after the Fortran-D normalisation pass); the non-critical fields (CM offsets, additional-stiffness scalars) still fall back to `0.0`. Pre-1.0 review (pass 2).
- **`campbell_sweep` — defensive bound on returned mode count.** The blade-sweep loop indexed `f_step[k]` for `k in range(n_modes)` after slicing whatever the eigensolver returned. On the rare general-eig fallback path (floating platforms with non-symmetric `K` at certain rotor speeds, dropping NaN eigenvalues) this could index past the slice. The loop now raises `RuntimeError` with a clear message naming the offending rotor speed if the solver returns fewer than `n_modes` rows. Pre-1.0 review (pass 2).
- **`Tower.from_elastodyn_with_mooring` — BMI radius / draft length mismatch.** The cantilever adapter sets `bmi.radius = TowerHt - TowerBsHt` (the flexible-tower length), but the floating BMI convention pairs `radius = TowerHt` with `draft = -TowerBsHt` so that `radius + draft` recovers the flexible length after the nondim step. The 0.4.0 `from_elastodyn_with_mooring` set the signed draft without overriding the radius, so for OC3 the flexible length came out as `TowerHt - 2·TowerBsHt = 67.6 m` instead of the intended 77.6 m. The constructor now overrides `bmi.radius = TowerHt` to match the bundled `OC3Hywind.bmi` convention. Pre-1.0 review.
- **`Tower.from_elastodyn_with_mooring` — `Ptfm*Stiff` scalars folded into `mooring_K`.** ElastoDyn carries six additional platform linear-stiffness scalars (`PtfmSurgeStiff` / `PtfmSwayStiff` / `PtfmHeaveStiff` / `PtfmRollStiff` / `PtfmPitchStiff` / `PtfmYawStiff`) that act on top of HydroDyn / MoorDyn contributions. The OC3 spec carries the delta-line crowfoot's yaw spring via `PtfmYawStiff` (~ 9.83e7 N·m/rad) and it isn't in the MoorDyn `.dat`; the 0.4.0 constructor ignored these so the coupled OC3 yaw frequency came out ~ 8× low. They are now scanned alongside the geometry/inertia scalars and added to the diagonal of `mooring_K`. Pre-1.0 review.
- **`_scan_platform_fields` — Fortran D-exponent notation.** ElastoDyn `.dat` scalars may be written as `7.466D+06` rather than `7.466E+06`; the scanner used `float(value)` directly and silently dropped any field that hit the Fortran form, producing a zero in `PtfmMass` / `PtfmRIner` / etc. The scanner now normalises `D` / `d` to `E` before parsing. Pre-1.0 review.
- **`pybmodes.campbell._solve_tower_sweep` — restore caller's `rot_rpm`.** The tower-only Campbell branch was setting `tbmi.rot_rpm = 0.0` without restoring it, mutating the caller's BMI. The blade-sweep path already used `try / finally`; the tower path now mirrors it. Cosmetic — tower modes are rotor-speed-independent so the mutation didn't change any computed value, but the API hygiene matters. Pre-1.0 review.
- **`Tower.from_elastodyn_with_mooring` — i_matrix double parallel-axis.** The previous build added `M·dz²` to `i_matrix[3,3]` / `i_matrix[4,4]` to "transfer the platform inertia from CM to body origin," but `pybmodes.fem.nondim.nondim_platform` already applies the rigid-arm transform itself using `cm_pform - draft` — so the parallel-axis term was being counted twice (~6e10 kg·m² for OC3, ~3.6e9 for IEA-15 UMaine), overstating roll/pitch inertia. The same bug also wrote spurious cross-coupling terms into `i_matrix[0,4]` / `i_matrix[1,3]`. The `i_matrix` is now stored AT THE CM exactly as the BMI parser expects (bottom-right 3×3 of the rotational block, no coupling). Reported by Pre-1.0 review.
- **`Tower.from_elastodyn_with_mooring` — BMI sign convention for `cm_pform` / `draft` / `ref_msl`.** The previous build stored these in ElastoDyn signed-z (so OC3 came out with `cm_pform = -89.9155`, `draft = 0`), but the downstream BMI consumer reads them in BModes file convention (positive distance below MSL for `cm_pform` and `ref_msl`; signed `draft` with negative = above MSL). For OC3 the corrected values match the canonical `OC3Hywind.bmi` deck: `draft = -10`, `cm_pform = 89.9155`, `ref_msl = 0`. Pulls `TowerBsHt` from the ElastoDyn main file to compute `draft = -TowerBsHt`. Reported by Pre-1.0 review.
- **`MooringSystem.from_moordyn` — MoorDyn v1 LINE PROPERTIES column order.** Older MoorDyn v1 line-properties rows use the column order `ID LineType UnstrLen NumSegs NodeAnch NodeFair`, not v2's `ID LineType AttachA AttachB UnstrLen ...`. The previous parser tried to read v2 columns unconditionally, so v1 rows like `1 main 902.2 20 1 4` failed with a ValueError on `int("902.2")` and got silently skipped — leaving the system with zero or incorrect lines. The parser now probes v2 column order first and validates AttachA/AttachB as known point IDs; on mismatch it falls back to v1 column order. `Point.__post_init__` also accepts MoorDyn v1 attachment aliases (`Fix` → `Fixed`, `Connect` → `Free`, `Body` / `Coupled` → `Vessel`, `Anchor` → `Fixed`). Reported by Pre-1.0 review.

### Added

- **`pybmodes.mooring`** — new module with `LineType`, `Point`, `Line`, and `MooringSystem`. Solves the extensible elastic catenary per line (Jonkman 2007 NREL/TP-500-41958 Appendix B equations B-1 / B-2 fully-suspended; B-7 / B-8 with `CB = 0` for the seabed-contact branch; damped Newton on `(H, V_F)` with analytical 2×2 Jacobian, `tol = 1e-6` m, MaxIter = 100). Multi-line platform restoring force assembled from world-frame fairlead positions through 3-2-1 intrinsic rotations; central-difference 6×6 linearisation about an arbitrary or zero offset, trans-rot off-diagonals symmetrised. `MooringSystem.from_moordyn(...)` parses MoorDyn v1 (`CONNECTION`) and v2 (`POINT`) `.dat` files; OC3 Hywind surge stiffness reproduced to better than 0.01 % vs Jonkman 2010 Table 5-1 (41,180 N/m).
- **`Tower.from_elastodyn_with_mooring(main_dat, moordyn_dat, hydrodyn_dat=None)`** — new classmethod that assembles a free-free (`hub_conn = 2`) floating-tower BMI with a populated `PlatformSupport` block: mooring K from MoorDyn, hydrodynamic A_inf + C_hst from HydroDyn / WAMIT (optional), platform inertia from ElastoDyn `PtfmMass` / `PtfmRIner` etc. with parallel-axis transfer from CM to body origin. End-to-end OC3 Hywind solve hits the 1st tower-bending FA pair within 1.2 % of Jonkman 2010's published 0.482 Hz. For ElastoDyn polynomial-coefficient generation use the standard cantilever `Tower.from_elastodyn` instead — that path is unchanged.
- **`pybmodes.io.wamit_reader`** — new module with `WamitReader`, `WamitData`, and `HydroDynReader`. Parses the WAMIT v7 `.1` (added mass / radiation damping) and `.hst` (hydrostatic restoring) output files an OpenFAST floating-platform deck points at via the HydroDyn `PotFile` value, redimensionalises them per the WAMIT v7 convention (`ρ · L^k` for added mass, `ρ · g · L^k` for hydrostatic stiffness — exponents pick up +1 per rotational DOF in the index pair), and returns SI 6 × 6 `A_inf` / `A_0` / `C_hst` matrices in a `WamitData` dataclass. `HydroDynReader` surfaces the four scalars needed to drive `WamitReader` from a HydroDyn `.dat` (`WAMITULEN`, `PotMod`, `PotFile`, `PtfmRefzt`) and chains them via `read_platform_matrices()`. `WtrDens` and `Gravity` defaults are ISO sea-water values since HydroDyn ≥ v2.03 delegates those to the paired SeaState input file. Path resolution handles surrounding quotes, Windows-style backslashes, and relative-vs-absolute `PotFile` values. Integration tests under `tests/test_wamit_reader.py` validate against the upstream IEA-15-240-RWT-UMaineSemi WAMIT files at the 1 % tolerance.

## [0.4.0] — 2026-05-11

### Added

- **`pybmodes examples --copy <dir> [--kind all|samples|decks] [--force]`** — new CLI subcommand. Vendors the bundled `sample_inputs/` and/or `reference_decks/` trees from the installed package into a user-supplied directory so a `pip install pybmodes` user can seed a working tree without keeping a git clone around. Resolves bundle paths relative to `pybmodes.__file__`. Destination conflicts return exit code 2 unless `--force` is set. Tests under `tests/test_examples_cli.py`.
- **Example bundles ship inside the wheel.** The previously top-level `cases/sample_inputs/` (analytical references + 7 RWT samples) and `reference_decks/` (6 patched ElastoDyn decks — 3 fixed + 3 floating) trees were moved into `src/pybmodes/_examples/sample_inputs/` and `src/pybmodes/_examples/reference_decks/` and declared as `package-data` in `pyproject.toml`. Every wheel and editable install now carries the trees alongside the package; the `pybmodes examples --copy` CLI uses this to work regardless of installation source. Delivers the *Repo assets accessible from a wheel install* item on the README *1.0 milestone* list.

### Changed

- **Breaking — bundle paths moved into the package tree.** Anything that hard-coded `repo_root / "cases" / "sample_inputs"` or `repo_root / "reference_decks"` now needs `repo_root / "src" / "pybmodes" / "_examples" / "sample_inputs"` (resp. `... / "_examples" / "reference_decks"`). The cleanest replacement is `pybmodes.cli._resolve_examples_root() / "sample_inputs"` (resp. `... / "reference_decks"`), which works for both source and wheel installs. Tests, scripts, and docs in the repo were updated mechanically. The bundles themselves are byte-identical to the 0.3.x payload — only the on-disk location changed.
- **`pybmodes report` no longer accepts `--rated-rpm`.** The flag was reserved / informational only in 0.3.0 and never surfaced in textual report output; it was removed for 0.4.0 to keep the CLI surface honest ahead of the 1.0 freeze. `pybmodes campbell --rated-rpm` (where the value is wired through to `plot_campbell`) is unchanged.

## [0.3.0] — 2026-05-11

### Added

- **Pre-solve sanity checks** (`pybmodes.checks`). New module shipping `check_model(model, n_modes=None) -> list[ModelWarning]` with eight gated checks: non-monotonic span stations (WARN), zero / negative mass density (ERROR), stiffness jumps > 5× between adjacent stations (WARN per FA + SS axis), EI_FA / EI_SS ratio outside `[0.1, 10]` (INFO), RNA mass > integrated tower mass (INFO), singular `PlatformSupport` 6×6 matrix (`cond > 1e10`, ERROR), `n_modes > 6 × n_nodes` (ERROR), polynomial-fit design-matrix condition number > 1e4 / 1e6 (WARN / ERROR, computed pre-solve from `bmi.el_loc`). `Tower.run()` / `RotatingBlade.run()` gain a keyword-only `check_model: bool = True` parameter; when True, WARN + ERROR findings emit `UserWarning`s and INFO findings stay silent. Internal validator / batch / patch service paths pass `check_model=False` to avoid duplicate warnings.
- **Mode-by-mode comparison** (`pybmodes.mac`). New module with `mac_matrix(shapes_A, shapes_B) -> ndarray (n, m)` (the standard `MAC_ij = |φ_i·φ_j|² / ((φ_i·φ_i)(φ_j·φ_j))` formula on the concatenated `[flap_disp, lag_disp, twist]` vector), `compare_modes(result_A, result_B, label_A, label_B) -> ModeComparison` (full MAC + Hungarian-optimal pairing via `scipy.optimize.linear_sum_assignment` + per-pair `(f_B − f_A)/f_A × 100` shift), and `plot_mac(comparison, ax=None) -> Figure` heatmap with paired cells outlined in red. The Campbell tracker's existing `_mac_matrix` is now a thin wrapper around the public function.
- **Bundled analysis report** (`pybmodes.report`). `generate_report(result, output_path, format='md'|'html'|'csv', model=…, validation=…, check_warnings=…, tower_params=…, blade_params=…, campbell=…, source_file=…)` builds a structured eight-section report (model summary, assumptions, frequencies, mode classification with FA / SS / twist participation, polynomial coefficients with fit residuals, validation verdict, `check_model` warnings, Campbell sweep). HTML output is emitted directly as self-contained HTML5 with inline CSS; no runtime dependency on the `markdown` package. CSV output is narrower (frequencies + coefficient rows) and suitable for spreadsheet ingestion.
- **Result serialisation.** `ModalResult` gains `save(path)` / `load(path)` (compressed NPZ) and `to_json(path)` / `from_json(path)` (UTF-8 JSON with `"schema_version": "1"`); new optional fields `participation` (N × 3) and `fit_residuals` (`dict[str, float]`); metadata block (pyBmodes version, UTC timestamp, source-file path, best-effort git hash) auto-populated at save time. `CampbellResult` gains `save(path)` / `load(path)` (NPZ) and `to_csv(path)`; per-step MAC tracking confidence rides in the new `mac_to_previous` array (NaN on row 0 and on tower columns). Shared `pybmodes.io._serialize` helper captures the metadata dict; `git rev-parse --short HEAD` runs with a 2-second timeout and silently records `None` on any failure.
- **`pybmodes report` CLI subcommand.** `pybmodes report ElastoDyn.dat --format md|html|csv --out PATH [--campbell --rated-rpm R --max-rpm R --n-steps N --n-blade-modes N --n-tower-modes N] [--n-modes N] [--no-validate]` runs the modal solve, optional coefficient validation, and optional Campbell sweep on one deck and writes a single bundled report via `pybmodes.report.generate_report`.
- **`pybmodes batch` CLI subcommand.** `pybmodes batch ROOT [--kind elastodyn] [--out OUT/] [--n-modes N] [--validate] [--patch]` walks `ROOT` recursively for ElastoDyn main `.dat` files (two-stage filter: name heuristic plus parse confirmation), runs `validate` and / or `patch` per deck, writes a per-deck validation report under `OUT/`, and emits a `summary.csv` with columns `filename, overall_verdict, TwFAM2Sh_ratio, TwSSM2Sh_ratio, n_fail, n_warn`. Exits 0 when every deck reaches PASS or WARN; 1 if any FAIL or ERROR remains; 2 on unsupported `--kind` or missing `ROOT`.
- **Sparse shift-invert eigensolver path** in `pybmodes.fem.solver`. When the FEM matrices are effectively symmetric AND `ngd > 500` AND the caller asked for a small subset of modes, `solve_modes` routes through `scipy.sparse.linalg.eigsh(K, k=n_modes, M=M, sigma=0, which='LM', mode='normal')`. The `mode='normal'` choice (not `'buckling'`) is documented inline — `mode='buckling'` with `sigma=0` reduces to `OP = K⁻¹ K = I` (degenerate). On ARPACK non-convergence or any other failure the solver logs a `WARNING` and falls back to dense `eigh`. The selected path is announced as a `logging.INFO` message on the `pybmodes.fem.solver` logger. `scripts/benchmark_sparse_solver.py` reports 5-18× speedups across `n_elements ∈ {20, 50, 100, 200, 500}` and asserts sparse beats dense for `n_elements > 100` within a 10 % margin.
- **Torsion-contamination filter** in `_select_tower_family` (`pybmodes.elastodyn.params`). Tower family candidates whose modal-kinetic-energy torsion fraction `T_tor ≥ 0.10` are dropped from the selection. New helper `_kinetic_participation(shape) -> (T_FA, T_SS, T_tor)` computes per-mode energy fractions under the unit-mass approximation. `TowerFamilyMemberReport` gains `fa_participation` / `ss_participation` / `torsion_participation` / `torsion_rejected` fields; `TowerSelectionReport` gains `rejected_fa_modes` / `rejected_ss_modes`; `CoeffBlockResult` gains the four corresponding fields for tower blocks (NaN / empty defaults on blade blocks).
- **`pybmodes patch` safe-review modes.** `--dry-run` computes the patched coefficients and prints a per-block change summary without writing; `--diff` prints a PR-ready coefficient-only unified-diff format (`old → new` lines per block plus a per-block `RMS improvement: file_rms → pyb_rms (Nx better)` annotation) and also writes nothing; `--output-dir DIR` (alias `--output DIR`) writes the patched tower + blade `.dat` to `DIR/` instead of in-place, leaving the originals untouched. Combining `--output*` with `--dry-run` or `--diff` exits 2 with a clear "incompatible flags" message. The default in-place path with no `--backup` emits a one-line first-time-run hint pointing at `--dry-run --diff`; suppressed when any of `--backup`, `--output-dir`, `--dry-run`, or `--diff` is set.
- **Hungarian MAC tracking on the Campbell sweep.** The greedy `argmax(mac)` mode-pairing inside `_solve_blade_sweep` is replaced with a global Hungarian assignment via `scipy.optimize.linear_sum_assignment(maximize=True)`. The old `_greedy_assignment` symbol is kept as a deprecated alias. `CampbellResult.mac_to_previous` (new field, `(N, n_total_modes)`) exposes per-step tracking confidence — NaN on row 0 (no previous step) and on tower columns (tower modes don't change with rotor speed). `_solve_blade_sweep` now restores `bbmi.rot_rpm` via `try`/`finally` so the caller's BMI is unmutated by the sweep.
- **Campbell input-validation hardening.** `campbell_sweep` rejects NaN, inf, negative, and unsorted `omega_rpm` arrays with explicit `ValueError`s naming the offending element.
- **`pybmodes.io._elastodyn` sub-package.** The 1315-line `elastodyn_reader.py` is split into `types.py` (dataclasses), `lex.py` (line + token scanning helpers), `parser.py` (line-driven flavour parsers), `writer.py` (canonical re-emitters), and `adapter.py` (`to_pybmodes_tower` / `to_pybmodes_blade` plus the `_stack_*` / `_rotary_inertia_floor` / `_build_bmi_skeleton` / `_tower_top_assembly_mass` helpers). `pybmodes.io.elastodyn_reader` becomes a re-export shim — every public name plus the private helpers `pybmodes.io.subdyn_reader` depends on (`_rotary_inertia_floor`, `_stack_*_section_props`, `_tower_top_assembly_mass`, `_build_bmi_skeleton`, `_resolve_relative`) stay importable from the historical dotted path.
- **`scripts/audit_validation_claims.py`.** Parses every `tests/...` link in `VALIDATION.md`, asserts each path exists and contains at least one `def test_…` method. Runs as a required CI step alongside ruff and mypy, plus step 4.5 of `https://pybmodes.readthedocs.io/en/latest/release_checklist.html`. Gates "claim ahead of test" drift mechanically.
- **`https://pybmodes.readthedocs.io/en/latest/release_checklist.html`** — 11-step pre-tag verification sequence (default + integration pytest, ruff + mypy, sample-input verifier, validation-matrix audit, reference-deck regeneration, notebook smoke, case-script regen, version + CHANGELOG promotion, tag + push, GitHub Release, post-release sanity).
- **Three floating reference decks** under `reference_decks/`: `nrel5mw_oc3spar/` (NREL 5MW on the OC3 Hywind spar), `nrel5mw_oc4semi/` (NREL 5MW on the OC4 DeepCwind semi), `iea15mw_umainesemi/` (IEA-15-240-RWT on the UMaine VolturnUS-S semi). All generated via the existing `Tower.from_elastodyn(...)` cantilever path with no platform / hydro / mooring matrices in the modal eigenproblem — matching what ElastoDyn assumes at runtime. The IEA-15 UMaine case ends at `Overall: WARN` on `TwSSM2Sh` (1.6 % RMS, ratio 1.00) — an unavoidable representation limit of the constrained 6th-order polynomial form for that tower's section-property gradient, documented inline in the deck's `validation_report.txt`.
- **`pybmodes.io._elastodyn` cantilever path used for floating polynomial generation.** OpenFAST ElastoDyn source-code audit (May 2026) established that the polynomial ansatz `SHP = Σ c_i · (h/H)^(i+1)` algebraically forces `SHP(0) = SHP'(0) = 0` and the modal eigenproblem in `Coeff` (lines 5141-5267) integrates only the tower beam plus `TwrTpMass` — no platform / hydro / mooring matrices. Platform 6-DOF motion is added at runtime via the rigid-body sum (lines 7485-7540). The correct polynomial basis for every ElastoDyn configuration (land, monopile, floating) is therefore the clamped-base cantilever. Findings published in `reference_decks/FLOATING_CASES.md` (rewritten) and `cases/ECOSYSTEM_FINDING.md` (new "Floating-deck polynomials" section).

### Changed

- **CI step hardening.** The integration-test step no longer uses `continue-on-error: true`; instead it tolerates pytest exit code 5 ("no tests collected" — the normal case on the default GA runner) but fails the build on any other non-zero exit, so a custom workflow run that does have the data surfaces real failures. Ruff scope expanded from `src/ tests/` to `src/ tests/ scripts/`; user-facing workflow scripts (`build_reference_decks`, `audit_validation_claims`, `benchmark_sparse_solver`, `campbell`, `visualise_polynomial_comparison_*`) are gated alongside the package and tests. The validation-matrix audit (`scripts/audit_validation_claims.py`) is now a required CI step between tests and lint.
- **Validator service paths skip pre-solve checks.** `validate_dat_coefficients` passes `check_model=False` to its internal `Tower.run()` / `RotatingBlade.run()` calls. Without this, batch over real RWT decks emitted noisy stiffness-jump warnings on every blade (the wind-turbine blades genuinely have stiffness gradients at the tip transition — not a bug worth re-reporting once per deck × per validate call). Direct `Tower.run()` calls from user code keep the default-on behaviour.
- **Standard engineering-paper plot palette.** `pybmodes.plots.style.apply_style` switches from the MATLAB R2014b lines colour order to a black / red / blue / green / magenta / orange / cyan ordering. Black first so single-line plots read as line art; red / blue / green next for grayscale-printability. Backwards-compatibility alias `MATLAB_LINES` points at the new palette. All committed plot-producing scripts under `cases/` and `scripts/` were regenerated; the visualise-polynomial-comparison scripts had hardcoded MATLAB RGB triples that were also updated.
- **README documentation.** New top-level *Sample inputs* section listing the four analytical-reference cases and the seven RWT samples; new *Validation* + *Compatibility policy* / *1.0 milestone* sections; refreshed *Quick Start* with mode-comparison, save/load, report, and batch examples.

### Fixed

- **Float reference-deck polynomial coefficient story corrected.** The earlier `FLOATING_CASES.md` claim that floating tower polynomials need `Tower.from_bmi()` with `hub_conn=2` and a populated `PlatformSupport` block was technically wrong: that path solves the coupled tower + platform eigenproblem (correct for matching BModes JJ frequency, validated to ~ 0.0003 % per `test_certtest_oc3hywind`) but produces eigenvectors that include platform rigid-body motion — incompatible with ElastoDyn's `SHP(0) = SHP'(0) = 0` ansatz. The correct path is the cantilever solve `Tower.from_elastodyn(...)`, the same as the land and monopile sides.
- **CHANGELOG and README test-count drift.** Earlier README / CHANGELOG entries quoted hardcoded test counts that aged out on every test addition. Both files now defer to `VALIDATION.md` (the structured single-source-of-truth matrix) and `pytest --collect-only` for the current count.

### Known limitations

- **`iea15mw_umainesemi/TwSSM2Sh`** stays at `Overall: WARN` (1.6 % RMS) after patching — an unavoidable representation limit of the constrained 6th-order polynomial form for that tower's section-property gradient. The patched polynomial IS pyBmodes' best constrained fit (ratio = 1.00 against pyBmodes' own reference); improving it further would require a higher polynomial order or a piecewise basis, neither of which ElastoDyn's `SHP` ansatz supports. Auto-emitted explanatory footer ships in the deck's `validation_report.txt`.

## [0.2.0] — 2026-05-09

### Added

- **Sample-input library** (`cases/sample_inputs/`) — pyBmodes-authored, Apache 2.0-licensed `.bmi` and section-property `.dat` files committed to the repo. Four analytical-reference cases at the top level (uniform isotropic cantilever blade, uniform tower with concentrated top mass, rotating uniform blade per Wright 1982 / Bir 2009 Table 3a, rotating pinned-free cable per Bir 2009 Eq. 8) exercising all four `hub_conn` BCs plus tower / blade and rotating / non-rotating splits. `cases/sample_inputs/verify.py` runs all four against closed-form references at < 1 % RMS. Plus `reference_turbines/` sub-directory with seven RWT samples (NREL 5MW land + OC3 monopile + OC3 Hywind, IEA-3.4-130-RWT land, IEA-10-198-RWT / IEA-15-240-RWT / IEA-22-280-RWT monopile), each shipping tower BMI + blade BMI + per-side section-properties + per-turbine README; regenerable from upstream ElastoDyn decks via `reference_turbines/build.py`.
- **Three floating reference decks** under `reference_decks/`: NREL 5MW on the OC3 Hywind floating spar (`nrel5mw_oc3spar/`), NREL 5MW on the OC4 DeepCwind semi-submersible (`nrel5mw_oc4semi/`), and IEA-15-240-RWT on the UMaine VolturnUS-S semi (`iea15mw_umainesemi/`). All three generated via the existing `Tower.from_elastodyn(...)` cantilever path; post-patch validation reports shipped per case. Two reach `Overall: PASS`; the IEA-15 UMaine case reaches `Overall: WARN` on `TwSSM2Sh` (1.6 % RMS) — a representation limit of the constrained 6th-order polynomial form for that specific tower's section-property gradient, with an auto-emitted explanatory footer in the report.
- **Canonical `tow_support = 1` block on monopile BMI samples** in `cases/sample_inputs/reference_turbines/` — full CS_Monopile.bmi-format section structure (3×3 platform-inertia, 6×6 hydro_M / hydro_K / mooring_K, distributed added-mass + distributed elastic-stiffness, tension wires) with zero-valued matrices for the rigid-clamp combined pile + tower model. Layout is BModes-JJ-readable unmodified; the all-zero matrices add nothing to the eigenvalue problem.
- **OpenFAST ElastoDyn and WISDEM source-code audit** documenting the load-bearing question of which boundary condition ElastoDyn assumes for the tower modal basis at runtime. Findings published in `cases/ECOSYSTEM_FINDING.md` (new "Floating-deck polynomials" section) and `reference_decks/FLOATING_CASES.md` (rewritten end-to-end). Conclusion: the ElastoDyn polynomial ansatz `SHP = Σ_{i=1..PolyOrd-1} c_i · (h/H)^(i+1)` (`ElastoDyn.f90:2486-2495`) algebraically forces `SHP(0) = SHP'(0) = 0`; the modal eigenproblem in `Coeff` (lines 5141-5267) integrates only the tower beam plus `TwrTpMass` with no platform / hydro / mooring matrices; platform 6-DOF motion enters at runtime via the rigid-body sum (lines 7485-7540). Therefore the **correct polynomial basis for ALL ElastoDyn configurations — land, monopile, floating — is the clamped-base cantilever in the platform-attached frame**.
- **Bir (2010) NREL/CP-500-47953 reproduction suite.** Four new things reproduce the canonical BModes verification paper:
  - **Closed-form regression tests against Wright et al. 1982 / Bir 2009 Tables 2a + 3a and Bir 2010 Table 5.** `tests/fem/test_rotating_uniform_blade.py` gates flap modes 1-3 of a uniform rotating cantilever blade (L = 31.623 m, m = 100 kg/m, EI_flap = 1e8, EI_lag = 1e9, GJ = 1e5) at ≤ 0.5 % across Ω ∈ {0..12} rad/s. `tests/fem/test_rotating_blade_with_tip_mass.py` gates flap modes 1-2 of the same blade plus a μ = 1 tip mass against Bir 2010 Table 5 at ≤ 0.1 %. The latter wires up the previously-missing tip-mass centrifugal-tension contribution to `cfe`; without it the rotating-tip-mass frequencies are 14-50 % low at moderate Ω. `tests/fem/test_rotating_cable.py` gates the inextensible spinning cable (Bir 2009 §III.B / Eq. 8: ω = Ω·√(k(2k−1))) on the new `hub_conn=4` BC at ≤ 0.5 %. Closes the "Centrifugal-stiffening validation" roadmap item.
  - **`hub_conn=4` (pinned-free) tower-base BC.** Locks axial, lag/flap deflections, and twist at the root while leaving the bending slopes FREE. Matches the implicit BC of Bir 2009's Legendre-polynomial cable solution. Implemented in `pybmodes.fem.boundary` (`build_connectivity`, `n_free_dof`, `active_dof_indices`).
  - **`pybmodes.plots.bir_mode_shape_plot` and `bir_mode_shape_subplot`.** Plot mode shapes with modal displacement on the x-axis (mass-normalised, *not* unit-tip) and normalised height $z/H$ on the y-axis, matching Bir 2010 Figs 4, 5a, 5b, 6a-c, 8. Optional horizontal annotation lines (Mean Sea Level, Mud Line) for offshore configurations and dashed coupling overlays for hybrid modes.
  - **Three case-study scripts** (`cases/bir_2010_land_tower/`, `cases/bir_2010_monopile/`, `cases/bir_2010_floating/`) reproduce Bir's figures using the cert-test decks. The scripts render Fig 4 (synthetic uniform cantilever, no head mass), Fig 5a / 5b (Test03 = land tower with head mass), Fig 8 (CS_Monopile with MSL marker), and Fig 6a / 6b / 6c (OC3Hywind floating spar). The monopile case classifies hybrid modes (e.g. CS_Monopile mode 4 is a 2nd-FA + twist coupled hybrid) with explicit "(+ F-A part)" labels. Frequencies on the cert-test decks are already validated against BModes JJ at ≤ 0.01 %; these PNGs are the visual companion.
- Self-contained walkthrough notebook (`notebooks/walkthrough.ipynb`) demonstrating the full public API on synthetic uniform blade and tower cases.
- Inline synthetic-fixture helpers (`tests/_synthetic_bmi.py`) that build `.bmi` and section-property files at test time, with numbers freely chosen by the project author.
- Closed-form analytical regression suite for the cantilever-with-tip-mass configuration (`tests/fem/test_uniform_tower_analytical.py`), validating the FEM solver against the Blevins (1979) / Karnovsky & Lebed (2001) frequency equation across tip-mass ratios from 0 to 5.
- Comprehensive unit-test coverage of FEM building blocks: boundary conditions, generalised eigensolver, non-dimensionalisation, mode-shape extraction, polynomial-fit edge cases, and parser primitives.
- **OpenFAST deck adapters.** New classmethod constructors that consume OpenFAST input files directly:
  - `Tower.from_elastodyn(main_dat_path)` — parses the ElastoDyn main file plus the tower file referenced via `TwrFile` and the first blade file via `BldFile(1)` (the latter only to lump rotor mass into the tower-top assembly). Lands the *NREL 5MW Reference Turbine* (Jonkman et al. 2009) tower modal solve within ~ 1 % of the published target.
  - `Tower.from_elastodyn_with_subdyn(main_dat_path, subdyn_dat_path)` — splices a SubDyn pile geometry below the ElastoDyn tower into a single combined cantilever clamped at the SubDyn reaction joint. Designed for OC3-style fixed-base monopiles (no soil flexibility, no hydrodynamic added mass).
  - `RotatingBlade.from_elastodyn(main_dat_path)` — synthesises a BMI-equivalent from the ElastoDyn main + blade files, including centrifugal stiffening from the deck's `RotSpeed`.
  - `Tower.from_bmi(bmi_path)` — explicit classmethod alias of `Tower(...)` for symmetry with the other constructors.
- **`pybmodes.io.elastodyn_reader`** module — full ElastoDyn `.dat` parser + canonical writer + adapter helpers. Three dataclasses (`ElastoDynMain`, `ElastoDynTower`, `ElastoDynBlade`); label-based scanning robust across FAST v8 / OpenFAST v3+ format drift; semantic round-trip via `write_elastodyn_*` (parse → emit → re-parse equality with `np.allclose` rtol = 1e-12). Adapter helpers `to_pybmodes_tower` and `to_pybmodes_blade` synthesise BMI / SectionProperties in memory; `run_fem` accepts an optional pre-built `SectionProperties` so adapter paths skip the on-disk round-trip.
- **`pybmodes.io.subdyn_reader`** module — minimal SubDyn parser + pile/tower combiner (joints, members, circular cross-section properties, base reaction joint, interface joint). Sufficient for OC3-style monopiles; non-circular sections and SSI files are not parsed.
- **Cross-solver certification suite (`tests/test_certtest.py`).** Six certification cases now compared against the BModes Fortran reference solver (Bir 2010) at strict tolerances:
  - BModes v3.00 CertTest Test01-04 (rotating blades, cantilever tower with top mass, tension-wire-supported tower) at < 1 % / < 3 % per-mode.
  - `CS_Monopile.bmi` — *NREL 5MW Reference Turbine* on the *OC3 Monopile* (Jonkman & Musial 2010) at 0.01 %, < 0.005 % observed.
  - `OC3Hywind.bmi` — *NREL 5MW* on the *OC3 Hywind* floating spar (Jonkman 2010) at 0.01 %, ≤ 0.0003 % observed across the first 9 modes.
- **Degenerate-eigenpair resolver (`pybmodes.elastodyn.params._rotate_degenerate_pairs`).** Detects consecutive modes whose relative frequency gap is below 1e-4 and rotates the pair inside its 2D eigenspace so the first comes out FA-pure and the second SS-pure. Handles the symmetric-tower case where the eigensolver returns an arbitrary basis of the degenerate subspace.
- **Polynomial-fit conditioning instrumentation.** `PolyFitResult.cond_number` reports the 2-norm condition number of the reduced design matrix solved by `lstsq`. `compute_tower_params_report` emits a `RuntimeWarning` above 1e4 (WARN) and a stronger one above 1e6 (FAIL) so basis-conditioning artefacts on poorly-sampled meshes don't pass silently.
- **Case studies** (`cases/` directory). Three exploratory case directories — `nrel5mw_land/` (*NREL 5MW Reference Turbine*, Jonkman et al. 2009), `iea3mw_land/` (*IEA-3.4-130-RWT*, Bortolotti et al. 2019, IEA Wind Task 37), and `nrel5mw_monopile/` (*NREL 5MW* on rigid OC3-style monopile) — each with a `run.py` that prints a coefficient-comparison table (`coefficients.txt`) and writes mode-shape PNGs. `cases/ECOSYSTEM_FINDING.md` is the cross-deck summary documenting that the polynomial-coefficient blocks shipped in industry `_ElastoDyn.dat` files are typically not reproducible from the structural-property blocks in the same files.

### Changed

- **FEM core vectorisation.** Element-matrix construction is now vectorised over both Gauss points and elements via `numpy.einsum`, replacing the per-element Python loop. Inner double sums over Gauss points and local DOF pairs collapse to a single tensor contraction. Net speedup is ~2–3× on small cases and ~1.6× on larger meshes where the dense `eigh` solve dominates.
- **Validation contract.** Switched from bundled reference data files to published closed-form formulas as the source of truth for FEM accuracy. The reference list now contains only textbook material (Euler-Bernoulli cantilever frequency series; Blevins / Karnovsky cantilever-with-tip-mass equation), supplemented by cross-solver certification against BModes (see "Added" above).
- README rewritten to drop external-program framing; Windows + conda install instructions added.
- **Tower-top mass kinematic coupling for offshore / free-base towers.** `nondim_tip_mass` now uses the BMI's literal `cm_loc` / `cm_axial` pair directly when `hub_conn ∈ {2, 3}`. The previous code path applied the cantilever convention (which folds `cm_axial` into the internal `cm_loc` lever arm and drops the literal `cm_loc`) regardless of `hub_conn`, which on OC3 Hywind effectively dropped the `cm_axial` bending lever arm and made the 1st tower-bending pair too stiff — 0.4997 / 0.5087 Hz instead of BModes' 0.4816 / 0.4908 Hz (~ 3.8 % high). The cantilever path is preserved for `hub_conn = 1` because the four BModes v3.00 CertTest cases depend on the older convention to pass at 6-digit precision.
- **Eigensolver dispatch for asymmetric platform support.** OC3 Hywind has genuinely asymmetric platform-support contributions after the rigid-arm transformation. `solver.py` now detects asymmetry in the assembled `K` / `M` and routes those cases through `scipy.linalg.eig` (general dense eigensolver), matching BModes. Symmetric problems — all cantilever cases plus the soft-monopile CS_Monopile case — still use `scipy.linalg.eigh`.
- **PlatformSupport detection** in `models/_pipeline.py` keys off `isinstance(bmi.support, PlatformSupport)` rather than `bmi.tow_support == 2`. Both BMI dialects (legacy `tow_support = 2` and inline `tow_support = 1` with a numeric draft follow-up) get normalised to `PlatformSupport` by the parser; the new check picks up both consistently and also handles hand-built `BMIFile` instances that don't set `tow_support`.
- **Reference-turbine naming convention clarified.** Citable published reference turbines (*NREL 5MW Reference Turbine*, *OC3 Monopile* / *OC3 Hywind*, *IEA-3.4-130-RWT* and the wider IEA Wind Task 37 family) are now explicitly named in validation tables, README content, and case-study reports — they're standard citations in the field. Restraint on ambient name-dropping in source comments and commit messages is unchanged.
- Test count expanded from 159 to 364 across this release window (159 → 197 with the analytical-validation pass; 197 → 252 with the cross-solver certification + offshore work; 252 → 338 with the Bir 2010 reproduction suite + the new `hub_conn=4` cable test; 338 → 364 with the coefficient-validator + reference-decks deliverable + professional-polish pass that landed test markers, public-API declaration, the unified plot style, and per-module mypy strict overrides).

### Removed

- All bundled reference-data files under `tests/data/` (`.bmi`, `.dat`, `.out`). The library is now a self-contained Python implementation validated only against analytical references and locally-supplied (uncommitted) BModes / OpenFAST decks.
- `examples/` directory — the demo scripts depended on the removed reference data; the walkthrough notebook supersedes them.

### Fixed

- **OC3 Hywind 1st tower-bending pair** — was running ≈ 3.7-3.8 % HIGH versus the BModes JJ reference (pyBmodes 0.4997 / 0.5087 Hz vs BModes 0.4816 / 0.4908 Hz). The fix combined the three changes listed under "Changed" above: the literal `cm_loc` / `cm_axial` interpretation for `hub_conn = 2`, the asymmetric-eigensolver routing, and the `PlatformSupport`-keyed pipeline. Post-fix, OC3 Hywind matches BModes JJ across the first 9 modes to **0.0000 – 0.0003 %** — > 30× headroom under the 0.01 % cert tolerance. CS_Monopile (which has zero `hydro_M` and a symmetric support matrix) was already exact; it remains so.
- **`patch_dat` no longer demotes CRLF line endings to LF** on Windows OpenFAST `.dat` files. The writer used to rstrip the matched line and rewrite it with a hardcoded `\n`, silently mixing endings; now the original line ending is captured per line and re-applied, with `newline=''` set on both read and write to defeat Python's universal-newline translation.
- Removed README claim of distributed-hydrodynamic-added-mass support for monopile towers — `distr_m` is parsed but not yet wired into the mass matrix; only distributed soil stiffness flows through to the FEM assembly.

## [0.1.0] — 2025-04-22

### Added

- Rotating blade modal analysis (flap, edge, torsion modes)
- Onshore tower analysis — cantilevered and tension-wire supported
- Offshore tower analysis — floating spar (`hub_conn=2`) and bottom-fixed monopile (`hub_conn=3`)
- Constrained 6th-order polynomial mode shape fitting (C₂ + C₃ + C₄ + C₅ + C₆ = 1)
- In-place patching of OpenFAST ElastoDyn `.dat` files
- Initial validation against bundled reference cases (later removed in the independence pass)
