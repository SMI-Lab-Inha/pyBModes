<!-- markdownlint-disable MD013 -->
# pyBmodes validation matrix

This document is the **single structured source of truth** for what
pyBmodes is validated against, at what tolerance, with what worst
observed error, and which test file enforces it. Prose-heavy reports
elsewhere in the repo (`cases/ECOSYSTEM_FINDING.md`,
`src/pybmodes/_examples/reference_decks/VALIDATION_SUMMARY.md`, the README's *Validation*
section) refer back to this matrix.

The validation work is split into **two tracks** with different
metrics:

- **Track A — frequency accuracy.** Eigenvalue agreement against an
  external reference (closed-form formula, published table, or the
  BModes Fortran solver). Tolerance is a relative frequency error.
- **Track B — coefficient consistency.** The mode-shape polynomial
  blocks shipped in OpenFAST ElastoDyn `.dat` files. Tolerance is
  the RMS residual of the polynomial against the FEM mode shape.

## Track A — frequency-accuracy cases

| Case | Source / reference | Quantity | Tolerance | Worst observed | Test file | Needs external data |
| --- | --- | --- | ---: | ---: | --- | :---: |
| Uniform Euler-Bernoulli cantilever, modes 1-3 | $\beta_n L = \{1.875,\,4.694,\,7.855,\ldots\}$ (textbook) | flap frequency | < 0.5 % | < 0.005 % | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Same, mode 4 | textbook | flap frequency | < 1 % | (within tol) | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Same, mode 5 | textbook | flap frequency | < 2 % | (within tol) | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Hermite cubic mesh-convergence | analytical $h^4$ rate | error ratio at h × 2 | > 5 (i.e. ≥ 5×) | confirmed | [`tests/fem/test_cantilever.py`](tests/fem/test_cantilever.py) | no |
| Cantilever + tip mass, $\mu \in [0, 5]$ | Blevins (1979) / Karnovsky & Lebed (2001) frequency equation | 1st bending frequency | < 0.5 % | (within tol) | [`tests/fem/test_uniform_tower_analytical.py`](tests/fem/test_uniform_tower_analytical.py) | no |
| Rotating uniform blade, flap modes 1-3, Ω ∈ [0, 12] rad/s | Wright et al. (1982) — *Vibration Modes of Centrifugally Stiffened Beams*, J. Appl. Mech. (transcribed from Bir 2009 Table 3a) | flap frequency | < 0.5 % | (within tol) | [`tests/fem/test_rotating_uniform_blade.py`](tests/fem/test_rotating_uniform_blade.py) | no |
| Rotating uniform blade + tip mass, flap modes 1-2, Ω ∈ [0, 12] rad/s | Wright et al. (1982) (Bir 2010 Table 5) | flap frequency | < 0.1 % | (within tol) | [`tests/fem/test_rotating_blade_with_tip_mass.py`](tests/fem/test_rotating_blade_with_tip_mass.py) | no |
| Inextensible spinning cable, flap modes 1-3, Ω ∈ [2, 30] rad/s | Bir 2009 §III.B / Eq. 8: $\omega_k = \Omega\sqrt{k(2k-1)}$ | flap frequency | < 0.5 % | (within tol) | [`tests/fem/test_rotating_cable.py`](tests/fem/test_rotating_cable.py) | no |
| Uniform steel tube cantilever via `Tower.from_geometry` (D, t, L → derived EI / mass) | Euler-Bernoulli closed form, $\beta_1 L = 1.875104$ | 1st bending frequency | < 0.1 % | machine-exact (forward-derived from geometry) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| `n_nodes` mesh refinement is bias-free + self-convergent (issue #35) | uniform tube: identical to native grid + Euler-Bernoulli; taper: self-convergence $n{=}200$ vs $400$ | 1st & 4th bending frequency; mode-shape sample count | uniform = EB < 0.1 %; taper 1st < 0.2 %, 4th < 2 % ($200$ vs $400$) | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| BModes v3.00 CertTest **Test01** — non-uniform rotating blade, 60 rpm | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 | < 1 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test01**, modes 7+ | BModes Fortran solver `.out` | per-mode frequency | < 3 % | (within tol) | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test02** — rotating blade + tip mass + offsets | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 / 7+ | < 1 % / < 3 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test03** — 82.4 m tower with top mass + c.m. offsets | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 / 7+ | < 1 % / < 3 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| BModes v3.00 CertTest **Test04** — Test03 + tension-wire support | BModes Fortran solver `.out` | per-mode frequency, modes 1-6 / 7+ | < 1 % / < 3 % | < 0.005 % across 20 modes | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| **CS_Monopile** — *NREL 5MW Reference Turbine* on the *OC3 Monopile* (`hub_conn = 3`, soft monopile, mooring stiffness) | BModes JJ (v1.03.01) `.out` | per-mode frequency, first 10 modes | < 0.01 % | < 0.005 % | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| **OC3Hywind** — *NREL 5MW* on the *OC3 Hywind* floating spar (`hub_conn = 2`, full hydro + mooring + 6×6 platform inertia) | BModes JJ (v1.03.01) `.out` | per-mode frequency, first 9 modes | < 0.01 % | ≤ 0.0003 % across surge / sway / yaw / roll / pitch / heave + 1st-2nd tower bending | [`tests/test_certtest.py`](tests/test_certtest.py) | yes |
| Degenerate-pair resolver, symmetric tower | construction (`EI_FA == EI_SS`) | post-rotation FA / SS purity | $p_{\text{FA}}, p_{\text{SS}} > 0.99$ | (within tol) | [`tests/test_classifier.py`](tests/test_classifier.py) | no |
| IEA-3.4 modes 1-2 — degenerate-pair resolver fires no warning | IEA Wind Task 37 RWT deck | classifier verdict | no `RuntimeWarning` | clean | [`tests/test_classifier.py`](tests/test_classifier.py) | yes |

**Citations** (full author / year forms used in the table above):

- Blevins (1979). *Formulas for Natural Frequency and Mode Shape*. Krieger Publishing.
- Karnovsky & Lebed (2001). *Formulas for Structural Dynamics*. McGraw-Hill.
- Wright, Smith, Thresher & Wang (1982). *Vibration Modes of Centrifugally Stiffened Beams*. *Journal of Applied Mechanics*, Vol. 104, March 1982.
- Bir (2009). *Blades and Towers Modal Analysis Code (BModes): Verification of Blade Modal Analysis Capability*. AIAA 2009-1035.
- Bir (2010). *Verification of BModes: Rotary Beam and Tower Modal Analysis Code*. NREL/CP-500-47953.
- Jonkman, Butterfield, Musial & Scott (2009). *Definition of a 5-MW Reference Wind Turbine for Offshore System Development*. NREL/TP-500-38060.
- Jonkman & Musial (2010). *Offshore Code Comparison Collaboration (OC3) for IEA Wind Task 23*. NREL/TP-5000-48191.
- Jonkman (2010). *Definition of the Floating System for Phase IV of OC3*. NREL/TP-500-47535.
- Bortolotti, Tarrés, Dykes, Merz, Sethuraman, Verelst & Zahle (2019). *IEA Wind TCP Task 37: Systems Engineering in Wind Energy — WP2.1 Reference Wind Turbines*. NREL/TP-5000-73492.

## Track B — coefficient-consistency cases

The metric here is the RMS residual of an ElastoDyn polynomial block
evaluated against the FEM-computed mode shape produced by the deck's
own structural inputs. See `pybmodes.elastodyn.validate` for the
implementation. Per-block verdicts: PASS < 0.01, WARN < 0.10, FAIL ≥ 0.10.

| Case | Source / reference | Quantity | Tolerance | Worst observed (file_rms) | Test file | Needs external data |
| --- | --- | --- | ---: | ---: | --- | :---: |
| Validator on stock NREL 5MW r-test deck — TwFAM2Sh / TwSSM2Sh | OpenFAST r-test (commit `dd5feaaa`) | file polynomial RMS vs pyBmodes mode shape (detection target) | verdict = FAIL | 5.08 (TwFAM2Sh) / 5.90 (TwSSM2Sh) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| Validator on same deck — 1st tower modes + blade modes | OpenFAST r-test | file polynomial RMS | verdict = PASS | 0.0081 / 0.0075 / 0.0020-0.0090 | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| Patch round-trip on staged NREL 5MW copy | self (post-`patch_dat`) | file polynomial RMS after patching | verdict = PASS, ratio ≈ 1.0 | ratio drift < 1 % (text-precision artefact) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| `reference_decks/nrel5mw_land/` patched deck | committed deliverable | per-block verdict | PASS or WARN, no FAIL | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no (artefact tracked) |
| `reference_decks/nrel5mw_oc3monopile/` patched deck | committed deliverable | per-block verdict | PASS or WARN, no FAIL | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |
| `reference_decks/iea34_land/` patched deck | committed deliverable | per-block verdict | PASS or WARN, no FAIL | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |
| `reference_decks/nrel5mw_oc3spar/` patched deck (OC3 Hywind floating spar; cantilever basis) | committed deliverable | per-block verdict | PASS or WARN, no FAIL | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |
| `reference_decks/nrel5mw_oc4semi/` patched deck (OC4 DeepCwind semi; cantilever basis) | committed deliverable | per-block verdict | PASS or WARN, no FAIL | all PASS | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |
| `reference_decks/iea15mw_umainesemi/` patched deck (UMaine VolturnUS-S; cantilever basis) | committed deliverable | per-block verdict | PASS or WARN, no FAIL | WARN on TwSSM2Sh (1.6 % RMS — ElastoDyn-basis representation limit; see footer in the deck's `validation_report.txt`) | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |
| Pre-patch sanity — at least one `before_patch.txt` shows FAIL or WARN | committed before-patch reports | per-deck overall verdict | ≥ 1 FAIL/WARN | 6/6 FAIL | [`tests/test_reference_decks.py`](tests/test_reference_decks.py) | no |

## Track C — supporting-pipeline behavioural cases

These tests gate the **workflow** layers that sit between the FEM
core and the user — pre-solve sanity checks, mode-by-mode
comparison, result serialisation, bundled report generation, batch
directory processing, sparse-solver dispatch, Campbell-diagram
orchestration, the ElastoDyn-compatibility blade adapter, polynomial-
fit conditioning, and parser / writer round-trips. They don't have a
separate external-reference frequency to compare against; the gate
is behavioural / contract-style.

| Case | Source / reference | Quantity | Tolerance | Worst observed | Test file | Needs external data |
| --- | --- | --- | ---: | ---: | --- | :---: |
| `check_model` — non-monotonic span detected | construction | `ModelWarning(severity='WARN')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `check_model` — zero / negative mass density detected | construction | `ModelWarning(severity='ERROR')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `check_model` — stiffness jump > 5× detected | construction | `ModelWarning(severity='WARN')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `check_model` — EI_FA / EI_SS extreme ratio detected | construction | `ModelWarning(severity='INFO')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `check_model` — RNA mass > tower mass detected | construction | `ModelWarning(severity='INFO')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `check_model` — singular `PlatformSupport` matrix detected | construction | `ModelWarning(severity='ERROR')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `check_model` — n_modes > 6 × n_nodes detected | construction | `ModelWarning(severity='ERROR')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `check_model` — polynomial-fit cond > 1e4 detected | construction | `ModelWarning(severity='WARN' or 'ERROR')` raised | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `Tower.run(check_model=True/False)` auto-run integration | construction | `UserWarning` emitted on True / suppressed on False | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `mac_matrix` — identity test (shapes vs themselves) | construction | diagonal entries equal 1, off-diag < 1 for distinct shapes | exact / `np.allclose` | (within tol) | [`tests/test_mac.py`](tests/test_mac.py) | no |
| `mac_matrix` — orthogonal shapes (FA-only vs SS-only) | construction | every entry = 0 | exact / `np.allclose` | (within tol) | [`tests/test_mac.py`](tests/test_mac.py) | no |
| `compare_modes` — frequency-shift sign matches input direction | construction | sign of `frequency_shift` matches sign of (f_B − f_A) | exact / `np.allclose` | (within tol) | [`tests/test_mac.py`](tests/test_mac.py) | no |
| `compare_modes` — Hungarian-optimal mode pairing | construction | output pair = identity when shapes match | exact | (within tol) | [`tests/test_mac.py`](tests/test_mac.py) | no |
| `plot_mac` — matplotlib smoke test | construction | Figure has ≥ 1 Axes; title carries labels | structural | (within tol) | [`tests/test_mac.py`](tests/test_mac.py) | no |
| `ModalResult.save / load` — NPZ round-trip | self | per-field `np.allclose` vs original | `rtol = 1e-12` | (within tol) | [`tests/test_serialize.py`](tests/test_serialize.py) | no |
| `ModalResult.to_json / from_json` — JSON round-trip | self | per-field equality vs original | exact / `np.allclose` | (within tol) | [`tests/test_serialize.py`](tests/test_serialize.py) | no |
| `ModalResult` — metadata capture (version + timestamp + git hash) | self | metadata dict populated with non-empty pybmodes_version | always | (within tol) | [`tests/test_serialize.py`](tests/test_serialize.py) | no |
| `CampbellResult.save / load` — NPZ round-trip | self | per-field `np.allclose` vs original | `rtol = 1e-12` | (within tol) | [`tests/test_serialize.py`](tests/test_serialize.py) | no |
| `CampbellResult.to_csv` — spec column order | self | header = `[rpm, <labels>, <labels>_mac]` | exact match | (within tol) | [`tests/test_serialize.py`](tests/test_serialize.py) | no |
| NPZ load is pickle-free on modern archives; legacy `dtype=object` `__meta__` takes a warned `allow_pickle=True` fallback (F2) | construction (forged legacy archive) | modern load emits no warning; legacy load raises `UserWarning` + still round-trips | exact / structural | (within tol) | [`tests/test_serialize.py`](tests/test_serialize.py) | no |
| `_validate_lengths` / `CampbellResult._validate` reject malformed schemas before any export (F5) | construction (bad participation / label shapes) | `ValueError` before save / to_json / to_csv | always raises | (within tol) | [`tests/test_serialize.py`](tests/test_serialize.py) | no |
| `check_model` n_modes guard uses exact `n_free_dof`, not `6·n_nodes` — no false ERROR in (6·n_nodes, n_free_dof] (F3) | construction (nselt=10 ⇒ n_free_dof=90) | n_modes=80 clean; n_modes=200 ERROR | always | (within tol) | [`tests/test_checks.py`](tests/test_checks.py) | no |
| `pybmodes patch` rejects conflicting `--output` / `--output-dir` (F4); silent-agree preserved for equal / single | construction | exit 2 + clear message only when paths differ | exact match | (within tol) | [`tests/test_cli_patch_args.py`](tests/test_cli_patch_args.py) | no |
| `kaimal_spectrum` — IEC 61400-1 longitudinal closed form | textbook: `S(0)=4σ²L/U`, strictly decreasing in `f` | low-freq plateau + monotonicity | exact / `pytest.approx` | (within tol) | [`tests/test_environmental.py`](tests/test_environmental.py) | no |
| `jonswap_spectrum` — peak at `1/Tp`; `m0 = Hs²/16` significant-wave-height identity | textbook JONSWAP; the Hs identity is exact by construction | `f_peak`, `∫S df` | peak `abs=2e-3`; `m0` `rel=2 %` | (within tol) | [`tests/test_environmental.py`](tests/test_environmental.py) | no |
| `plot_environmental_spectra` — structure (bands + tower lines + spectra + legend) and input guards | construction | axvspans / vlines / legend entries; `ValueError` on bad `freq_max` / `harmonics` | structural / always raises | (within tol) | [`tests/test_environmental.py`](tests/test_environmental.py) | no |
| `read_out(strict=True)` — fail-loud on short / non-numeric / non-finite / duplicate-mode / empty `.out`; default stays tolerant | construction (malformed `.out` fixtures) | `BModeOutParseError` with file/line/mode context vs tolerant skip | exact match / always raises | (within tol) | [`tests/test_out_parser.py`](tests/test_out_parser.py) | no |
| `generate_report` — markdown contains frequencies | construction | every 4-dp frequency appears in body | exact match | (within tol) | [`tests/test_report.py`](tests/test_report.py) | no |
| `generate_report` — HTML is well-formed | construction | DOCTYPE + balanced `<table>` / `<tr>` / `<td>` tags | structural | (within tol) | [`tests/test_report.py`](tests/test_report.py) | no |
| `generate_report` — CSV has coefficient columns | construction | second header row has `C2..C6, rms_residual, cond_number` | exact column match | (within tol) | [`tests/test_report.py`](tests/test_report.py) | no |
| `pybmodes batch` — discovery filter excludes Tower / Blade / SubDyn `.dat` | self (committed `reference_decks/`) | 6 main decks found; 0 aux files | exact | (within tol) | [`tests/test_batch.py`](tests/test_batch.py) | no |
| `pybmodes batch` — summary CSV column set | construction | header = `[filename, overall_verdict, TwFAM2Sh_ratio, TwSSM2Sh_ratio, n_fail, n_warn]` | exact column match | (within tol) | [`tests/test_batch.py`](tests/test_batch.py) | no |
| `pybmodes batch --patch` — every block reaches PASS / WARN | OpenFAST r-test 5MW deck | per-deck `overall_verdict` post-patch | no FAIL | (within tol) | [`tests/test_batch.py`](tests/test_batch.py) | yes |
| `pybmodes examples --copy --kind samples` — vendors `cases/sample_inputs/` | self (committed bundle) | four analytical-reference subdirs + `reference_turbines/` present at dest | exact match | (within tol) | [`tests/test_examples_cli.py`](tests/test_examples_cli.py) | no |
| `pybmodes examples --copy --kind decks` — vendors `reference_decks/` | self (committed bundle) | `nrel5mw_land/` present at dest; `sample_inputs/` not vendored | exact match | (within tol) | [`tests/test_examples_cli.py`](tests/test_examples_cli.py) | no |
| `pybmodes examples --copy` (default `--kind all`) — vendors both bundles | self (committed bundle) | both `sample_inputs/` and `reference_decks/` present at dest | exact match | (within tol) | [`tests/test_examples_cli.py`](tests/test_examples_cli.py) | no |
| `pybmodes examples` — destination conflict guard | construction (pre-existing target subdir) | exit code 2 + preexisting file preserved | exact match | (within tol) | [`tests/test_examples_cli.py`](tests/test_examples_cli.py) | no |
| `pybmodes examples --force` — overwrites stale destination | construction (pre-existing target subdir + stale file) | stale file removed; real bundle present | exact match | (within tol) | [`tests/test_examples_cli.py`](tests/test_examples_cli.py) | no |
| `WamitReader.read` — surge A_inf vs manual redimensionalisation | IEA-15-240-RWT-UMaineSemi WAMIT `.1` | `A_inf[0,0]` (dim, kg) | `rel = 1 %` | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader.read` — heave A_inf vs manual redim | IEA-15-240-RWT-UMaineSemi WAMIT `.1` | `A_inf[2,2]` (dim, kg) | `rel = 1 %` | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader.read` — pitch A_inf vs manual redim | IEA-15-240-RWT-UMaineSemi WAMIT `.1` | `A_inf[4,4]` (dim, kg·m²) | `rel = 1 %` | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader.read` — heave C_hst vs manual redim | IEA-15-240-RWT-UMaineSemi WAMIT `.hst` | `C_hst[2,2]` (dim, N/m) | `rel = 1 %` | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader.read` — pitch C_hst positive (stable semi) | IEA-15-240-RWT-UMaineSemi WAMIT `.hst` | `C_hst[3,3] > 0` and ≈ 2.193e9 N·m/rad | `rel = 1 %` | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader.read` — A_inf and C_hst symmetric (no transposed-index bug) | IEA-15-240-RWT-UMaineSemi WAMIT pair | `max(\|M − M.T\|) < 1e-3 · max(\|M\|)` | always | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader._resolve_pot_path` — quoting / separator robustness | construction (4 spellings of one path) | all forms resolve to the same Path | exact match | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader.read` — clear error on missing PotFile | construction (non-existent root) | `FileNotFoundError` naming `<root>.1` | exact match | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `HydroDynReader` — ULEN / PotMod / PotFile / PtfmRefzt / ρ / g | IEA-15-240-RWT-UMaineSemi `_HydroDyn.dat` | 6 scalar fields | exact / default | (within tol) | [`tests/test_wamit_reader.py`](tests/test_wamit_reader.py) | yes |
| `WamitReader` — upper-triangle-only `.hst` / `.1` mirrored | synthetic 2-entry input | `C[2,4] == C[4,2]` and `A_inf[0,2] == A_inf[2,0]` after parse | exact | (within tol) | [`tests/test_wamit_reader_synth.py`](tests/test_wamit_reader_synth.py) | no |
| `WamitReader` — Fortran `D` / `d` exponents accepted | synthetic `4.430486D+02` / `1.233416d+04` | dimensional values match the C-form equivalents | `rel = 1e-12` | (within tol) | [`tests/test_wamit_reader_synth.py`](tests/test_wamit_reader_synth.py) | no |
| `_scan_platform_fields` — raises on malformed critical scalars | synthetic deck with `not_a_number   PtfmMass` | `ValueError` naming the offending field | exact match | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | no |
| `Line.solve_static` — inextensible limit matches Irvine 1981 §2.3 | construction (EA → ∞) | residuals on Irvine eq. (2.27)–(2.29) | `rel = 1e-4` | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | no |
| `Line.solve_static` — horizontal-line symmetry | construction (ΔZ = 0) | `V_F = W·L/2` at fairlead | `rel = 1e-6` | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | no |
| `Line.solve_static` — Newton residuals satisfied to ≤ solver tol | construction (typical slack-line geometry) | both catenary residuals after solve | `< 1e-6` m | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | no |
| `MooringSystem.restoring_force` — 3-fold-symmetric zero offset | construction (synthetic 3-line layout) | F\_x, F\_y, M\_x, M\_y, M\_z all `< 1e-3·\|F\_z\|` | `rel = 1e-3` | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | no |
| `MooringSystem.stiffness_matrix` — diagonal positive | construction (synthetic 3-line) | every diag entry > 0 | always | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | no |
| `MooringSystem.stiffness_matrix` — symmetric after symmetrisation | construction (synthetic 3-line) | `max(\|K − K.T\|) < 1e-10 · max(\|K\|)` | always | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | no |
| `MooringSystem.from_moordyn` — OC3 Hywind layout round-trip | OpenFAST r-test 5MW_OC3Spar MoorDyn `.dat` | 3 lines / 6 points / fairlead radius 5.2 / anchor radius 853.87 / L 902.2 | `rel = 1e-3` | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | yes |
| `MooringSystem.stiffness_matrix` — OC3 Hywind surge stiffness | OC3 r-test MoorDyn `.dat` | `K[0,0] ≈ 41,180 N/m` (Jonkman 2010 Table 5-1) | `rel = 5%` | within 0.01 % | [`tests/test_mooring.py`](tests/test_mooring.py) | yes |
| `MooringSystem.stiffness_matrix` — OC3 yaw stiffness (catenary-only) | OC3 r-test MoorDyn `.dat` | `K[5,5] ≈ 1.156e7 N·m/rad` (catenary lines only; OC3's delta-line crowfoot adds the bulk of the published 9.83e7) | `rel = 5%` | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | yes |
| `MooringSystem.stiffness_matrix` — 3-fold-symmetry signatures | OC3 r-test MoorDyn `.dat` | K[0,0] = K[1,1], K[3,3] = K[4,4], K[0,4]·K[1,3] < 0 | `rel = 1e-4` | (within tol) | [`tests/test_mooring.py`](tests/test_mooring.py) | yes |
| `Tower.from_elastodyn_with_mooring` — end-to-end OC3 coupled solve | OC3 ElastoDyn + r-test MoorDyn `.dat` | hub\_conn=2, PlatformSupport populated, 1st tower-bending in 0.4–0.6 Hz | `rel = 5%` | within 1.2 % of Jonkman 2010's 0.482 Hz | [`tests/test_mooring.py`](tests/test_mooring.py) | yes |
| Sparse `eigsh` shift-invert — matches dense `eigh` | construction (SPD problem at threshold + 100 DOFs) | lowest-k eigenvalues | `rtol = 1e-8` | (within tol) | [`tests/fem/test_sparse_solver.py`](tests/fem/test_sparse_solver.py) | no |
| Sparse path triggered above threshold | construction | log message announces "sparse shift-invert" | always | (within tol) | [`tests/fem/test_sparse_solver.py`](tests/fem/test_sparse_solver.py) | no |
| Sparse path skipped below threshold | construction | log message does not mention "sparse" | always | (within tol) | [`tests/fem/test_sparse_solver.py`](tests/fem/test_sparse_solver.py) | no |
| Asymmetric problems fall back to dense general `eig` | construction (asymmetric K) | log message announces "dense general"; sparse not invoked | always | (within tol) | [`tests/fem/test_sparse_solver.py`](tests/fem/test_sparse_solver.py) | no |
| `pybmodes patch --dry-run` writes nothing (mtime check) | construction | mtime of every staged file unchanged | exact match | (within tol) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| `pybmodes patch --diff` PR format contains `×` improvement ratio | construction | stdout contains `×`, `RMS improvement:`, `better` literals | exact match | (within tol) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| `pybmodes patch --output` leaves source byte-identical | construction | source-file sha256 unchanged | exact match | (within tol) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| Default in-place `pybmodes patch` emits first-time-run hint | construction | stdout contains `--dry-run --diff` | exact match | (within tol) | [`tests/test_validate.py`](tests/test_validate.py) | yes |
| Tower torsion-contamination filter — rejects T_tor ≥ 10 % | construction (synthetic torsion-contaminated mode) | `rejected_modes` carries the contaminated mode | always | (within tol) | [`tests/test_classifier.py`](tests/test_classifier.py) | no |
| Tower torsion-contamination filter — accepts pure bending | construction (synthetic clean modes) | `rejected_modes` is empty | always | (within tol) | [`tests/test_classifier.py`](tests/test_classifier.py) | no |
| IEA-3.4 deck — torsion participations populated, summing to 1 | OpenFAST IEA-3.4-130-RWT deck | per-mode `(T_FA, T_SS, T_tor)` triple | `Σ = 1` to `abs_tol = 1e-9` | (within tol) | [`tests/test_classifier.py`](tests/test_classifier.py) | yes |
| Campbell sweep — Hungarian MAC tracking on bundled NREL 5MW reference deck | committed `reference_decks/nrel5mw_land/` | `mac_to_previous` ≥ 0.90 between consecutive steps on a smooth sweep | always | (within tol) | [`tests/test_campbell.py`](tests/test_campbell.py) | no |
| Campbell sweep — input validation (NaN / inf / negative / unsorted RPM) | construction | `ValueError` raised | always raises | (within tol) | [`tests/test_campbell.py`](tests/test_campbell.py) | no |
| Campbell sweep — restores `bbmi.rot_rpm` after sweep (clean + on-exception) | construction | post-sweep / post-exception BMI state | unchanged | (within tol) | [`tests/test_campbell.py`](tests/test_campbell.py) | no |
| Campbell sweep — tower modes constant across all rotor speeds | construction | tower frequency vs rotor speed | exactly constant (no `rpm` dependence in tower modal eigenproblem) | (within tol) | [`tests/test_campbell.py`](tests/test_campbell.py) | no |
| `plot_campbell` — 6-DOF platform overlay: degenerate-pair merge + Hz/period labels + log axis (issue #39); default-off byte-identical; non-finite/≤0 skipped | construction (synthetic CampbellResult + platform_modes) | horizontal refs + right-margin labels + `yscale`; regression invariant w/o platform_modes | structural / exact | (within tol) | [`tests/test_campbell.py`](tests/test_campbell.py) | no |
| ElastoDyn-compat blade adapter — strips `str_tw`, `tw_iner`, offsets | Jonkman 2015 NREL forum guidance | resulting BMI fields | zeroed for compat-on, preserved for compat-off | (within tol) | [`tests/test_elastodyn_compatible.py`](tests/test_elastodyn_compatible.py) | no |
| ElastoDyn-compat — frequency drift on flap modes is small | construction | rel freq diff vs compat-off | ≤ ~ few % on flap modes | (within tol) | [`tests/test_elastodyn_compatible.py`](tests/test_elastodyn_compatible.py) | no |
| ElastoDyn `.dat` parse → write → parse round-trip | self | per-field `np.allclose` vs original | `rtol = 1e-12` | (within tol) | [`tests/test_elastodyn_reader.py`](tests/test_elastodyn_reader.py) | yes |
| Polynomial-fit design-matrix cond-number reporting | construction | `RuntimeWarning` above thresholds | WARN > 1e4, FAIL > 1e6 | (within tol) | [`tests/test_fitting.py`](tests/test_fitting.py) | no |
| BMI / section-properties parser primitives | construction (synthetic fixtures) | round-trip equality | exact / `np.allclose` | (within tol) | [`tests/test_io.py`](tests/test_io.py) | no |
| FEM building blocks — boundary conditions, eigensolver, normalisation | construction | per-DOF / per-mode invariants | exact / `np.allclose` | (within tol) | [`tests/fem/test_*.py`](tests/fem/) | no |
| `tubular_section_props` — closed-form tube identities | $A=\pi(R_o^2-R_i^2)$, $I=\tfrac{\pi}{4}(R_o^4-R_i^4)$, $J=2I$, $G=E/2(1+\nu)$ (textbook) | mass\_den / flp\_stff / tor\_stff / axial\_stff / flp\_iner | exact | `pytest.approx` | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| `outfitting_factor` — scales mass + rotary inertia only, never stiffness | construction (factor 1.07) | mass × 1.07; EI / GJ / EA / structural ρI unchanged | exact | `pytest.approx` | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| Geometry guard — rejects D ≤ 0 / t ≤ 0 / 2t ≥ D | construction (degenerate sections) | `ValueError` raised | always raises | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| WindIO dialect equivalence — modern `outer_shape`/`structure` vs older `outer_shape_bem`/`internal_structure_2d_fem` | construction (numerically identical fixtures) | same `WindIOTubular` from both key dialects | exact / `np.allclose` | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| WindIO duplicate-anchor tolerance — IEA-10's redefined-anchor habit | construction (`&g` redefined; stock `SafeLoader` raises `ComposerError`) | parses (last-wins) where strict PyYAML fails | exact | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| WindIO orthotropic wall material — clear out-of-scope error | construction (list-valued composite E) | `ValueError` naming "orthotropic", not bare `float(list)` `TypeError` | exact match | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| WindIO thickness interp — linear vs piecewise-constant differs on a taper | construction (smoothly tapered wall) | mid-station wall thickness diverges by > 0.01 m | exact | 0.035 vs 0.05 m | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| WindIO PyYAML-absent guard — names the `[windio]` extra | construction (monkeypatched import) | `ModuleNotFoundError` matching `pybmodes\[windio\]` | exact match | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | no |
| **IEA-15 base WindIO yaml → derived mass / EI vs IEA-15 *Monopile* ElastoDyn tower table** (RNA-independent like-for-like; that deck was generated from this geometry) | IEA Wind Task 37 IEA-15-240-RWT ontology + OpenFAST Monopile deck | distributed mass\_den & FA EI, station-by-station | < 0.5 % | **7.5 × 10⁻¹² (machine-exact)** | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | yes |
| Upstream WindIO corpus — IEA-3.4 / 10 / 15 / 22 + WISDEM examples (both dialects + IEA-10 dup-anchor) parse to a physically sane tube | IEA Wind Task 37 RWT ontologies + WISDEM example yamls | grid ∈ [0,1] monotone, D / t > 0, 2t < D, steel E / ρ / ν range, span > 0 | structural | 12 yamls clean | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | yes |
| `Tower.from_windio` — older-dialect modal smoke (IEA-3.4 / 10 / 22) | IEA Wind Task 37 RWT ontologies | full FEM solve → bare-member spectrum | finite, positive, ascending | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | yes |
| Older-dialect yaml-derived mass / EI vs same-turbine ElastoDyn tower table (ballpark — those decks were not 1:1 geometry round-trips, unlike IEA-15) | IEA-3.4 / 10 / 22 ontology + own OpenFAST tower deck | distributed mass\_den & FA EI envelope | mass < 25 %, EI < 30 % | (within tol) | [`tests/test_geometry_windio.py`](tests/test_geometry_windio.py) | yes |
| CLT laminate primitives — reduced stiffness / ply transform / ABD / membrane condensation | Jones, *Mechanics of Composite Materials* (2nd ed.) §2.5–4.3 closed forms | $Q$, $\bar Q(\theta)$, $A/B/D$, $\tilde A$ | exact | `pytest.approx` (12 cases) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | no |
| Airfoil `nd_arc` profile — arc spine / LE-TE landmarks / chord↔arc / blend | construction (analytic circle & ellipse) | `s_le`, `tc`, landmark `(x,y)`, blend weights | exact (≤ 2e-3) | (within tol) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | no |
| WindIO web/layer `nd_arc` resolver — older-explicit ≡ modern-anchor-registry dialect equivalence | construction (numerically identical fixtures) | identical resolved bands from both dialects | exact / `np.allclose` | (within tol) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | no |
| Thin-wall reduction — isotropic tube & box vs exact closed form | textbook thin-ring / thin-wall-box EA / EI / GJ / mass | section properties | exact | < 2 % (discretisation) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | no |
| Multi-cell Bredt–Batho — symmetric interior web carries zero shear flow | analytic (a symmetric diametral web ⇒ GJ unchanged vs webless) | `GJ`, `n_cells` | exact | < 2 % | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | no |
| `windio_blade` dual-dialect end-to-end | construction (tube blade, both WindIO dialects) | identical `SectionProperties`; physically sane | exact / `np.allclose` | (within tol) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | no |
| **6×6 cross-sectional decoupling** (issue #50) — rigid-offset congruence → tension centre → principal-axis eigen-decomposition → shear centre / `GJ`; analytic round-trips | construction (build a 6×6 *from* known decoupled props at a known offset / rotation) | recovered `EA`/`EI_flap`/`EI_edge`/`GJ`/`mass`/centres + defining decoupling property | exact (`rtol≤1e-6`) round-trip | (within tol) | [`tests/test_decouple.py`](tests/test_decouple.py) | no |
| **WindIO published distributed elastic properties → decoupled beam** (issue #48/#50) — `elastic="auto"` prefers them over PreComp; both dialects parsed onto the span grid as the **full** 6×6 then decoupled (elastic/shear centre + principal elastic axes — *not* raw diagonal); `"file"`/`"precomp"`/bad-value guards | construction (tube blade with embedded published properties, both dialects) | `SectionProperties` from the decoupled 6×6; mode-string dispatch | exact / `np.allclose`; `ValueError` on bad/missing | (within tol) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | no |
| **WindIO published elastic properties vs the turbine's own BeamDyn 6×6**, IEA-15, span 0.15–0.90 (issue #48/#50) — *decoupled-vs-decoupled* (the BeamDyn 6×6 is also referenced at the blade axis, so the apples-to-apples oracle is the decoupled BeamDyn, not its raw diagonal); also pins that raw≠decoupled so the case genuinely exercises #50, and that the published path stays far tighter than PreComp | IEA Wind Task 37 IEA-15 ontology + companion `*_BeamDyn_blade.dat` | decoupled `mass` / `EA` / `EI_flap` / `EI_edge` / `GJ` | mass & EA med < 3 %; EI med < 5 %; GJ med < 8 %; raw-vs-decoupled EI > 5 %; strictly tighter than `elastic="precomp"` | (within tol) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | yes |
| **WindIO composite blade → distributed beam props vs the turbine's own BeamDyn 6×6** (WISDEM-PreComp-generated; `elastic="precomp"` — the PreComp-fallback path's true scope), IEA-3.4 / 10 / 15 / 22, span 0.15–0.90 | IEA Wind Task 37 RWT ontology + companion `*_BeamDyn_blade.dat` | `mass` / `EA` (PreComp-class) ; `GJ` & `EI` (diagonal-reduction approximate — documented limitation) | mass med < 6 % / max < 15 %; EA med < 10 % / max < 18 %; GJ med < 22 % / max < 47 %; EI med < 35 % / max < 55 % | mass ≈ 1.5–4 % med ; EA ≈ 1–8 % med ; GJ ≈ 3–18 % med (IEA-10 worst, composite multi-cell torsion) ; EI ≈ 2–27 % med (weak axis worst — omits spar-cap-offset / bend-twist coupling) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | yes |
| WindIO `elastic_properties_mb` ecosystem-drift caveat — a turbine's published `_mb` 6×6 is not always mutually consistent with its own companion BeamDyn deck (IEA-10 `_mb` ≈ 50 % from its BeamDyn 6×6); `auto` treats the WindIO ontology as canonical per #48 intent | IEA-10 ontology `_mb` vs `*_BeamDyn_blade.dat` | documented divergence (not a parser bug — same pattern as `cases/ECOSYSTEM_FINDING.md`) | n/a (documented, not gated) | ≈ 50 % (intrinsic to upstream data) | [`cases/ECOSYSTEM_FINDING.md`](cases/ECOSYSTEM_FINDING.md) | yes |
| `RotatingBlade.from_windio` — modal smoke, all 4 RWTs (both dialects + parametric-layer blades) | IEA Wind Task 37 RWT ontologies | full FEM → parked-blade spectrum | finite, positive, ascending | (within tol) | [`tests/test_windio_blade.py`](tests/test_windio_blade.py) | yes |
| WindIO floating reader — joints (cartesian + cylindrical `[r,θ°,z]`), member geometry, axial-joint resolution, transition | construction | parsed `WindIOFloating` | exact / `np.allclose` | (within tol) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | no |
| Hydrostatic restoring — single surface-piercing cylinder & off-axis column vs closed form | textbook waterplane + buoyancy (WAMIT/`.hst` convention) | `C_hst` 6×6 | exact | < 1 % (discretisation) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | no |
| Morison added mass + RAFT end-cap & rigid-body inertia vs closed form | thin-ring transverse + RAFT `Ca_End` end disc; thin-wall steel | `A_inf` / mass 6×6 | exact | `pytest.approx` | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | no |
| `from_windio_mooring` — topology / props / 120°-symmetric stiffness / regression-warn / bad-ref | construction | `MooringSystem` + 6×6 `K` | exact / symmetry | (within tol) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | no |
| `from_windio_floating` — yaml-only coupled modal smoke (stable single-column FOWT) | construction | full coupled FEM spectrum | finite, ascending, soft RB ≪ tower | (within tol) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | no |
| **WindIO floating C_hst vs the turbine's own potential-flow WAMIT `.hst`** (geometry-exact anchor) | IEA-15 UMaine VolturnUS-S ontology + companion HydroDyn/WAMIT | heave / roll / pitch restoring | < 3 % / < 4 % | **heave 0.8 %, roll/pitch 1.6 %** | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | yes |
| WindIO *screening-preview* Morison `A_inf` & structural mass vs WAMIT / ElastoDyn (the yaml-only tier — NOT industry-grade, emits a `UserWarning`) | IEA-15 UMaine VolturnUS-S + companion HydroDyn / ElastoDyn | added-mass diagonal / `PtfmMass` | surge/sway/yaw < 45 %; heave/roll/pitch factor ~2; mass a lower bound | surge 22 %, heave 53 % (Morison ≠ BEM, as RAFT/WISDEM also find — supply a HydroDyn deck for industry grade); struct+fixed mass ≈ 0.36·PtfmMass (trim ballast excluded by design) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | yes |
| `from_windio_mooring` vs `from_moordyn` (the yaml-only *screening* mooring tier) | IEA-15 UMaine VolturnUS-S ontology + companion MoorDyn | deck-fallback line props; 6×6 `K` | props exact (rel<1e-9); roll/pitch<15 %, heave<15 %, yaw<20 %, surge/sway<40 % | props exact; roll/pitch ~3 %, heave ~9 %, surge/sway ~32 % (WindIO centreline vs MoorDyn surface-attachment radius — the screening path; the coupled industry path below uses the full MoorDyn system) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | yes |
| **`Tower.from_windio_floating` (companion decks present) vs the BModes-JJ-validated `from_elastodyn_with_mooring`** — the industry-grade tier (full deck model: MoorDyn system + WAMIT A_inf/C_hst + ElastoDyn PtfmMass/RIner+RNA; only the tower is the Phase-1 machine-exact WindIO one) | IEA Wind Task 37 ontology + companion HydroDyn/MoorDyn/ElastoDyn | coupled rigid-body + tower-bending frequencies | every platform RB mode + 1st tower bending < 1 %; 2nd+ tower harmonics < 8 % | **all 6 platform rigid-body modes + 1st tower bending 0.0–0.3 %** (reference grade); 2nd+ tower harmonics ≤ 6 % (the Phase-1 WindIO-vs-ElastoDyn tower-discretisation residual, orthogonal to the platform) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | yes |
| `Tower.from_windio_floating(..., platform_support=...)` — injected-platform tier (WindIO tower + caller-supplied PlatformSupport; floater designed separately, issue #35) | construction (synthetic FOWT yaml + stable diagonal PlatformSupport) | byte-equivalence to the manual WindIO-tower-BMI + attach-PlatformSupport recipe; no screening warning; decks mutually-exclusive `ValueError`; soft RB ≪ tower | `rtol=1e-9` / exact / structural | (within tol) — adds no new numerics; the free-free PlatformSupport FEM is the OC3-Hywind-certified one (`test_certtest`) | [`tests/test_windio_floating.py`](tests/test_windio_floating.py) | no |
| `pybmodes windio` one-click — companion-deck auto-discovery scoping (bare yaml ⇒ no decks/screening; RWT layout ⇒ scoped, config-matched, aux-excluded) + fixed-tower / floating-screening / bad-input | construction | discovery dict + report file + exit code + SCREENING warning | exact / structural | (within tol) | [`tests/test_windio_cli.py`](tests/test_windio_cli.py) | no |
| `pybmodes windio <IEA-15 VolturnUS-S>` end-to-end — discovery picks the UMaineSemi *floating* decks ⇒ industry-grade deck-backed coupled model | IEA Wind Task 37 IEA-15 RWT tree (ontology + companion HydroDyn/MoorDyn/ElastoDyn) | discovered config + exit code + report + no screening warning | exact / structural | (within tol) | [`tests/test_windio_cli.py`](tests/test_windio_cli.py) | yes |

## What "needs external data" means — and how integration coverage is gated

`yes` rows skip in the default `pytest` run (they carry the
`integration` marker). Run them locally with `pytest -m integration`
once you have the upstream sources:

- `external/OpenFAST_files/r-test/` — clone of the OpenFAST regression-test
  corpus (any recent commit; pyBmodes was last validated against
  `dd5feaaa`).
- `external/OpenFAST_files/IEA-3.4-130-RWT/`, `IEA-10.0-198-RWT/`,
  `IEA-15-240-RWT/`, `IEA-22-280-RWT/` — clones of the IEA Wind Task 37
  reference-turbine repositories (each ships a WindIO ontology `.yaml`
  plus matching OpenFAST/ElastoDyn decks; the WindIO validation rows
  consume both).
- `external/OpenFAST_files/WISDEM/examples/` — clone of the WISDEM
  examples tree (modern-dialect WindIO ontology yamls for the
  corpus-parse coverage).
- `external/BModes/CertTest/` — BModes v3.00 CertTest reference outputs.
- `external/BModes/docs/examples/` — the bundled `CS_Monopile.bmi` and
  `OC3Hywind.bmi` example decks plus their BModes JJ `.out` files.

These directories are gitignored under the **Independence stance**.
The data is not bundled in the repo because the
licence terms of the upstream NREL / IEA Wind Task 37 packages
include attribution / indemnification obligations that pyBmodes can't
inherit by republication. The contributor clones them locally.

**Integration-track coverage is therefore developer-local + manual
pre-tag, not CI-gated.** Specifically:

- The default GitHub Actions runner has no upstream decks checked
  out, so `pytest -m integration` exits with code 5 ("no tests
  collected"). The CI workflow treats that single exit code as a
  pass, so the job stays green; **any other non-zero exit** (i.e.
  a genuine integration-test failure when data IS provided by a
  custom workflow run) fails the build.
- The pre-tag release sequence (see
  [`docs/release_checklist.rst`](docs/release_checklist.rst), step 2)
  requires the maintainer to run `pytest -m integration` locally on
  a checkout that has the upstream sources cloned, and confirm
  every case passes, before tagging a new version. The tag therefore
  represents a state that's been integration-verified by hand even
  though CI couldn't see it.
- `scripts/audit_validation_claims.py` (run as part of the release
  checklist) scans this matrix and asserts that every test-file path
  named in a row actually exists and contains at least one collected
  test method. This is the gate that catches "claim ahead of test"
  drift: a row may legitimately need external data (and thus skip
  in CI), but its referenced test file must always be present and
  populated.

**For users**: when you depend on pyBmodes, the integration-track
guarantee is "the maintainer has run these locally before tagging
this release." If you need stronger CI-level evidence (i.e. a
re-runnable workflow against the same upstream commits), follow the
audit script's instructions to mirror the upstream sources into a
private repo and dispatch the integration workflow there.

## Reproducing every row

Track A (no external data):

```bash
pytest tests/fem/test_cantilever.py
pytest tests/fem/test_uniform_tower_analytical.py
pytest tests/fem/test_rotating_uniform_blade.py
pytest tests/fem/test_rotating_blade_with_tip_mass.py
pytest tests/fem/test_rotating_cable.py
pytest tests/test_classifier.py        # 3 of 4 pass without external data
pytest tests/test_geometry_windio.py   # closed-form tube + dialect/anchor robustness
pytest tests/test_windio_blade.py      # CLT + thin-wall + multi-cell closed forms
pytest tests/test_windio_floating.py   # hydrostatic / Morison / mooring closed forms
```

Track A (external data needed):

```bash
pytest tests/test_certtest.py -m integration
pytest tests/test_geometry_windio.py -m integration  # WindIO corpus + IEA-15 anchor
pytest tests/test_windio_blade.py -m integration     # blade vs BeamDyn 6×6 (4 RWTs)
pytest tests/test_windio_floating.py -m integration  # IEA-15 VolturnUS-S vs WAMIT/MoorDyn/ED
```

Track B:

```bash
pytest tests/test_validate.py -m integration   # validator round-trip
pytest tests/test_reference_decks.py            # always self-contained
```

Or run the full matrix at once:

```bash
pytest -m ""        # default + integration combined
```
