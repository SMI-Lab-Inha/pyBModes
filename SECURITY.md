# Security policy

## Supported versions

pyBmodes follows semantic versioning with a frozen 1.x public API. Security fixes are applied to the most recent minor release on the 1.x line. Pre-1.0 versions are unsupported.

| Version | Supported |
| --- | --- |
| 1.x latest | ✅ |
| 1.x older | Best effort if the fix is trivial |
| 0.x | ❌ |

## Reporting a vulnerability

pyBmodes is a numerical-modelling library, not a network service. The realistic attack surface is:

- **Maliciously crafted input files** (`.bmi`, `.dat`, `.yaml`, `.out`) that could trigger uncontrolled resource use, file-system writes outside expected paths, or arbitrary-code execution via deserialisation.
- **Supply-chain compromise** of the published wheel on PyPI.

If you believe you have found a vulnerability:

1. **Do not** open a public GitHub issue.
2. Email **jaehoon.seo@inha.ac.kr** with subject `pyBmodes security:` and a minimal reproducer.
3. We will acknowledge within 7 days and provide a remediation timeline.

If a coordinated disclosure window is needed, we will agree one explicitly.

## What is in scope

- Parsing logic in `pybmodes.io.*` (every reader we ship).
- The `pybmodes` CLI argument-handling path.
- Anything that touches the file system, deserialises YAML / JSON / NPZ, or invokes subprocess.

## What is out of scope

- Numerical accuracy disagreements vs other modal-analysis codes — those go to a normal issue under [`VALIDATION.md`](VALIDATION.md)'s framework.
- DoS-by-large-input on a beam mesh you control (FEM solves are O(n³) by design).
- Issues in upstream OpenFAST / BModes / WindIO data files — report those upstream.
