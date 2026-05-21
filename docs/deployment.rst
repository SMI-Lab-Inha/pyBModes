Deployment
==========

How the documentation site at
https://pybmodes.readthedocs.io/ is built, deployed, and maintained.

Where the site lives
--------------------

The docs are hosted on **Read the Docs** (RTD). RTD reads the
``.readthedocs.yaml`` at the repository root, installs the package
plus its ``[docs,plots,windio]`` extras, then runs the strict
Sphinx build (``fail_on_warning: true``) and serves the resulting
HTML.

Two consequences worth remembering:

- **The rendered docs are what users should read.** Browsing the
  raw ``.rst`` source on GitHub will show Sphinx-specific roles
  (``:math:``, ``:doc:``, ``:func:``, ``:class:``) as literal text
  — that's by design, GitHub's RST viewer doesn't process those
  roles. Always link readers to the RTD URL, not the source file.
- **The strict build is the gate.** A docs PR that introduces any
  Sphinx warning fails RTD's build (matching the CI ``docs`` job's
  ``-W --keep-going`` flag). The deployed site only updates after a
  green build on master.

One-time setup (maintainer)
---------------------------

Done once. Until this is complete the RTD URLs in README /
CHANGELOG / VALIDATION / CONTRIBUTING will 404.

1. **Sign in to Read the Docs**

   Go to https://readthedocs.org/ and sign in with the GitHub
   account that owns the ``SMI-Lab-Inha/pyBModes`` repository.

2. **Import the project**

   From the dashboard click *Import a Project → Import from GitHub*,
   pick ``pyBModes`` from the list, accept the default project slug
   ``pybmodes`` (lowercase, no S — this becomes the URL prefix).

3. **First build**

   The import triggers an immediate build. RTD reads
   ``.readthedocs.yaml`` from the repo root and follows the steps
   it declares — installs ``ubuntu-22.04`` + Python 3.11, ``pip
   install .[docs,plots,windio]``, runs ``sphinx-build`` with
   ``fail_on_warning: true``.

   Expected build time on a cold cache: 3–5 minutes. Watch the
   live log under *Builds → <build id>* on RTD's dashboard.

4. **Verify the deployed URL**

   After the first green build, the site is live at::

       https://pybmodes.readthedocs.io/en/latest/

   The ``latest`` URL tracks ``master``; tagged releases become
   versioned URLs like ``/en/v1.7.0/``.

5. **Webhook is auto-installed**

   RTD installs a GitHub webhook on the repo as part of import,
   so every push to ``master`` (and every tag matching the
   versioning policy) auto-triggers a fresh build. No further
   per-release action required.

What to do when a build fails
-----------------------------

The strict gate (``fail_on_warning: true``) makes RTD builds
mirror local ``nox -s docs`` and the CI ``docs`` job. Any time
the strict CI step passes but RTD fails, the cause is one of:

- **A package missing from the ``[docs]`` extra.** Sphinx extensions
  that work locally (because they're already installed in the dev
  env) fail to import on RTD if not declared in
  ``pyproject.toml``. Add them to the extra and the next build
  recovers.
- **A version-pinning skew.** If a published Sphinx / MyST /
  furo release breaks an extension, RTD's ``latest`` line picks it
  up first. Pin the offending version in ``pyproject.toml``.
- **A LaTeX-required block** (mostly when adding ``imgmath``). RTD's
  ``ubuntu-22.04`` image carries TeX Live but builds with imgmath
  are slower; consider switching to ``mathjax`` (the default) which
  doesn't need LaTeX.

Versioning + the docs site
--------------------------

RTD supports per-tag versioning, so every release tag adds a new
URL prefix that pins the docs at that point in time. Recommended
policy:

- ``latest`` — tracks ``master``; what users land on by default.
- ``stable`` — tracks the most recent tag matching ``v\d+\.\d+\.\d+``
  (no pre-release suffix); what most external links should point at.
- ``v1.7.0`` (etc.) — per-tag versioned URL for citation and
  reproducibility.

The version selector at the bottom of every page lets readers
switch. Pre-release tags (e.g. ``v1.8.0-rc1``) can be marked as
"hidden" on RTD so they don't appear in the selector but stay
reachable by direct URL.

Custom domain (optional)
------------------------

If the lab acquires ``pybmodes.smi-lab.org`` or similar, point a
CNAME record at ``readthedocs.io`` and add the custom domain
under *Admin → Domains* on the RTD dashboard. RTD issues a Let's
Encrypt certificate automatically. The ``readthedocs.io``
subdomain keeps working as a fallback.
