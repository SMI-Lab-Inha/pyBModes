Changelog
=========

The release history of ``pybmodes``. The format follows
`Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_:
each release block lists changes under *Added*, *Changed*,
*Fixed*, *Deprecated*, *Removed*, and *Security* as
appropriate.

The version on ``master`` between tagged releases is the
``[Unreleased]`` block — work-in-progress that will land in
the next minor or patch release.

Versioning policy
-----------------

See :doc:`api_contract` for the full semver contract. Quick
reminder:

- **Major** (X.y.z) — breaks the public API contract.
- **Minor** (x.Y.z) — adds entry points, keyword arguments
  with defaults, dataclass fields with defaults.
- **Patch** (x.y.Z) — bug fixes, numerical-accuracy
  improvements (always called out), docs, internal refactors.

Numerical outputs may shift between minor and patch releases
when validation tightens or a modelling correction lands.
**Every numerical shift is called out** in the entry below
under *Fixed* / *Changed* with magnitude and affected case.

How to find what changed for a specific name
--------------------------------------------

Use the GitHub-rendered changelog and your browser's
``Find on page`` (Ctrl-F / ⌘-F) against the symbol or
deck name. Every release entry is structured around concrete
public-API names and deck filenames; searching for
``Tower.from_windio_floating`` or
``IEA-15-240-RWT-UMaineSemi`` will surface every entry that
touched it.

The full history
----------------

.. include:: ../CHANGELOG.md
   :parser: myst_parser.sphinx_
