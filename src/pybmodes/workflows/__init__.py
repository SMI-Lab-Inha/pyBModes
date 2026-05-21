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

"""Library-callable workflow functions for every ``pybmodes`` CLI subcommand.

Background
----------

Before Phase 2 of the v1.x architecture refactor, each
:command:`pybmodes` subcommand was implemented inline in a ~60 KB
:mod:`pybmodes.cli` module: argument parsing, business logic, error
formatting, and exit-code mapping were all mixed together. That made
the workflows useful only via subprocess — a notebook or external
script that wanted to *run* the validate / patch / batch / report /
windio / examples flow had to ``subprocess.run(["pybmodes", ...])``
and parse stdout.

This sub-package separates the layers:

- **Workflow functions** (one per subcommand) live here. Each is a
  pure-typed library entry point with explicit parameters and a
  typed result dataclass.
- The CLI parser stays in :mod:`pybmodes.cli` — it parses
  ``sys.argv``, calls the workflow, formats the result to stdout /
  stderr, and translates the result into an exit code. Nothing
  more.

Result-dataclass pattern
------------------------

Every workflow returns a :class:`WorkflowResult` subclass carrying:

- ``exit_code: int`` — what the CLI should return (0 = success,
  1 = verdict failure, 2 = usage / IO error)
- ``messages: list[str]`` — info-level output the CLI prints to
  stdout
- ``errors: list[str]`` — error-level output the CLI prints to
  stderr
- ... plus the workflow's typed payload (validation report,
  patched-deck path, modal frequencies, etc.)

Exception handling
------------------

Workflow functions **raise** on unrecoverable input errors —
:class:`FileNotFoundError` when a path doesn't exist,
:class:`pybmodes.io.errors.ParseError` (and subclasses) when an
input file is structurally invalid. The CLI catches both and maps
to exit code 2. This keeps the workflow signatures clean for
library callers (a notebook can ``try / except`` the typed
exception); the CLI's translation layer absorbs the messy
``argparse.Namespace``-to-function-arg shape.
"""
from __future__ import annotations

from pybmodes.workflows._base import WorkflowResult
from pybmodes.workflows.batch import (
    BatchResult,
    run_batch,
)
from pybmodes.workflows.campbell import (
    CampbellWorkflowResult,
    run_campbell,
)
from pybmodes.workflows.examples import (
    ExamplesResult,
    run_examples_copy,
)
from pybmodes.workflows.patch import (
    PatchResult,
    run_patch,
)
from pybmodes.workflows.report import (
    ReportResult,
    run_report,
)
from pybmodes.workflows.validate import (
    ValidateResult,
    run_validate,
)
from pybmodes.workflows.windio import (
    WindioDiscovery,
    WindioResult,
    discover_windio_inputs,
    run_windio,
)

__all__ = [
    "WorkflowResult",
    "ValidateResult",
    "ExamplesResult",
    "PatchResult",
    "ReportResult",
    "BatchResult",
    "CampbellWorkflowResult",
    "WindioResult",
    "WindioDiscovery",
    "run_validate",
    "run_examples_copy",
    "run_patch",
    "run_report",
    "run_batch",
    "run_campbell",
    "run_windio",
    "discover_windio_inputs",
]
