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

"""Shared base dataclass for ``pybmodes.workflows.*`` results.

Each workflow function returns a :class:`WorkflowResult` subclass.
The base carries the three fields the CLI needs to translate a
workflow outcome into terminal output + an exit code; subclasses
add the workflow-specific typed payload.

Exit-code convention (matches the existing CLI semantics so any
downstream caller scripting ``pybmodes ...`` keeps working):

- ``0`` — success. Includes "WARN-verdict but tolerated" cases
  for validate; warnings are informational, not failures.
- ``1`` — verdict failure (e.g. ``validate`` returned FAIL,
  ``patch`` ran but a block ended at FAIL after re-fit).
- ``2`` — usage / IO error (missing input, malformed deck,
  conflicting flags). Workflows raise rather than return for
  these; the CLI catches and translates.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkflowResult:
    """Base class for every ``pybmodes.workflows.*`` return type.

    Attributes
    ----------
    exit_code : int
        Process exit code the CLI should return after this workflow
        runs. ``0`` for success / informational, ``1`` for verdict
        failure, ``2`` for usage / IO error (rare on this path —
        workflows usually raise instead).
    messages : list[str]
        Info-level lines the CLI prints to stdout. Each entry is a
        complete line (no trailing newline expected; the CLI adds
        one). Empty when the workflow has nothing to say.
    errors : list[str]
        Error-level lines the CLI prints to stderr. Same line-per-
        entry convention. Empty when there are no errors.
    """

    exit_code: int = 0
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
