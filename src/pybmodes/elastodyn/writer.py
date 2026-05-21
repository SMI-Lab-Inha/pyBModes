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

"""Patch ElastoDyn .dat files with computed polynomial coefficients.

Each coefficient occupies a separate line of the form
``<value>   <ParamName(k)>   - comment text``. The writer finds each
line by matching the parameter name token, then replaces the leading
value in-place, preserving indentation and all trailing text.
"""

from __future__ import annotations

import pathlib
import re

from pybmodes.elastodyn.params import BladeElastoDynParams, TowerElastoDynParams

_MISSING: list[str] = []   # sentinel for type checking


def patch_dat(
    path: str | pathlib.Path,
    params: BladeElastoDynParams | TowerElastoDynParams,
) -> None:
    """Patch named mode-shape coefficient lines in an ElastoDyn .dat file.

    Each parameter in *params* is located by searching for its exact name as a
    whitespace-delimited token.  The value on the same line (the token before
    the name) is replaced with the computed coefficient.  All other content
    (indentation, comment text) is left unchanged.

    Parameters
    ----------
    path   : path to the ElastoDyn .dat file (modified in place).
    params : BladeElastoDynParams or TowerElastoDynParams.

    Raises
    ------
    KeyError  if a required parameter name is not found in the file.
    """
    path = pathlib.Path(path)
    # newline='' disables Python's universal-newline translation on read so we
    # see the file's original \n / \r\n / \r endings; we restore them verbatim
    # below.  Without this, on Windows both the read and write sides would
    # silently coerce LF into CRLF.  (pathlib's read_text/write_text only got
    # the ``newline`` argument in 3.13, so we use open() for compatibility
    # with 3.11.)
    with path.open(encoding='utf-8', newline='') as f:
        text = f.read()
    lines = text.splitlines(keepends=True)

    replacements = params.as_dict()
    missing = list(replacements.keys())

    for i, line in enumerate(lines):
        # Capture the original line ending (\n, \r\n, \r, or none for EOF) so
        # we preserve the file's existing convention — important on Windows
        # OpenFAST files where mixing \n and \r\n in the same file confuses
        # downstream tooling.
        body = line.rstrip('\n\r')
        ending = line[len(body):]
        for name in list(missing):
            # Match: optional whitespace, any value, whitespace, then the exact
            # parameter name as a whole token (word boundary or end-of-field).
            pattern = re.compile(
                r'^(\s*)\S+(\s+' + re.escape(name) + r')(\s.*)?$'
            )
            m = pattern.match(body)
            if m:
                value = replacements[name]
                tail  = m.group(3) if m.group(3) is not None else ''
                new_line = f"{m.group(1)}{value: .7E}{m.group(2)}{tail}{ending}"
                lines[i] = new_line
                missing.remove(name)
                break

    if missing:
        raise KeyError(
            f"The following parameter names were not found in {path}:\n"
            + "\n".join(f"  {n}" for n in missing)
        )

    with path.open('w', encoding='utf-8', newline='') as f:
        f.write(''.join(lines))
