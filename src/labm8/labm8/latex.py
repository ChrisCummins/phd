# Copyright (C) 2015 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
import re

import labm8 as lab
from labm8 import fs
from labm8 import io

if lab.is_python3():
    from io import StringIO
else:
    from StringIO import StringIO


def escape(text):
    return re.sub(r"(_)", r"\\\g<1>", str(text))


def wrap_bold(text):
    return "\\textbf{" + text + "}"


def write_table_body(data, output=None, headers=None,
                     header_fmt=wrap_bold, hline_after_header=True,
                     hline_before=False, hline_after=False):
    def _write_row(row):
        output.write(" & ".join(escape(column) for column in row) + "\\\\\n")

    # Determine if we're writing to a file or returning a string.
    isfile = output is not None
    output = output or StringIO()

    # Write hline before body.
    if hline_before:
        output.write("\\hline\n")

    # Write headers.
    if headers:
        _write_row(header_fmt(str(column)) for column in headers)
        # Write hline after headers.
        if hline_after_header:
            output.write("\\hline\n")

    # Write table entries.
    for row in data:
        _write_row(row)

    # Write hline after body.
    if hline_after:
        output.write("\\hline\n")

    return None if isfile else output.getvalue()
