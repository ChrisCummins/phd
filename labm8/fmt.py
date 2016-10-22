# Copyright (C) 2015, 2016 Chris Cummins.
#
# This file is part of labm8.
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
#
"""
String formatting utils.
"""
from __future__ import print_function

import labm8 as lab


class Error(Exception):
    """
    Module-level error.
    """
    pass


def table(rows, columns=None, output=None, data_args={}, **kwargs):
    """
    Return a formatted string of "list of list" table data.

    See: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html

    Examples:

        >>> fmt.print([("foo", 1), ("bar", 2)])
             0  1
        0  foo  1
        1  bar  2

        >>> fmt.print([("foo", 1), ("bar", 2)], columns=("type", "value"))
          type  value
        0  foo      1
        1  bar      2

    Arguments:

        rows (list of list): Data to format, one row per element,
          multiple columns per row.
        columns (list of str, optional): Column names.
        output (str, optional): Path to output file.
        data_args (dict, optional): Any additional kwargs to pass to
          pandas.DataFrame constructor.
        **kwargs: Any additional arguments to pass to
          pandas.DataFrame.to_string().

    Returns:

        str: Formatted data as table.

    Raises:

        Error: If number of columns (if provided) does not equal
          number of columns in rows; or if number of columns is not
          consistent across all rows.
    """
    import pandas

    # Number of columns.
    num_columns = len(rows[0])

    # Check that each row is the same length.
    for i,row in enumerate(rows[1:]):
        if len(row) != num_columns:
            raise Error("Number of columns in row {i_row} ({c_row}) "
                        "does not match number of columns in row 0 ({z_row})"
                        .format(i_row=i, c_row=len(row), z_row=num_columns))

    if columns is None:
        # Default parameters.
        if "header" not in kwargs:
            kwargs["header"] = False
    elif len(columns) != num_columns:
        # Check that number of columns matches number of columns in
        # rows.
        raise Error("Number of columns in header ({c_header}) does not "
                    "match the number of columns in the data ({c_rows})"
                    .format(c_header=len(columns), c_rows=num_columns))

    # Default arguments.
    if "index" not in kwargs:
        kwargs["index"] = False

    data_args["columns"] = columns

    string = pandas.DataFrame(list(rows), **data_args).to_string(**kwargs)
    if output is None:
        return string
    else:
        print(string, file=open(output, "w"))
        io.info("Wrote", output)
