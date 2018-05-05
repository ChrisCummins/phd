"""String formatting utils.
"""


class Error(Exception):
  """
  Module-level error.
  """
  pass


def table(rows, columns=None, output=None, data_args={}, **kwargs) -> str:
  """
  Return a formatted string of "list of list" table data.

  See: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html

  Examples:
    >>> table([("foo", 1), ("bar", 2)])
         0  1
    0  foo  1
    1  bar  2

    >>> table([("foo", 1), ("bar", 2)], columns=("type", "value"))
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
  for i, row in enumerate(rows[1:]):
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
