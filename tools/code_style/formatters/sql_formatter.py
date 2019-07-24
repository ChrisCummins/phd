"""Module to format SQL code."""

import sqlparse

from labm8 import app

FLAGS = app.FLAGS


def FormatSql(text: str) -> str:
  """Format a string of SQL queries.
  
  This function does not validate the SQL, and will try its best to format 
  invalid inputs.
  
  Args:
    text: The SQL text to format.
  
  Returns:
    The formatted SQL string.
  """
  return sqlparse.format(text, reindent=True, keyword_case='upper')
