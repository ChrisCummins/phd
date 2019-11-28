# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""A command-line interface for printing results."""
import random

import deeplearning.deepsmith.result
from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS


def _SelectRandomRow(query: db.query_t):
  """Select a random row from the query results.

  Args:
    query: The query.

  Returns:
    A randomly selected row from the query results.
  """
  number_of_rows = int(query.count())
  return query.offset(int(number_of_rows * random.random())).first()


def PrintRandomResult(session: db.session_t) -> None:
  """Pretty print a random result.

  Args:
    session: A database session.
  """
  query = session.query(deeplearning.deepsmith.result.Result)
  result = _SelectRandomRow(query)
  print(pbutil.PrettyPrintJson(result.ToProto()))


def main(argv):
  del argv
  ds = datastore.DataStore.FromFlags()
  with ds.Session(commit=True) as session:
    PrintRandomResult(session)


if __name__ == "__main__":
  app.RunWithArgs(main)
