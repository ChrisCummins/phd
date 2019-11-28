# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Utility code for data importers."""
import collections
import multiprocessing
import pathlib
import typing

from datasets.me_db import me_pb2
from labm8.py import app
from labm8.py import labtypes

FLAGS = app.FLAGS

# An inbox importer is a function that takes a path to a directory (the inbox)
# and a Queue. When called, the function places a SeriesCollection proto on the
# queue.
InboxImporter = typing.Callable[
  [pathlib.Path, multiprocessing.Queue], me_pb2.SeriesCollection
]


class ImporterError(EnvironmentError):
  """Error raised if an importer fails."""

  def __init__(
    self,
    name: str,
    source: typing.Union[str, pathlib.Path],
    error_message: typing.Optional[str],
  ):
    self._name = name
    self._source = source
    self._error_message = error_message

  @property
  def name(self) -> str:
    return self._name

  @property
  def source(self) -> str:
    return str(self._source)

  @property
  def error_message(self) -> str:
    return self._error_message

  def __repr__(self) -> str:
    if self.error_message:
      return (
        f"{self.name} importer failed for source `{self.source}` with "
        f"error: {self.error_message}"
      )
    else:
      return f"{self.name} importer failed for source `{self.source}`"

  def __str__(self) -> str:
    return repr(self)


def ConcatenateSeries(
  series: typing.Iterator[me_pb2.SeriesCollection],
) -> me_pb2.SeriesCollection:
  if len({s.name for s in series}) != 1:
    raise ValueError("Multiple names")
  if len({s.family for s in series}) != 1:
    raise ValueError("Multiple families")
  if len({s.unit for s in series}) != 1:
    raise ValueError("Multiple units")

  concat_series = me_pb2.Series()
  concat_series.CopyFrom(series[0])
  for s in series[1:]:
    series[0].measurement.extend(s.measurement)
  return concat_series


def MergeSeriesCollections(
  series: typing.Iterator[me_pb2.SeriesCollection],
) -> me_pb2.SeriesCollection:
  """Merge the given series collections into a single SeriesCollection.

  Args:
    series: The SeriesCollection messages to merge.

  Returns:
    A SeriesCollection message.

  Raises:
    ValueError: If there are Series with duplicate names.
  """
  series = list(labtypes.flatten(list(f.series) for f in series))

  # Create a map from series name to a list of series protos.
  names_to_series = collections.defaultdict(list)
  [names_to_series[s.name].append(s) for s in series]

  # Concatenate each list of series with the same name.
  concatenated_series = [ConcatenateSeries(s) for s in names_to_series.values()]
  return me_pb2.SeriesCollection(
    series=sorted(concatenated_series, key=lambda s: s.name)
  )
