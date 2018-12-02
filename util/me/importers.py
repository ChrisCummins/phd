"""Utility code for data importers."""
import collections
import pathlib
import sys
import typing
from absl import flags

from lib.labm8 import labtypes
from util.me import me_pb2


FLAGS = flags.FLAGS

# An importer task is a function that takes no arguments, and returns a
# generator
ImporterTask = typing.Callable[[], typing.Iterator[me_pb2.SeriesCollection]]

# A iterator of ImporterTasks.
ImporterTasks = typing.Iterator[ImporterTask]


class ImporterError(EnvironmentError):
  """Error raised if an importer fails."""

  def __init__(self, name: str, source: typing.Union[str, pathlib.Path],
               error_message: typing.Optional[str]):
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
      return (f'{self.name} importer failed for source `{self.source}` with '
              f'error: {self.error_message}')
    else:
      return f'{self.name} importer failed for source `{self.source}`'

  def __str__(self) -> str:
    return repr(self)


def ConcatenateSeries(series: typing.Iterator[
  me_pb2.SeriesCollection]) -> me_pb2.SeriesCollection:
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
    series: typing.Iterator[
      me_pb2.SeriesCollection]) -> me_pb2.SeriesCollection:
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
  for s in series:
    names_to_series[s.name].append(s)

  # Concatenate each list of series with the same name.
  concatenated_series = [
    ConcatenateSeries(s) for s in names_to_series.values()
  ]
  return me_pb2.SeriesCollection(
      series=sorted(concatenated_series, key=lambda s: s.name))


def RunTasksAndExit(tasks: typing.Iterator[ImporterTask]) -> None:
  series_collections = []
  for task in tasks:
    series_collections += list(task())
  MergeAndPrintSeriesAndExit(series_collections)


def MergeAndPrintSeriesAndExit(
    series: typing.Iterator[me_pb2.SeriesCollection]) -> None:
  """Merge the given series to a SeriesCollection and exit.

  This is convenience function wrap MergeSeriesCollections() with printing to
  stdout / stderr.
  """
  try:
    print(MergeSeriesCollections(series))
    sys.exit(0)
  except ImporterError as e:
    print(e, file=sys.stderr)
    sys.exit(1)
