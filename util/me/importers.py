"""Utility code for data importers."""
import pathlib
import sys
import typing
from absl import flags
from absl import logging

from lib.labm8 import labtypes
from util.me import me_pb2


FLAGS = flags.FLAGS


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


def MergeSeriesFromPaths(
    series: typing.Iterator[me_pb2.SeriesFromPath]) -> me_pb2.SeriesCollection:
  """Merge the given series into a SeriesCollection.

  Args:
    series: An iterator of SeriesFromPath messages.

  Returns:
    A SeriesCollection message.
  """
  series = list(labtypes.flatten(list(f.series) for f in series))
  s_names = set(s.name for s in series)

  if len(s_names) != len(series):
    for s in series:
      logging.error('  Merging series: `%s`', s.name)
    raise ValueError("Duplicate series names")

  return me_pb2.SeriesCollection(series=sorted(series, key=lambda s: s.name))


def MergeSeriesAndExit(
    series: typing.Iterator[me_pb2.SeriesFromPath]) -> None:
  """Merge the given series to a SeriesCollection and exit."""
  try:
    print(MergeSeriesFromPaths(series))
    sys.exit(0)
  except ImporterError as e:
    print(e, file=sys.stderr)
    sys.exit(1)
