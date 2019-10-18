"""This module defines log writer objects."""

import datetime
import pathlib
import typing

from labm8 import app
from labm8 import jsonutil

FLAGS = app.FLAGS


class FormattedJsonLogWriter(object):

  def __init__(self,
               outpath: pathlib.Path,
               time_format: str = '%Y.%m.%dT%H:%M:%S.%f'):
    """Constructor.

    Args:
      outpath: The directory to write to.
      time_format: The strftime() format string to produce file names. Be sure
        to provide sufficient time granularity to match the expected frequency
        of log writes - this method will overwrite existing files with
        timestamp clashes.
    """
    self.outpath = outpath
    self.outpath.mkdir(exist_ok=True, parents=True)
    self.time_format = time_format

  def Log(self, log: typing.Dict[str, typing.Any]) -> pathlib.Path:
    """Log the given data to file and return the path of the log file."""
    now = datetime.datetime.now()
    log_file = self.outpath / f'{now.strftime(self.time_format)}.log.json'
    jsonutil.write_file(log_file, log)
    return log_file

  def Logs(self) -> typing.Iterable[typing.Dict[str, typing.Any]]:
    """Return an iterator over the log files."""
    for path in sorted(self.outpath.iterdir()):
      if path.name.endswith('.log.json'):
        yield jsonutil.read_file(path)
