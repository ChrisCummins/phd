"""Base class for loggers that write to filesystem."""

import datetime
import pathlib
import typing

from labm8 import app
from labm8 import fs


FLAGS = app.FLAGS


class FilesystemLoggerBase(object):

  def __init__(self,
               outpath: pathlib.Path,
               log_name_format: str = '%Y-%m-%dT%H-%M-%S.%f',
               use_system_local_time: bool = False):
    """Constructor.

    Args:
      outpath: The directory to write to.
      log_name_format: The format string used to generate file names, passed to
        strftime(). Be sure to provide sufficient time granularity to match the
        expected frequency of log writes - loggers will overwrite existing
        files with timestamp clashes.
      use_system_local_time: If True, use local system time to generate
        timestamps. Default is UTC.
    """
    self.outpath = outpath
    self.outpath.mkdir(exist_ok=True, parents=True)
    self.log_name_format = log_name_format
    if use_system_local_time:
      self.now = datetime.datetime.now
    else:
      self.now = datetime.datetime.utcnow

  def WriteLog(self, data_to_log: typing.Any, path: pathlib.Path) -> None:
    """Write the given log data to file. Implemented by subclasses."""
    raise NotImplementedError("abstract class")

  def ReadLog(self, path: pathlib.Path) -> typing.Any:
    """Read the given log data. Implemented by subclasses."""
    raise NotImplementedError("abstract class")

  def Log(self, data_to_log: typing.Any) -> pathlib.Path:
    """Log the given data to file and return the path of the log file."""
    now = self.now()
    log_file = self.outpath / f'{now.strftime(self.log_name_format)}'
    self.WriteLog(data_to_log, log_file)
    return log_file

  def Logs(self) -> typing.Any:
    """Return an iterator over the log files."""
    for path in sorted(self.outpath.iterdir()):
      if path.name not in fs.COMMONLY_IGNORED_FILE_NAMES:
        yield self.ReadLog(path)
