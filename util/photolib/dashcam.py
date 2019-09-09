"""Library code for working with dash-cam files."""
import re
import datetime

from labm8 import app

FLAGS = app.FLAGS


def ParseDatetimeFromFilenameOrDire(name: str):
  """Decode a YYMMDD_HHMMSS_SEQ.MOV format file name produced by the dash cam
  into a python datetime object."""
  datestring = re.sub(r"_\d\d\d\.MOV$", "", name)
  if datestring == name:
    app.FatalWithoutStackTrace("Failed to parse datetime: `%s`", name)
  return datetime.datetime.strptime(datestring, "%y%m%d_%H%M%S")


def DatetimeToFilename(date: datetime.datetime) -> str:
  """Encode a datetime to a YYMMDD_HHMMSS_SEQ.MOV format file name."""
  return date.strftime("%y%m%d_%H%M%S_000.MOV")
