"""Unit tests for //labm8/logutil.py."""
import datetime
import pathlib
import tempfile

import pytest
from absl import flags
from absl import logging

from labm8 import labdate
from labm8 import logutil
from labm8 import test
from labm8.proto import logging_pb2


FLAGS = flags.FLAGS


def test_ABSL_LOGGING_PREFIX_RE_match():
  """Test that absl logging regex matches a log line."""
  m = logutil.ABSL_LOGGING_LINE_RE.match(
      'I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!')
  assert m
  assert m.group('lvl') == 'I'
  assert m.group('timestamp') == '0527 23:14:18.903151'
  assert m.group('thread_id') == '140735784891328'
  assert m.group('filename') == 'log_to_file.py'
  assert m.group('lineno') == '31'
  assert m.group('contents') == 'Hello, info!'


def test_ABSL_LOGGING_PREFIX_RE_not_match():
  """Test that absl logging regex doesn't match a line."""
  m = logutil.ABSL_LOGGING_LINE_RE.match('Hello world!')
  assert not m


# DatetimeFromAbslTimestamp() tests.

def test_DatetimeFromAbslTimestamp():
  dt = logutil.DatetimeFromAbslTimestamp('0527 23:14:18.903151')
  assert dt.year == datetime.datetime.utcnow().year
  assert dt.month == 5
  assert dt.day == 27
  assert dt.hour == 23
  assert dt.minute == 14
  assert dt.second == 18
  assert dt.microsecond == 903151


# ConvertAbslLogToProtos() tests.

def test_ConvertAbslLogToProtos_empty_input():
  assert not logutil.ConertAbslLogToProtos('')


def test_ConvertAbslLogToProtos_num_records():
  p = logutil.ConertAbslLogToProtos("""\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
""")
  assert 5 == len(p)


def test_ConvertAbslLogToProtos_levels():
  p = logutil.ConertAbslLogToProtos("""\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
""")
  assert p[0].level == logging_pb2.LogRecord.INFO
  assert p[1].level == logging_pb2.LogRecord.WARNING
  assert p[2].level == logging_pb2.LogRecord.INFO
  assert p[3].level == logging_pb2.LogRecord.ERROR
  assert p[4].level == logging_pb2.LogRecord.FATAL


def test_ConvertAbslLogToProtos_date_utc_epoch_ms():
  p = logutil.ConertAbslLogToProtos("""\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, fatal!
""")
  assert p[0].date_utc_epoch_ms == 1527462858903
  assert p[1].date_utc_epoch_ms == 1527462858903
  assert p[2].date_utc_epoch_ms == 1527462858903
  assert p[3].date_utc_epoch_ms == 1527462858903
  assert p[4].date_utc_epoch_ms == 1527462858903
  dt = labdate.DatetimeFromMillisecondsTimestamp(p[0].date_utc_epoch_ms)
  assert dt.year == datetime.datetime.utcnow().year
  assert dt.month == 5
  assert dt.day == 27
  assert dt.hour == 23
  assert dt.minute == 14
  assert dt.second == 18
  # Microsecond precision has been reduced to millisecond.
  assert dt.microsecond == 903000


def test_ConvertAbslLogToProtos_levels():
  p = logutil.ConertAbslLogToProtos("""\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
""")
  assert p[0].thread_id == 140735784891328
  assert p[1].thread_id == 140735784891328
  assert p[2].thread_id == 140735784891328
  assert p[3].thread_id == 1407
  assert p[4].thread_id == 1


def test_ConvertAbslLogToProtos_line_number():
  p = logutil.ConertAbslLogToProtos("""\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
""")
  assert p[0].line_number == 31
  assert p[1].line_number == 31
  assert p[2].line_number == 31
  assert p[3].line_number == 31
  assert p[4].line_number == 31


def test_ConvertAbslLogToProtos_message():
  p = logutil.ConertAbslLogToProtos("""\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Goodbye ...
multiline!
""")
  assert p[0].message == 'Hello, info!'
  assert p[1].message == 'Hello, warning!'
  assert p[2].message == 'Hello ...\n\nmultiline!'
  assert p[3].message == 'Hello, error!'
  assert p[4].message == 'Hello, fatal!'
  assert p[5].message == 'Goodbye ...\nmultiline!'


# StartTeeLogsToFile() and StopTeeLogsToFile() tests.

def test_TeeLogsToFile_dir_not_found():
  """Test that FileNotFoundError is raised if log_dir does not exist"""
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(FileNotFoundError) as e_info:
      logutil.StartTeeLogsToFile('test', pathlib.Path(d) / 'notadir')
    assert "Log directory not found: '{d}/notadir'"


def test_TeeLogsToFile(capsys):
  """Test that StartTeeLogs also logs to file, and StopTeeLogs prevents that."""
  with tempfile.TemporaryDirectory() as d:
    FLAGS.logtostderr = True
    logging.info('This is not going in a file')
    logutil.StartTeeLogsToFile('test', d)
    logging.info('Hello, file!')
    logutil.StopTeeLogsToFile()
    logging.info('This is not going in a file')
    # Test file contents.
    with open(pathlib.Path(d) / 'test.INFO') as f:
      lines = f.read().rstrip().split('\n')
      assert len(lines) == 1
      assert lines[0].endswith('Hello, file!')
    out, err = capsys.readouterr()
    assert not out
    # Test stderr contents.
    lines = err.rstrip().split('\n')
    assert len(lines) == 3
    assert lines[0].endswith('This is not going in a file')
    assert lines[1].endswith('Hello, file!')
    assert lines[2].endswith('This is not going in a file')


def test_TeeLogsToFile_contextmanager(capsys):
  """Test that contextmanager temporarily also logs to file."""
  with tempfile.TemporaryDirectory() as d:
    FLAGS.logtostderr = True
    logging.info('This is not going in a file')
    with logutil.TeeLogsToFile('test', d):
      logging.info('Hello, file!')
    logging.info('This is not going in a file')
    # Test file contents.
    with open(pathlib.Path(d) / 'test.INFO') as f:
      lines = f.read().rstrip().split('\n')
      assert len(lines) == 1
      assert lines[0].endswith('Hello, file!')
    out, err = capsys.readouterr()
    assert not out
    # Test stderr contents.
    lines = err.rstrip().split('\n')
    assert len(lines) == 3
    assert lines[0].endswith('This is not going in a file')
    assert lines[1].endswith('Hello, file!')
    assert lines[2].endswith('This is not going in a file')


if __name__ == '__main__':
  test.Main()
