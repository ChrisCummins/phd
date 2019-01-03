"""Unit tests for //deeplearning/clgen/telemetry.py."""
import pathlib
import tempfile

from absl import flags

from deeplearning.clgen import telemetry
from labm8 import test


FLAGS = flags.FLAGS


# TrainingLogger tests.

def test_TrainingLogger_create_file():
  """Test that EpochEndCallback() produces a file."""
  with tempfile.TemporaryDirectory() as d:
    logger = telemetry.TrainingLogger(pathlib.Path(d))
    logger.KerasEpochBeginCallback(0, {})
    logger.KerasEpochEndCallback(0, {'loss': 1})
    # Note that one is added to the epoch number.
    assert (pathlib.Path(d) / 'epoch_001_telemetry.pbtxt').is_file()


def test_TrainingLogger_EpochTelemetry():
  """Test that EpochTelemetry() returns protos."""
  with tempfile.TemporaryDirectory() as d:
    logger = telemetry.TrainingLogger(pathlib.Path(d))
    assert not logger.EpochTelemetry()
    logger.KerasEpochBeginCallback(0, {})
    logger.KerasEpochEndCallback(0, {'loss': 1})
    assert len(logger.EpochTelemetry()) == 1
    logger.KerasEpochBeginCallback(1, {})
    logger.KerasEpochEndCallback(1, {'loss': 1})
    assert len(logger.EpochTelemetry()) == 2


if __name__ == '__main__':
  test.Main()
