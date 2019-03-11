# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/clgen/telemetry.py."""
import pathlib
import tempfile

from deeplearning.clgen import telemetry
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS

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
