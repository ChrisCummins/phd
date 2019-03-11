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
"""This file defines telemetry data gathers."""
import pathlib
import re
import typing

from deeplearning.clgen.proto import telemetry_pb2
from labm8 import app
from labm8 import jsonutil
from labm8 import labdate
from labm8 import pbutil

FLAGS = app.FLAGS


class TrainingLogger(object):
  """A TrainingLogger produces telemetry data of a CLgen model as it is trained.

  Telemetry data is gathered after every epoch of training. It includes a
  timestamp, the model's loss, and the time spent training the epoch.

  See the Keras callback docs: https://keras.io/callbacks/#lambdacallback
  """

  def __init__(self, logdir: pathlib.Path):
    self.logdir = logdir
    self.last_epoch_begin_timestamp = None

  def EpochBeginCallback(self) -> None:
    self.last_epoch_begin_timestamp = labdate.MillisecondsTimestamp()

  def EpochEndCallback(self, epoch: int, loss: float):
    now = labdate.MillisecondsTimestamp()
    epoch_time_ms = now - self.last_epoch_begin_timestamp
    telemetry = telemetry_pb2.ModelEpochTelemetry(
        timestamp_utc_epoch_ms=now,
        epoch_num=epoch,
        epoch_wall_time_ms=epoch_time_ms,
        loss=loss,
    )
    pbutil.ToFile(telemetry, self.logdir / f'epoch_{epoch:03d}_telemetry.pbtxt')

  def KerasEpochBeginCallback(self, epoch: int, logs: jsonutil.JSON) -> None:
    """A Keras "on_epoch_end" callback."""
    del epoch
    del logs
    self.EpochBeginCallback()

  def KerasEpochEndCallback(self, epoch: int, logs: jsonutil.JSON) -> None:
    """A Keras "on_epoch_end" callback."""
    # Keras epoch numbers are zero indexed.
    self.EpochEndCallback(epoch + 1, logs['loss'])

  def KerasCallback(self, keras):
    """Returns the keras callback to passed to a model's fit() function."""
    return keras.callbacks.LambdaCallback(
        on_epoch_begin=self.KerasEpochBeginCallback,
        on_epoch_end=self.KerasEpochEndCallback)

  def EpochTelemetry(self) -> typing.List[telemetry_pb2.ModelEpochTelemetry]:
    """Return the epoch telemetry files."""
    return [
        pbutil.FromFile(self.logdir / p, telemetry_pb2.ModelEpochTelemetry())
        for p in sorted(self.logdir.iterdir())
        if re.match(r'epoch_\d\d+_telemetry\.pbtxt', str(p.name))
    ]
