"""This file defines telemetry data gathers."""
import pathlib
import re
import typing

from absl import flags
from phd.lib.labm8 import jsonutil
from phd.lib.labm8 import labdate
from phd.lib.labm8 import pbutil

from deeplearning.clgen.proto import telemetry_pb2


FLAGS = flags.FLAGS


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
