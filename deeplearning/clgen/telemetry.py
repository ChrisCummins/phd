"""This file defines telemetry data gathers."""
import pathlib

from absl import flags

from deeplearning.clgen.proto import internal_pb2
from lib.labm8 import jsonutil
from lib.labm8 import labdate
from lib.labm8 import pbutil


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

  def EpochBeginCallback(self, epoch: int, logs: jsonutil.JSON) -> None:
    """A Keras "on_epoch_end" callback."""
    del epoch
    del logs
    self.last_epoch_begin_timestamp = labdate.MillisecondsTimestamp()

  def EpochEndCallback(self, epoch: int, logs: jsonutil.JSON) -> None:
    """A Keras "on_epoch_end" callback."""
    epoch += 1
    now = labdate.MillisecondsTimestamp()
    epoch_time_ms = now - self.last_epoch_begin_timestamp
    log = internal_pb2.ModelEpochTelemetry(
        timestamp_utc_epoch_ms=now,
        epoch_num=epoch,
        epoch_wall_time_ms=epoch_time_ms,
        loss=logs['loss'],
    )
    pbutil.ToFile(log, self.logdir / f'epoch_{epoch}_end.pbtxt')

  def KerasCallback(self, keras):
    """Returns the keras callback to passed to a model's fit() function."""
    return keras.callbacks.LambdaCallback(
        on_epoch_begin=self.EpochBeginCallback,
        on_epoch_end=self.EpochEndCallback)
