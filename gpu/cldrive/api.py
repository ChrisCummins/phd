"""Public API for cldrive."""
import csv
import io

import pandas as pd
from absl import flags

from gpu.cldrive.legacy import env as _env
from gpu.cldrive.proto import cldrive_pb2
from gpu.oclgrind import oclgrind
from labm8 import bazelutil
from labm8 import pbutil

FLAGS = flags.FLAGS

_NATIVE_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_driver')


def _GetCommand(instances):
  if (len(instances.instance) and
      instances.instance[0].device.name == _env.OclgrindOpenCLEnvironment().name
     ):
    return [str(oclgrind.OCLGRIND_PATH), str(_NATIVE_DRIVER)]
  else:
    return [str(_NATIVE_DRIVER)]


def Drive(instances: cldrive_pb2.CldriveInstances,
          timeout_seconds: int = 360) -> cldrive_pb2.CldriveInstances:
  pbutil.RunProcessMessageInPlace(
      _GetCommand(instances), instances, timeout_seconds=timeout_seconds)
  return instances


def DriveToDataFrame(instances: cldrive_pb2.CldriveInstances,
                     timeout_seconds: int = 360) -> pd.DataFrame:
  stdout = pbutil.RunProcessMessage(
      _GetCommand(instances), instances, timeout_seconds=timeout_seconds)
  return pd.read_csv(
      io.StringIO(stdout.decode('utf-8')), sep=',', quoting=csv.QUOTE_NONE)
