"""Public API for cldrive."""
import csv
import io

import numpy as np
import pandas as pd
from absl import flags

from gpu.cldrive.legacy import env as _env
from gpu.cldrive.proto import cldrive_pb2
from gpu.oclgrind import oclgrind
from labm8 import bazelutil
from labm8 import pbutil

FLAGS = flags.FLAGS

_NATIVE_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_driver')
_NATIVE_CSV_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_csv_driver')


def _GetCommand(driver, instances):
  if (len(instances.instance) and
      instances.instance[0].device.name == _env.OclgrindOpenCLEnvironment().name
     ):
    return [str(oclgrind.OCLGRIND_PATH), str(driver)]
  else:
    return [str(driver)]


def Drive(instances: cldrive_pb2.CldriveInstances,
          timeout_seconds: int = 360) -> cldrive_pb2.CldriveInstances:
  """Run cldrive with the given instances proto."""
  pbutil.RunProcessMessageInPlace(
      _GetCommand(_NATIVE_DRIVER, instances),
      instances,
      timeout_seconds=timeout_seconds)
  return instances


def DriveToDataFrame(instances: cldrive_pb2.CldriveInstances,
                     timeout_seconds: int = 360) -> pd.DataFrame:
  """Run cldrive with the given instances and read results to dataframe."""
  stdout = pbutil.RunProcessMessage(
      _GetCommand(_NATIVE_CSV_DRIVER, instances),
      instances,
      timeout_seconds=timeout_seconds)
  df = pd.read_csv(
      io.StringIO(stdout.decode('utf-8')),
      sep=',',
      quoting=csv.QUOTE_NONE,
      dtype={
          # Specify the data types explicitly for consistency.
          # Support for NaN values in integer arrays is new in pandas. See:
          # https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.24.0.html#optional-integer-na-support
          'instance': np.int32,
          'device': str,
          'build_opts': str,
          'kernel': str,
          'work_item_local_mem_size': 'Int64',
          'work_item_private_mem_size': 'Int64',
          'global_size': 'Int32',
          'local_size': 'Int32',
          'outcome': str,
          'transferred_bytes': 'Int64',
          'runtime_ms': np.float64,
      })

  # Pandas will interpret empty string as NaN. Replace NaN with empty strings.
  df['build_opts'].fillna('', inplace=True)

  # These columns may never contain null values.
  assert not pd.isna(df['instance']).any()
  assert not pd.isna(df['device']).any()
  assert not pd.isna(df['outcome']).any()

  return df
