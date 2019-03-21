# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""Public API for cldrive."""
import csv
import io

import numpy as np
import pandas as pd

from gpu.cldrive.legacy import env as _env
from gpu.cldrive.proto import cldrive_pb2
from gpu.oclgrind import oclgrind
from labm8 import app
from labm8 import bazelutil
from labm8 import pbutil

FLAGS = app.FLAGS

_NATIVE_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_driver')
_NATIVE_CSV_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_csv_driver')


class CldriveCrash(OSError):
  """Exception raised if native cldrive binary fails."""
  pass


def _GetCommand(driver, instances):
  if (len(instances.instance) and
      instances.instance[0].device.name == _env.OclgrindOpenCLEnvironment().name
     ):
    return [str(oclgrind.OCLGRIND_PATH), str(driver)]
  else:
    return [str(driver)]


def Drive(instances: cldrive_pb2.CldriveInstances,
          timeout_seconds: int = 300) -> cldrive_pb2.CldriveInstances:
  """Run cldrive with the given instances proto."""
  pbutil.RunProcessMessageInPlace(
      _GetCommand(_NATIVE_DRIVER, instances),
      instances,
      timeout_seconds=timeout_seconds)
  return instances


def DriveToDataFrame(instances: cldrive_pb2.CldriveInstances,
                     timeout_seconds: int = 300) -> pd.DataFrame:
  """Run cldrive with the given instances and read results to dataframe."""
  try:
    stdout = pbutil.RunProcessMessage(
        _GetCommand(_NATIVE_CSV_DRIVER, instances),
        instances,
        timeout_seconds=timeout_seconds).decode('utf-8')
  except subprocess.CalledProcessError as e:
    raise CldriveCrash(e)
  except UnicodeDecodeError:
    raise CldriveCrash("Failed to decode output")
  df = pd.read_csv(
      io.StringIO(stdout),
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
  df['kernel'].fillna('', inplace=True)

  # These columns may never contain null values.
  assert not pd.isna(df['instance']).any()
  assert not pd.isna(df['device']).any()
  assert not pd.isna(df['outcome']).any()

  return df
