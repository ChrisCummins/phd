"""A dataset of OpenCL Device Mappings."""
import pandas as pd
from absl import flags

from labm8 import bazelutil
from labm8 import decorators


FLAGS = flags.FLAGS

AMD_CSV = bazelutil.DataPath('phd/datasets/opencl/device_mapping/amd.csv')
NVIDIA_CSV = bazelutil.DataPath('phd/datasets/opencl/device_mapping/amd.csv')


class OpenClDeviceMappingsDataset(object):

  def __init__(self):
    self._amd_df = pd.read_csv(AMD_CSV)
    self._nvidia_df = pd.read_csv(NVIDIA_CSV)

    # Drop the redunant columns that shouldn't be there.
    self._amd_df.drop('Unnamed: 0', axis='columns', inplace=True)
    self._nvidia_df.drop('Unnamed: 0', axis='columns', inplace=True)

  @decorators.memoized_property
  def programs_df(self) -> pd.DataFrame:
    return self._amd_df
