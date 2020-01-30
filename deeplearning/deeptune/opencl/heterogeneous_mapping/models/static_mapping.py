# Copyright (c) 2017-2020 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.
"""Static mapping model."""
import pickle

import numpy as np
import pandas as pd

from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from labm8.py import app

FLAGS = app.FLAGS


class StaticMapping(base.HeterogeneousMappingModel):
  __name__ = "Static mapping"
  __basename__ = "static"

  def __init__(self):
    self.model = None

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    return self

  def save(self, outpath):
    with open(outpath, "wb") as outfile:
      pickle.dump(self.model, outfile)

  def restore(self, inpath):
    with open(inpath, "rb") as infile:
      self.model = pickle.load(infile)

  def train(self, df: pd.DataFrame, platform_name: str, verbose: bool = False):
    del verbose

    if np.mean(df["y"]) >= 0.5:
      self.model = "GPU"
    else:
      self.model = "CPU"

  def predict(
    self, df: pd.DataFrame, platform_name: str, verbose: bool = False
  ):
    del platform_name
    del verbose
    if self.model == "GPU":
      return np.ones(len(df), dtype=np.int32)
    elif self.model == "CPU":
      return np.zeros(len(df), dtype=np.int32)
    else:
      raise LookupError
