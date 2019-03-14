# Copyright (c) 2017, 2018, 2019 Chris Cummins.
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
"""Grewe et. al model."""
import pickle

import pandas as pd
from sklearn import tree as sktree

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from labm8 import app

FLAGS = app.FLAGS


class Grewe(base.HeterogeneousMappingModel):
  """Grewe et al. predictive model for heterogeneous device mapping.

  The Grewe et al. predictive model uses decision trees and hand engineered
  features to predict optimal device mapping, described in publication:

    ﻿Grewe, D., Wang, Z., & O’Boyle, M. (2013). Portable Mapping of Data
    Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
    https://doi.org/10.1109/CGO.2013.6494993
  """
  __name__ = "Grewe et al."
  __basename__ = "grewe"

  def __init__(self):
    self.model = None

  def init(self, seed: int, atomizer: atomizers.AtomizerBase):
    self.model = sktree.DecisionTreeClassifier(
        random_state=seed,
        splitter="best",
        criterion="entropy",
        max_depth=5,
        min_samples_leaf=5)
    return self

  def save(self, outpath):
    with open(outpath, "wb") as outfile:
      pickle.dump(self.model, outfile)

  def restore(self, inpath):
    with open(inpath, "rb") as infile:
      self.model = pickle.load(infile)

  def train(self, df: pd.DataFrame, platform_name: str, verbose: bool = False):
    del verbose
    features = opencl_device_mapping_dataset.ComputeGreweFeaturesForGpu(
        platform_name, df).values
    self.model.fit(features, df["y"])

  def predict(self, df: pd.DataFrame, platform_name: str,
              verbose: bool = False):
    del verbose
    features = opencl_device_mapping_dataset.ComputeGreweFeaturesForGpu(
        platform_name, df).values
    return self.model.predict(features)
