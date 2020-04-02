# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run script for machine learning models.

This defines the schedules for running training / validation / testing loops
of a machine learning model.
"""
import copy
import sys
import time
import warnings
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import pyfiglet
from sklearn.exceptions import UndefinedMetricWarning

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from labm8.py import app
from labm8.py import ppar
from programl.ml.batch.rolling_results import RollingResults
from programl.ml.epoch.epoch import EpochResults
from programl.ml.epoch.epoch import EpochType


FLAGS = app.FLAGS


def TrainModel(model, data_loader, batch_builder, epoch_count):
  """Run the model with the requested flags actions.

  Args:
    model_class: The model to run.

  Returns:
    A DataFrame of k-fold results, or a single series of results.
  """
  for epoch in range(epoch_count):
    rolling_results = RollingResults()
    graphs = ppar.ThreadedIterator(
      data_loader.LoadGraphs(EpochType.TRAIN), max_queue_size=5
    )
    batches = ppar.ThreadedIterator(
      batch_builder.BuildBatches(graphs), max_queue_size=5
    )
    for batch_data in batches:
      batch_results = model.RunBatches(batch_data, EpochType.TRAIN)
      rolling_results.Update(batch_data, batch_results, weight=None)
    epoch_results = EpochResults.FromRollingResults(rolling_results)
    print(epoch_results)
