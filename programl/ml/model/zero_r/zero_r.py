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
"""A Zero-R baseline classifier."""
import time
from typing import Any
from typing import Iterable

import numpy as np

import programl.ml.batch.batch_results
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from labm8.py import app
from labm8.py import progress
from programl.ml.batch import batch_data as batchs
from programl.ml.epoch import epoch
from programl.ml.model import classifier_base
from programl.ml.model import run


FLAGS = app.FLAGS

app.DEFINE_integer(
  "zero_r_batch_size",
  100000,
  "The number of graphs to process in a single batch.",
)


class ZeroR(classifier_base.ClassifierBase):
  """A Zero-R classifier that supports node-level or graph-level labels.

  Zero-R classifiers predict the mode value from the training set. It is used
  as a baseline for comparing the performance of other classifiers.
  """

  def __init__(self, *args, **kwargs):
    super(ZeroR, self).__init__(*args, **kwargs)
    # The table used to count training labels.
    self.class_counts = np.zeros(self.y_dimensionality, dtype=np.int64)

  @property
  def y(self) -> np.array:
    """Return the prediction array."""
    a = np.zeros(self.y_dimensionality, dtype=np.int64)
    a[np.argmax(self.class_counts)] = 1
    return a

  def MakeBatch(
    self,
    epoch_type: epoch.EpochType,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batchs.BatchData:
    del epoch_type  # Unused.
    del ctx  # Unused.

    batch_size = 0
    graph_ids = []
    targets = []

    # Limit batch size to 10 million elements.
    while batch_size < FLAGS.zero_r_batch_size:
      # Read the next graph.
      try:
        graph = next(graphs)
      except StopIteration:
        # We have run out of graphs.
        if len(graph_ids) == 0:
          return batchs.EndOfBatches()
        break

      # Add the graph data to the batch.
      graph_ids.append(graph.id)
      if self.graph_db.node_y_dimensionality:
        batch_size += graph.tuple.node_y.size
        targets.append(graph.tuple.node_y)
      else:
        batch_size += graph.tuple.graph_y.size
        targets.append(graph.tuple.graph_y)

    return batchs.BatchData(
      graph_ids=graph_ids, data=np.vstack(targets) if targets else None
    )

  def RunBatch(
    self,
    epoch_type: epoch.EpochType,
    batch: batchs.BatchData,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> programl.ml.batch.batch_results.BatchResults:
    """Run a batch.

    Args:
      epoch_type: The type of epoch.
      batch: The batch data.
      ctx: A logging context.

    Returns:
      A Results instance.
    """
    del ctx

    time.sleep(0.05)
    targets = batch.model_data

    # "Training" step updates the class frequency counts.
    if epoch_type == epoch.EpochType.TRAIN:
      bincount = np.bincount(
        np.argmax(targets, axis=1), minlength=self.y_dimensionality
      )
      self.class_counts += bincount

      assert targets.shape[1] == self.y.shape[0]

    # The 1-hot predicted value.
    predictions = np.tile(self.y, targets.shape[0]).reshape(targets.shape)

    return programl.ml.batch.batch_results.BatchResults.Create(
      targets=targets, predictions=predictions
    )

  def GetModelData(self) -> Any:
    return self.class_counts

  def LoadModelData(self, data_to_load: Any) -> None:
    self.class_counts = data_to_load


def main():
  """Main entry point."""
  run.Run(ZeroR)


if __name__ == "__main__":
  app.Run(main)
