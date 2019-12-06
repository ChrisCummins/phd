"""A Zero-R baseline classifier."""
import time
from typing import Any
from typing import Iterable

import numpy as np

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batchs
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import run
from labm8.py import app
from labm8.py import progress

FLAGS = app.FLAGS


class ZeroR(classifier_base.ClassifierBase):
  """A Zero-R classifier that supports node-level or graph-level labels.

  Zero-R classifiers predict the mode value from the training set. It is used
  as a baseline for comparing the performance of other classifiers.
  """

  def __init__(self, *args, **kwargs):
    super(ZeroR, self).__init__(*args, **kwargs)
    # The table used to count training labels.
    self.class_counts = np.zeros(self.y_dimensionality, dtype=np.int32)

  @property
  def y(self) -> np.array:
    """Return the prediction array."""
    a = np.zeros(self.y_dimensionality, dtype=np.int32)
    a[np.argmax(self.class_counts)] = 1
    return a

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batchs.Data:
    del ctx

    batch_size = 0
    graph_ids = []
    targets = []

    # Limit batch size to 10 million elements.
    while batch_size < 10000000:
      # Read the next graph.
      try:
        graph = next(graphs)
      except StopIteration:
        # We have run out of graphs.
        break

      # Add the graph data to the batch.
      graph_ids.append(graph.id)
      if self.node_y_dimensionality:
        batch_size += graph.tuple.node_y.size
        targets.append(graph.tuple.node_y)
      else:
        batch_size += graph.tuple.graph_y.size
        targets = graph.tuple.graph_y

    # Flatten per-graph node y to a single array.
    if self.node_y_dimensionality and targets:
      targets = np.vstack(targets)

    return batchs.Data(graph_ids=graph_ids, data=targets)

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batchs.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batchs.Results:
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
    targets = batch.data

    # "Training" step updates the class frequency counts.
    if epoch_type == epoch.Type.TRAIN:
      bincount = np.bincount(np.argmax(targets, axis=1))
      self.class_counts += bincount

    assert targets.shape[1] == self.y.shape[0]

    # The 1-hot predicted value.
    predictions = np.tile(self.y, targets.shape[0]).reshape(targets.shape)

    return batchs.Results.Create(targets=targets, predictions=predictions)

  def ModelDataToSave(self) -> Any:
    return self.class_counts

  def LoadModelData(self, data_to_load: Any) -> None:
    self.class_counts = data_to_load


def main():
  """Main entry point."""
  run.Run(ZeroR)


if __name__ == "__main__":
  app.Run(main)
