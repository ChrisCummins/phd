"""A Zero-R baseline classifier."""
import typing

import numpy as np

from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from labm8 import app

FLAGS = app.FLAGS


class ZeroRClassifier(classifier_base.ClassifierBase):
  """A Zero-R classifier that supports node-level or graph-level labels.

  A Zero-R classifier always predicts the mode value from the training set. It
  is used as a baseline for comparing the performance of other classifiers.
  """

  def __init__(self, *args):
    super(ZeroRClassifier, self).__init__(*args)
    # The table used to count training labels.
    self.class_counts = np.zeros(self.labels_dimensionality, dtype=np.int32)

  def MakeMinibatchIterator(
      self, epoch_type: str, groups: typing.List[str], print_context: typing.Any = None
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, np.array]]:
    options = graph_batcher.GraphBatchOptions(max_nodes=FLAGS.batch_size,
                                              groups=groups)
    max_instance_count = (
        FLAGS.max_train_per_epoch if epoch_type == 'train' else
        FLAGS.max_val_per_epoch if epoch_type == 'val' else None)
    for batch_tuple in self.batcher.MakeGraphBatchIterator(
        options, max_instance_count, print_context):
      if batch_tuple.has_node_y:
        targets = batch_tuple.node_y
      elif batch_tuple.has_graph_y:
        targets = batch_tuple.graph_y
      else:
        app.FatalWithoutStackTrace("Could not determine label type")
      yield batch_tuple.log, targets

  def RunMinibatch(self, log: log_database.BatchLogMeta, targets: np.array
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    log.loss = 0
    # "Training" step updates the class frequency counts.
    if log.type == "train":
      y_true = np.argmax(targets, axis=1)
      freqs = np.bincount(y_true)
      for i, n in enumerate(freqs):
        self.class_counts[i] += n

    # The 1-hot predicted value.
    pred = np.zeros(self.labels_dimensionality, dtype=np.int32)
    pred[np.argmax(self.class_counts)] = 1

    return self.MinibatchResults(
        y_true_1hot=targets,
        y_pred_1hot=np.tile(pred, len(targets)).reshape(
            len(targets), self.labels_dimensionality))

  def ModelDataToSave(self) -> typing.Any:
    return self.class_counts

  def LoadModelData(self, data_to_load: typing.Any) -> None:
    self.class_counts = data_to_load


def main():
  """Main entry point."""
  classifier_base.Run(ZeroRClassifier)


if __name__ == '__main__':
  app.Run(main)
