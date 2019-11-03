"""A Zero-R baseline classifier."""
import typing

import numpy as np
from labm8 import app

from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database

FLAGS = app.FLAGS


class ZeroRNodeClassifier(classifier_base.ClassifierBase):
  """A Zero-R classifier that supports node-level or graph-level labels.

  A Zero-R classifier always predicts the mode value from the training set. It
  is used as a baseline for comparing the performance of other classifiers.
  """

  def __init__(self, *args):
    super(ZeroRNodeClassifier, self).__init__(*args)
    self.class_counts = np.zeros(self.labels_dimensionality, dtype=np.int32)

  def MakeMinibatchIterator(
      self, epoch_type: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLog, np.array]]:
    for batch_tuple in self.batcher.MakeGaphBatchIterator(epoch_type):
      if batch_tuple.has_node_y:
        targets = batch_tuple.node_y
      elif batch_tuple.has_graph_y:
        targets = batch_tuple.graph_y
      else:
        raise ValueError("Could not determine label type")
      yield batch_tuple.log, targets

  def RunMinibatch(self, log: log_database.BatchLog, targets: np.array
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
  classifier_base.Run(ZeroRNodeClassifier)


if __name__ == '__main__':
  app.Run(main)
