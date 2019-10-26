"""A Zero-R baseline classifier."""

import numpy as np
import typing

from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from labm8 import app

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
    for batch in self.batcher.MakeGroupBatchIterator(epoch_type):
      if 'node_y' in batch:
        targets = batch['node_y']
      elif 'graph_y' in batch:
        targets = batch['graph_y']
      else:
        raise ValueError("Only node or graph labels are supported")
      yield batch['log'], targets

  def RunMinibatch(self, epoch_type: str, targets: np.array
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    # "Training" step updates the class frequence counts.
    if epoch_type == "train":
      y_true = np.argmax(targets, axis=1)
      freqs = np.bincount(y_true)
      for i, n in enumerate(freqs):
        self.class_counts[i] += n

    # The 1-hot predicted value.
    pred = np.zeros(self.labels_dimensionality, dtype=np.int32)
    pred[np.argmax(self.class_counts)] = 1

    return self.MinibatchResults(
        loss=0,
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
