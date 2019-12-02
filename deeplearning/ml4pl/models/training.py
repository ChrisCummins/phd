"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import enum

from labm8.py import app


FLAGS = app.FLAGS

app.DEFINE_integer("epoch_count", 300, "The number of epochs to train for.")
app.DEFINE_integer(
  "patience",
  300,
  "The number of epochs to train for without any improvement in validation "
  "accuracy before stopping.",
)
app.DEFINE_boolean(
  "test_on_improvement",
  True,
  "If true, test model accuracy on test data when the validation accuracy "
  "improves.",
)
app.DEFINE_integer(
  "max_train_per_epoch",
  None,
  "Use this flag to limit the maximum number of instances used in a single "
  "training epoch. For k-fold cross-validation, each of the k folds will "
  "train on a maximum of this many graphs.",
)
app.DEFINE_integer(
  "max_val_per_epoch",
  None,
  "Use this flag to limit the maximum number of instances used in a single "
  "validation epoch.",
)
app.DEFINE_list(
  "val_split",
  ["val"],
  "The names of the splits to be used for validating model performance. All "
  "splits except --val_split and --test_split will be used for training.",
)
app.DEFINE_string(
  "test_split",
  ["test"],
  "The name of the hold-out splits to be used for testing. All splits "
  "except --val_split and --test_split will be used for training.",
)


class EpochType(enum.Enum):
  TRAIN = 0
  VAL = 1
  TEST = 2
