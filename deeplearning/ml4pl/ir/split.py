"""Module for splitting IR datasets for training/validation/testing."""
import random
from typing import List
from typing import Tuple

import numpy as np
import sklearn.model_selection
import sqlalchemy as sql

from deeplearning.ml4pl.ir import ir_database
from labm8.py import app
from labm8.py import humanize

FLAGS = app.FLAGS


app.DEFINE_string(
  "split_type", "train_val_test", "The type of split to generate."
)
app.DEFINE_list(
  "train_val_test_ratio",
  [3, 1, 1],
  "The ratio of training to validation to test dataset sizes when "
  "--split_type=train_val_test.",
)


class Splitter(object):
  """Base class for implementing IR splitters."""

  def Split(self, db: ir_database.Database) -> List[np.array]:
    """Split the given database.

    Returns:
      A list of IR id arrays, one for each split.
    """
    raise NotImplementedError("abstract class")

  @classmethod
  def CreateFromFlags(cls) -> "Splitter":
    """Construct a splitter from flag values."""
    if FLAGS.split_type == "train_val_test":
      train_val_test_ratio = [float(x) for x in FLAGS.train_val_test_ratio]
      return TrainValTestSplitter(train_val_test_ratio)
    elif FLAGS.split_type == "poj104":
      return Poj104TrainValTestSplitter()
    elif FLAGS.split_type == "pact17_10_fold":
      return Pact17KFoldSplitter()
    else:
      raise app.UsageError("Unknown split type")


class Poj104TrainValTestSplitter(Splitter):
  """Split the POJ-104 dataset using the same train/val/test splits as in:

      Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
      Comprehension: A Learnable Representation of Code Semantics. In NeurIPS.
  """

  def Split(self, db: ir_database.Database) -> List[np.array]:
    """Get the bytecode IDs for the POJ-104 app classification experiment."""

    def GetBytecodeIds(session, filter_cb) -> np.array:
      """Return the IDs for the given filtered query."""
      return np.array(
        [
          row.id
          for row in (
            session.query(ir_database.IntermediateRepresentation.id).filter(
              ir_database.IntermediateRepresentation.compilation_succeeded
              == True,
              filter_cb(),
            )
          )
        ],
        dtype=np.int32,
      )

    with db.Session() as session:
      return [
        GetBytecodeIds(
          session,
          lambda: (
            ir_database.IntermediateRepresentation.source == "poj-104:train"
          ),
        ),
        GetBytecodeIds(
          session,
          lambda: (
            ir_database.IntermediateRepresentation.source == "poj-104:val"
          ),
        ),
        GetBytecodeIds(
          session,
          lambda: (
            ir_database.IntermediateRepresentation.source == "poj-104:test"
          ),
        ),
      ]


class TrainValTestSplitter(Poj104TrainValTestSplitter):
  """A generator train/val/test splits."""

  def __init__(
    self, train_val_test_ratio: Tuple[float, float, float] = (3, 1, 1)
  ):
    """Constructor.

    Args:
      train_val_test_ratio: A triplet of ratios for the training, validation,
        and test sets. E.g. with the triplet (3, 1, 1), the training set will be
        3/5 of the dataset, and the validation and test sets will each by 1/5 of
        the dataset.
    """
    if len(train_val_test_ratio) != 3:
      raise ValueError("len(train_val_test_ratio) != 3")

    # Normalize the ratios to sum to 1.
    self.ratios = np.array(list(train_val_test_ratio), dtype=np.float32)
    self.ratios /= sum(self.ratios)

  def Split(self, db: ir_database.Database) -> List[np.array]:
    """Split the database."""
    poj104 = super(TrainValTestSplitter, self).Split(db)

    # Get the IDs of non-POJ-104 IRs.
    with db.Session() as session:
      total_count = (
        session.query(sql.func.count(ir_database.IntermediateRepresentation.id))
        .filter(
          ir_database.IntermediateRepresentation.compilation_succeeded == True,
          ~ir_database.IntermediateRepresentation.source.like("poj-104:%"),
        )
        .scalar()
      )

      # Scale the train/val/test ratio to the total IR count.
      train_val_test_counts = np.floor(self.ratios * total_count).astype(
        np.int32
      )
      # Round up if there were missing values.
      while train_val_test_counts.sum() < total_count:
        train_val_test_counts[random.randint(0, 2)] += 1

      assert total_count == train_val_test_counts.sum()
      app.Log(
        1,
        "Splitting %s IRs into splits: %s train, %s val, %s test",
        humanize.Commas(total_count + sum(len(s) for s in poj104)),
        humanize.Commas(train_val_test_counts[0] + len(poj104[0])),
        humanize.Commas(train_val_test_counts[1] + len(poj104[1])),
        humanize.Commas(train_val_test_counts[2] + len(poj104[2])),
      )

      ir_ids = [
        row.id
        for row in session.query(ir_database.IntermediateRepresentation.id)
        .filter(
          ir_database.IntermediateRepresentation.compilation_succeeded == True,
          ~ir_database.IntermediateRepresentation.source.like("poj-104:%"),
        )
        .order_by(db.Random())
      ]

    return [
      np.concatenate((poj104[0], ir_ids[: train_val_test_counts[0]])),
      np.concatenate(
        (
          poj104[1],
          ir_ids[train_val_test_counts[0] : sum(train_val_test_counts[:2])],
        )
      ),
      np.concatenate((poj104[2], ir_ids[sum(train_val_test_counts[:2]) :])),
    ]


class Pact17KFoldSplitter(Splitter):
  """Split the OpenCL sources into 10-fold cross validation sets as in:

      Cummins, C., Petoumenos, P., Wang, Z., & Leather, H. (2017). End-to-end
      Deep Learning of Optimization Heuristics. In PACT. IEEE.
  """

  def __init__(self, k: int = 10):
    self.k = k

  def Split(self, db: ir_database.Database) -> List[np.array]:
    """Split the database."""
    with db.Session() as session:
      all_ids = np.array(
        [
          row.id
          for row in session.query(
            ir_database.IntermediateRepresentation.id
          ).filter(
            ir_database.IntermediateRepresentation.compilation_succeeded
            == True,
            ir_database.IntermediateRepresentation.source
            == "pact17_opencl_devmap",
          )
        ],
        dtype=np.int32,
      )

    kfold = sklearn.model_selection.KFold(self.k).split(all_ids)
    return [all_ids[test] for (train, test) in kfold]


def ApplySplitSizeLimit(splits: List[np.array]):
  """Limit the size of ID lists per group if --max_split_size > 0.

  Ags:
    groups: A mapping of group name to IDs.

  Returns:
    The groups dictionary, where each ID list has been limited to
    --max_split_size elements, if --max_split_size > 0. Else,
    the groups are returned unmodified.
  """
  if FLAGS.max_split_size:
    for i, split in enumerate(splits):
      if len(split) > FLAGS.max_split_size:
        app.Log(
          1,
          "Limiting the size of split %s to %s elements from %s",
          i,
          humanize.Commas(FLAGS.max_split_size),
          humanize.Commas(len(split)),
        )
        random.shuffle(split)
        splits[i] = split[: FLAGS.max_split_size]

  return splits
