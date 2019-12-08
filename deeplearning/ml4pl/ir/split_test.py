"""Unit tests for //deeplearning/ml4pl/bytecode:split."""
import numpy as np

from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.ir import splitters
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(
  scope="session",
  params=(
    splitters.TrainValTestSplitter,
    splitters.Poj104TrainValTestSplitter,
    splitters.Pact17KFoldSplitter,
  ),
)
def splitter(request) -> splitters.Splitter:
  """Test fixture which yields a splitter."""
  return request.param()


@test.Fixture(
  scope="session",
  params=(splitters.TrainValTestSplitter, splitters.Poj104TrainValTestSplitter),
)
def train_val_test_splitter(request) -> splitters.Splitter:
  """Test fixture which yields a train/val/test splitter."""
  return request.param()


def test_train_val_test_split_counts(
  populated_db: ir_database.Database,
  train_val_test_splitter: splitters.Splitter,
):
  """Test that train/val/test splitter produces 3 splits."""
  splits = train_val_test_splitter.Split(populated_db)
  assert len(splits) == 3


@test.Parametrize("k", (None, 5, 10))
def test_k_fold_split_counts(populated_db: ir_database.Database, k: int):
  """Test that K-fold splitter produces k splits."""
  splitter = splitters.Pact17KFoldSplitter(k=k)
  splits = splitter.Split(populated_db)
  assert len(splits) == (k or 10)


def test_TrainValTestSplitter_includes_all_irs(
  populated_db: ir_database.Database,
):
  """Test that all IRs are included in splits."""
  splitter = splitters.TrainValTestSplitter()
  splits = splitter.Split(populated_db)
  assert sum(len(split) for split in splits) == populated_db.ir_count


@test.Parametrize("n", (3, 5))
def test_TrainValTestSplitter_ratios(
  populated_db: ir_database.Database, n: int
):
  """Test the ratio of train/val/test splits."""
  splitter = splitters.TrainValTestSplitter(train_val_test_ratio=(n, 1, 1))
  splits = splitter.Split(populated_db)
  assert len(splits[0]) == len(splits[1]) * n
  assert len(splits[0]) == len(splits[2]) * n


def test_unique_irs(
  populated_db: ir_database.Database, splitter: splitters.Splitter
):
  """Test that all IR IDs are unique."""
  splits = splitter.Split(populated_db)
  all_ids = np.concatenate(splits)
  assert len(set(all_ids)) == len(all_ids)


def test_CreateFromFlags():
  splitters.Splitter.CreateFromFlags()


if __name__ == "__main__":
  test.Main()
