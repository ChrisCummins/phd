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
"""Reset the "group" column on a database using train/validation/test split."""
import typing

import numpy as np
import sqlalchemy as sql
from sklearn import model_selection

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import graph_batcher
from labm8.py import app
from labm8.py import humanize
from labm8.py import prof


FLAGS = app.FLAGS

app.DEFINE_database(
  "graph_db",
  graph_database.Database,
  None,
  "URL of database to modify.",
  must_exist=True,
)
app.DEFINE_integer("k_fold", 0, "If > 0, use stratified k-fold split.")

###############################################################################
# Splitters copied from //deeplearning/ml4pl/bytecode:splitters
# This is a straight copy-and-paste, but adapted to work on graph databases
# rather than LLVM bytecode databases. This requires removing references to
# clang_returncode columns.


def GetPoj104BytecodeGroups(
  db: graph_database.Database,
) -> typing.Dict[str, typing.List[int]]:
  """Get the bytecode IDs for the POJ-104 app classification experiment."""

  def GetBytecodeIds(filter_cb) -> typing.List[int]:
    """Return the bytecode IDs from the given filtered query."""
    with db.Session() as session:
      q = session.query(graph_database.GraphMeta.id).filter(filter_cb())
      return [r[0] for r in q]

  train = lambda: graph_database.GraphMeta.source_name == "poj-104:train"
  test = lambda: graph_database.GraphMeta.source_name == "poj-104:test"
  val = lambda: graph_database.GraphMeta.source_name == "poj-104:val"
  return {
    "train": GetBytecodeIds(train),
    "val": GetBytecodeIds(val),
    "test": GetBytecodeIds(test),
  }


def GetTrainValTestSplits(
  db: graph_database.Database,
  train_val_test_ratio: typing.Iterable[float] = (3, 1, 1),
) -> typing.Dict[str, typing.List[int]]:
  """Get the bytecode IDs split into train, val, and test groups.

  This concatenates the POJ-104 sources with the other sources split into
  train/val/test segments.

  Args:
    db: The database to read IDs from.
    train_val_test_ratio: A triplet of ratios for the training, validation, and
      test sets. E.g. with the triplet (3, 1, 1), the training set will be 3/5
      of the dataset, and the validation and test sets will each by 1/5 of the
      dataset.

  Returns:
    A dictionary of bytecode IDs with "train", "val", and "test" keys.
  """
  # Normalize the ratios to sum to 1.
  ratios = np.array(list(train_val_test_ratio), dtype=np.float32)
  ratios /= sum(ratios)

  with db.Session() as s:
    poj104 = GetPoj104BytecodeGroups(db)

    # Get the bytecode IDs of non-POJ-104
    num_bytecodes = (
      s.query(sql.func.count(graph_database.GraphMeta.id))
      .filter(~graph_database.GraphMeta.source_name.like("poj-104:%"))
      .one()[0]
    )
    train_val_test_counts = np.floor(ratios * num_bytecodes).astype(np.int32)
    total_count = train_val_test_counts.sum()
    app.Log(
      1,
      "Splitting %s bytecodes into groups: %s train, %s val, %s test",
      humanize.Commas(total_count + sum(len(s) for s in poj104.values())),
      humanize.Commas(train_val_test_counts[0] + len(poj104["train"])),
      humanize.Commas(train_val_test_counts[1] + len(poj104["val"])),
      humanize.Commas(train_val_test_counts[2] + len(poj104["test"])),
    )

    q = (
      s.query(graph_database.GraphMeta.id)
      .filter(~graph_database.GraphMeta.source_name.like("poj-104:%"))
      .order_by(db.Random())
    )
    ids = [r[0] for r in q]

  return {
    "train": ids[: train_val_test_counts[0]] + poj104["train"],
    "val": (
      ids[train_val_test_counts[0] : sum(train_val_test_counts[:2])]
      + poj104["val"]
    ),
    "test": ids[sum(train_val_test_counts[:2]) :] + poj104["test"],
  }


# End of splitters copied from //deeplearning/ml4pl/bytecode:splitters
###############################################################################


def StratifiedKFold(db: graph_database.Database, num_splits: int):
  """Apply a stratified K-fold split on the graph database."""
  with db.Session() as session:
    num_graphs = session.query(
      sql.func.count(graph_database.GraphMeta.id)
    ).one()[0]
    with prof.Profile(f"Loaded labels from {num_graphs} graphs"):
      # Load all graphs as a single batch. WARNING this will not work for large
      # datasets!
      batcher = graph_batcher.GraphBatcher(db)
      options = graph_batcher.GraphBatchOptions(max_graphs=num_graphs + 1)
      graph_batches = list(batcher.MakeGraphBatchIterator(options))
      assert len(graph_batches) == 1
      graph_batch = graph_batches[0]

      graph_ids = np.array(
        graph_batch.log._transient_data["graph_indices"], dtype=np.int32
      )
      # Compute the dense labels from one-hot vectors.
      labels = np.argmax(graph_batch.graph_y, axis=1)

  # Split the graphs
  seed = 0xCEC
  splitter = model_selection.StratifiedKFold(
    n_splits=num_splits, shuffle=True, random_state=seed
  )
  dataset_splits = splitter.split(graph_ids, labels)

  groups = {
    str(i): graph_ids[test] for i, (train, test) in enumerate(dataset_splits)
  }
  return groups


def main():
  """Main entry point."""
  graph_db = FLAGS.graph_db()

  with graph_db.Session(commit=True) as session:
    app.Log(1, "Unsetting all graph groups")
    update = sql.update(graph_database.GraphMeta).values(group="")
    graph_db.engine.execute(update)

  if FLAGS.k_fold:
    groups = StratifiedKFold(graph_db, FLAGS.k_fold)
  else:
    groups = GetTrainValTestSplits(graph_db)

  for group, ids in groups.items():
    with graph_db.Session(commit=True) as session:
      app.Log(
        1, "Setting `%s` group on %s graphs", group, humanize.Commas(len(ids))
      )
      update = (
        sql.update(graph_database.GraphMeta)
        .where(graph_database.GraphMeta.id.in_(ids))
        .values(group=group)
      )
      graph_db.engine.execute(update)

  app.Log(1, "done")


if __name__ == "__main__":
  app.Run(main)
