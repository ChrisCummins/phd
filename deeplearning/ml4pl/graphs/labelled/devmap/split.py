"""Split a labelled graph database for K-fold cross-validation."""
from typing import List

import numpy as np
import sqlalchemy as sql
from sklearn import model_selection

from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_integer("k", 10, "The number of splits for K-fold splitter.")
app.DEFINE_database(
  "copy_splits_to",
  graph_tuple_database.Database,
  None,
  "If set, this is a database to copy the split column to.",
)


class StratifiedGraphLabelKFold(object):
  """Stratified K-fold cross validation using graph labels."""

  def __init__(self, k: int):
    self.k = k

  def Split(self, db: graph_tuple_database.Database) -> List[np.array]:
    """Apply K-fold split on a graph database stratified over graph_y labels."""
    assert db.graph_y_dimensionality

    with prof.Profile(f"Loaded labels from {db.graph_count} graphs"):
      # Load all of the graph IDs and their labels.
      reader = graph_database_reader.BufferedGraphReader(db)

      graph_ids: List[int] = []
      graph_y: List[int] = []
      for graph in reader:
        graph_ids.append(graph.id)
        graph_y.append(np.argmax(graph.tuple.graph_y))

    splitter = model_selection.StratifiedKFold(n_splits=self.k, shuffle=True)
    dataset_splits = splitter.split(graph_ids, graph_y)

    return [
      np.array(graph_ids[test], dtype=np.int32)
      for i, (train, test) in enumerate(dataset_splits)
    ]

  def ApplySplit(self, db: graph_tuple_database.Database) -> None:
    """Set the split values on the given database."""
    for split, ids in self.Split(db):
      with prof.Profile(
        f"Set {split} split on {humanize.Plural(len(ids), 'row')}"
      ):
        update = (
          sql.update(graph_tuple_database.GraphTuple)
          .where(graph_tuple_database.GraphTuple.id.in_(ids))
          .values(split=split)
        )
        db.engine.execute(update)


def CopySplits(
  input_db: graph_tuple_database.Database,
  output_db: graph_tuple_database.Database,
):
  """Propagate the `split` column from one database to another."""
  # Unset splits on output database.
  with prof.Profile(f"Unset splits on {output_db.graph_count} graphs"):
    update = sql.update(graph_tuple_database.GraphTuple).values(split=None)
    output_db.engine.execute(update)

  # Copy each split one at a time.
  for split in input_db.splits:
    with prof.Profile(1, f"Copied split {split}"):
      with input_db.Session() as in_session:
        ids_to_set = [
          row.id
          for row in in_session.query(
            graph_tuple_database.GraphTuple.id
          ).filter(graph_tuple_database.GraphTuple.split == split)
        ]

      update = (
        sql.update(graph_tuple_database.GraphTuple)
        .where(graph_tuple_database.GraphTuple.id.in_(ids_to_set))
        .values(split=split)
      )
      output_db.engine.execute(update)


def main():
  """Main entry point."""
  if FLAGS.copy_splits_to:
    CopySplits(FLAGS.graph_db(), FLAGS.copy_splits_to())
  else:
    StratifiedGraphLabelKFold(FLAGS.k).ApplySplit(FLAGS.graph_db())


if __name__ == "__main__":
  app.Run(main)
