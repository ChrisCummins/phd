"""Split a graph database using IR IDs for training/validation/testing."""
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.ir import split as ir_split
from labm8.py import app
from labm8.py import humanize
from labm8.py import prof

FLAGS = app.FLAGS


def ApplySplit(
  ir_db: ir_database.Database,
  proto_db: unlabelled_graph_database.Database,
  splitter: ir_split.Splitter,
):
  """Split the IR database and apply the split to the graph database."""
  # Unset all splits.
  with prof.Profile(f"Unset splits on {proto_db.proto_count} protos"):
    update = sql.update(unlabelled_graph_database.ProgramGraph).values(
      split=None
    )
    proto_db.engine.execute(update)

  # Split the IR database and assign the splits to the unlabelled graphs.
  for split, ir_ids in enumerate(splitter.Split(ir_db)):
    with prof.Profile(
      f"Set {split} split on {humanize.Plural(len(ir_ids), 'IR ID')}"
    ):
      update = (
        sql.update(unlabelled_graph_database.ProgramGraph)
        .where(unlabelled_graph_database.ProgramGraph.ir_id.in_(ir_ids))
        .values(split=split)
      )
      proto_db.engine.execute(update)


def Main():
  """Main entry point."""
  ApplySplit(
    FLAGS.ir_db(), FLAGS.proto_db(), ir_split.Splitter.CreateFromFlags()
  )


if __name__ == "__main__":
  app.Run(Main)
