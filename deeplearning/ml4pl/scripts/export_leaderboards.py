"""Export all leaderboards."""
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.eval import google_sheets
from deeplearning.ml4pl.models.eval import leaderboard
from labm8 import app
from labm8 import prof

FLAGS = app.FLAGS


def main():
  """Main entry point."""
  g = google_sheets.GoogleSheets.CreateFromFlagsOrDie()
  s = g.GetOrCreateSpreadsheet('ml4pl', 'chrisc.101@gmail.com')

  dbs = [
      "reachability",
      "domtree",
      "datadep",
      "liveness",
      "subexpressions",
      "alias_set",
      "polyhedra",
      "devmap_unbalanced_split",
  ]

  for db in dbs:
    with prof.Profile(f"Exported {db} leaderboard"):
      df = leaderboard.GetLeaderboard(log_db=log_database.Database(
          f"file:///var/phd/db/cc1.mysql?ml4pl_{db}_logs?charset=utf8",
          must_exist=True))
      ws = g.GetOrCreateWorksheet(s, db)
      g.ExportDataFrame(ws, df)


if __name__ == '__main__':
  app.Run(main)
