"""Export a leaderboard of model results."""
from deeplearning.ml4pl.models.eval import google_sheets
from deeplearning.ml4pl.models.eval import leaderboard
from labm8 import app
from labm8 import prof

FLAGS = app.FLAGS


def main():
  """Main entry point."""
  with prof.Profile("Created google worksheet"):
    g = google_sheets.GoogleSheets.CreateFromFlagsOrDie()
    s = g.GetOrCreateSpreadsheet('ml4pl', 'chrisc.101@gmail.com')
    ws = g.GetOrCreateWorksheet(s, 'Leaderboard')

  with prof.Profile("Created leaderboard"):
    df = leaderboard.GetLeaderboard(log_db=FLAGS.log_db())

  with prof.Profile("Exported dataset"):
    g.ExportDataFrame(ws, df)


if __name__ == '__main__':
  app.Run(main)
