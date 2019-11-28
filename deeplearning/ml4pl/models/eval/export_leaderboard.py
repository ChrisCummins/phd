"""Export a leaderboard of model results."""
from deeplearning.ml4pl.models.eval import google_sheets
from deeplearning.ml4pl.models.eval import leaderboard
from labm8.py import app
from labm8.py import prof

app.DEFINE_string(
  "worksheet", "Leaderboard", "The name of the worksheet to export to."
)
FLAGS = app.FLAGS


def main():
  """Main entry point."""
  with prof.Profile("Created google worksheet"):
    g = google_sheets.GoogleSheets.CreateFromFlagsOrDie()
    s = g.GetOrCreateSpreadsheet(
      "ProGraML_Leaderboard_export", "zacharias.vf@gmail.com"
    )
    ws = g.GetOrCreateWorksheet(s, FLAGS.worksheet)

  with prof.Profile("Created leaderboard"):
    df = leaderboard.GetLeaderboard(log_db=FLAGS.log_db())

  with prof.Profile("Exported dataset"):
    g.ExportDataFrame(ws, df)


if __name__ == "__main__":
  app.Run(main)
