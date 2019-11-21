"""Export a leaderboard of model results."""
from deeplearning.ml4pl.models.eval import google_sheets
from deeplearning.ml4pl.models.eval import leaderboard
from labm8 import app
from labm8 import prof
import pandas as pd

app.DEFINE_string('worksheet', 'Leaderboard',
                  'The name of the worksheet to export to.')
FLAGS = app.FLAGS


def main():
  """Main entry point."""
  with prof.Profile("Created google worksheet"):
    g = google_sheets.GoogleSheets.CreateFromFlagsOrDie()
    s = g.GetOrCreateSpreadsheet('ProGraML_Leaderboard_export', 'zacharias.vf@gmail.com')
    ws = g.GetOrCreateWorksheet(s, FLAGS.worksheet)

  with prof.Profile("Created leaderboard"):
    log_db_str = FLAGS.log_db.url
    dfs = []
    for i in range(10):
      FLAGS.log_db.url = log_db_str.replace('.db', f'_{i}.db')
      df = leaderboard.GetLeaderboard(log_db=FLAGS.log_db())
      dfs.append(df)
    df = pd.concat(dfs)

  with prof.Profile("Exported dataset"):
    g.ExportDataFrame(ws, df)


if __name__ == '__main__':
  app.Run(main)
