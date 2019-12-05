"""Print a summary table of model results."""
import io
import pickle

import pandas as pd
import sqlalchemy as sql

from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import pdutil

app.DEFINE_database(
  "log_db",
  log_database.Database,
  None,
  "The input log database.",
  must_exist=True,
)
app.DEFINE_string(
  "format", "txt", "The format to print the result table. One of {txt,csv}"
)
app.DEFINE_boolean(
  "human_readable", True, "Format the column data in a human-readable format."
)
app.DEFINE_list("extra_model_flags", [], "Additional model flags to print.")
app.DEFINE_list("extra_flags", [], "Additional flags to print.")
FLAGS = app.FLAGS


def GetProblemFromPickledGraphDbUrl(pickled_column_value: bytes):
  db_url = pickle.loads(pickled_column_value)
  if "reachability" in db_url:
    return "reachability"
  elif "domtree" in db_url:
    return "domtree"
  elif "datadep" in db_url:
    return "datadep"
  elif "liveness" in db_url:
    return "liveness"
  elif "subexpressions" in db_url:
    return "subexpressions"
  elif "alias_set" in db_url:
    return "alias_sets"
  elif "polyhedra" in db_url:
    return "polyhedras"
  elif "devmap_amd_unbalanced_split" in db_url:
    return "devmap_amd_unbalanced_split"
  elif "devmap_nvidia_unbalanced_split" in db_url:
    return "devmap_nvidia_unbalanced_split"
  elif "devmap_amd" in db_url:
    return "devmap_amd"
  elif "devmap_nvidia" in db_url:
    return "devmap_nvidia"
  else:
    raise ValueError(f"Could not interpret database URL '{db_url}'")


def GetLeaderboard(
  log_db: log_database.Database, human_readable: bool = False
) -> pd.DataFrame:
  """Compute a leaderboard."""
  with log_db.Session() as session:
    # Create a table with batch log stats.
    query = session.query(
      log_database.BatchLogMeta.run_id,
      sql.func.max(log_database.BatchLogMeta.date_added).label("last_log"),
      log_database.BatchLogMeta.epoch,
      sql.func.count(log_database.BatchLogMeta.run_id).label("num_batches"),
      sql.func.avg(log_database.BatchLogMeta.accuracy).label("accuracy"),
      sql.func.avg(log_database.BatchLogMeta.precision).label("precision"),
      sql.func.avg(log_database.BatchLogMeta.recall).label("recall"),
      sql.func.avg(log_database.BatchLogMeta.iteration_count).label(
        "avg_iteration_count"
      ),
      sql.func.avg(log_database.BatchLogMeta.model_converged).label(
        "avg_model_converged"
      ),
    )
    query = query.filter(log_database.BatchLogMeta.type == "test")
    query = query.group_by(log_database.BatchLogMeta.run_id)
    query = query.group_by(log_database.BatchLogMeta.epoch)
    df = pdutil.QueryToDataFrame(session, query)
    df.set_index("run_id", inplace=True)

    # Create a table with the names of the graph databases.
    query = session.query(
      log_database.Parameter.run_id,
      log_database.Parameter.pickled_value.label("problem"),
    )
    query = query.filter(
      log_database.Parameter.type == log_database.ParameterType.FLAG
    )
    query = query.filter(
      log_database.Parameter.parameter
      == "deeplearning.ml4pl.models.classifier_base.graph_db"
    )
    aux_df = pdutil.QueryToDataFrame(session, query)
    # Un-pickle the parameter values and extract the database names from between
    # the `?` delimiters.
    aux_df["problem"] = [
      GetProblemFromPickledGraphDbUrl(x) for x in aux_df["problem"]
    ]
    aux_df.set_index("run_id", inplace=True)
    df = df.join(aux_df)

    # Add extra model flags.
    model_flags = ["restore_model"] + FLAGS.extra_model_flags
    for flag in model_flags:
      query = session.query(
        log_database.Parameter.run_id,
        log_database.Parameter.pickled_value.label(flag),
      )
      query = query.filter(
        sql.func.lower(log_database.Parameter.type) == "model_flag"
      )
      query = query.filter(log_database.Parameter.parameter == flag)
      aux_df = pdutil.QueryToDataFrame(session, query)
      # Un-pickle flag value.
      pdutil.RewriteColumn(aux_df, flag, lambda x: pickle.loads(x))
      aux_df.set_index("run_id", inplace=True)
      df = df.join(aux_df)

    extra_flags = [
      "ggnn_unroll_strategy",
      "ggnn_unroll_factor",
      "ggnn_layer_timesteps",
    ] + FLAGS.extra_flags
    for flag in extra_flags:
      # Strip the fully qualified flag name, e.g. "foo.bar.flag" -> "flag".
      flag_name = flag.split(".")[-1]
      query = session.query(
        log_database.Parameter.run_id,
        log_database.Parameter.pickled_value.label(flag_name),
      )
      query = query.filter(
        sql.func.lower(log_database.Parameter.type) == "flag"
      )
      query = query.filter(log_database.Parameter.parameter.like(f"%.{flag}"))
      aux_df = pdutil.QueryToDataFrame(session, query)
      # Un-pickle flag values.
      pdutil.RewriteColumn(aux_df, flag_name, lambda x: pickle.loads(x))
      aux_df.set_index("run_id", inplace=True)
      df = df.join(aux_df)

    def Time(x):
      """Humanize or default to '-' on failure."""
      try:
        return humanize.Time(x)
      except:
        return "-"

    def Duration(x):
      """Humanize or default to '-' on failure."""
      try:
        return humanize.Duration(x)
      except:
        return "-"

    def Percent(x):
      try:
        return f"{x:.2%}"
      except:
        return "-"

    def Float(x):
      try:
        return f"{x:.3f}"
      except:
        return "-"

    df.fillna("-", inplace=True)

    # Rewrite columns to be more user friendly.
    if human_readable:
      pdutil.RewriteColumn(df, "accuracy", Percent)
      pdutil.RewriteColumn(df, "precision", Float)
      pdutil.RewriteColumn(df, "recall", Float)

    return df


def main():
  """Main entry point."""
  df = GetLeaderboard(FLAGS.log_db(), human_readable=FLAGS.human_readable)
  if FLAGS.format == "csv":
    buf = io.StringIO()
    df.to_csv(buf)
    print(buf.getvalue())
  elif FLAGS.format == "txt":
    print(pdutil.FormatDataFrameAsAsciiTable(df))
  else:
    raise app.UsageError(f"Unknown --format='{FLAGS.format}'")


if __name__ == "__main__":
  app.Run(main)
