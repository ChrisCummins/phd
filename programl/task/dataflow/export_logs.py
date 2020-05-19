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
"""Export per-epoch statistics from machine learning logs.

Machine learning jobs write log files summarizing the performance of the model
after each training epoch. The log directory is printed at the start of
execution of a machine learning job, for example:

  $ bazel run //programl/task/dataflow:train_ggnn
  Writing logs to ~/programl/dataflow/logs/ggnn/reachability/foo@20:05:16T12:53:42
  ...

This script reads one of these log directories and prints a table of per-epoch
stats to stdout. For example:

  $ export-ml-logs --path=~/programl/dataflow/ml/logs/foo@20:05:16T12:53:42

CSV format can be exported using --fmt=csv:

  $ export-ml-logs --path=~/programl/dataflow/ml/logs/foo@20:05:16T12:53:42 \\
      --fmt=csv > stats.csv

Alternatively the summary can be uploaded to Google Sheets using:

  $ export-ml-logs --path=~/programl/dataflow/ml/logs/foo@20:05:16T12:53:42 \\
      --google_sheet='My Spreadsheet' \\
      --worksheet='reachability' \\
      --google_sheets_credentials=/tmp/credentials.json \\
      --google_sheets_default_share_with=joe@example.com
"""
import pathlib
import sys

import pandas as pd

from labm8.py import app
from labm8.py import google_sheets
from labm8.py import pbutil
from labm8.py import pdutil
from programl.proto import epoch_pb2

app.DEFINE_input_path(
  "path", None, "The directory containing logs to export.", is_dir=True
)
app.DEFINE_string(
  "google_sheet",
  None,
  "The name of a Google Sheets spreadsheet to export tables to. If it does "
  "not exist, the spreadsheet is created and shared with "
  "--google_sheets_default_share_with. See --google_sheets_credentials for "
  "setting the credentials required to use the Google Sheets API.",
)
app.DEFINE_string("fmt", "txt", "Stdout format.")
app.DEFINE_string(
  "worksheet", "Sheet1", "The name of the worksheet to export to"
)
FLAGS = app.FLAGS


def ReadEpochLogs(path: pathlib.Path):
  epochs = []
  for path in (path / "epochs").iterdir():
    epoch = pbutil.FromFile(path, epoch_pb2.EpochList())
    # Skip files without data.
    if not len(epoch.epoch):
      continue
    # Skip files without training results (e.g. test runs).
    if not epoch.epoch[0].train_results.batch_count:
      continue
    epochs += list(epoch.epoch)
  return epoch_pb2.EpochList(epoch=sorted(epochs, key=lambda x: x.epoch_num))


def EpochsToDataFrame(epochs: epoch_pb2.EpochList) -> pd.DataFrame:
  rows = []
  for e in epochs.epoch:
    rows.append(
      {
        "epoch_num": e.epoch_num,
        "walltime_seconds": e.walltime_seconds,
        "train_graph_count": e.train_results.graph_count,
        "train_batch_count": e.train_results.batch_count,
        "train_target_count": e.train_results.target_count,
        "train_learning_rate": e.train_results.mean_learning_rate,
        "train_loss": e.train_results.mean_loss,
        "train_accuracy": e.train_results.mean_accuracy,
        "train_precision": e.train_results.mean_precision,
        "train_recall": e.train_results.mean_recall,
        "train_f1": e.train_results.mean_f1,
        "train_walltime_seconds": e.train_results.walltime_seconds,
        "val_graph_count": e.val_results.graph_count,
        "val_batch_count": e.val_results.batch_count,
        "val_target_count": e.val_results.target_count,
        "val_learning_rate": e.val_results.mean_learning_rate,
        "val_loss": e.val_results.mean_loss,
        "val_accuracy": e.val_results.mean_accuracy,
        "val_precision": e.val_results.mean_precision,
        "val_recall": e.val_results.mean_recall,
        "val_f1": e.val_results.mean_f1,
        "val_walltime_seconds": e.val_results.walltime_seconds,
      }
    )
  df = pd.DataFrame(rows)
  df = df.set_index("epoch_num", drop=True).sort_index()

  # Add columns for cumulative totals.
  df["train_graph_count_cumsum"] = df.train_graph_count.cumsum()
  df["train_batch_count_cumsum"] = df.train_batch_count.cumsum()
  df["train_target_count_cumsum"] = df.train_target_count.cumsum()
  df["train_walltime_seconds_cumsum"] = df.train_walltime_seconds.cumsum()
  df["walltime_seconds_cumsum"] = df.walltime_seconds.cumsum()

  # Re-order columns.
  return df[
    [
      "train_batch_count",
      "train_batch_count_cumsum",
      "train_graph_count",
      "train_graph_count_cumsum",
      "train_target_count",
      "train_target_count_cumsum",
      "train_learning_rate",
      "train_loss",
      "train_accuracy",
      "train_precision",
      "train_recall",
      "train_f1",
      "train_walltime_seconds",
      "train_walltime_seconds_cumsum",
      "val_batch_count",
      "val_graph_count",
      "val_target_count",
      "val_learning_rate",
      "val_loss",
      "val_accuracy",
      "val_precision",
      "val_recall",
      "val_f1",
      "val_walltime_seconds",
    ]
  ]


def Main():
  path = FLAGS.path
  fmt = FLAGS.fmt
  spreadsheet_name = FLAGS.google_sheet
  worksheet_name = FLAGS.worksheet

  epochs = ReadEpochLogs(path)
  df = EpochsToDataFrame(epochs)

  if spreadsheet_name:
    gsheets = google_sheets.GoogleSheets.FromFlagsOrDie()
    spreadsheet = gsheets.GetOrCreateSpreadsheet(spreadsheet_name)
    worksheet = gsheets.GetOrCreateWorksheet(spreadsheet, worksheet_name)
    gsheets.ExportDataFrame(worksheet, df)
    print(f"Exported Google Sheet to {spreadsheet_name}:{worksheet_name}")
  elif fmt == "csv":
    df.to_csv(sys.stdout, header=True)
  elif fmt == "txt":
    print(pdutil.FormatDataFrameAsAsciiTable(df))
  else:
    raise app.UsageError(f"Unknown --fmt: {fmt}")


if __name__ == "__main__":
  app.Run(Main)
