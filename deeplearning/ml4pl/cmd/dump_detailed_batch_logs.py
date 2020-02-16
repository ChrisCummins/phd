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
"""Write detailed batch logs to files.

This script dumps per-graph files for a specific epoch of a model's logs. It
dumps the raw input and output graphs, stats summarizing the model's performance
on each graph, and finally whole-epoch stats.

Usage:

    $ bazel run //deeplearning/ml4pl/cmd:dump_detailed_batch_logs -- \
        --log_db=sqlite:////path/to/log/db \
        --outdir=/path/to/export \
        --checkpoint=run_id@epoch_num \
        --epoch_type=test

This requires joining the log database and labelled graph databases.
"""
from deeplearning.ml4pl.models import batch_details_exporter
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from labm8.py import app
from labm8.py import progress


FLAGS = app.FLAGS

app.DEFINE_output_path(
  "outdir", "/tmp/reports", "The directory to write files to.",
)
app.DEFINE_string(
  "checkpoint",
  None,
  "The checkpoint to export graphs from, in the form <run_id>@<epoch_num>.",
)
app.DEFINE_enum(
  "epoch_type",
  epoch.Type,
  epoch.Type.TEST,
  "The type of epoch to export graphs from.",
)
app.DEFINE_integer(
  "export_batches_per_query",
  32,
  "Tuning parameter. The number of batches to read from the database at a time.",
)


def Main():
  """Main entry point."""
  exporter = batch_details_exporter.BatchDetailsExporter(
    log_db=FLAGS.log_db(),
    checkpoint_ref=checkpoints.CheckpointReference.FromString(FLAGS.checkpoint),
    epoch_type=FLAGS.epoch_type(),
    outdir=FLAGS.outdir,
    export_batches_per_query=FLAGS.export_batches_per_query,
  )
  progress.Run(exporter)


if __name__ == "__main__":
  app.Run(Main)
