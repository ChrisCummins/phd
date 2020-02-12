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
"""Dump the contents of an unlabelled graph database as files.

Write each unique graph to a file, named for its checksum. Supports dumping to
all graph representation formats: protocol buffers, networkx graphs, Graphviz
dotfiles, etc.

For example, to dump the contents of a local sqlite database as networkx graphs:

  $ bazel run //deeplearning/ml4pl/cmd:dump_unlabelled_graph_db -- \
      --proto_db=sqlite:////path/to/db \
      --outdir=/path/to/outdir \
      --fmt=nx

Partial exports are supported - exporting will resume where it left off.
"""
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.unlabelled import (
  unlabelled_graph_database_exporter,
)
from labm8.py import app
from labm8.py import progress

app.DEFINE_output_path(
  "outdir",
  "/tmp/phd/ml4pl/graphs",
  "The directory to write output files to.",
  is_dir=True,
)
app.DEFINE_enum(
  "fmt",
  programl.StdoutGraphFormat,
  programl.StdoutGraphFormat.PB,
  "The file type for graphs to dump.",
)
app.DEFINE_integer(
  "batch_size",
  1024,
  "Tuning parameter. The number of graphs to read in a batch.",
)
FLAGS = app.FLAGS


def Main():
  """Main entry point."""
  exporter = unlabelled_graph_database_exporter.GraphDatabaseExporter(
    db=FLAGS.proto_db(),
    outdir=FLAGS.outdir,
    fmt=FLAGS.fmt(),
    batch_size=FLAGS.batch_size,
  )

  progress.Run(exporter)


if __name__ == "__main__":
  app.Run(Main)
