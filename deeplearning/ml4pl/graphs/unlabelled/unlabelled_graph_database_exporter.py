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
"""This module defines a class for exporting unlabelled graph databases."""
import pathlib

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from labm8.py import app
from labm8.py import fs
from labm8.py import labtypes
from labm8.py import ppar
from labm8.py import progress

FLAGS = app.FLAGS


class GraphDatabaseExporter(progress.Progress):
  """A class for exporting a database of unlabelled program graphs to files.

  Serialized graph protocol buffers are read from the database in order of their
  checksum string and converted to a different format if required. Partial
  exports are supported by first checking the output directory to see if any
  files have already been exported there, and if so, resuming the export from
  that checksum.
  """

  def __init__(
    self,
    db: unlabelled_graph_database.Database,
    outdir: pathlib.Path,
    fmt: programl.StdoutGraphFormat,
    batch_size: int = 512,
  ):
    """Constructor.

    Args:
      db: A database of unlabelled graphs.
      outdir: The directory to write files to.
      fmt: The file format to dump. One of
      batch_size:
    """
    self.db = db
    self.outdir = outdir
    self.fmt = fmt
    self.batch_size = batch_size

    self.outdir.mkdir(exist_ok=True, parents=True)
    self.file_suffix = programl.StdoutGraphFormatToFileExtension(self.fmt)

    # Find the most recently exported file, if any.
    exported_count = 0
    self.most_recent_export = ""
    for path in self.outdir.iterdir():
      if path.name.endswith(self.file_suffix):
        exported_count += 1
        self.most_recent_export = max(path.name, self.most_recent_export)

    # Compute the number of graphs that to be exported.
    with self.db.Session() as session:
      query = session.query(
        sql.func.count(
          sql.func.distinct(unlabelled_graph_database.ProgramGraphData.sha1)
        )
      )
      if self.most_recent_export:
        query = query.filter(
          unlabelled_graph_database.ProgramGraphData.sha1
          > self.most_recent_export
        )
      max_rows = query.scalar()

    super(GraphDatabaseExporter, self).__init__(
      name="export graphs", i=exported_count, n=max_rows
    )

  def Run(self):
    """Run the exporter."""
    with self.db.Session() as session:
      # Get the IDs of the unique graphs.
      ids_to_export = (
        session.query(
          sql.func.min(unlabelled_graph_database.ProgramGraphData.ir_id).label(
            "ir_id"
          ),
        )
        .group_by(unlabelled_graph_database.ProgramGraphData.sha1)
        .order_by(unlabelled_graph_database.ProgramGraphData.sha1)
      )

      if self.most_recent_export:
        ids_to_export = ids_to_export.filter(
          unlabelled_graph_database.ProgramGraphData.sha1
          > self.most_recent_export
        )

      ids_to_export = [r.ir_id for r in ids_to_export]

    def ReadGraphs(ids_chunk):
      """Read the checksum and serialized protos for a list of IR IDs."""
      # We must create a disposable session to perform this query in since it
      # will be executed in a background thread and SQLite requires session
      # objects to be used only in the thread in which they are created.
      with self.db.Session() as session:
        batch = (
          session.query(
            unlabelled_graph_database.ProgramGraphData.sha1,
            unlabelled_graph_database.ProgramGraphData.serialized_proto,
          )
          .filter(
            unlabelled_graph_database.ProgramGraphData.ir_id.in_(ids_chunk)
          )
          .order_by(unlabelled_graph_database.ProgramGraphData.sha1)
          .all()
        )
      return batch

    # An iterator of queries to run. Each query reads a chunk of the ID list
    # and returns the graphs.
    queries = map(ReadGraphs, labtypes.Chunkify(ids_to_export, self.batch_size))

    # Overlap the database reading and file writing in a background thread.
    queries = ppar.ThreadedIterator(queries, max_queue_size=5)

    for query in queries:
      for sha1, serialized_proto in query:
        self.ctx.i += 1
        fs.Write(
          self.outdir / f"{sha1}{self.file_suffix}",
          programl.SerializedProgramGraphToBytes(serialized_proto, self.fmt),
        )
