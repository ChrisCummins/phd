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
"""This module defines a class for exporting detailed batch logs."""
import pathlib
from typing import List
from typing import Tuple

import numpy as np
import sqlalchemy as sql

from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_analysis
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logger_lib
from labm8.py import fs
from labm8.py import jsonutil
from labm8.py import labtypes
from labm8.py import ppar
from labm8.py import progress


class BatchDetailsExporter(progress.Progress):
  """A class to export detailed batch logs to local files for analysis.

  This job exports detailed batch logs to the following files:

    DIR/graph_stats/RUN.EPOCH.TYPE.BATCH.GRAPH.json: A JSON file summarizing the
      performance of the model on a single graph.
    DIR/in_graphs/RUN.EPOCH.TYPE.BATCH.GRAPH.GraphTuple.pickle: A pickled
      GraphTuple of the input graph.
    DIR/out_graphs/RUN.EPOCH.TYPE.BATCH.GRAPH.GraphTuple.pickle: A pickled
      GraphTuple of the output graph.
    DIR/RUN_ID.EPOCH.TYPE.json: Epoch-level stats.

  Where <name> is: <run_id>.<epoch_num>.<epoch_type>.<batch>.<graph>.
  """

  def __init__(
    self,
    log_db: log_database.Database,
    checkpoint_ref: checkpoints.CheckpointReference,
    epoch_type: epoch.Type,
    outdir: pathlib.Path,
    export_batches_per_query: int = 128,
  ):
    """Constructor.

    Args:
      log_db: The database to export logs from.
      checkpoint_ref: A run ID and epoch number.
      epoch_type: The type of epoch to export graphs from.
      outdir: The directory to write results to.
    """
    self.log_db = log_db
    self.logger = logger_lib.Logger(self.log_db)
    self.checkpoint = checkpoint_ref
    self.epoch_type = epoch_type
    self.analyzer = log_analysis.RunLogAnalyzer(
      self.log_db, self.checkpoint.run_id
    )
    self.export_batches_per_query = export_batches_per_query

    self.outdir = outdir
    self.outdir.mkdir(parents=True, exist_ok=True)
    (self.outdir / "graph_stats").mkdir(exist_ok=True)
    (self.outdir / "in_graphs").mkdir(exist_ok=True)
    (self.outdir / "out_graphs").mkdir(exist_ok=True)

    # Count the total number of graphs to export cross all batches.
    with self.log_db.Session() as session:
      num_graphs = self.FilterBatchesQuery(
        session.query(sql.func.sum(log_database.Batch.graph_count))
      ).scalar()

    if not num_graphs:
      raise ValueError("No graphs found!")

    super(BatchDetailsExporter, self).__init__(
      name=f"export {self.checkpoint} graphs", unit="graphs", i=0, n=num_graphs,
    )

  def FilterBatchesQuery(self, query):
    """Filter a query against log_database.Batch table to select only those
    batches which should be exported.
    """
    return query.filter(
      log_database.Batch.epoch_num == self.checkpoint.epoch_num,
      log_database.Batch.epoch_type_num == self.epoch_type.value,
      log_database.Batch.run_id == str(self.checkpoint.run_id),
    ).join(log_database.BatchDetails)

  def LoadBatch(self, batch_ids: List[int]) -> List[log_database.Batch]:
    """Read batch data from the log database.

    Args:
      batch_ids: The batches to read.

    Returns:
      A list of batch logs, with detailed batches loaded. The order of the
      batches is undefined.
    """
    # Load the batch data from the database.
    with self.log_db.Session() as session:
      batches = (
        session.query(log_database.Batch)
        .filter(log_database.Batch.id.in_(batch_ids))
        .options(sql.orm.joinedload(log_database.Batch.details))
        .all()
      )
    return batches

  def BuildGraphsFromBatches(
    self, batches: List[log_database.Batch]
  ) -> List[
    Tuple[int, List[Tuple[int, graph_tuple.GraphTuple, graph_tuple.GraphTuple]]]
  ]:
    """An iterator which reads the graphs for a list of batches.

    Returns:
      An iterator of <batch_id, graphs> tuples, where batch_id is the ID of the
      batch in the log database, and graphs is a list of
      <graph_id, input_graph, output_graph> tuples.
    """
    return [
      (
        batch.id,
        [
          (graph_id, input_graph, output_graph)
          for graph_id, (input_graph, output_graph) in zip(
            batch.graph_ids, self.analyzer.GetInputOutputGraphs(batch)
          )
        ],
      )
      for batch in batches
    ]

  def Run(self):
    """Export the batch graphs."""
    # Get the full list of batches to export.
    with self.log_db.Session() as session:
      batch_ids = self.FilterBatchesQuery(
        session.query(log_database.Batch.id)
      ).all()

    # A confusion matrix for the entire set of batches.
    cm = np.zeros((2, 2), dtype=np.int64)

    name_prefix = f"{self.checkpoint.run_id}.{self.checkpoint.epoch_num}.{self.epoch_type.name.lower()}"

    # Split the batches into chunks.
    batch_id_chunks = labtypes.Chunkify(
      batch_ids, self.export_batches_per_query
    )
    # Read the batches in a background thread.
    batches = ppar.ThreadedIterator(
      map(self.LoadBatch, batch_id_chunks), max_queue_size=5
    )
    # Process the batches in a background thread.
    graphs_batches = ppar.ThreadedIterator(
      map(self.BuildGraphsFromBatches, batches), max_queue_size=5
    )

    for batch in graphs_batches:
      for batch_id, graphs in batch:
        for graph_id, ingraph, outgraph in graphs:
          self.ctx.i += 1

          name = f"{name_prefix}.{batch_id:04d}.{graph_id:04d}"

          statspath = self.outdir / "graph_stats" / f"{name}.json"
          ingraph_path = self.outdir / "in_graphs" / f"{name}.GraphTuple.pickle"
          outgraph_path = (
            self.outdir / "out_graphs" / f"{name}.GraphTuple.pickle"
          )

          # Write the input and output graph tuples.
          ingraph.ToFile(ingraph_path)
          outgraph.ToFile(outgraph_path)

          # Write the graph-level stats.
          graph_cm = log_analysis.BuildConfusionMatrix(
            ingraph.node_y, outgraph.node_y
          )
          fs.Write(
            statspath,
            jsonutil.format_json(
              {
                "accuracy": (graph_cm[0][0] + graph_cm[1][1]) / graph_cm.sum(),
                "node_count": ingraph.node_count,
                "edge_count": ingraph.edge_count,
                "confusion_matrix": graph_cm.tolist(),
              }
            ).encode("utf-8"),
          )

          cm = np.add(cm, graph_cm)

    # Write the epoch-level stats.
    fs.Write(
      self.outdir / f"{name_prefix}.json",
      jsonutil.format_json(
        {"graph_count": self.ctx.i, "confusion_matrix": cm.tolist(),}
      ).encode("utf-8"),
    )

    self.ctx.i = self.ctx.n
