# Copyright 2019 the ProGraML authors.
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
"""Rewrite the 'node_x' values of graph tuples to use embedding indices.

Instead of np.array([0, 1]) for True and np.array([1, 0]) for False, use
1 for True and 0 for False.
"""
import pickle

import numpy as np
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import graph_tuple as graph_tuples
from labm8.py import app
from labm8.py import humanize
from labm8.py import prof


FLAGS = app.FLAGS

app.DEFINE_database(
  "graph_db",
  graph_database.Database,
  None,
  "URL of database to modify.",
  must_exist=True,
)


def ReplaceOneHotNodeFeaturesWithEmbeddings(
  graph_tuple: graph_tuples.GraphTuple,
) -> graph_tuples.GraphTuple:
  """Swap out the 1-hot binary `node_x_indices` values for integers."""
  node_x_indices = np.array(
    [
      (1 if x[1] else 0) if isinstance(x, np.ndarray) else x
      for x in graph_tuple.node_x_indices
    ],
    dtype=np.int64,
  )

  return graph_tuples.GraphTuple(
    adjacency_lists=graph_tuple.adjacency_lists,
    edge_positions=graph_tuple.edge_positions,
    incoming_edge_counts=graph_tuple.incoming_edge_counts,
    node_x_indices=node_x_indices,
    node_y=graph_tuple.node_y,
    graph_x=graph_tuple.graph_x,
    graph_y=graph_tuple.graph_y,
  )


def main():
  """Main entry point."""

  graph_db = FLAGS.graph_db()

  buffer_size = 1024

  with graph_db.Session() as s:
    with prof.Profile(
      lambda t: (
        f"Selected {humanize.Commas(len(ids))} graphs " "from database"
      )
    ):
      q = s.query(graph_database.GraphMeta.id)
      ids = [r[0] for r in q]

    # Iterate through the IDs in batches.
    while ids:
      batch_ids = ids[:buffer_size]
      ids = ids[buffer_size:]
      app.Log(1, "%s remaining graphs to process", humanize.Commas(len(ids)))

      q = s.query(graph_database.GraphMeta)
      q = q.options(sql.orm.joinedload(graph_database.GraphMeta.graph))
      q = q.filter(graph_database.GraphMeta.id.in_(batch_ids))

      with prof.Profile(
        f"Fixed embedding indices of {len(batch_ids)} graph tuples"
      ):
        for graph_meta in q:
          graph_meta.graph.pickled_data = pickle.dumps(
            ReplaceOneHotNodeFeaturesWithEmbeddings(graph_meta.data)
          )
      with prof.Profile("Committed changes"):
        s.commit()


if __name__ == "__main__":
  app.Run(main)
