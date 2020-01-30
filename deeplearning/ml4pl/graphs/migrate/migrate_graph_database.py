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
"""Migrate the graph tuple databases.

This updates the graph tuple representation based on my experience in initial
experiments.
"""
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from labm8.py import app

FLAGS = app.FLAGS


def RemoveBackwardEdges(graph: graph_tuple.GraphTuple):
  """Graph tuples store redundant backward edges. Remove those."""
  return graph_tuple.GraphTuple(
    adjacency_lists=graph.adjacency_lists[:3],
    edge_positions=graph.edge_positions[:3],
    incoming_edge_counts=graph.incoming_edge_counts[:3],
    node_x_indices=graph.node_x_indices,
    node_y=graph.node_y,
    graph_x=graph.graph_x,
    graph_y=graph.graph_y,
  )


def UpdateGraphMetas(graph: graph_database.GraphMeta):
  """Update column values on graph metas."""
  # TODO(github.com/ChrisCummins/ProGraML/issues/5): Implement!


def main():
  """Main entry point."""
  # TODO(github.com/ChrisCummins/ProGraML/issues/5): Implement!


if __name__ == "__main__":
  app.Run(main)
