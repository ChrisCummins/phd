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
