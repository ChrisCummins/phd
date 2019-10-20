"""Smoke test for //deeplearning/ml4pl/models/ggnn:ggnn_graph_classifier."""
import pickle

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models.ggnn import ggnn_graph_classifier
from deeplearning.ml4pl.models.ggnn.smoke_tests import smoke_test_base
from labm8 import app


FLAGS = app.FLAGS


class SmokeTester(smoke_test_base.SmokeTesterBase):

  def GetModelClass(self):
    return ggnn_graph_classifier.GgnnGraphClassifierModel

  def PopulateDatabase(self, db: graph_database.Database):
    with db.Session(commit=True) as session:
      session.add_all([
        graph_database.GraphMeta(
            group="train",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            node_count=3,
            edge_count=2,
            edge_type_count=3,
            edge_features_dimensionality=1,
            graphs_labels_dimensionality=1,
            graph=graph_database.Graph(
                data=pickle.dumps({
                  'adjacency_lists': [[(0, 1)], [(1, 2)], []],
                  'edge_x': [[1], [2]],
                  'graph_y': [1],
                }))),
        graph_database.GraphMeta(
            group="train",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            node_count=3,
            edge_count=2,
            edge_type_count=3,
            edge_features_dimensionality=1,
            graphs_labels_dimensionality=1,
            graph=graph_database.Graph(
                data=pickle.dumps({
                  'adjacency_lists': [[(0, 1)], [(1, 2)], []],
                  'edge_x': [[1], [2]],
                  'graph_y': [1],
                }))),
        graph_database.GraphMeta(
            group="train",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            node_count=3,
            edge_count=2,
            edge_type_count=3,
            edge_features_dimensionality=1,
            graphs_labels_dimensionality=1,
            graph=graph_database.Graph(
                data=pickle.dumps({
                  'adjacency_lists': [[(0, 1)], [(1, 2)], []],
                  'edge_x': [[1], [2]],
                  'graph_y': [1],
                }))),
      ])


def main():
  smoke_tester = SmokeTester()
  smoke_tester.Run()


if __name__ == '__main__':
  app.Run(main)
