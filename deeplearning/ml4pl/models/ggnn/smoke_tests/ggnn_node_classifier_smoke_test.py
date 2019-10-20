"""Smoke test for //deeplearning/ml4pl/models/ggnn:ggnn_node_classifier."""
import pickle

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models.ggnn import ggnn_node_classifier
from deeplearning.ml4pl.models.ggnn.smoke_tests import smoke_test_base
from labm8 import app


FLAGS = app.FLAGS


class SmokeTester(smoke_test_base.SmokeTesterBase):

  def GetModelClass(self):
    return ggnn_node_classifier.GgnnGraphClassifierModel

  def PopulateDatabase(self, db: graph_database.Database):
    with db.Session(commit=True) as session:
      session.add_all([
        graph_database.GraphMeta(
            group="train",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            edge_type_count=1,
            node_count=3,
            edge_count=2,
            node_features_dimensionality=2,
            node_labels_dimensionality=2,
            data_flow_max_steps_required=2,
            graph=graph_database.Graph(
                data=pickle.dumps({
                  'adjacency_lists': [[(0, 1), (1, 2)]],
                  'node_x': [
                    (1, 0),
                    (0, 1),
                    (0, 0),
                  ],
                  'node_y': [
                    (1, 0),
                    (0, 1),
                    (0, 0),
                  ],
                }))),
        graph_database.GraphMeta(
            group="val",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            edge_type_count=1,
            node_count=3,
            edge_count=2,
            node_features_dimensionality=2,
            node_labels_dimensionality=2,
            data_flow_max_steps_required=2,
            graph=graph_database.Graph(
                data=pickle.dumps({
                  'adjacency_lists': [[(0, 1), (1, 2)]],
                  'node_x': [
                    (1, 0),
                    (0, 1),
                    (0, 0),
                  ],
                  'node_y': [
                    (1, 0),
                    (0, 1),
                    (0, 0),
                  ],
                }))),
        graph_database.GraphMeta(
            group="test",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            edge_type_count=1,
            node_count=3,
            edge_count=2,
            node_features_dimensionality=2,
            node_labels_dimensionality=2,
            data_flow_max_steps_required=2,
            graph=graph_database.Graph(
                data=pickle.dumps({
                  'adjacency_lists': [[(0, 1), (1, 2)]],
                  'node_x': [
                    (1, 0),
                    (0, 1),
                    (0, 0),
                  ],
                  'node_y': [
                    (1, 0),
                    (0, 1),
                    (0, 0),
                  ],
                }))),
      ])


def main():
  smoke_tester = SmokeTester()
  smoke_tester.Run()


if __name__ == '__main__':
  app.Run(main)
