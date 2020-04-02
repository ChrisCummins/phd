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
"""Unit tests for //program/graph:graph_tuple."""
import numpy as np

from labm8.py import test
from programl.graph.py import graph_tuple_builder
from programl.proto import edge_pb2
from programl.proto import features_pb2
from programl.proto import node_pb2
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2

FLAGS = test.FLAGS


def test_missing_node_feature():
  builder = graph_tuple_builder.GraphTupleBuilder(
    features=graph_tuple_builder.Feature("node", "x", "int64"),
    labels=graph_tuple_builder.Feature("node", "y", "int64"),
  )

  proto = program_graph_pb2.ProgramGraph(
    node=[node_pb2.Node(), node_pb2.Node(),],
    edge=[edge_pb2.Edge(source=0, target=1),],
  )
  with test.Raises(ValueError) as e_ctx:
    builder.AddProgramGraph(proto)
  assert str(e_ctx.value) == "node feature not found: x (int64)"


def test_missing_node_label():
  builder = graph_tuple_builder.GraphTupleBuilder(
    features=graph_tuple_builder.Feature("node", "x", "int64"),
    labels=graph_tuple_builder.Feature("graph", "y", "float"),
  )

  proto = program_graph_pb2.ProgramGraph(
    node=[
      node_pb2.Node(
        features=features_pb2.Features(
          feature={
            "x": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[1, 2, 3])
            ),
          }
        )
      ),
      node_pb2.Node(
        features=features_pb2.Features(
          feature={
            "x": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[1, 2, 3])
            ),
          }
        )
      ),
    ],
    edge=[edge_pb2.Edge(source=0, target=1),],
  )
  with test.Raises(ValueError) as e_ctx:
    builder.AddProgramGraph(proto)
  assert str(e_ctx.value) == "graph feature not found: y (float)"


def test_node_features_and_node_labels():
  """Build from an empty proto."""
  builder = graph_tuple_builder.GraphTupleBuilder(
    features=graph_tuple_builder.Feature("node", "x", "int64"),
    labels=graph_tuple_builder.Feature("node", "y", "int64"),
  )

  proto = program_graph_pb2.ProgramGraph(
    node=[
      node_pb2.Node(
        features=features_pb2.Features(
          feature={
            "x": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[1, 2, 3])
            ),
            "y": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[0, 1])
            ),
          }
        )
      ),
      node_pb2.Node(
        features=features_pb2.Features(
          feature={
            "x": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[4, 5, 6])
            ),
            "y": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[1, 0])
            ),
          }
        )
      ),
    ],
    edge=[edge_pb2.Edge(source=0, target=1),],
  )
  builder.AddProgramGraph(proto)
  builder.AddProgramGraph(proto)

  gt = builder.Build()
  assert len(gt.adjacencies) == 3
  assert gt.adjacencies[0].tolist() == [[0, 1], [2, 3]]
  assert gt.adjacencies[1].tolist() == []
  assert gt.adjacencies[2].tolist() == []
  assert gt.features.tolist() == [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
  assert gt.labels.tolist() == [[0, 1], [1, 0], [0, 1], [1, 0]]

  assert gt.disjoint_nodes_sections.tolist() == [2]
  assert gt.disjoint_features_sections.tolist() == [2]
  assert gt.disjoint_labels_sections.tolist() == [2]
  assert gt.node_count == 4
  assert gt.edge_count == 2


def test_node_features_and_graph_labels():
  """Build from an empty proto."""
  builder = graph_tuple_builder.GraphTupleBuilder(
    features=graph_tuple_builder.Feature("node", "x", "int64"),
    labels=graph_tuple_builder.Feature("graph", "target", "float"),
  )

  proto = program_graph_pb2.ProgramGraph(
    node=[
      node_pb2.Node(
        features=features_pb2.Features(
          feature={
            "x": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[1, 2, 3])
            ),
          }
        )
      ),
      node_pb2.Node(
        features=features_pb2.Features(
          feature={
            "x": features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[4, 5, 6])
            ),
          }
        )
      ),
    ],
    edge=[edge_pb2.Edge(source=0, target=1),],
    features=features_pb2.Features(
      feature={
        "target": features_pb2.Feature(
          float_list=features_pb2.FloatList(value=[0.5, 0.3, 0.1])
        ),
      }
    ),
  )
  builder.AddProgramGraph(proto)
  builder.AddProgramGraph(proto)

  gt = builder.Build()
  assert gt.features.tolist() == [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
  np.testing.assert_almost_equal(gt.labels, [[0.5, 0.3, 0.1], [0.5, 0.3, 0.1]])

  assert gt.disjoint_nodes_sections.tolist() == [2]
  assert gt.disjoint_features_sections.tolist() == [2]
  assert gt.disjoint_labels_sections.tolist() == [1]


def test_graph_features_lists():
  """Build from an empty proto."""
  builder = graph_tuple_builder.GraphTupleBuilder(
    features=graph_tuple_builder.Feature("node", "x", "int64"),
    labels=graph_tuple_builder.Feature("node", "x", "int64"),
  )

  proto = program_graph_pb2.ProgramGraph(
    node=[node_pb2.Node(), node_pb2.Node()],
    edge=[edge_pb2.Edge(source=0, target=1)],
  )

  features = program_graph_features_pb2.ProgramGraphFeatures(
    node_features=features_pb2.FeatureLists(
      feature_list={
        "x": features_pb2.FeatureList(
          feature=[
            features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[1, 2, 3])
            ),
            features_pb2.Feature(
              int64_list=features_pb2.Int64List(value=[4, 5, 6])
            ),
          ]
        ),
      }
    )
  )
  builder.AddProgramGraphFeatures(proto, features)
  builder.AddProgramGraphFeatures(proto, features)

  gt = builder.Build()
  assert gt.features.tolist() == [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
  assert gt.labels.tolist() == [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]

  assert gt.disjoint_nodes_sections.tolist() == [2]
  assert gt.disjoint_features_sections.tolist() == [2]
  assert gt.disjoint_labels_sections.tolist() == [2]


if __name__ == "__main__":
  test.Main()
