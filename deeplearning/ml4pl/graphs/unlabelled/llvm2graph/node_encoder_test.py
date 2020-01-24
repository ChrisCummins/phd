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
"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled/llvm2graph:node_encoder."""
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import node_encoder
from labm8.py import test


FLAGS = test.FLAGS


def test_EncodeNodes_equivalent_preprocessed_text():
  """Test equivalence of nodes that pre-process to the same text."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="%7 = add nsw i32 %5, -1")
  b = builder.AddNode(text="%9 = add nsw i32 %5, -2")
  g = builder.g

  encoder = node_encoder.GraphNodeEncoder()
  encoder.EncodeNodes(g)

  assert g.nodes[a]["preprocessed_text"] == "<%ID> = add nsw i32 <%ID>, <INT>"
  assert g.nodes[b]["preprocessed_text"] == "<%ID> = add nsw i32 <%ID>, <INT>"


def test_EncodeNodes_encoded_values():
  """Test that "x" attribute of a node matches dictionary value."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="br label %4")
  g = builder.g

  encoder = node_encoder.GraphNodeEncoder()
  encoder.EncodeNodes(g)

  assert g.nodes[a]["x"][0] == encoder.dictionary["br label <%ID>"]


def test_EncodeNodes_encoded_values_differ_between_statements():
  """Test that "x" attribute of nodes differ between different texts."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="%7 = add nsw i32 %5, -1")
  b = builder.AddNode(text="br label %4")
  g = builder.g

  encoder = node_encoder.GraphNodeEncoder()
  encoder.EncodeNodes(g)

  assert g.nodes[a]["x"][0] != g.nodes[b]["x"][0]


if __name__ == "__main__":
  test.Main()
