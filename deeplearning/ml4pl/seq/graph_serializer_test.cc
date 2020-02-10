// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "deeplearning/ml4pl/seq/graph_serializer.h"

#include "deeplearning/ml4pl/graphs/programl.pb.h"

#include <vector>

#include "labm8/cpp/test.h"

namespace ml4pl {
namespace {

TEST(SerializeStatements, EmptyGraph) {
  ProgramGraphProto graph;

  auto serialized = SerializeStatements(graph);
  ASSERT_EQ(0, serialized.size());
}

TEST(SerializeStatements, RootNodeOnly) {
  ProgramGraphProto graph;
  Node* root = graph.add_node();
  root->set_type(Node::STATEMENT);

  auto serialized = SerializeStatements(graph);
  ASSERT_EQ(0, serialized.size());
}

TEST(SerializeStatements, SingleFunction) {
  ProgramGraphProto graph;
  Node* root = graph.add_node();
  root->set_type(Node::STATEMENT);
  Node* a = graph.add_node();
  a->set_type(Node::STATEMENT);
  a->set_function(0);
  Edge* root_to_a = graph.add_edge();
  root_to_a->set_flow(Edge::CALL);
  root_to_a->set_source_node(0);
  root_to_a->set_destination_node(1);

  auto serialized = SerializeStatements(graph);
  ASSERT_EQ(1, serialized.size());
  ASSERT_EQ(1, serialized[0]);
}

TEST(SerializeStatements, SingleFunctionWithLoop) {
  ProgramGraphProto graph;
  Node* root = graph.add_node();
  root->set_type(Node::STATEMENT);
  Node* a = graph.add_node();
  a->set_type(Node::STATEMENT);
  a->set_function(0);
  Edge* root_to_a = graph.add_edge();
  root_to_a->set_flow(Edge::CALL);
  root_to_a->set_source_node(0);
  root_to_a->set_destination_node(1);
  Node* b = graph.add_node();
  b->set_type(Node::STATEMENT);
  Edge* a_to_b = graph.add_edge();
  a_to_b->set_flow(Edge::CONTROL);
  a_to_b->set_source_node(1);
  a_to_b->set_destination_node(2);
  Edge* b_to_a = graph.add_edge();
  b_to_a->set_flow(Edge::CONTROL);
  b_to_a->set_source_node(2);
  b_to_a->set_destination_node(1);

  auto serialized = SerializeStatements(graph);
  ASSERT_EQ(2, serialized.size());
  ASSERT_EQ(1, serialized[0]);
  ASSERT_EQ(2, serialized[1]);
}

}  // namespace
}  // namespace ml4pl

TEST_MAIN();
