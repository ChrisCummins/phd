// This file defines a class for constructing program graphs.
//
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
#pragma once

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/string.h"

namespace ml4pl {

// An <entry, exits> pair which records the node numbers for a function's entry
// and exit statement nodes, respectively.
using FunctionEntryExits = std::pair<size_t, std::vector<size_t>>;

// A module for constructing a single program graph.
class GraphBuilder {
 public:
  GraphBuilder();

  // Construct a new function and return its number.
  std::pair<int, Function*> AddFunction(const string& name);

  // Node factories.
  std::pair<int, Node*> AddStatement(const string& text, int function);

  std::pair<int, Node*> AddIdentifier(const string& text, int function);

  std::pair<int, Node*> AddImmediate(const string& text);

  // Edge factories.
  void AddControlEdge(int sourceNode, int destinationNode);

  void AddCallEdge(int sourceNode, int destinationNode);

  void AddDataEdge(int sourceNode, int destinationNode, int position);

  // Access the graph.
  const ProgramGraphProto& GetGraph();

 protected:
  std::pair<int, Node*> AddNode(const Node::Type& type);

  Edge* AddEdge(const Edge::Flow& flow, int sourceNode, int destinationNode,
                int position);

  size_t NextNodeNumber() const { return graph_.node_size(); }

  // Add outgoing and return call edges from a node to a function.
  void AddCallEdges(const size_t callingNode,
                    const FunctionEntryExits& calledFunction);

 private:
  ProgramGraphProto graph_;

  void AddEdges(const std::vector<std::vector<size_t>>& adjacencies,
                const Edge::Flow& flow, std::vector<bool>* visitedNodes);

  void AddReverseEdges(
      const std::vector<std::vector<std::pair<size_t, int>>>& adjacencies,
      const Edge::Flow& flow, std::vector<bool>* visitedNodes);

  // Insert the string into the strings table, or return its value if already
  // present.
  int AddString(const string& s);

  // Return the string from the strings table at the given index. This CHECK
  // fails if the requested index is out of bound.
  const string& GetString(int index) const;

  // Adjacency lists.
  std::vector<std::vector<size_t>> control_adjacencies_;
  std::vector<std::vector<std::pair<size_t, int>>> data_reverse_adjacencies_;
  std::vector<std::vector<size_t>> call_adjacencies_;

  // A map from unique strings to their position in the flattened string list.
  // E.g. {"a": 0, "b": 1} flattens to the strings list ["a", "b"].
  absl::flat_hash_map<string, int> strings_;

  bool finalized_;
};

}  // namespace ml4pl
