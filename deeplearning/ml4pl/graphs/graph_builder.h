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
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"

using labm8::Status;
using labm8::StatusOr;

namespace ml4pl {

// An <entry, exits> pair which records the node numbers for a function's entry
// and exit statement nodes, respectively.
using FunctionEntryExits = std::pair<size_t, std::vector<size_t>>;

// A module for constructing a single program graph.
class GraphBuilder {
 public:
  GraphBuilder();

  // Construct a new function and return its number.
  StatusOr<int> AddFunction(const string& name);

  // Node factories.
  StatusOr<int> AddStatement(const string& text, int function);

  StatusOr<int> AddIdentifier(const string& text, int function);

  StatusOr<int> AddImmediate(const string& text);

  // Edge factories.
  Status AddControlEdge(int sourceNode, int destinationNode);

  Status AddCallEdge(int sourceNode, int destinationNode);

  Status AddDataEdge(int sourceNode, int destinationNode, int position);

  // Access the graph.
  StatusOr<ProgramGraphProto> GetGraph();

 protected:
  StatusOr<std::pair<int, Node*>> AddNode(const Node::Type& type);

  StatusOr<std::pair<int, Node*>> AddNode(const Node::Type& type, int function);

  // Check that source and destination nodes are in-range.
  Status ValidateEdge(int sourceNode, int destinationNode) const;

  int NextNodeNumber() const { return graph_.node_size(); }

  // Add outgoing and return call edges from a node to a function.
  Status AddCallEdges(const size_t callingNode,
                      const FunctionEntryExits& calledFunction);

  int FunctionCount() const { return graph_.function_size(); }

  const Node& GetNode(int i) const { return graph_.node(i); }

 private:
  ProgramGraphProto graph_;

  void AddEdges(const std::vector<std::vector<size_t>>& adjacencies,
                const Edge::Flow& flow, std::vector<bool>* visitedNodes);

  Status AddReverseEdges(
      const std::vector<std::vector<std::pair<size_t, int>>>& adjacencies,
      const Edge::Flow& flow, std::vector<bool>* visitedNodes);

  // Insert the string into the strings table, or return its value if already
  // present.
  int AddString(const string& s);

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
