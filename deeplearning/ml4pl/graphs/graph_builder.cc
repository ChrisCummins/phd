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

#include "deeplearning/ml4pl/graphs/graph_builder.h"

#include <sstream>
#include <utility>
#include "fmt/format.h"

#include "absl/container/flat_hash_set.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

namespace error = labm8::error;

namespace ml4pl {

GraphBuilder::GraphBuilder() : finalized_(false) {
  // Create the graph root node.
  auto node = AddNode(Node::STATEMENT).ValueOrDie();
  node.second->set_text(AddString("; root"));
}

StatusOr<int> GraphBuilder::AddFunction(const string& name) {
  if (!name.size()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Empty function name is invalid");
  }
  int functionNumber = graph_.function_size();
  Function* function = graph_.add_function();
  function->set_name(AddString(name));
  return functionNumber;
}

StatusOr<int> GraphBuilder::AddStatement(const string& text, int function) {
  std::pair<int, Node*> node;
  ASSIGN_OR_RETURN(node, AddNode(Node::STATEMENT, function));
  node.second->set_text(AddString(text));
  return node.first;
}

StatusOr<int> GraphBuilder::AddIdentifier(const string& text, int function) {
  std::pair<int, Node*> node;
  ASSIGN_OR_RETURN(node, AddNode(Node::IDENTIFIER));
  node.second->set_text(AddString(text));
  return node.first;
}

StatusOr<int> GraphBuilder::AddImmediate(const string& text) {
  std::pair<int, Node*> node;
  ASSIGN_OR_RETURN(node, AddNode(Node::IMMEDIATE));
  node.second->set_text(AddString(text));
  return node.first;
}

Status GraphBuilder::AddControlEdge(int sourceNode, int destinationNode) {
  RETURN_IF_ERROR(ValidateEdge(sourceNode, destinationNode));

  if (graph_.node(sourceNode).type() != Node::STATEMENT ||
      graph_.node(destinationNode).type() != Node::STATEMENT) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Control edge must connect statement nodes");
  }

  int sourceFunction = graph_.node(sourceNode).function();
  int destinationFunction = graph_.node(destinationNode).function();

  if (sourceFunction != destinationFunction) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Control edge must connect statements in the same function. "
                  "Received source node {} is in function {} and destination "
                  "node {} is in function {}",
                  sourceNode, sourceFunction, destinationNode,
                  destinationFunction);
  }

  control_adjacencies_[sourceNode].push_back(destinationNode);

  return Status::OK;
}

Status GraphBuilder::AddCallEdge(int sourceNode, int destinationNode) {
  RETURN_IF_ERROR(ValidateEdge(sourceNode, destinationNode));

  Node::Type sourceType = graph_.node(sourceNode).type();
  Node::Type destinationType = graph_.node(destinationNode).type();

  if (sourceType != Node::STATEMENT || destinationType != Node::STATEMENT) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Call edge must connect statements. Received source node {} "
                  "has type {} and destination node {} has type {}",
                  sourceNode, Node::Type_Name(sourceType), destinationNode,
                  Node::Type_Name(destinationType));
  }

  if (graph_.node(sourceNode).has_function() &&
      graph_.node(destinationNode).has_function() &&
      graph_.node(sourceNode).function() ==
          graph_.node(destinationNode).function()) {
    return Status(
        error::Code::INVALID_ARGUMENT,
        "Call edge must connect statements in different functions. "
        "Attempting to connect statements {} and {} both in function {}.",
        sourceNode, destinationNode, graph_.node(sourceNode).function());
  }

  call_adjacencies_[sourceNode].push_back(destinationNode);

  return Status::OK;
}

Status GraphBuilder::AddDataEdge(int sourceNode, int destinationNode,
                                 int position) {
  RETURN_IF_ERROR(ValidateEdge(sourceNode, destinationNode));

  Node::Type sourceType = graph_.node(sourceNode).type();
  Node::Type destinationType = graph_.node(destinationNode).type();

  bool sourceIsData =
      (sourceType == Node::IDENTIFIER || sourceType == Node::IMMEDIATE);
  bool destinationIsData = (destinationType == Node::IDENTIFIER ||
                            destinationType == Node::IMMEDIATE);

  if (!((sourceIsData && destinationType == Node::STATEMENT) ||
        (sourceType == Node::STATEMENT && destinationIsData))) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Data edge must connect either a statement with data "
                  "OR data with a statement. Received source node {} and "
                  "destination node {}",
                  Node::Type_Name(sourceType),
                  Node::Type_Name(destinationType));
  }

  data_reverse_adjacencies_[destinationNode].push_back({sourceNode, position});
  return Status::OK;
}

Status GraphBuilder::AddCallEdges(const size_t callingNode,
                                  const FunctionEntryExits& calledFunction) {
  RETURN_IF_ERROR(AddCallEdge(callingNode, calledFunction.first));
  for (auto exitNode : calledFunction.second) {
    RETURN_IF_ERROR(AddCallEdge(exitNode, callingNode));
  }
  return Status::OK;
}

void GraphBuilder::AddEdges(const std::vector<std::vector<size_t>>& adjacencies,
                            const Edge::Flow& flow,
                            std::vector<bool>* visitedNodes) {
  for (size_t sourceNode = 0; sourceNode < adjacencies.size(); ++sourceNode) {
    for (size_t position = 0; position < adjacencies[sourceNode].size();
         ++position) {
      size_t destinationNode = adjacencies[sourceNode][position];
      Edge* edge = graph_.add_edge();
      edge->set_flow(flow);
      edge->set_source_node(sourceNode);
      edge->set_destination_node(destinationNode);
      edge->set_position(position);

      // Record the source and destination nodes in the node set.
      (*visitedNodes)[sourceNode] = true;
      (*visitedNodes)[destinationNode] = true;
    }
  }
}

Status GraphBuilder::AddReverseEdges(
    const std::vector<std::vector<std::pair<size_t, int>>>& adjacencies,
    const Edge::Flow& flow, std::vector<bool>* visitedNodes) {
  for (size_t destinationNode = 0; destinationNode < adjacencies.size();
       ++destinationNode) {
    // Track the positions to ensure that they are unique.
    absl::flat_hash_set<int> positionsSet;

    for (auto source : adjacencies[destinationNode]) {
      size_t sourceNode = source.first;
      int position = source.second;

      // Ensure that the position is unique.
      auto it = positionsSet.find(position);
      if (it != positionsSet.end()) {
        return Status(error::Code::INVALID_ARGUMENT, "Duplicate position {}",
                      position);
      };
      positionsSet.insert(position);

      Edge* edge = graph_.add_edge();
      edge->set_flow(flow);
      edge->set_source_node(sourceNode);
      edge->set_destination_node(destinationNode);
      edge->set_position(position);

      // Record the source and destination nodes in the node set.
      (*visitedNodes)[source.first] = true;
      (*visitedNodes)[destinationNode] = true;
    }
  }

  return Status::OK;
}

StatusOr<ProgramGraphProto> GraphBuilder::GetGraph() {
  if (finalized_) {
    return graph_;
  }
  std::vector<bool> visitedNodes(graph_.node_size(), false);

  AddEdges(control_adjacencies_, Edge::CONTROL, &visitedNodes);
  RETURN_IF_ERROR(
      AddReverseEdges(data_reverse_adjacencies_, Edge::DATA, &visitedNodes));
  AddEdges(call_adjacencies_, Edge::CALL, &visitedNodes);

  // Check that all nodes except the root are connected. The root is allowed to
  // have no connections in the case where it is an empty graph.
  for (size_t i = 1; i < visitedNodes.size(); ++i) {
    if (!visitedNodes[i]) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "Graph contains node with no connections: {}",
                    graph_.node(i).DebugString());
    }
  }

  finalized_ = true;
  return graph_;
}

StatusOr<std::pair<int, Node*>> GraphBuilder::AddNode(const Node::Type& type) {
  int nodeNumber = NextNodeNumber();
  Node* node = graph_.add_node();
  node->set_type(type);

  // Create empty adjacency lists for the new node.
  DCHECK(control_adjacencies_.size() == static_cast<size_t>(nodeNumber))
      << control_adjacencies_.size() << " != " << nodeNumber;
  DCHECK(data_reverse_adjacencies_.size() == static_cast<size_t>(nodeNumber))
      << data_reverse_adjacencies_.size() << " != " << nodeNumber;
  DCHECK(call_adjacencies_.size() == static_cast<size_t>(nodeNumber))
      << call_adjacencies_.size() << " != " << nodeNumber;

  control_adjacencies_.push_back({});
  data_reverse_adjacencies_.push_back({});
  call_adjacencies_.push_back({});

  return std::make_pair(nodeNumber, node);
}

StatusOr<std::pair<int, Node*>> GraphBuilder::AddNode(const Node::Type& type,
                                                      int function) {
  if (function < 0 || function >= graph_.function_size()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Function {} is out of bounds for graph with {} functions",
                  function, graph_.function_size());
  }

  std::pair<int, Node*> node;
  ASSIGN_OR_RETURN(node, AddNode(type));
  node.second->set_function(function);
  return node;
}

Status GraphBuilder::ValidateEdge(int sourceNode, int destinationNode) const {
  if (sourceNode < 0 || sourceNode >= graph_.node_size()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Edge source node {} out of range for graph with {} nodes",
                  sourceNode, graph_.node_size());
  }

  if (destinationNode < 0 || destinationNode >= graph_.node_size()) {
    return Status(
        error::Code::INVALID_ARGUMENT,
        "Edge destination node {} out of range for graph with {} nodes",
        destinationNode, graph_.node_size());
  }

  return Status::OK;
}

int GraphBuilder::AddString(const string& s) {
  auto it = strings_.find(s);
  if (it == strings_.end()) {
    int index = graph_.string_size();
    strings_.insert({s, index});
    graph_.add_string(s);
    return index;
  }

  return it->second;
}

}  // namespace ml4pl
