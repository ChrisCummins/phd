#include "deeplearning/ml4pl/seq/graph_serializer.h"

#include "absl/container/flat_hash_set.h"

#include <deque>

namespace ml4pl {

static const int ROOT_NODE = 0;

std::vector<int> SerializeStatements(const ProgramGraph& graph) {
  std::vector<int> serialized;

  // An array of function entries to visit, where each function entry is a node
  // that is the destination of an outgoing call edge from the root node.
  std::vector<int> function_entries_to_visit;
  // A map from source node to a list of destination control edges.
  absl::flat_hash_map<int, std::vector<int>> forward_control_edges;

  // First traverse the graph edges to build the list of function entries to
  // visit and forward control edge map.
  for (int i = 0; i < graph.edge_size(); ++i) {
    const auto& edge = graph.edge(i);

    switch (edge.flow()) {
      case Edge::CONTROL: {
        // Insert an empty list in the forward control edge map if there is not
        // one already.
        auto existing = forward_control_edges.find(edge.source_node());
        if (existing == forward_control_edges.end()) {
          std::vector<int> new_edge_list;
          forward_control_edges.insert({edge.source_node(), new_edge_list});
        }

        // Get the forward control edge list for this node.
        auto edge_list = forward_control_edges.find(edge.source_node());

        edge_list->second.push_back(edge.destination_node());
      } break;
      case Edge::CALL:
        if (edge.source_node() == ROOT_NODE &&
            graph.node(edge.destination_node()).has_function()) {
          function_entries_to_visit.push_back(edge.destination_node());
        }
        break;
      case Edge::DATA:
        break;
    }
  }

  for (const auto& function_entry : function_entries_to_visit) {
    std::vector<int> serialized_function =
        SerializeStatements(function_entry, forward_control_edges);
    serialized.insert(serialized.end(), serialized_function.begin(),
                      serialized_function.end());
  }

  return serialized;
}

std::vector<int> SerializeStatements(
    const int& root,
    const absl::flat_hash_map<int, std::vector<int>>& forward_control_edges) {
  std::vector<int> serialized;

  // A set of visited nodes.
  absl::flat_hash_set<int> visited;
  // A queue of nodes to visit.
  std::deque<int> q({root});

  while (q.size()) {
    const int node = q.front();
    q.pop_front();

    // Mark the node as visited.
    visited.insert(node);

    // Emit the node in the serialized node list.
    serialized.push_back(node);

    // Add the unvisited outgoing control edges to the queue.
    const auto& outgoing_edges = forward_control_edges.find(node);
    if (outgoing_edges != forward_control_edges.end()) {
      for (const auto& successor : outgoing_edges->second) {
        if (visited.find(successor) == visited.end()) {
          q.push_back(successor);
        }
      }
    }
  }

  return serialized;
}

}  // namespace ml4pl
