#pragma once

#include <vector>

#include "deeplearning/ml4pl/graphs/programl.pb.h"

#include "absl/container/flat_hash_map.h"

namespace ml4pl {

// Serialize the statements in a graph, returning a list of node indices,
// ordered by a depth first traversal of the statements in each.
std::vector<int> SerializeStatements(const ProgramGraph& graph);

// Serialize the statements in a function using a depth first traversal of the
// forward control edges starting at the given root node.
std::vector<int> SerializeStatements(
    const int& root,
    const absl::flat_hash_map<int, std::vector<int>>& forward_control_edges);

// Serialize the statements in a graph, returning a list of node indices,
// ordered by a depth first traversal of each functions.
std::vector<std::vector<int>> SerializeStatementsByIdentifier(
    const ProgramGraph& graph);

}  // namespace ml4pl
