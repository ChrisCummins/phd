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
