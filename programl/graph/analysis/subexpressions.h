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

#include "labm8/cpp/status.h"
#include "programl/graph/analysis/data_flow_pass.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace programl {
namespace graph {
namespace analysis {

using absl::flat_hash_map;
using absl::flat_hash_set;
using labm8::Status;
using std::pair;
using std::vector;

// Common subexpressions analysis.
//
// Starting at instruction `n`, label all instructions which compute the same
// expression.
class SubexpressionsAnalysis : public RoodNodeDataFlowAnalysis {
 public:
  using RoodNodeDataFlowAnalysis::RoodNodeDataFlowAnalysis;

  virtual Status RunOne(int rootNode, ProgramGraphFeatures* features) override;

  virtual vector<int> GetEligibleRootNodes() override;

  virtual Status Init() override;

  string ToString() const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const SubexpressionsAnalysis& dt);

  // Return a list of instruction node indices which compute the same
  // subexpression.
  const vector<flat_hash_set<int>>& subexpression_sets() const {
    return subexpressionSets_;
  };

 private:
  vector<flat_hash_set<int>> subexpressionSets_;
};

}  // namespace analysis
}  // namespace graph
}  // namespace programl
