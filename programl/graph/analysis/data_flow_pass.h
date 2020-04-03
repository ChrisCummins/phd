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

#include <chrono>
#include <vector>

#include "labm8/cpp/status.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"

using labm8::Status;
using std::vector;

namespace programl {
namespace graph {
namespace analysis {

// A data flow analysis pass.
class DataFlowPass {
 public:
  explicit DataFlowPass(const ProgramGraph& graph) : DataFlowPass(graph, 10) {}

  DataFlowPass(const ProgramGraph& graph, int maxInstancesPerGraph)
      : graph_(graph),
        maxInstancesPerGraph_(maxInstancesPerGraph),
        seed_(std::chrono::system_clock::now().time_since_epoch().count()){}

            [[nodiscard]] virtual Status
        Run(ProgramGraphFeaturesList * featuresList) = 0;

  int max_instances_per_graph() const { return maxInstancesPerGraph_; }

  const ProgramGraph& graph() const { return graph_; }

  const vector<vector<int>>& control_adjacencies();

  unsigned seed() const { return seed_; }
  void seed(unsigned seed) { seed_ = seed; }

 private:
  const ProgramGraph& graph_;
  const int maxInstancesPerGraph_;
  unsigned seed_;
  vector<vector<int>> controlAdjacencies_;
};

class InstructionRootDataFlowAnalysis : public DataFlowPass {
 public:
  using DataFlowPass::DataFlowPass;

  [[nodiscard]] virtual Status Run(
      ProgramGraphFeaturesList* featuresList) override;

 protected:
  virtual Status RunOne(int rootNode, ProgramGraphFeatures* features) = 0;
};

}  // namespace analysis
}  // namespace graph
}  // namespace programl
