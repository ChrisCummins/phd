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

struct AdjacencyListOptions {
  bool control;
  bool reverse_control;
  bool data;
  bool reverse_data;
};

struct AdjacencyLists {
  vector<vector<int>> control;
  vector<vector<int>> reverse_control;
  vector<vector<int>> data;
  vector<vector<int>> reverse_data;
};

// A data flow analysis pass.
class DataFlowPass {
 public:
  explicit DataFlowPass(const ProgramGraph& graph)
      : graph_(graph){}

            [[nodiscard]] virtual Status
        Run(ProgramGraphFeaturesList * featuresList) = 0;

  const ProgramGraph& graph() const { return graph_; }

 protected:
  const AdjacencyLists& ComputeAdjacencies(const AdjacencyListOptions& options);
  const AdjacencyLists& adjacencies() const;

 private:
  const ProgramGraph& graph_;
  AdjacencyLists adjacencies_;
};

class InstructionRootDataFlowAnalysis : public DataFlowPass {
 public:
  InstructionRootDataFlowAnalysis(const ProgramGraph& graph)
      : InstructionRootDataFlowAnalysis(graph, 10) {}

  InstructionRootDataFlowAnalysis(const ProgramGraph& graph,
                                  int maxInstancesPerGraph)
      : DataFlowPass(graph),
        maxInstancesPerGraph_(maxInstancesPerGraph),
        seed_(std::chrono::system_clock::now().time_since_epoch().count()){}

            [[nodiscard]] virtual Status
        Run(ProgramGraphFeaturesList * featuresList) override;

  [[nodiscard]] virtual Status Init();

  int max_instances_per_graph() const { return maxInstancesPerGraph_; }

  unsigned seed() const { return seed_; }
  void seed(unsigned seed) { seed_ = seed; }

 protected:
  virtual Status RunOne(int rootNode, ProgramGraphFeatures* features) = 0;

 private:
  const int maxInstancesPerGraph_;
  unsigned seed_;
};

}  // namespace analysis
}  // namespace graph
}  // namespace programl
