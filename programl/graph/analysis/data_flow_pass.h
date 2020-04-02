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
