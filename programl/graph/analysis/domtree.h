#pragma once

#include "labm8/cpp/status.h"
#include "programl/graph/analysis/data_flow_pass.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {

class DomtreeAnalysis : public InstructionRootDataFlowAnalysis {
 public:
  using InstructionRootDataFlowAnalysis::InstructionRootDataFlowAnalysis;

  Status Run(int rootNode, ProgramGraphFeatures* features);
};

}  // namespace analysis
}  // namespace graph
}  // namespace programl
