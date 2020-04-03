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

using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {

class ReachabilityAnalysis : public InstructionRootDataFlowAnalysis {
 public:
  using InstructionRootDataFlowAnalysis::InstructionRootDataFlowAnalysis;

  virtual Status RunOne(int rootNode, ProgramGraphFeatures* features) override;
};

}  // namespace analysis
}  // namespace graph
}  // namespace programl
