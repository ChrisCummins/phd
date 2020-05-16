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

#include "programl/graph/analysis/analysis.h"

#include "programl/graph/analysis/liveness.h"
#include "programl/graph/analysis/reachability.h"

namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status RunAnalysis(const string& analysisName, const ProgramGraph& graph,
                   ProgramGraphFeaturesList* featuresList) {
  if (analysisName == "reachability") {
    ReachabilityAnalysis analysis(graph);
    return analysis.Run(featuresList);
  } else if (analysisName == "liveness") {
    LivenessAnalysis analysis(graph);
    return analysis.Run(featuresList);
  } else {
    return Status(error::Code::INVALID_ARGUMENT, "Invalid analysis: {}",
                  analysisName);
  }
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl