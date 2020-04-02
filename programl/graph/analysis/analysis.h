#pragma once

#include "labm8/cpp/status.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {

Status RunAnalysis(const string& analysisName, const ProgramGraph& graph,
                   ProgramGraphFeaturesList* featuresList);

}  // namespace analysis
}  // namespace graph
}  // namespace programl
