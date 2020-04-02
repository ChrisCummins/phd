#include "programl/graph/analysis/analysis.h"

#include "programl/graph/analysis/reachability.h"

namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status RunAnalysis(const string& analysisName, const ProgramGraph& graph,
                   ProgramGraphFeaturesList* featuresList) {
  if (analysisName == "reachability") {
    programl::graph::analysis::ReachabilityAnalysis reachability(graph);
    return reachability.Run(featuresList);
  } else {
    return Status(error::Code::INVALID_ARGUMENT, "Invalid analysis: {}",
                  analysisName);
  }
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
