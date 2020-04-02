#include "programl/graph/analysis/domtree.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "programl/graph/features.h"

#include <queue>
#include <utility>
#include <vector>

using std::pair;
using std::vector;
namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status DomtreeAnalysis::Run(int rootNode, ProgramGraphFeatures* features) {
  // TODO:
  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
