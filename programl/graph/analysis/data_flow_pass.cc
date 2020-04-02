#include "programl/graph/analysis/data_flow_pass.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>

#include "labm8/cpp/status_macros.h"

namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

namespace {

// Compute a Control Flow Graph as an adjacency list.
vector<vector<int>> GetControlAdjacencies(const ProgramGraph& graph) {
  vector<vector<int>> adjacencies;
  adjacencies.reserve(graph.node_size());
  for (int i = 0; i < graph.node_size(); ++i) {
    adjacencies.emplace_back();
  }

  for (int i = 0; i < graph.edge_size(); ++i) {
    const Edge& edge = graph.edge(i);
    if (edge.flow() == Edge::CONTROL) {
      adjacencies[edge.source()].push_back(edge.target());
    }
  }
  return adjacencies;
}

}  // anonymous namespace

const vector<vector<int>>& DataFlowPass::control_adjacencies() {
  if (!controlAdjacencies_.size()) {
    controlAdjacencies_ = GetControlAdjacencies(graph());
  }
  return controlAdjacencies_;
}

Status InstructionRootDataFlowAnalysis::Run(
    ProgramGraphFeaturesList* featuresList) {
  vector<int> rootNodes;

  for (int i = 1; i < graph().node_size(); ++i) {
    if (graph().node(i).type() == Node::INSTRUCTION) {
      rootNodes.push_back(i);
    }
  }
  if (!rootNodes.size()) {
    return Status(error::Code::FAILED_PRECONDITION, "No valid root nodes");
  }
  std::shuffle(rootNodes.begin(), rootNodes.end(),
               std::default_random_engine(seed()));

  int numRoots = std::min(static_cast<int>(ceil(rootNodes.size() / 10.0)),
                          max_instances_per_graph());
  for (int i = 0; i < numRoots; ++i) {
    ProgramGraphFeatures features;
    RETURN_IF_ERROR(RunOne(rootNodes[i], &features));
    *featuresList->add_graph() = features;
  }

  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
