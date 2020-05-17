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

#include "programl/graph/analysis/liveness.h"

#include "labm8/cpp/logging.h"
#include "programl/graph/features.h"

#include <queue>

using std::pair;
using std::vector;
namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status LivenessAnalysis::Init() {
  ComputeAdjacencies({.control = true,
                      .reverse_control = true,
                      .data = true,
                      .reverse_data = true});

  // Pre-compute all live-out sets.
  liveInSets_.reserve(graph().node_size());
  liveOutSets_.reserve(graph().node_size());
  for (int i = 0; i < graph().node_size(); ++i) {
    liveInSets_.push_back({});
    liveOutSets_.push_back({});
  }

  const auto& cfg = adjacencies().control;
  const auto& rcfg = adjacencies().reverse_control;
  const auto& dfg = adjacencies().data;
  const auto& rdfg = adjacencies().reverse_data;
  DCHECK(cfg.size() == graph().node_size())
      << "CFG size: " << cfg.size() << " != "
      << " graph size: " << graph().node_size();

  std::queue<int> workList;
  absl::flat_hash_set<int> workSet;

  // Liveness is computed backwards starting from the program exit.
  // Build a work list of exit nodes to begin from.
  for (int i = 0; i < graph().node_size(); ++i) {
    if (graph().node(i).type() == Node::INSTRUCTION && !cfg[i].size()) {
      LOG(INFO) << "Node " << i << " is exit";
      workList.push(i);
      workSet.insert(i);
    }
  }
  LOG(INFO) << "Worklist init " << workList.size();

  // A graph may not have any exit nodes.
  if (workList.empty()) {
    return Status::OK;
  }

  dataFlowStepCount_ = 0;
  while (!workList.empty()) {
    ++dataFlowStepCount_;
    int node = workList.front();
    workList.pop();
    workSet.erase(node);

    // Get immediate control and data neighbors.
    const auto& successors = cfg[node];
    const auto& predecessors = rcfg[node];
    const auto& defs = dfg[node];
    const auto& uses = rdfg[node];

    // LiveOut(n) = U {LiveIn(p) for p in succ(n)}
    absl::flat_hash_set<int> newOutSet{};
    for (const auto& successor : successors) {
      newOutSet.merge(liveInSets_[successor]);
    }

    // LiveIn(n) = Gen(n) U {LiveOut(n) - Kill(n)}
    absl::flat_hash_set<int> newInSet{uses.begin(), uses.end()};
    absl::flat_hash_set<int> newOutSetMinusDefs = newOutSet;
    for (const auto& d : defs) {
      newOutSetMinusDefs.erase(d);
    }
    newInSet.merge(newOutSetMinusDefs);

    // No need to visit predecessors if the in-set is non-empty and has not
    // changed.
    if (newInSet != liveInSets_[node]) {
      for (const auto& predecessor : predecessors) {
        if (!workSet.contains(predecessor)) {
          workSet.insert(predecessor);
          workList.push(predecessor);
        }
      }
      liveInSets_[node] = newInSet;
    }

    liveOutSets_[node] = newOutSet;
  }

  return Status::OK;
}

Status LivenessAnalysis::RunOne(int rootNode, ProgramGraphFeatures* features) {
  if (rootNode < 0 || rootNode >= graph().node_size()) {
    return Status(error::INVALID_ARGUMENT, "Root node is out-of-range");
  }
  if (graph().node(rootNode).type() != Node::INSTRUCTION) {
    return Status(error::INVALID_ARGUMENT, "Root node must be an instruction");
  }

  Feature falseFeature = CreateFeature(0);
  Feature trueFeature = CreateFeature(1);

  Features notRootFeatures;
  SetFeature(&notRootFeatures, "data_flow_root_node", falseFeature);
  SetFeature(&notRootFeatures, "data_flow_value", falseFeature);

  Features rootNodeFeatures;
  SetFeature(&rootNodeFeatures, "data_flow_root_node", trueFeature);

  // We have already pre-computed the live-out sets, so just add the
  // annotations.
  const auto& outSet = liveOutSets_[rootNode];
  int dataFlowActiveNodeCount = 0;
  for (int i = 0; i < graph().node_size(); ++i) {
    AddNodeFeature(features, "data_flow_root_node",
                   i == rootNode ? trueFeature : falseFeature);
    if (outSet.contains(i)) {
      ++dataFlowActiveNodeCount;
      AddNodeFeature(features, "data_flow_value", trueFeature);
    } else {
      AddNodeFeature(features, "data_flow_value", falseFeature);
    }
  }

  AddScalarFeature(features, "data_flow_step_count", dataFlowStepCount_);
  AddScalarFeature(features, "data_flow_active_node_count",
                   dataFlowActiveNodeCount);

  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
