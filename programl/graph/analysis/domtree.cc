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

#include "programl/graph/analysis/domtree.h"

#include "labm8/cpp/status.h"
#include "programl/graph/features.h"

#include <utility>

using std::pair;
using std::vector;
namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status DomtreeAnalysis::Init() {
  ComputeAdjacencies({.reverse_control = true});
  return Status::OK;
}

std::pair<int, absl::flat_hash_map<int, absl::flat_hash_set<int>>>
DomtreeAnalysis::ComputeDominators(const int rootNode) {
  const int function = graph().node(rootNode).function();
  const auto& rcfg = adjacencies().reverse_control;

  // Because a node may only be dominated by a node from within the same
  // function, we need only consider the statements nodes within the same
  // function as the root node.
  vector<int> instructionsInFunction;
  for (int i = 0; i < graph().node_size(); ++i) {
    if (i && graph().node(i).function() == function) {
      instructionsInFunction.push_back(i);
    }
  }

  // Initialize the dominator sets. These map nodes to the set of nodes that
  // dominate it.
  absl::flat_hash_set<int> initialDominators{instructionsInFunction.begin(),
                                             instructionsInFunction.end()};
  initialDominators.erase(rootNode);

  absl::flat_hash_map<int, absl::flat_hash_set<int>> dominators;
  for (const auto& node : instructionsInFunction) {
    if (node == rootNode) {
      dominators[node].insert(rootNode);
    } else {
      dominators[node] = initialDominators;
    }
  }

  bool changed = true;
  int dataFlowSteps = 0;
  while (changed) {
    changed = false;
    for (const auto& node : instructionsInFunction) {
      if (node == rootNode) {
        continue;
      }

      // Get the predecessor nodes.
      const auto& predecessors = rcfg[node];

      // Intersect the dominators of all predecessors.
      absl::flat_hash_map<int, int> domPred;
      for (const auto& predecessor : predecessors) {
        for (const auto& d : dominators[predecessor]) {
          ++domPred[d];
        }
      }
      absl::flat_hash_set<int> newDom;
      newDom.insert(node);
      for (const auto& it : domPred) {
        if (it.second == predecessors.size()) {
          newDom.insert(it.first);
        }
      }

      if (newDom != dominators[node]) {
        ++dataFlowSteps;
        dominators[node] = newDom;
        changed = true;
      }
    }
  }

  return {dataFlowSteps, dominators};
}

Status DomtreeAnalysis::RunOne(int rootNode, ProgramGraphFeatures* features) {
  Feature falseFeature = CreateFeature(0);
  Feature trueFeature = CreateFeature(1);

  const auto result = ComputeDominators(rootNode);
  const int stepCount = result.first;
  const auto& dominators = result.second;

  int activeNodeCount = 0;
  for (int i = 0; i < graph().node_size(); ++i) {
    AddNodeFeature(features, "data_flow_root_node",
                   i == rootNode ? trueFeature : falseFeature);

    auto it = dominators.find(i);
    if (it != dominators.end() &&
        it->second.find(rootNode) != it->second.end()) {
      ++activeNodeCount;
      AddNodeFeature(features, "data_flow_value", trueFeature);
    } else {
      AddNodeFeature(features, "data_flow_value", falseFeature);
    }
  }

  SetFeature(features->mutable_features(), "data_flow_step_count",
             CreateFeature(stepCount));
  SetFeature(features->mutable_features(), "data_flow_active_node_count",
             CreateFeature(activeNodeCount));

  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
