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

#include "programl/graph/analysis/reachability.h"

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

Status ReachabilityAnalysis::RunOne(int rootNode,
                                    ProgramGraphFeatures* features) {
  Feature falseFeature = CreateFeature(0);
  Feature trueFeature = CreateFeature(1);

  Features notRootFeatures;
  SetFeature(&notRootFeatures, "data_flow_root_node", falseFeature);
  SetFeature(&notRootFeatures, "data_flow_value", falseFeature);

  Features rootNodeFeatures;
  SetFeature(&rootNodeFeatures, "data_flow_root_node", trueFeature);

  // Create the selector node features.
  for (int i = 0; i < graph().node_size(); ++i) {
    (*(*features->mutable_node_features()
            ->mutable_feature_list())["data_flow_root_node"]
          .add_feature()) = i == rootNode ? trueFeature : falseFeature;
    (*(*features->mutable_node_features()
            ->mutable_feature_list())["data_flow_value"]
          .add_feature()) = falseFeature;
  }

  int dataFlowStepCount = 0;
  vector<bool> visited(graph().node_size(), false);
  std::queue<pair<int, int>> q;
  q.push({rootNode, 0});

  const vector<vector<int>>& adjacencies = control_adjacencies();

  int activeNodeCount = 0;
  while (!q.empty()) {
    int current = q.front().first;
    dataFlowStepCount = q.front().second;
    q.pop();

    visited[current] = true;
    ++activeNodeCount;

    (*(*features->mutable_node_features()
            ->mutable_feature_list())["data_flow_value"]
          .mutable_feature(current)) = trueFeature;

    for (int neighbour : adjacencies[current]) {
      if (!visited[neighbour]) {
        q.push({neighbour, dataFlowStepCount + 1});
      }
    }
  }

  SetFeature(features->mutable_features(), "data_flow_step_count",
             CreateFeature(dataFlowStepCount));
  SetFeature(features->mutable_features(), "data_flow_active_node_count",
             CreateFeature(activeNodeCount));

  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
