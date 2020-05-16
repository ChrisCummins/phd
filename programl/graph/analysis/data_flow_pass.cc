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

#include "programl/graph/analysis/data_flow_pass.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

const AdjacencyLists& DataFlowPass::ComputeAdjacencies(
    const AdjacencyListOptions& options) {
  if (options.control) {
    adjacencies_.control.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.control.emplace_back();
    }
  }
  if (options.reverse_control) {
    adjacencies_.reverse_control.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.reverse_control.emplace_back();
    }
  }
  if (options.data) {
    adjacencies_.data.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.data.emplace_back();
    }
  }
  if (options.reverse_data) {
    adjacencies_.reverse_data.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.reverse_data.emplace_back();
    }
  }

  for (int i = 0; i < graph().edge_size(); ++i) {
    const Edge& edge = graph().edge(i);
    if (edge.flow() == Edge::CONTROL) {
      if (options.control) {
        adjacencies_.control[edge.source()].push_back(edge.target());
      }
      if (options.reverse_control) {
        adjacencies_.reverse_control[edge.target()].push_back(edge.source());
      }
    } else if (edge.flow() == Edge::DATA) {
      if (options.data) {
        adjacencies_.data[edge.source()].push_back(edge.target());
      }
      if (options.reverse_data) {
        adjacencies_.reverse_data[edge.target()].push_back(edge.source());
      }
    }
  }

  return adjacencies_;
}

const AdjacencyLists& DataFlowPass::adjacencies() const { return adjacencies_; }

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

  RETURN_IF_ERROR(Init());

  int numRoots = std::min(static_cast<int>(ceil(rootNodes.size() / 10.0)),
                          max_instances_per_graph());
  for (int i = 0; i < numRoots; ++i) {
    ProgramGraphFeatures features;
    RETURN_IF_ERROR(RunOne(rootNodes[i], &features));
    *featuresList->add_graph() = features;
  }

  return Status::OK;
}

Status InstructionRootDataFlowAnalysis::Init() {
  LOG(INFO) << "Base init";
  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
