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
#include <cstring>
#include <iomanip>
#include <iostream>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

#include "labm8/cpp/app.h"
#include "labm8/cpp/status.h"
#include "programl/graph/analysis/analysis.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"

using labm8::Status;
namespace error = labm8::error;

const char* usage = R"(Usage: analyze-graph

Run an analysis pass on a graph.
)";

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 2) {
    std::cerr << "Usage: graph-analysis <analysis>" << std::endl;
    return 4;
  }

  google::protobuf::io::IstreamInputStream istream(&std::cin);
  programl::ProgramGraph graph;
  if (!google::protobuf::TextFormat::Parse(&istream, &graph)) {
    std::cerr << "fatal: failed to parse ProgramGraph from stdin" << std::endl;
    return 3;
  }

  programl::ProgramGraphFeaturesList featuresList;
  Status status =
      programl::graph::analysis::RunAnalysis(argv[1], graph, &featuresList);
  if (!status.ok()) {
    std::cerr << "fatal: " << status.error_message() << std::endl;
    return 4;
  }

  std::cout << featuresList.DebugString();
  return 0;
}
