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
#include "programl/graph/format/node_link_graph.h"
#include "programl/proto/program_graph.pb.h"

const char* usage = R"(Usage: graph2json

Convert a ProgramGraph message to JSON node link graph.
)";

DEFINE_bool(pretty_print, false, "Pretty-print the generated JSON.");

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 1) {
    std::cerr << "Usage: graph2json" << std::endl;
    return 4;
  }

  google::protobuf::io::IstreamInputStream istream(&std::cin);
  programl::ProgramGraph graph;
  if (!google::protobuf::TextFormat::Parse(&istream, &graph)) {
    std::cerr << "fatal: failed to parse ProgramGraph from stdin\n";
    return 3;
  }

  auto nodeLinkGraph = json({});
  Status status = programl::graph::format::ProgramGraphToNodeLinkGraph(
      graph, &nodeLinkGraph);
  if (!status.ok()) {
    std::cerr << "fatal: failed to convert ProgramGraph to node link graph ("
              << status.error_message() << ')' << std::endl;
    return 2;
  }

  if (FLAGS_pretty_print) {
    std::cout << std::setw(2) << nodeLinkGraph << std::endl;
  } else {
    std::cout << nodeLinkGraph << std::endl;
  }

  return 0;
}
