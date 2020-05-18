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

#include "programl/graph/format/graph_serializer.h"
#include "programl/proto/node.pb.h"
#include "programl/proto/program_graph.pb.h"

const char* usage = R"(Usage: graph2seq < program_graph.pbtxt

Read a ProgramGraph message from stdin and print a serialized list
of instruction node indices to stdout.
)";

int main(int argc, char** argv) {
  if (argc == 2 && !strcmp(argv[1], "--help")) {
    std::cerr << usage;
    return 0;
  }

  if (argc != 1) {
    std::cerr << usage;
    return 4;
  }

  google::protobuf::io::IstreamInputStream istream(&std::cin);
  programl::ProgramGraph graph;
  if (!google::protobuf::TextFormat::Parse(&istream, &graph)) {
    std::cerr << "fatal: failed to parse ProgramGraph from stdin";
    return 3;
  }

  programl::NodeIndexList serialized;
  programl::graph::format::SerializeInstructionsInProgramGraph(graph,
                                                               &serialized);
  std::cout << serialized.DebugString();

  return 0;
}
