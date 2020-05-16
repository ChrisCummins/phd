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
#include <iostream>

#include "programl/proto/checkpoint.pb.h"
#include "programl/proto/ir.pb.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"
#include "programl/proto/src.pb.h"

const char* usage = R"(Usage: pb2pbtxt <Protobuf>

Decode a binary protocol buffer to text format. For example, to
decode a ProgramGraph protocol buffer, use:

  $ pb2pbtxt ProgramGraph myfile.pb

Or to decode a IrList protocol buffer to a text file:

  $ pb2pbtxt IrList ir.pb > ir.pbtxt
)";

template <typename ProtocolBuffer>
void Decode() {
  ProtocolBuffer proto;
  if (!proto.ParseFromIstream(&std::cin)) {
    std::cerr << "fatal: failed to parse stdin";
    exit(3);
  }
  std::cout << proto.DebugString();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << usage;
    return 4;
  }

  const std::string name(argv[1]);

  if (name == "--help") {
    std::cerr << usage;
    return 4;
  }

  if (name == "ProgramGraph") {
    Decode<programl::ProgramGraph>();
  } else if (name == "ProgramGraphList") {
    Decode<programl::ProgramGraphList>();
  } else if (name == "ProgramGraphFeatures") {
    Decode<programl::ProgramGraphFeatures>();
  } else if (name == "ProgramGraphFeaturesList") {
    Decode<programl::ProgramGraphFeaturesList>();
  } else if (name == "Ir") {
    Decode<programl::Ir>();
  } else if (name == "IrList") {
    Decode<programl::IrList>();
  } else if (name == "SourceFile") {
    Decode<programl::SourceFile>();
  } else if (name == "Checkpoint") {
    Decode<programl::Checkpoint>();
  }
}
