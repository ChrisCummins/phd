// Generate a program graph from a HLO module.
//
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
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "programl/ir/xla/hlo_module_graph_builder.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"

#include <fstream>
#include <sstream>
#include <streambuf>

using labm8::Status;
namespace error = labm8::error;

static const char* usage = R"(
Generate program graph from a HLO module.

Read a HloProto message from file and print the program graph to stdout.

Tensorflow, JAX, Julia, and PyTorch can all be used as XLA frontends. To
run TensorFlow using XLA and dump HloProto files, run:

  $ TF_XLA_FLAGS=--tf_xla_auto_jit=2 \
    XLA_FLAGS="--xla_dump_hlo_as_proto --xla_dump_to=/tmp/hlo" \
    path/to/your/tf/program

Then read and convert the HloProto to a ProgramGraph using:

  $ xla2graph /tmp/hlo/module_0000.before_optimizations.hlo.pb
)";

labm8::StatusOr<xla::HloProto> GetProtoFromFileorSTDIN(const string& path) {
  xla::HloProto proto;

  // Read from stdin if path is '-'.
  if (path == "-") {
    google::protobuf::io::IstreamInputStream istream(&std::cin);
    if (!google::protobuf::TextFormat::Parse(&istream, &proto)) {
      return Status(error::INVALID_ARGUMENT, "Failed to parse HloProto");
    }
    return proto;
  }

  std::ifstream file(path, std::ios::ate);
  if (!file) {
    return Status(error::INVALID_ARGUMENT, "File not found: {}", path);
  }

  string str;
  str.reserve(file.tellg());
  file.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(file)),
             std::istreambuf_iterator<char>());

  if (!proto.ParseFromString(str)) {
    return Status(error::INVALID_ARGUMENT, "Failed to parse HloProto");
  }

  return proto;
}

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 2) {
    std::cerr << "Usage: xla2graph <filepath>" << std::endl;
    return 4;
  }

  auto protoOr = GetProtoFromFileorSTDIN(argv[1]);
  if (!protoOr.ok()) {
    std::cerr << protoOr.status().error_message() << std::endl;
    return 1;
  }

  programl::ir::xla::HloModuleGraphBuilder builder;
  auto graphOr = builder.Build(protoOr.ValueOrDie());
  if (!graphOr.ok()) {
    std::cerr << graphOr.status().error_message() << std::endl;
    return 1;
  }

  std::cout << graphOr.ValueOrDie().DebugString();

  return 0;
}
