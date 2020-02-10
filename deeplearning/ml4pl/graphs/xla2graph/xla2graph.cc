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
#include "tensorflow/compiler/xla/service/hlo.pb.h"

#include "deeplearning/ml4pl/graphs/graphviz_converter.h"
#include "deeplearning/ml4pl/graphs/xla2graph/hlo_module_graph_builder.h"
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"

#include <fstream>
#include <sstream>
#include <streambuf>

static const char* usage =
    "Generate program graph from a HLO module.\n"
    "\n"
    "Read a HloProto message from file and print the program graph to stdout.\n"
    "\n"
    "Tensorflow, JAX, Julia, and PyTorch can all be used as XLA frontends. To\n"
    "run TensorFlow using XLA and dump HloProto files, run:\n"
    "\n"
    "  $ TF_XLA_FLAGS=--tf_xla_auto_jit=2 \\\n"
    "    XLA_FLAGS=\"--xla_dump_hlo_as_proto --xla_dump_to=/tmp/hlo\" \\\n"
    "    path/to/your/tf/program\n"
    "\n"
    "Then read and convert the HloProto to a ProgramGraphProto using:\n"
    "\n"
    "  $ xla2graph /tmp/hlo/module_0000.before_optimizations.hlo.pb\n"
    "\n"
    "The output format is the textual representation of a ProgramGraphProto \n"
    "protocol buffer. A binary protocol buffer can be printed using:\n"
    "\n"
    "  $ xla2graph /tmp/hlo/module_0000.before_optimizations.hlo.pb \\\n"
    "        --stdout_fmt=pb > /tmp/llvm.pb\n"
    "\n"
    "Or a dot string using:\n"
    "\n"
    "  $ xla2graph /tmp/hlo/module_0000.before_optimizations.hlo.pb \\\n"
    "        --stdout_fmt=dot > /tmp/module_0000.before_optimizations.dot\n"
    "\n"
    "The output can then be processed by Graphviz.";

DEFINE_string(stdin_fmt, "pb",
              "The type of input format to use. Valid options are: "
              "\"pb\" which reads binary protocol buffers, or \"pbtxt\" which "
              "reads a text format protocol buffer.");

// Assert that the stdin format is legal.
static bool ValidateStdinFormat(const char* flagname, const string& value) {
  if (value == "pb" || value == "pbtxt") {
    return true;
  }

  LOG(FATAL) << "Unknown --" << flagname << ": `" << value << "`. Supported "
             << "formats: pb,pbtxt";
  return false;
}
DEFINE_validator(stdin_fmt, &ValidateStdinFormat);

DEFINE_string(stdout_fmt, "pbtxt",
              "The type of output format to use. Valid options are: "
              "\"pb\" which prints binary protocol buffer, \"pbtxt\" which "
              "prints a text format protocol buffer, or \"dot\" which prints a "
              "graphviz dot string.");

// Assert that the stdout format is legal.
static bool ValidateStdoutFormat(const char* flagname, const string& value) {
  if (value == "pb" || value == "pbtxt" || value == "dot") {
    return true;
  }

  LOG(FATAL) << "Unknown --" << flagname << ": `" << value << "`. Supported "
             << "formats: pb,pbtxt,dot";
  return false;
}
DEFINE_validator(stdout_fmt, &ValidateStdoutFormat);

labm8::StatusOr<string> GetFileOrSTDIN(const string& path) {
  std::ifstream file(path, std::ios::ate);
  if (!file) {
    std::stringstream err;
    err << "File not found: " << path;
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT, err.str());
  }

  string str;
  str.reserve(file.tellg());
  file.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(file)),
             std::istreambuf_iterator<char>());

  return str;
}

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 2) {
    std::cerr << "Usage: xla2graph <filepath>" << std::endl;
    return 4;
  }

  auto serializedProtoOr = GetFileOrSTDIN(argv[1]);
  if (!serializedProtoOr.ok()) {
    std::cerr << serializedProtoOr.status().error_message() << std::endl;
    return 1;
  }

  xla::HloProto proto;
  if (!proto.ParseFromString(serializedProtoOr.ValueOrDie())) {
    std::cerr << "Failed to parse HloProto\n";
    return 1;
  }

  ml4pl::HloModuleGraphBuilder builder;
  auto graphOr = builder.Build(proto);
  if (!graphOr.ok()) {
    std::cerr << graphOr.status().error_message() << std::endl;
    return 1;
  }

  const auto graph = graphOr.ValueOrDie();
  if (FLAGS_stdout_fmt == "pb") {
    graph.SerializeToOstream(&std::cout);
  } else if (FLAGS_stdout_fmt == "pbtxt") {
    std::cout << graph.DebugString();
  } else if (FLAGS_stdout_fmt == "dot") {
    ml4pl::SerializeGraphVizToString(graph, &std::cout);
  } else {
    LOG(FATAL) << "unreachable";
  }

  return 0;
}
