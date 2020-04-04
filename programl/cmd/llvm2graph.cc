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
#include <iomanip>
#include <iostream>

#include "labm8/cpp/app.h"
#include "labm8/cpp/statusor.h"
#include "programl/ir/llvm/llvm.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_options.pb.h"

#include "llvm/Support/SourceMgr.h"

using labm8::StatusOr;

const char* usage = R"(Generate program graph from an IR.

Read an LLVM-IR module from file and print the program graph to stdout:

  $ llvm2graph /path/to/llvm.ll

Use '-' to read the input from stdin:

  $ clang foo.c -emit-llvm -o - | llvm2graph -
)";

DEFINE_bool(instructions_only, false,
            "Include only instructions in the generated program graph.");
DEFINE_bool(ignore_call_returns, false,
            "Include only instructions in the generated program graph.");

StatusOr<programl::ProgramGraphOptions> GetProgramGraphOptionsFromFlags() {
  programl::ProgramGraphOptions options;
  if (FLAGS_instructions_only) {
    options.set_instructions_only(true);
  }
  if (FLAGS_ignore_call_returns) {
    options.set_ignore_call_returns(true);
  }
  return options;
}

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 2) {
    std::cerr << "Usage: llvm2graph <filepath>" << std::endl;
    return 4;
  }

  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (!buffer) {
    std::cerr << "File not found: " << argv[1] << std::endl;
    return 1;
  }

  auto options = GetProgramGraphOptionsFromFlags();
  if (!options.ok()) {
    std::cerr << options.status().error_message() << std::endl;
    return 4;
  }

  programl::ProgramGraph graph;
  Status status = programl::ir::llvm::BuildProgramGraph(*buffer.get(), &graph,
                                                        options.ValueOrDie());
  if (!status.ok()) {
    std::cerr << status.error_message() << std::endl;
    return 2;
  }

  std::cout << graph.DebugString();

  return 0;
}
