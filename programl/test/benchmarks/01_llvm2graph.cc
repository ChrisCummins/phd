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

#include "boost/filesystem.hpp"
#include "labm8/cpp/bazelutil.h"
#include "labm8/cpp/fsutil.h"
#include "labm8/cpp/logging.h"
#include "programl/ir/llvm/llvm.h"
#include "programl/proto/program_graph.pb.h"

using std::vector;
namespace fs = boost::filesystem;

const fs::path irdir =
    labm8::BazelDataPathOrDie("phd/programl/test/data/llvm_ir");

vector<string> ReadIrs() {
  vector<string> irs;
  for (const auto& it : fs::directory_iterator(irdir)) {
    string ir;
    CHECK(labm8::fsutil::ReadFile(it, &ir).ok());
    irs.push_back(ir);
  }
  return irs;
}

int main(int argc, char** argv) {
  CHECK(argc == 1);

  vector<string> irs = ReadIrs();
  vector<programl::ProgramGraph> graphs(irs.size());

  for (size_t i = 0; i < irs.size(); ++i) {
    programl::ir::llvm::BuildProgramGraph(irs[i], &graphs[i]);
  }

  std::cout << "done";

  return 0;
}
