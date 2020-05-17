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
#include <sys/stat.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "boost/filesystem.hpp"
#include "labm8/cpp/app.h"
#include "labm8/cpp/fsutil.h"
#include "labm8/cpp/logging.h"

#include "programl/graph/format/cdfg.h"
#include "programl/proto/node.pb.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"
#include "programl/task/dataflow/dataset/graph_map.h"

#include "absl/strings/str_format.h"

namespace fs = boost::filesystem;
using std::vector;

const char* usage = R"(Create CDFGs from program graphs.)";

DEFINE_string(
    path,
    (labm8::fsutil::GetHomeDirectoryOrDie() / "programl/dataflow").string(),
    "The directory to write generated files to.");

namespace programl {
namespace task {
namespace dataflow {

void ProcessProgramGraph(const fs::path& root, const fs::path& path) {
  const string baseName = path.string().substr(path.string().rfind("/") + 1);
  const string outPath = absl::StrFormat("%s/cdfg/%s", root.string(), baseName);

  if (FileExists(outPath)) {
    return;
  }

  const string nameStem =
      baseName.substr(0, baseName.size() - StrLen("ProgramGraph.pb"));
  const string nodeIndexPath =
      absl::StrFormat("%s/cdfg/%sNodeIndexList.pb", root.string(), nameStem);

  std::ifstream file(path.string());
  ProgramGraph graph;
  CHECK(graph.ParseFromIstream(&file));

  graph::format::CDFGBuilder builder;
  ProgramGraph cdfg = builder.Build(graph);

  NodeIndexList nodeList;
  for (const auto& nodeIndex : builder.GetNodeList()) {
    nodeList.add_node(nodeIndex);
  }

  std::ofstream out(outPath);
  cdfg.SerializeToOstream(&out);

  std::ofstream nodeListOut(nodeIndexPath);
  nodeList.SerializeToOstream(&nodeListOut);
}

}  // namespace dataflow
}  // namespace task
}  // namespace programl

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv);
  if (argc > 1) {
    std::cerr << "fatal: Unrecognized arguments" << std::endl;
    return 4;
  }

  const fs::path path(FLAGS_path);
  fs::create_directory(path / "cdfg");

  programl::task::dataflow::ParallelMap<
      programl::task::dataflow::ProcessProgramGraph, 128>(path);
  LOG(INFO) << "done";

  return 0;
}
