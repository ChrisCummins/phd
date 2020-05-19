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
#include "labm8/cpp/strutil.h"

#include "programl/graph/analysis/domtree.h"
#include "programl/graph/analysis/reachability.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"
#include "programl/task/dataflow/dataset/parallel_file_map.h"

#include "absl/strings/str_format.h"

namespace fs = boost::filesystem;
using std::vector;

const char* usage = R"(Create dataflow labels for program graphs.)";

DEFINE_string(
    path,
    (labm8::fsutil::GetHomeDirectoryOrDie() / "programl/dataflow").string(),
    "The directory to write generated files to.");

namespace programl {
namespace task {
namespace dataflow {

// Instantiate and run an analysis on the given graph, writing resulting
// features to file.
template <typename Analysis>
void RunAnalysis(const ProgramGraph& graph, const string& root,
                 const string& analysisName, const string& nameStem) {
  const string outPath =
      absl::StrFormat("%s/labels/%s/%sProgramGraphFeaturesList.pb", root,
                      analysisName, nameStem);

  // Do nothing if the file already exists. This is to allow for incremental
  // runs in which the dataset is only partially exported.
  if (labm8::fsutil::FileExists(outPath)) {
    return;
  }

  ProgramGraphFeaturesList labels;
  Analysis analysis(graph);
  CHECK(analysis.Run(&labels).ok());
  auto status = analysis.Run(&labels);
  // If the analysis failed, log the error and move on.
  if (!status.ok()) {
    std::cerr << "\r\033[K" << analysisName << " analysis failed on "
              << nameStem << "ProgramGraph.pb: 🙈 " << status.error_message()
              << std::endl
              << std::flush;
    return;
  }

  // Only write the files if features were created.
  if (labels.graph_size()) {
    std::ofstream out(outPath);
    labels.SerializeToOstream(&out);
  }
}

void ProcessProgramGraph(const fs::path& root, const fs::path& path) {
  std::ifstream file(path.string());
  ProgramGraph graph;
  CHECK(graph.ParseFromIstream(&file));

  const string baseName = path.string().substr(path.string().rfind("/") + 1);
  const string nameStem =
      baseName.substr(0, baseName.size() - labm8::StrLen("ProgramGraph.pb"));

  // Run the analyses to produce graph features.
  RunAnalysis<graph::analysis::ReachabilityAnalysis>(graph, root.string(),
                                                     "reachability", nameStem);
  RunAnalysis<graph::analysis::DomtreeAnalysis>(graph, root.string(), "domtree",
                                                nameStem);
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
  const vector<fs::path> files =
      programl::task::dataflow::EnumerateProgramGraphFiles(path / "graphs");

  programl::task::dataflow::ParallelFileMap<
      programl::task::dataflow::ProcessProgramGraph, 16>(path, files);
  LOG(INFO) << "done";

  return 0;
}
