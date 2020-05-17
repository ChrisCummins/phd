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
#include "labm8/cpp/logging.h"
#include "programl/graph/analysis/domtree.h"
#include "programl/graph/analysis/reachability.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"

#include "absl/strings/str_format.h"

#include "tbb/parallel_for.h"

namespace fs = boost::filesystem;
using std::vector;

const char* usage = R"(Create dataflow labels for program graphs.)";

DEFINE_int32(limit, 0,
             "If --limit > 0, limit the number of input graphs processed to "
             "this number.");
DEFINE_string(path, "/tmp/programl/poj104",
              "The directory to write generated files to.");

namespace programl {
namespace task {
namespace dataflow {

inline bool EndsWith(const string& value, const string& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

vector<fs::path> EnumerateProgramGraphFiles(const fs::path& root) {
  vector<fs::path> files;
  for (auto it : fs::directory_iterator(root)) {
    if (EndsWith(it.path().string(), ".ProgramGraph.pb")) {
      files.push_back(it.path());
    }
  }

  // Randomize the order of files to crudely load balancing a
  // bunch of parallel workers iterating through this list in order
  // as the there is a high variance in the size / complexity of files.
  std::srand(unsigned(std::time(0)));
  std::random_shuffle(files.begin(), files.end());
  return files;
}

int constexpr StrLen(const char* str) { return *str ? 1 + StrLen(str + 1) : 0; }

// Return true if the given file exists.
inline bool FileExists(const string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

// Instantiate and run an analysis on the given graph, writing resulting
// features to file.
template <typename Analysis>
void RunAnalysis(const ProgramGraph& graph, const string& root,
                 const string& analysisName, const string& nameStem) {
  const string outPath = absl::StrFormat("%s/%s/%sProgramGraphFeaturesList.pb",
                                         root, analysisName, nameStem);

  // Do nothing if the file already exists. This is to allow for incremental
  // runs in which the dataset is only partially exported.
  if (FileExists(outPath)) {
    return;
  }

  ProgramGraphFeaturesList labels;
  Analysis analysis(graph);
  CHECK(analysis.Run(&labels).ok());
  // Only write the files if features were exported.
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
      baseName.substr(0, baseName.size() - StrLen("ProgramGraph.pb"));

  // Run the analyses to produce graph features.
  RunAnalysis<graph::analysis::ReachabilityAnalysis>(graph, root.string(),
                                                     "reachability", nameStem);
  RunAnalysis<graph::analysis::DomtreeAnalysis>(graph, root.string(), "domtree",
                                                nameStem);
}

std::chrono::milliseconds Now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
}

void CreateDataflowLabels(const fs::path& path) {
  // Create a directory for each of the analyses that we will run.
  fs::create_directory(path / "reachability");
  fs::create_directory(path / "domtree");
  std::chrono::milliseconds startTime = Now();

  const vector<fs::path> files = EnumerateProgramGraphFiles(path / "graphs");

  std::atomic_uint64_t fileCount{0};

  // The size of file path chunks to execute in worker thread inner loops.
  // A larger chunk size creates more infrequent status updates.
  const int chunkSize = 16;

  const size_t n = FLAGS_limit
                       ? std::min(size_t(files.size()), size_t(FLAGS_limit))
                       : files.size();

#pragma omp parallel for
  for (size_t j = 0; j < n; j += chunkSize) {
    for (size_t i = j; i < std::min(n, j + chunkSize); ++i) {
      ProcessProgramGraph(path, files[i]);
    }
    fileCount += chunkSize;
    uint64_t localFileCount = fileCount;
    std::chrono::milliseconds now = Now();
    int msPerGraph = ((now - startTime) / localFileCount).count();
    std::cout << "\r\033[K" << localFileCount << " of " << n
              << " files processed (" << msPerGraph << " ms / graph, "
              << std::setprecision(3)
              << (localFileCount / static_cast<float>(n)) * 100 << "%)"
              << std::flush;
  }
  std::cout << std::endl;
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
  programl::task::dataflow::CreateDataflowLabels(fs::path(FLAGS_path));
  LOG(INFO) << "done";

  return 0;
}
