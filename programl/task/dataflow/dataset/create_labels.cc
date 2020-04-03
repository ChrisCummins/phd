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
#include <fstream>
#include <iomanip>
#include <iostream>

#include "boost/filesystem.hpp"
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "programl/graph/analysis/reachability.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"

#include "absl/strings/str_format.h"

#include "tbb/parallel_for.h"

namespace fs = boost::filesystem;
using std::vector;

const char* usage = R"(Create dataflow labels for program graphs.)";

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
  return files;
}

int constexpr StrLen(const char* str) { return *str ? 1 + StrLen(str + 1) : 0; }

void ProcessProgramGraph(const fs::path& root, const fs::path& path) {
  std::ifstream file(path.string());
  ProgramGraph graph;
  CHECK(graph.ParseFromIstream(&file));

  const string baseName = path.string().substr(path.string().rfind("/") + 1);
  const string nameStem =
      baseName.substr(0, baseName.size() - StrLen("ProgramGraph.pb"));

  // Reachability labels.
  {
    ProgramGraphFeaturesList labels;
    graph::analysis::ReachabilityAnalysis analysis(graph);
    CHECK(analysis.Run(&labels).ok());
    if (labels.graph_size()) {
      const string outPath =
          absl::StrFormat("%s/reachability/%sProgramGraphFeaturesList.pb",
                          root.string(), nameStem);
      std::ofstream out(outPath);
      labels.SerializeToOstream(&out);
    }
  }
}

std::chrono::microseconds Now() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch());
}

void CreateDataflowLabels(const fs::path& path) {
  fs::create_directory(path / "reachability");
  std::chrono::microseconds startTime = Now();

  const vector<fs::path> files = EnumerateProgramGraphFiles(path / "graphs");

  std::atomic_uint64_t fileCount{0};
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, files.size()),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t index = r.begin(); index != r.end(); ++index) {
          ProcessProgramGraph(path, files[index]);
          ++fileCount;
          uint64_t f = fileCount;
          if (f && !(f % 8)) {
            std::chrono::microseconds now = Now();
            int usPerGraph = ((now - startTime) / f).count();
            std::cout << "\r\033[K" << f << " of " << files.size()
                      << " files processed (" << usPerGraph << " μs / graph, "
                      << std::setprecision(3)
                      << (f / static_cast<float>(files.size())) * 100 << "%)"
                      << std::flush;
          }
        }
      });
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
