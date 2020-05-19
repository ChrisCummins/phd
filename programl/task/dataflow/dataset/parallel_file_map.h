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
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "boost/filesystem.hpp"
#include "labm8/cpp/app.h"

namespace fs = boost::filesystem;
using std::string;
using std::vector;

DECLARE_int32(limit);

namespace programl {
namespace task {
namespace dataflow {

vector<fs::path> EnumerateProgramGraphFiles(const fs::path& root);

inline std::chrono::milliseconds Now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
}

// chunk_size: The size of file path chunks to execute in worker
// thread inner loops. A larger chunk size creates more infrequent
// status updates.
template <void (*ProcessOne)(const fs::path&, const fs::path&),
          size_t chunkSize = 16>
void ParallelFileMap(const fs::path& path, const vector<fs::path>& files) {
  std::chrono::milliseconds startTime = Now();

  std::atomic_uint64_t fileCount{0};

  const size_t n = FLAGS_limit
                       ? std::min(size_t(files.size()), size_t(FLAGS_limit))
                       : files.size();

#pragma omp parallel for
  for (size_t j = 0; j < n; j += chunkSize) {
    for (size_t i = j; i < std::min(n, j + chunkSize); ++i) {
      ProcessOne(path, files[i]);
    }
    fileCount += chunkSize;
    uint64_t localFileCount = fileCount;
    std::chrono::milliseconds now = Now();
    int msPerGraph = ((now - startTime) / localFileCount).count();
    std::cout << "\r\033[K" << localFileCount << " of " << n
              << " files processed (" << msPerGraph << " ms / file, "
              << std::setprecision(3)
              << (localFileCount / static_cast<float>(n)) * 100 << "%)"
              << std::flush;
  }
  std::cout << std::endl;
}

}  // namespace dataflow
}  // namespace task
}  // namespace programl
