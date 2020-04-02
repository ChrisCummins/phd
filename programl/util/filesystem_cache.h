#pragma once

#include <cstdlib>
#include <vector>
#include "boost/filesystem.hpp"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/string.h"

using std::vector;
namespace fs = boost::filesystem;

namespace programl {
namespace util {

class FilesystemCache {
 public:
  FilesystemCache();

  fs::path operator[](const vector<string>& components) const;

 private:
  const fs::path root_;
};

}  // namespace util
}  // namespace programl
