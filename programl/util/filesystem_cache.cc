#include "programl/util/filesystem_cache.h"

#include "labm8/cpp/fsutil.h"

namespace programl {
namespace util {

FilesystemCache::FilesystemCache()
    : root_(labm8::fsutil::GetHomeDirectoryOrDie() / ".cache" / "programl") {
  fs::create_directories(root_);
};

fs::path FilesystemCache::operator[](const vector<string>& components) const {
  fs::path path(root_);
  CHECK(components.size());
  for (auto& component : components) {
    path /= component;
  }
  return path;
}

}  // namespace util
}  // namespace programl
