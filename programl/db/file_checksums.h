#pragma once

#include "boost/filesystem.hpp"
#include "labm8/cpp/statusor.h"

using labm8::StatusOr;
namespace fs = boost::filesystem;

namespace programl {
namespace db {

StatusOr<string> Sha1(const fs::path &path);
StatusOr<string> Sha256(const fs::path &path);

}  // namespace db
}  // namespace programl
