#include "programl/db/file_checksums.h"

#include <stdio.h>
#include <array>
#include <iostream>
#include <memory>
#include "boost/filesystem.hpp"
#include "labm8/cpp/status.h"

using labm8::Status;
namespace error = labm8::error;
namespace fs = boost::filesystem;

namespace programl {
namespace db {

namespace {

StatusOr<string> Exec(const string &cmd, size_t expectedLength) {
  std::array<char, 128> buffer;
  string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    return Status(error::Code::FAILED_PRECONDITION, "popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  if (result.size() != expectedLength) {
    return Status(error::Code::FAILED_PRECONDITION, "invalid return length");
  }
  return result.erase(result.size() - 1);
}

}  // anonymous namespace

StatusOr<string> Sha1(const fs::path &path) {
  return Exec("sha1sum " + path.string() + " | cut -f1 -d' '", 41);
}

StatusOr<string> Sha256(const fs::path &path) {
  return Exec("sha256sum " + path.string() + " | cut -f1 -d' '", 65);
}

}  // namespace db
}  // namespace programl
