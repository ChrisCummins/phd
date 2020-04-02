#pragma once

#include <pqxx/pqxx>
#include "boost/filesystem.hpp"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "programl/db/db.h"

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;
namespace fs = boost::filesystem;

namespace programl {
namespace db {

namespace detail {

string ReadFile(const fs::path &path);

}  // namespace detail

[[nodiscard]] StatusOr<uint64_t> ImportSourceFromFile(pqxx::work &transaction,
                                                      const fs::path &root,
                                                      const fs::path &path,
                                                      uint64_t repoId);

[[nodiscard]] Status ImportFromDirectory(Database &db, pqxx::work &transaction,
                                         const fs::path root, uint64_t repoId,
                                         uint64_t *exportedFileCount);

}  // namespace db
}  // namespace programl
