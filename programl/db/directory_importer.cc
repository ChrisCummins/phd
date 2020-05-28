// https://sites.google.com/site/treebasedcnn/
#include "programl/db/directory_importer.h"

#include <stdio.h>
#include <iostream>
#include <pqxx/pqxx>
#include "boost/filesystem.hpp"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/statusor.h"
#include "programl/db/file_checksums.h"
#include "programl/db/text_column.h"

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;
namespace fs = boost::filesystem;

namespace programl {
namespace db {

namespace detail {

string ReadFile(const fs::path &path) {
  std::ifstream ifs(path.string().c_str(),
                    std::ios::in | std::ios::binary | std::ios::ate);

  std::ifstream::pos_type fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<char> bytes(fileSize);
  ifs.read(bytes.data(), fileSize);

  return string(bytes.data(), fileSize);
}

}  // namespace detail

StatusOr<uint64_t> ImportSourceFromFile(pqxx::work &transaction,
                                        const fs::path &root,
                                        const fs::path &path, uint64_t repoId) {
  string relpath = path.string().substr(root.string().size() + 1);
  string sha256;
  ASSIGN_OR_RETURN(sha256, Sha256(path));
  string text = StripNonUtf8(detail::ReadFile(path));

  pqxx::row result(
      transaction.exec1("INSERT INTO src (\n"
                        "    repo_id,\n"
                        "    relpath,\n"
                        "    src_language,\n"
                        "    sha256\n"
                        ") VALUES (\n"
                        "    " +
                        transaction.quote(repoId) +
                        ",\n"
                        "    " +
                        transaction.quote(relpath) +
                        ",\n"
                        "    'C',\n"
                        "    " +
                        transaction.quote(sha256) +
                        "\n"
                        ") RETURNING src_id;"));
  uint64_t srcId = result[0].as<int>();
  transaction.prepared("insert_text")(srcId)(text).exec();

  return srcId;
}

Status ImportFromDirectory(Database &db, pqxx::work &transaction,
                           const fs::path root, uint64_t repoId,
                           uint64_t *exportedFileCount) {
  db.connection()->prepare(
      "insert_text", "INSERT INTO src_text (src_id, text) VALUES ($1, $2)");

  for (auto it : fs::recursive_directory_iterator(root)) {
    if (!fs::is_regular_file(it)) {
      continue;
    }
    ++(*exportedFileCount);
    RETURN_IF_ERROR(
        ImportSourceFromFile(transaction, root, it.path(), repoId).status());
    LOG_IF(INFO, !((*exportedFileCount) % 100))
        << *exportedFileCount << " files imported ...";
  }

  return Status::OK;
}

}  // namespace db
}  // namespace programl
