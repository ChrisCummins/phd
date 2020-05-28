#include "programl/db/dataset/poj104_import.h"

#include <stdio.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <pqxx/pqxx>
#include <queue>

#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "boost/filesystem.hpp"
#include "ctpl.h"
#include "labm8/cpp/bazelutil.h"
#include "labm8/cpp/crypto.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "programl/db/db.h"
#include "programl/db/directory_importer.h"
#include "programl/db/file_checksums.h"
#include "programl/db/text_column.h"
#include "programl/ir/llvm/clang.h"
#include "programl/ir/llvm/llvm.h"
#include "programl/proto/ir.pb.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/repo.pb.h"
#include "programl/proto/src.pb.h"
#include "programl/util/filesystem_cache.h"
#include "programl/util/nproc.h"
#include "subprocess/subprocess.hpp"

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;
namespace fs = boost::filesystem;

namespace programl {
namespace db {
namespace dataset {
namespace poj104 {

static fs::path tempdir;

static ir::llvm::Clang compiler("-xc++ -std=c++11");

namespace detail {

uint64_t InsertRepoOrDie(pqxx::work& transaction, const string& sha1) {
  try {
    pqxx::row result(
        transaction.exec1("INSERT INTO repo (\n"
                          "    url, sha1, created_date\n"
                          ") VALUES (\n"
                          "    'https://sites.google.com/site/treebasedcnn/',\n"
                          "    " +
                          transaction.quote(sha1) +
                          ",\n"
                          "    '2014-09-18 00:00:00'\n"
                          ")\n"
                          "RETURNING repo_id"));
    return result[0].as<uint64_t>();
  } catch (pqxx::unique_violation&) {
    LOG(FATAL) << "Nothing to do";
    return 0;
  }
}

void CleanUp() { fs::remove_all(tempdir); }

void PreprocessSrc(string* src) {
  // Clean up declaration of main function. Many are missing a return type
  // declaration, or use incorrect void return type.
  size_t n = src->find("void main");
  if (n != string::npos) {
    src->replace(n, 9, "int main ");
  }

  n = src->find("\nmain");
  if (n != string::npos) {
    src->insert(n + 1, "int ");
  }

  if (!src->compare(0, 4, "main")) {
    src->insert(0, "int ");
  }

  src->insert(0,
              "#include <cstdio>\n"
              "#include <cstdlib>\n"
              "#include <cmath>\n"
              "#include <cstring>\n"
              "#include <iostream>\n"
              "#include <algorithm>\n"
              "#define LEN 512\n"
              "#define MAX_LENGTH 512\n"
              "using namespace std;\n");
}

string ReadFile(const fs::path& path) {
  std::ifstream ifs(path.string().c_str(),
                    std::ios::in | std::ios::binary | std::ios::ate);

  std::ifstream::pos_type fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<char> bytes(fileSize);
  ifs.read(bytes.data(), fileSize);

  return string(bytes.data(), fileSize);
}

Status ImportFile(int _, const fs::path& root, const fs::path& path,
                  const fs::path& outpath, const uint64_t srcId) {
  string src = StripNonUtf8(ReadFile(path));
  PreprocessSrc(&src);

  std::ofstream srcOut(
      absl::StrFormat("%s/%d.SourceFile.pbtxt", outpath.string(), srcId));
  SourceFile srcMessage;
  srcMessage.set_relpath(path.string().substr(root.string().size() + 1));
  srcMessage.set_language(SourceFile::CXX);
  srcMessage.set_text(src);
  srcOut << srcMessage.DebugString();

  const IrList irs = compiler.Compile(src);
  std::ofstream irsOut(
      absl::StrFormat("%s/%d.IrList.pbtxt", outpath.string(), srcId));
  irsOut << irs.DebugString();

  for (int i = 0; i < irs.ir_size(); ++i) {
    std::ofstream irOut(
        absl::StrFormat("%s/%d.%d.ll", outpath.string(), srcId, i));
    irOut << irs.ir(i).text();

    ProgramGraph graph;
    ASSIGN_OR_RETURN(graph, ir::llvm::BuildProgramGraph(irs.ir(i).text());

    std::ofstream graphOut(
        absl::StrFormat("%s/%d.ProgramGraph.%d.pbtxt", outpath.string(), srcId, i));
    graphOut << graph.DebugString();
  }
}

// void StoreResult(pqxx::work& transaction, const Results& results,
//                 uint64_t repoId, const vector<uint64_t>& optIds) {
//  pqxx::row row(
//      transaction.exec1("INSERT INTO src (\n"
//                        "    repo_id,\n"
//                        "    relpath,\n"
//                        "    src_language,\n"
//                        "    sha256\n"
//                        ") VALUES (\n"
//                        "    " +
//                        transaction.quote(repoId) +
//                        ",\n"
//                        "    " +
//                        transaction.quote(results.relpath) +
//                        ",\n"
//                        "    'C',\n"
//                        "    " +
//                        transaction.quote(results.sha256) +
//                        "\n"
//                        ") RETURNING src_id;"));
//  uint64_t srcId = row[0].as<int>();
//  transaction.prepared("insert_src_text")(srcId)(results.src).exec();
//  for (size_t i = 0; i < results.results.size(); ++i) {
//    const auto& result = results.results[i];
//    uint64_t irOptsId = optIds[i];
//    pqxx::row row(
//        transaction.exec1("INSERT INTO ir (ir_type, ir_version, ir_opts_id, "
//                          "src_id, text_size, sha256)\n"
//                          "VALUES (\n"
//                          "    'LLVM',\n"
//                          "    '6_0',\n"
//                          "    " +
//                          transaction.quote(irOptsId) +
//                          ",\n"
//                          "    " +
//                          transaction.quote(srcId) +
//                          ",\n"
//                          "    " +
//                          transaction.quote(result.ir.size()) +
//                          ",\n"
//                          "    " +
//                          transaction.quote(result.sha256) +
//                          "\n"
//                          ") RETURNING ir_id;"));
//    uint64_t irId = row[0].as<uint64_t>();
//    transaction.exec("INSERT INTO ir_text (ir_id, text) VALUES (" +
//                     transaction.quote(irId) + ", " +
//                     transaction.quote(result.ir) + ")");
//  }
//}

size_t ImportFilesOrDie(const fs::path& root, const fs::path& outpathBase) {
  std::chrono::milliseconds startTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch());

  size_t fileCount = 0;
  //  db.connection()->prepare(
  //      "insert_src_text", "INSERT INTO src_text (src_id, text) VALUES ($1,
  //      $2)");
  //  db.connection()->prepare("insert_ir_text",
  //                           "INSERT INTO ir_text (ir_id, text) VALUES ($1,
  //                           $2)");

  ctpl::thread_pool pool(util::GetNumberOfProcessors());
  std::queue<std::future<Status>> futures;

  size_t totalFiles = std::count_if(
      fs::recursive_directory_iterator(root),
      fs::recursive_directory_iterator(),
      static_cast<bool (*)(const fs::path&)>(fs::is_regular_file));

  LOG(INFO) << "Processing " << totalFiles << " files in " << pool.size()
            << " threads ...";

  for (auto it : fs::recursive_directory_iterator(root)) {
    if (!fs::is_regular_file(it)) {
      continue;
    }

    ++fileCount;
    futures.push(
        pool.push(ImportFile, root, it.path(), outpathBase, fileCount));

    // Block until.
    while (futures.size() > pool.size() * static_cast<size_t>(4)) {
      futures.front().get();
      // StoreResult(transaction, futures.front().get(), repoId, optIds);
      futures.pop();
      if (!(fileCount % pool.size())) {
        std::chrono::milliseconds now =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch());
        int msPerGraph = ((now - startTime) / fileCount).count();
        std::cout << "\r\033[K" << fileCount << " of " << totalFiles
                  << " files processed (" << msPerGraph << " ms / src, "
                  << std::setprecision(3)
                  << (fileCount / static_cast<float>(totalFiles)) * 100 << "%)"
                  << std::flush;
      }
    }
  }

  while (!futures.empty()) {
    futures.front().get();
    //    StoreResult(transaction, futures.front().get(), repoId, optIds);
    futures.pop();
    // ++fileCount;
    std::chrono::milliseconds now =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    int msPerGraph = ((now - startTime) / fileCount).count();
    std::cout << "\r\033[K" << fileCount << " of " << totalFiles
              << " files processed (" << msPerGraph << " ms / src, "
              << std::setprecision(3)
              << (fileCount / static_cast<float>(totalFiles)) * 100 << "%)"
              << std::flush;
  }
  std::cout << std::endl;

  return fileCount;
}

}  // namespace detail

size_t CreateDataset(const string& url, const fs::path& outputPath) {
  fs::create_directories(outputPath);

  tempdir = fs::temp_directory_path() / fs::unique_path();
  CHECK(fs::create_directory(tempdir));
  std::atexit(detail::CleanUp);

  util::FilesystemCache fileCache;

  const fs::path archive = fileCache[{"poj104.tar.gz"}];
  if (!fs::is_regular_file(archive)) {
    LOG(INFO) << "Downloading dataset from " << url << " ...";
    string wget = "wget '" + url + "' -O " + archive.string();
    CHECK(!system(wget.c_str())) << "failed: $ " << wget;
    CHECK(fs::is_regular_file(archive));
  }
  string sha1 = programl::util::Sha1(archive).ValueOrDie();

  LOG(INFO) << "Extracting dataset archive ...";
  string tar = "tar -xf " + archive.string() + " -C " + tempdir.string();
  CHECK(!system(tar.c_str())) << "failed: $ " << tar;

  // uint64_t repoId = detail::InsertRepoOrDie(transaction, sha1);
  Repo repo;
  repo.set_url("https://sites.google.com/site/treebasedcnn/");
  repo.set_sha1(sha1);
  absl::CivilSecond ct(2014, 9, 18, 0, 0, 0);
  absl::Time created = absl::FromCivil(ct, absl::UTCTimeZone());
  repo.set_created_ms_timestamp(
      absl::ToInt64Milliseconds(absl::time_internal::ToUnixDuration(created)));
  {
    std::ofstream repoOut((outputPath / "Repo.pbtxt").string());
    repoOut << repo.DebugString();
  }

  fs::path root = tempdir / "ProgramData";
  CHECK(fs::is_directory(root));

  // vector<uint64_t> optIds = detail::InsertOptsOrDie(transaction);

  size_t fileCount = detail::ImportFilesOrDie(root, outputPath);
  CHECK(fileCount);

  // transaction.commit();

  LOG(INFO) << "Imported " << fileCount << " files";

  return fileCount;
}

}  // namespace poj104
}  // namespace dataset
}  // namespace db
}  // namespace programl
