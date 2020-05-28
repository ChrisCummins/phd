#include <stdio.h>
#include <array>
#include <iostream>
#include <pqxx/pqxx>
#include <vector>
#include "labm8/cpp/app.h"
#include "labm8/cpp/bazelutil.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "programl/proto/program_graph_options.pb.h"
#include "subprocess/subprocess.hpp"

const char* usage = R"(foo
)";

DEFINE_int32(batch_size, 10000, "Foo");

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;

const static string kClangPath =
    labm8::BazelDataPathOrDie("llvm_linux/bin/clang++").string();
const string kClangCommand =
    kClangPath + " -emit-llvm -c -S -xc++ - -o - -Wno-everything -std=c++11";

const std::vector<string> kClangOpts{
    "-O0",
    "-O1",
    "-O2",
    "-O3",
};

StatusOr<pqxx::connection*> Connect(const string& string) {
  try {
    pqxx::connection* con = new pqxx::connection(string);
    CHECK(con) << "Failed to allocate connection";
    CHECK(con->is_open()) << "Can't open database";
    LOG(INFO) << "Connected to database";
    return std::move(con);
  } catch (const std::exception& e) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Failed to connect to database");  //, e.what());
  }
}

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

void ProcessSource(pqxx::work& transaction, uint64_t srcId, string* src) {
  LOG(INFO) << srcId;
  PreprocessSrc(src);

  for (const auto& opt : kClangOpts) {
    auto cmd = subprocess::Popen(kClangCommand + " " + opt,
                                 subprocess::input{subprocess::PIPE},
                                 subprocess::output{subprocess::PIPE});

    auto out = cmd.communicate(src->c_str(), src->size());
    auto ir = string(out.first.buf.begin(), out.first.buf.end());

    if (cmd.retcode()) {
      LOG(INFO) << "Failed to process src #" << srcId;
    }
  }
}

uint64_t InsertClangOpts(pqxx::work& transaction, const string& opts) {
  pqxx::row result(
      transaction.exec1("INSERT OR UPDATE INTO ir_opts (\n"
                        "    url, sha1, created_date\n"
                        ") VALUES (\n"
                        "    'https://sites.google.com/site/treebasedcnn/',\n"
                        ")\n"
                        "RETURNING repo_id;"));
  return result[0].as<uint64_t>();
}

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  auto C = Connect("dbname = programl").ValueOrDie();

  pqxx::work transaction(*C);

  programl::ProgramGraphOptions options;

  string query =
      "SELECT src_text.src_id, text "
      "FROM src "
      "LEFT JOIN src_text "
      "  ON src.src_id=src_text.src_id";

  pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned>
      cursor(transaction, query, "cursor", /*hold=*/false);

  uint64_t idx = 0;
  while (true) {
    pqxx::result result = cursor.retrieve(idx, idx + FLAGS_batch_size);
    if (result.empty()) {
      break;
    }

    for (pqxx::result::const_iterator row : result) {
      uint64_t id = row["src_id"].as<uint64_t>();
      string text = row["text"].as<string>();
      ProcessSource(transaction, id, &text);
      ++idx;
    }

    LOG(INFO) << "idx " << idx;
  }

  // transaction.commit();

  LOG(INFO) << "Done";
  return 0;
}
