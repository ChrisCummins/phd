#pragma once

#include <pqxx/pqxx>
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;

namespace programl {
namespace db {

// Example usage:
//
//    Database db(FLAGS_db);
//    CHECK(db.Connect().ok());
//
//    db.connection().prepare(...);
//    pqxx::transaction transcation = db.transaction();
class Database {
 public:
  Database(const string& uri) : uri_(uri), connection_(nullptr){};

  [[nodiscard]] Status Connect();

  void ConnectOrDie();

  void Disconnect();

  pqxx::connection* connection();

 private:
  const string uri_;
  pqxx::connection* connection_;
};

}  // namespace db
}  // namespace programl
